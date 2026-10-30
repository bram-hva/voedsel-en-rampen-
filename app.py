"""Wat kost een droogte? Casus 8 (voedselzekerheid en natuurrampen), richting A.

Draaien met:   C:\\Users\\Waldo\\anaconda3\\python.exe -m streamlit run app.py

Bronnen (staan lokaal in data/, opgehaald in verkennen.ipynb):
    FAOSTAT  graanproductie in ton per land, gewas en jaar (3 downloads)
    FAOSTAT  landcodes (M49, ISO2, ISO3)
    EM-DAT   rampenregister 1970-2021 (Kaggle-mirror)
    GitHub   middelpunt per land (gavinr/world-countries-centroids)
"""

import math

import folium
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Wat kost een droogte?", layout="wide")

# ------------------------------------------------------------ data laden ---
@st.cache_data
def laad_productie():
    # Drie FAO-downloads onder elkaar plakken
    fao = pd.concat([
        pd.read_csv("data/fao_granen_totaal.csv"),
        pd.read_csv("data/fao_tarwe_mais_rijst.csv"),
        pd.read_csv("data/fao_sorghum_gierst_gerst.csv"),
    ], ignore_index=True)

    # Vertaaltabel M49 -> ISO3 (alleen echte landen, geen regiocodes als 2.03)
    landcodes = pd.read_csv("data/fao_landcodes.csv")
    landcodes = landcodes[landcodes["M49 Code"] % 1 == 0]
    landcodes = landcodes[["M49 Code", "ISO3 Code"]]

    fao = fao.merge(landcodes, left_on="Area Code (M49)", right_on="M49 Code",
                    how="left", validate="many_to_one")

    productie = fao[["ISO3 Code", "Area", "Item", "Year", "Value"]]
    productie = productie.rename(columns={
        "ISO3 Code": "ISO3", "Area": "Land", "Item": "Gewas",
        "Year": "Jaar", "Value": "Ton",
    })

    # Lege productiecijfers weg (niet op 0 zetten!)
    productie = productie.dropna(subset=["Ton"])

    # Oude FAO-landen doorzetten onder de code van het huidige land
    productie["ISO3"] = productie["ISO3"].replace({"F62": "ETH", "F206": "SDN"})
    return productie


@st.cache_data
def laad_rampen():
    rampen = pd.read_csv("data/emdat_rampen_1970_2021.csv")
    rampen = rampen[["ISO", "Continent", "Start Year", "Disaster Type"]]
    rampen = rampen.rename(columns={
        "ISO": "ISO3", "Continent": "Werelddeel",
        "Start Year": "Jaar", "Disaster Type": "Ramptype",
    })
    # Sommige ramptypes hebben een spatie aan het eind ("Extreme temperature ")
    rampen["Ramptype"] = rampen["Ramptype"].str.strip()
    return rampen


@st.cache_data
def laad_centroiden():
    """Middelpunt (lat/lon) per land, gekoppeld via ISO2 -> ISO3.
    keep_default_na=False: anders wordt Namibie ("NA") een lege waarde."""
    landcodes = pd.read_csv("data/fao_landcodes.csv", dtype=str, keep_default_na=False)
    landcodes = landcodes[["ISO2 Code", "ISO3 Code"]]

    centroiden = pd.read_csv("data/landen_centroiden.csv", keep_default_na=False)
    centroiden = centroiden.rename(columns={
        "ISO": "ISO2 Code", "latitude": "Lat", "longitude": "Lon"})
    centroiden["Lat"] = pd.to_numeric(centroiden["Lat"], errors="coerce")
    centroiden["Lon"] = pd.to_numeric(centroiden["Lon"], errors="coerce")

    centroiden = centroiden.merge(landcodes, on="ISO2 Code", how="inner")
    centroiden = centroiden.rename(columns={"ISO3 Code": "ISO3"})
    return centroiden[["ISO3", "Lat", "Lon"]].drop_duplicates("ISO3")


# ------------------------------------------------------------- bewerken ---
def naar_lang(tabel, naam):
    """Van breed (een kolom per jaar) terug naar lang (een rij per jaar)."""
    return tabel.reset_index().melt(id_vars=["ISO3", "Gewas"],
                                    var_name="Jaar", value_name=naam)


@st.cache_data
def bereken_veranderingen(productie):
    # Lang -> breed: een rij per land en gewas, een kolom per jaar
    breed = productie.pivot(index=["ISO3", "Gewas"], columns="Jaar", values="Ton")
    jaar_ervoor = breed.shift(1, axis=1)
    jaar_erna = breed.shift(-1, axis=1)

    # De afgeleide kolommen
    klap = (breed - jaar_ervoor) / jaar_ervoor * 100
    verandering = (jaar_erna - jaar_ervoor) / jaar_ervoor * 100

    # Breed -> lang, en de drie tabellen naast elkaar zetten
    lang = naar_lang(verandering, "Verandering_pct")
    lang = lang.merge(naar_lang(klap, "Klap_pct"), on=["ISO3", "Gewas", "Jaar"])
    lang = lang.merge(naar_lang(jaar_ervoor, "Ton_ervoor"), on=["ISO3", "Gewas", "Jaar"])

    # Delen door nul (inf) en lege veranderingen eruit
    lang = lang.replace([float("inf"), float("-inf")], float("nan"))
    lang = lang.dropna(subset=["Verandering_pct"])

    # Hersteld: een jaar na de ramp minstens zoveel als het jaar ervoor
    lang["Hersteld"] = lang["Verandering_pct"] >= 0
    return lang


@st.cache_data
def koppel_rampen(veranderingen, rampen, ramptype):
    # Alleen de jaren waarin EM-DAT meetelt
    veranderingen = veranderingen[veranderingen["Jaar"].between(1970, 2021)]

    # Een rij per land en jaar met minstens een ramp van dit type
    gekozen = rampen[rampen["Ramptype"] == ramptype]
    rampjaren = gekozen.groupby(["ISO3", "Jaar"]).size().reset_index(name="Aantal_rampen")

    analyse = veranderingen.merge(rampjaren, on=["ISO3", "Jaar"],
                                  how="left", validate="many_to_one")
    analyse["Rampjaar"] = analyse["Aantal_rampen"].notna()

    # Jaren direct vóór of ná een rampjaar. Het jaar na een droogte is vaak
    # een inhaaljaar; tel je dat als 'normaal', dan lijkt normaal beter dan het is.
    jaar_na_ramp = rampjaren[["ISO3", "Jaar"]].copy()
    jaar_na_ramp["Jaar"] = jaar_na_ramp["Jaar"] + 1
    jaar_voor_ramp = rampjaren[["ISO3", "Jaar"]].copy()
    jaar_voor_ramp["Jaar"] = jaar_voor_ramp["Jaar"] - 1
    buurjaren = pd.concat([jaar_na_ramp, jaar_voor_ramp]).drop_duplicates()
    buurjaren["Buurjaar"] = True

    analyse = analyse.merge(buurjaren, on=["ISO3", "Jaar"], how="left", validate="many_to_one")
    analyse["Buurjaar"] = analyse["Buurjaar"].notna()

    analyse["Groep"] = "Normaal jaar"
    analyse.loc[analyse["Buurjaar"], "Groep"] = "Jaar naast ramp"
    analyse.loc[analyse["Rampjaar"], "Groep"] = "Jaar met ramp"

    # Werelddeel per land, uit EM-DAT (een rij per land)
    werelddelen = rampen.drop_duplicates("ISO3")[["ISO3", "Werelddeel"]]
    analyse = analyse.merge(werelddelen, on="ISO3", how="left", validate="many_to_one")
    return analyse


def filter_data(data, gewas, drempel, periode, werelddeel):
    """Alle filters op één plek: gewas, minimale productie, periode en werelddeel.
    Een nieuw filter voeg je hier toe, dan werkt het overal."""
    selectie = data[
        (data["Gewas"] == gewas)
        & (data["Ton_ervoor"] >= drempel)
        & (data["Jaar"].between(periode[0], periode[1]))
    ]
    if werelddeel is not None:
        selectie = selectie[selectie["Werelddeel"] == werelddeel]
    return selectie


def splits_groepen(data, zuiver):
    """Splits in rampjaren en de vergelijkingsgroep.
    zuiver=True: jaren direct naast een ramp tellen niet als normaal jaar."""
    ramp = data[data["Rampjaar"]]
    if zuiver:
        normaal = data[data["Groep"] == "Normaal jaar"]
    else:
        normaal = data[~data["Rampjaar"]]
    return ramp, normaal


# ----------------------------------------------------------- statistiek ---
def vergelijk_groepen(ramp, normaal, kolom):
    """Verschil in gemiddelde tussen twee groepen, met een 95%-interval."""
    a = ramp[kolom].dropna()
    b = normaal[kolom].dropna()
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan"), float("nan")

    verschil = a.mean() - b.mean()
    standaardfout = math.sqrt(a.var() / len(a) + b.var() / len(b))
    laag = verschil - 1.96 * standaardfout
    hoog = verschil + 1.96 * standaardfout
    return verschil, laag, hoog


def toets(data, kolom, zuiver):
    """Splits in rampjaren en normale jaren en vergelijk die.
    Geeft het verschil, het interval en het aantal rampjaren terug."""
    ramp, normaal = splits_groepen(data, zuiver)
    verschil, laag, hoog = vergelijk_groepen(ramp, normaal, kolom)
    return verschil, laag, hoog, len(ramp)


def gemiddelde_met_interval(waarden):
    """Gemiddelde van een kolom, met een 95%-interval."""
    waarden = waarden.dropna()
    if len(waarden) < 2:
        return float("nan"), float("nan"), float("nan")
    gemiddelde = waarden.mean()
    standaardfout = waarden.std() / math.sqrt(len(waarden))
    return gemiddelde, gemiddelde - 1.96 * standaardfout, gemiddelde + 1.96 * standaardfout


def beschrijf(verschil, laag, hoog):
    """Een verschil met interval in gewone taal: minder, meer, of niet aantoonbaar."""
    interval = f"95%-interval {laag:.1f} tot {hoog:.1f}"
    if hoog < 0:
        return f"**{abs(verschil):.1f} procentpunt minder** groei dan normaal ({interval})"
    if laag > 0:
        return f"**{verschil:.1f} procentpunt méér** groei dan normaal ({interval})"
    return f"**geen aantoonbaar verschil** met normaal ({verschil:+.1f} procentpunt, {interval})"


def straal(waarde, laagste, hoogste, klein=5, groot=22):
    """Waarde naar straal in pixels. Via de wortel, want het oog
    vergelijkt oppervlakte en geen straal."""
    if hoogste == laagste:
        return (klein + groot) / 2
    deel = (waarde - laagste) / (hoogste - laagste)
    return klein + (groot - klein) * deel ** 0.5


# ------------------------------------------------------------ vaste waarden ---
RAMPTYPES = {
    "Droogte": "Drought",
    "Overstroming": "Flood",
    "Storm": "Storm",
    "Extreme temperatuur": "Extreme temperature",
}

GEWASSEN = {
    "Granen totaal": "Cereals, primary",
    "Tarwe": "Wheat",
    "Maïs": "Maize (corn)",
    "Rijst": "Rice",
    "Sorghum": "Sorghum",
    "Gierst": "Millet",
    "Gerst": "Barley",
}

# Vanaf het jaar vóór de ramp meten we tot ... Beide kolommen zijn zelf afgeleid.
MEETMOMENTEN = {
    "het rampjaar zelf (de klap)": "Klap_pct",
    "het jaar ná de ramp (herstel)": "Verandering_pct",
}

# Astitel per meetmoment, zodat een grafiek zegt wat hij laat zien
AS_TITELS = {
    "Klap_pct": "Rampjaar t.o.v. jaar ervóór (%)",
    "Verandering_pct": "Jaar ná de ramp t.o.v. jaar ervóór (%)",
}

DREMPELS = [0, 10_000, 100_000, 1_000_000]

AS_GRENS = 60               # histogram toont -60% tot +60%
MIN_RAMPJAREN_GEWAS = 10    # gewassen met minder rampjaren niet in de gewasgrafiek
AANTAL_SLECHTSTE = 5        # zoveel landen noemen bij 'slechtste herstel'

KLEUR_RAMP = "#C2410C"     # oranjerood: jaren met een ramp
KLEUR_NORMAAL = "#94A3B8"  # grijs: de vergelijkingsgroep
KLEUR_LIJN = "#334155"     # donkergrijs: productie door de tijd
KLEUR_LICHT = "#F4B79B"    # licht oranje: de rest, naast het gekozen item

# Bleke ondergrond zonder API-sleutel. Esri wil {z}/{y}/{x}, niet {z}/{x}/{y}.
ESRI = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
        "Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}")

# Vier vaste klassen voor herstel: oranje = slecht, blauw = goed.
KLASSEN = {
    "0-25% hersteld": "#C2410C",
    "25-50% hersteld": "#F59E0B",
    "50-75% hersteld": "#60A5FA",
    "75-100% hersteld": "#1D4ED8",
}

# Nederlandse naam -> naam in de kolom Continent van EM-DAT. None = geen filter.
WERELDDELEN = {
    "Wereld": None,
    "Afrika": "Africa",
    "Amerika": "Americas",
    "Azië": "Asia",
    "Europa": "Europe",
    "Oceanië": "Oceania",
}


# --------------------------------------------------------------- data inlezen ---
try:
    productie = laad_productie()
    rampen = laad_rampen()
    centroiden = laad_centroiden()
except FileNotFoundError as fout:
    st.error(f"Databestand niet gevonden: {fout.filename}. Staat de map data/ naast app.py?")
    st.stop()

veranderingen = bereken_veranderingen(productie)

# Een landnaam per ISO3-code. Sorteren op jaar en de laatste houden, zodat
# Ethiopie "Ethiopia" heet en niet "Ethiopia PDR".
landnamen = (productie.sort_values("Jaar")
             .drop_duplicates("ISO3", keep="last")[["ISO3", "Land"]])


# -------------------------------------------------------------------- sidebar ---
st.sidebar.header("Filters")

gekozen_ramp = st.sidebar.selectbox("Soort ramp", list(RAMPTYPES.keys()))
gekozen_gewas = st.sidebar.selectbox("Gewas", list(GEWASSEN.keys()))
gekozen_werelddeel = st.sidebar.selectbox(
    "Werelddeel", list(WERELDDELEN.keys()),
    help="Filtert alles: de cijfers, de grafieken, de kaart en het advies.",
)
meetmoment = st.sidebar.radio(
    "Vergelijk het jaar vóór de ramp met",
    list(MEETMOMENTEN.keys()),
    help="'Het rampjaar zelf' meet de klap. 'Het jaar ná de ramp' is de definitie "
         "uit de opdracht en meet vooral of het land hersteld is.",
)
minimale_productie = st.sidebar.select_slider(
    "Minimale productie in het jaar ervóór (ton)",
    options=DREMPELS,
    value=100_000,
    help="Bij kleine producenten worden percentages extreem: van 4 naar 700 ton is +17.000%.",
)
periode = st.sidebar.slider("Periode", 1970, 2021, (1970, 2021))
zuiver = st.sidebar.checkbox(
    "Jaren direct vóór en ná een ramp niet als normaal jaar tellen",
    value=True,
    help="Het jaar na een ramp is vaak een inhaaljaar. Telt dat mee als 'normaal', "
         "dan lijkt een normaal jaar beter dan het is en het verlies groter.",
)

st.sidebar.divider()
st.sidebar.caption("Bronnen: FAOSTAT (productie in ton, gedownload 28-10-2026) "
                   "en EM-DAT 1970-2021 via Kaggle (rampenregister).")

maat = MEETMOMENTEN[meetmoment]
werelddeel = WERELDDELEN[gekozen_werelddeel]
ramp_naam = gekozen_ramp.lower()
wanneer = "in het rampjaar zelf" if maat == "Klap_pct" else "tot een jaar na de ramp"


# ---------------------------------------------------- koppelen en filteren ---
analyse = koppel_rampen(veranderingen, rampen, RAMPTYPES[gekozen_ramp])
analyse = analyse.merge(landnamen, on="ISO3", how="left")

# Alle filters op één plek; alles hieronder gebruikt `selectie`
selectie = filter_data(analyse, GEWASSEN[gekozen_gewas], minimale_productie, periode,
                       werelddeel)
ramp, normaal = splits_groepen(selectie, zuiver)

if len(ramp) < 3 or len(normaal) < 3:
    st.warning("Te weinig waarnemingen in deze selectie. Verruim de filters.")
    st.stop()

# Hoe groeit elk land normaal? Zo vergelijken we een rampjaar met hetzelfde
# land, en niet met de rest van de wereld. Ethiopie groeit bijvoorbeeld hard.
normale_groei = normaal.groupby("ISO3")[maat].median().reset_index(name="Normale_groei")
ramp = ramp.merge(normale_groei, on="ISO3", how="left")
ramp["Afwijking"] = ramp[maat] - ramp["Normale_groei"]

# Precies de twee groepen die we vergelijken, voor de grafieken. Staat het vinkje
# uit, dan zitten er buurjaren in 'normaal'; die heten in de grafiek ook 'Normaal jaar'.
vergeleken = pd.concat([ramp, normaal])
vergeleken["Groep"] = vergeleken["Rampjaar"].map({True: "Jaar met ramp", False: "Normaal jaar"})

verschil, laag, hoog = vergelijk_groepen(ramp, normaal, maat)
afwijking, afwijking_laag, afwijking_hoog = gemiddelde_met_interval(ramp["Afwijking"])


# ------------------------------------------------------------ eerste laag ---
st.title(f"Wat kost een {ramp_naam}?")
drempel_tekst = f"{minimale_productie:,}".replace(",", ".")
st.caption(f"{gekozen_gewas} · {gekozen_werelddeel} · {periode[0]}-{periode[1]} · landen met minstens "
           f"{drempel_tekst} ton in het jaar ervóór · verandering {wanneer}"
           + (" · zonder jaren direct rond een ramp" if zuiver else ""))

kolom1, kolom2, kolom3, kolom4 = st.columns(4)
kolom1.metric("Rampjaren in selectie", len(ramp),
              help="Combinaties van land en jaar met minstens één geregistreerde ramp.")
kolom1.caption(f"tegen {len(normaal)} normale jaren")
kolom2.metric("Verschil met een normaal jaar", f"{verschil:+.1f} pp",
              help="Gemiddelde verandering in rampjaren min die in normale jaren, in procentpunt.")
kolom2.caption(f"95%-interval {laag:.1f} tot {hoog:.1f}")
kolom3.metric("Verschil met de eigen normale groei", f"{afwijking:+.1f} pp",
              help="Per rampjaar: verandering min de mediane verandering van datzelfde land "
                   "in normale jaren. Zo telt een snelgroeiend land niet als 'geen schade'.")
kolom3.caption(f"95%-interval {afwijking_laag:.1f} tot {afwijking_hoog:.1f}")
kolom4.metric("Binnen een jaar hersteld", f"{ramp['Hersteld'].mean():.0%}",
              help="Aandeel rampjaren waarna de oogst een jaar later minstens zo groot was "
                   "als in het jaar vóór de ramp.")
kolom4.caption("oogst jaar ná ≥ jaar vóór")

st.info(
    f"**Conclusie:** {wanneer} geeft een {ramp_naam} {beschrijf(verschil, laag, hoog)}. "
    f"Vergeleken met de normale groei van hetzelfde land: "
    f"{beschrijf(afwijking, afwijking_laag, afwijking_hoog)}."
)

tab_effect, tab_kaart, tab_land, tab_conclusie, tab_data = st.tabs(
    ["Effect", "Op de kaart", "Per land", "Conclusie en advies", "Over de data"])


# ----------------------------------------------------------------- effect ---
with tab_effect:
    st.subheader("Is het verschil groter dan toeval?")

    if hoog < 0 and afwijking_hoog < 0:
        st.success("Beide vergelijkingen geven verlies, en beide intervallen liggen onder 0: "
                   "het verlies is niet aan toeval toe te schrijven.")
    elif laag > 0 and afwijking_laag > 0:
        st.info("Beide intervallen liggen boven 0: in rampjaren groeit de productie juist méér.")
    elif hoog < 0 or afwijking_hoog < 0:
        st.warning("Maar één van de twee vergelijkingen geeft aantoonbaar verlies: "
                   "het verlies is niet overtuigend.")
    else:
        st.warning("Bij deze selectie is geen verlies aantoonbaar.")

    links, rechts = st.columns(2)

    with links:
        mediaan_ramp = ramp[maat].median()
        mediaan_normaal = normaal[maat].median()

        # Verdeling van beide groepen over elkaar, als percentage van de groep
        zichtbaar = vergeleken[vergeleken[maat].between(-AS_GRENS, AS_GRENS)]
        figuur = px.histogram(
            zichtbaar, x=maat, color="Groep", histnorm="percent",
            barmode="overlay", nbins=60, opacity=0.6,
            color_discrete_map={"Jaar met ramp": KLEUR_RAMP, "Normaal jaar": KLEUR_NORMAAL},
            title=f"Mediaan {mediaan_ramp:+.1f}% bij een {ramp_naam}, "
                  f"{mediaan_normaal:+.1f}% in een normaal jaar",
            labels={maat: AS_TITELS[maat], "Groep": ""},
        )
        figuur.add_vline(x=0, line_color="black", line_width=1)
        figuur.update_layout(yaxis_title="% van de jaren in die groep",
                             legend=dict(orientation="h", y=1.08))
        st.plotly_chart(figuur)
        st.caption(f"{len(vergeleken) - len(zichtbaar)} waarnemingen buiten -{AS_GRENS}% tot "
                   f"+{AS_GRENS}% zijn niet getekend, maar tellen wel mee in de berekening.")

    with rechts:
        # Per gewas hetzelfde verschil met interval, met dezelfde drempel en periode
        rijen_gewas = []
        for naam, code in GEWASSEN.items():
            deel = filter_data(analyse, code, minimale_productie, periode, werelddeel)
            verschil_g, laag_g, hoog_g, aantal_g = toets(deel, maat, zuiver)
            if aantal_g >= MIN_RAMPJAREN_GEWAS:
                rijen_gewas.append({
                    "Gewas": naam, "Verschil": verschil_g, "Hoog": hoog_g,
                    "Marge": hoog_g - verschil_g, "Rampjaren": aantal_g,
                    "Keuze": "Gekozen gewas" if naam == gekozen_gewas else "Ander gewas",
                })

        if rijen_gewas:
            per_gewas = pd.DataFrame(rijen_gewas).sort_values("Verschil")
            aangetoond = per_gewas[per_gewas["Hoog"] < 0]
            if aangetoond.empty:
                titel_gewas = f"Bij geen enkel gewas is verlies door een {ramp_naam} aangetoond"
            else:
                grootste = aangetoond.iloc[0]
                titel_gewas = (f"{grootste['Gewas']} lijdt het meest: "
                               f"{grootste['Verschil']:.1f} procentpunt")

            figuur = px.bar(
                per_gewas, x="Verschil", y="Gewas", orientation="h", error_x="Marge",
                color="Keuze",
                color_discrete_map={"Gekozen gewas": KLEUR_RAMP, "Ander gewas": KLEUR_LICHT},
                hover_data={"Rampjaren": True, "Marge": ":.1f", "Keuze": False, "Hoog": False},
                title=titel_gewas,
                labels={"Verschil": "Verschil met normaal jaar (procentpunt)", "Gewas": ""},
            )
            figuur.add_vline(x=0, line_color="black", line_width=1)
            figuur.update_yaxes(autorange="reversed")
            figuur.update_layout(showlegend=False)
            st.plotly_chart(figuur)
            st.caption("Het streepje is het 95%-interval: raakt het de nul, dan is het "
                       f"verschil voor dat gewas niet aangetoond. Gewassen met minder dan "
                       f"{MIN_RAMPJAREN_GEWAS} rampjaren staan er niet bij.")
        else:
            st.info(f"Geen enkel gewas heeft hier {MIN_RAMPJAREN_GEWAS} of meer rampjaren; "
                    "te weinig om per gewas te vergelijken.")

    # Robuustheid: blijft de uitkomst overeind bij een andere drempel?
    st.subheader("Hangt de uitkomst af van de drempel?")
    robuust = []
    for drempel in DREMPELS:
        deel = filter_data(analyse, GEWASSEN[gekozen_gewas], drempel, periode, werelddeel)
        verschil_d, laag_d, hoog_d, aantal_d = toets(deel, maat, zuiver)
        robuust.append({
            "Minimale productie (ton)": f"{drempel:,}".replace(",", "."),
            "Rampjaren": aantal_d,
            "Verschil (pp)": round(verschil_d, 1),
            "95%-interval": f"{laag_d:.1f} tot {hoog_d:.1f}",
            "Verlies aangetoond": "ja" if hoog_d < 0 else "nee",
        })
    st.dataframe(pd.DataFrame(robuust), hide_index=True)
    st.caption("Zonder drempel trekken kleine producenten met extreme percentages het "
               "gemiddelde alle kanten op. Blijft het verschil bij elke drempel onder 0, "
               "dan hangt de conclusie niet af van die keuze.")

    with st.expander("Hoe is dit getoetst?"):
        st.markdown(f"""
- **Verschil in gemiddelde** tussen rampjaren en normale jaren, met een **95%-interval**:
  verschil ± 1,96 × standaardfout. Ligt het hele interval onder 0, dan is het verlies
  niet aan toeval toe te schrijven.
- **Tweede toets**: elk rampjaar naast de **mediane groei van hetzelfde land** in normale jaren.
  Zo vergelijk je Ethiopië met Ethiopië, en niet met de rest van de wereld.
- **Gemiddelde en mediaan** kunnen verschillen: de verdeling heeft een lange staart naar rechts
  (inhaaljaren). De toets gebruikt het gemiddelde, de titels van de grafieken de mediaan.
- **Voorbehoud**: jaren van hetzelfde land zijn niet onafhankelijk van elkaar. Het echte
  interval is daardoor iets breder dan hier berekend.
""")


# ------------------------------------------------------------------ kaart ---
with tab_kaart:
    st.subheader(f"Welke landen herstellen binnen een jaar? ({gekozen_werelddeel})")

    kol1, kol2 = st.columns(2)
    minimaal_aantal = kol1.slider("Minimaal aantal rampjaren per land", 1, 10, 3,
                                  help="Een land met één ramp zegt weinig over hoe het herstelt.")
    # Een Streamlit-widget en geen Folium-lagenmenu: dat menu werkt alleen in de
    # browser, Python merkt het niet en de titel zou dan niet meeveranderen.
    gekozen_klassen = kol2.multiselect("Toon landen met", list(KLASSEN.keys()),
                                       default=list(KLASSEN.keys()))

    per_land = ramp.groupby("ISO3").agg(
        Land=("Land", "first"),
        Aantal_rampjaren=(maat, "size"),
        Mediane_verandering=(maat, "median"),
        Mediane_afwijking=("Afwijking", "median"),
        Hersteld=("Hersteld", "mean"),
    ).reset_index()
    per_land["Hersteld"] = per_land["Hersteld"] * 100
    per_land = per_land[per_land["Aantal_rampjaren"] >= minimaal_aantal]
    per_land = per_land.merge(centroiden, on="ISO3", how="left")
    geen_plek = per_land["Lat"].isna().sum()
    per_land = per_land.dropna(subset=["Lat", "Lon"])

    if per_land.empty:
        st.warning(f"Geen landen in {gekozen_werelddeel} met minstens {minimaal_aantal} "
                   "rampjaren. Zet de drempel lager.")
    else:
        # Vaste klassen: zo betekent een kleur altijd hetzelfde, welke filters je ook kiest
        per_land["Klasse"] = pd.cut(per_land["Hersteld"], bins=[0, 25, 50, 75, 100],
                                    labels=list(KLASSEN.keys()), include_lowest=True)
        # Grootte op basis van alle landen, zodat een stip niet groeit als je een klasse uitzet
        laagste = per_land["Aantal_rampjaren"].min()
        hoogste = per_land["Aantal_rampjaren"].max()

        # Alleen de gekozen klassen op de kaart en in de titel
        op_kaart = per_land[per_land["Klasse"].isin(gekozen_klassen)]

        if op_kaart.empty:
            st.warning("Kies minstens één klasse bij 'Toon landen met'.")
        else:
            slechtste = op_kaart.sort_values(["Hersteld", "Aantal_rampjaren"],
                                             ascending=[True, False]).iloc[0]
            st.markdown(f"**{gekozen_werelddeel}: van de {len(op_kaart)} landen op de kaart "
                        f"herstelt {slechtste['Land']} het slechtst, "
                        f"{slechtste['Hersteld']:.0f}% binnen een jaar, "
                        f"over {int(slechtste['Aantal_rampjaren'])} rampjaren.**")

            kaart = folium.Map(location=[20, 10], zoom_start=2, tiles=None)
            folium.TileLayer(ESRI, attr="Tiles &copy; Esri").add_to(kaart)

            # Zoom automatisch in op de landen die er staan (hoek linksonder, hoek rechtsboven)
            kaart.fit_bounds([[op_kaart["Lat"].min(), op_kaart["Lon"].min()],
                              [op_kaart["Lat"].max(), op_kaart["Lon"].max()]], max_zoom=5)

            for _, r in op_kaart.iterrows():
                folium.CircleMarker(
                    location=[r["Lat"], r["Lon"]],
                    radius=straal(r["Aantal_rampjaren"], laagste, hoogste),
                    color="white", weight=1, fill=True,
                    fill_color=KLASSEN[r["Klasse"]], fill_opacity=0.85,
                    tooltip=f"{r['Land']} — {r['Hersteld']:.0f}% hersteld",
                    popup=folium.Popup(
                        f"<b>{r['Land']}</b><br>"
                        f"{int(r['Aantal_rampjaren'])} rampjaren<br>"
                        f"mediane verandering {r['Mediane_verandering']:.1f}%<br>"
                        f"t.o.v. eigen normale groei {r['Mediane_afwijking']:.1f} pp<br>"
                        f"{r['Hersteld']:.0f}% binnen een jaar hersteld", max_width=250),
                ).add_to(kaart)

            # Legenda: een gekleurde stip per klasse
            legenda = ""
            for naam, kleur in KLASSEN.items():
                legenda += f'<span style="color:{kleur}; font-size:20px">●</span> {naam} &nbsp;&nbsp; '
            st.markdown(legenda, unsafe_allow_html=True)

            # Een Folium-kaart is gewoon HTML, dus die geven we aan Streamlit door
            st.components.v1.html(kaart._repr_html_(), height=560)
            st.caption(
                "Grootte is het aantal rampjaren, via de wortel. Kleur is het aandeel "
                "rampjaren waarna de oogst een jaar later minstens zo groot was als in het "
                "jaar vóór de ramp. Klik een stip aan voor details. "
                f"{geen_plek} landen hebben geen middelpunt en staan niet op de kaart.")


# --------------------------------------------------------------- per land ---
with tab_land:
    st.subheader("Productie door de tijd")

    # Alleen landen die in deze selectie minstens één rampjaar hebben
    landen = sorted(ramp["Land"].dropna().unique())
    standaard = landen.index("Ethiopia") if "Ethiopia" in landen else 0
    gekozen_land = st.selectbox("Land", landen, index=standaard)

    iso = landnamen.loc[landnamen["Land"] == gekozen_land, "ISO3"].iloc[0]
    reeks = productie[
        (productie["ISO3"] == iso)
        & (productie["Gewas"] == GEWASSEN[gekozen_gewas])
        & (productie["Jaar"].between(periode[0], periode[1]))
    ].sort_values("Jaar")
    rampjaren_land = ramp[ramp["ISO3"] == iso]

    figuur = px.line(
        reeks, x="Jaar", y="Ton",
        title=f"{gekozen_gewas} in {gekozen_land}: {len(rampjaren_land)} jaren met "
              f"{ramp_naam} (oranje), {rampjaren_land['Hersteld'].mean():.0%} "
              f"binnen een jaar hersteld",
        labels={"Ton": "Productie (ton)", "Jaar": ""},
    )
    figuur.update_traces(line=dict(color=KLEUR_LIJN, width=2))
    for jaar in rampjaren_land["Jaar"]:
        figuur.add_vrect(x0=jaar - 0.5, x1=jaar + 0.5,
                         fillcolor=KLEUR_RAMP, opacity=0.25, line_width=0)
    figuur.update_layout(height=430)
    st.plotly_chart(figuur)

    normaal_land = rampjaren_land["Normale_groei"].iloc[0]
    periode_tekst = "van het ene jaar op het volgende" if maat == "Klap_pct" else "over twee jaar"
    if pd.isna(normaal_land):
        st.markdown(f"{gekozen_land} heeft in deze selectie geen normale jaren om mee te vergelijken.")
    else:
        st.markdown(f"In een normaal jaar verandert de productie in {gekozen_land} {periode_tekst} "
                    f"mediaan **{normaal_land:+.1f}%**. Dat is de lat waar elk rampjaar "
                    f"hieronder tegen wordt gelegd.")

    tabel = rampjaren_land[["Jaar", "Klap_pct", "Verandering_pct", "Afwijking", "Hersteld"]]
    tabel = tabel.rename(columns={
        "Klap_pct": "Rampjaar t.o.v. ervoor (%)",
        "Verandering_pct": "Jaar erna t.o.v. ervoor (%)",
        "Afwijking": "Gekozen maat t.o.v. normale groei (pp)",
        "Hersteld": "Hersteld binnen een jaar",
    })
    st.dataframe(tabel.round(1), hide_index=True)
    st.caption(f"EM-DAT registreert het jaar waarin de {ramp_naam} begon. Begint een ramp laat "
               "in het jaar, dan raakt hij vaak pas de oogst van het jaar erna. Daarom kan het "
               "rampjaar zelf er soms goed uitzien; zet links het meetmoment om om dat te zien.")


# ------------------------------------------------------- conclusie en advies ---
with tab_conclusie:
    st.subheader("Welke ramp kost graan?")

    # Alle ramptypes naast elkaar, met hetzelfde gewas, dezelfde drempel en periode
    rijen_ramp = []
    for naam, code in RAMPTYPES.items():
        deel = koppel_rampen(veranderingen, rampen, code)
        deel = filter_data(deel, GEWASSEN[gekozen_gewas], minimale_productie, periode,
                           werelddeel)
        for moment, kolom in [("In het rampjaar zelf", "Klap_pct"),
                              ("Een jaar later", "Verandering_pct")]:
            verschil_r, laag_r, hoog_r, aantal_r = toets(deel, kolom, zuiver)
            rijen_ramp.append({"Ramp": naam, "Moment": moment, "Verschil": verschil_r,
                               "Marge": hoog_r - verschil_r, "Hoog": hoog_r,
                               "Rampjaren": aantal_r})
    per_ramp = pd.DataFrame(rijen_ramp)

    klap_per_ramp = per_ramp[per_ramp["Moment"] == "In het rampjaar zelf"]
    met_verlies = klap_per_ramp[klap_per_ramp["Hoog"] < 0]["Ramp"].str.lower().tolist()
    if met_verlies:
        titel_ramp = f"Aantoonbaar verlies in het rampjaar bij: {', '.join(met_verlies)}"
    else:
        titel_ramp = "Bij geen enkele ramp is verlies in het rampjaar aantoonbaar"

    figuur = px.bar(
        per_ramp, x="Verschil", y="Ramp", color="Moment", barmode="group",
        orientation="h", error_x="Marge",
        color_discrete_map={"In het rampjaar zelf": KLEUR_RAMP, "Een jaar later": KLEUR_NORMAAL},
        hover_data={"Rampjaren": True, "Marge": ":.1f", "Hoog": False},
        title=titel_ramp,
        labels={"Verschil": "Verschil met normaal jaar (procentpunt)", "Ramp": "", "Moment": ""},
    )
    figuur.add_vline(x=0, line_color="black", line_width=1)
    figuur.update_layout(legend=dict(orientation="h", y=1.1), height=380)
    st.plotly_chart(figuur)
    st.caption(f"{gekozen_gewas}, {gekozen_werelddeel}, minstens {drempel_tekst} ton, "
               f"{periode[0]}-{periode[1]}. "
               "Het streepje is het 95%-interval.")

    st.subheader("Wat betekent dit voor een hulporganisatie?")

    # Hier altijd beide meetmomenten, los van de keuze in de zijbalk
    klap_verschil, klap_laag, klap_hoog = vergelijk_groepen(ramp, normaal, "Klap_pct")
    erna_verschil, erna_laag, erna_hoog = vergelijk_groepen(ramp, normaal, "Verandering_pct")

    slecht_herstel = per_land.sort_values(["Hersteld", "Aantal_rampjaren"],
                                          ascending=[True, False]).head(AANTAL_SLECHTSTE)
    namen = ", ".join(slecht_herstel["Land"]) if len(slecht_herstel) else "geen landen"

    if klap_hoog < 0:
        advies = (f"reken in een jaar met {ramp_naam} op ongeveer "
                  f"**{abs(klap_verschil):.0f}% minder graan** dan in een normaal jaar "
                  f"(gemeten als deel van de oogst van het jaar ervóór). Hulp in het jaar ná de "
                  f"ramp is vooral nodig in de landen die niet binnen een jaar herstellen.")
    else:
        advies = (f"een {ramp_naam} kost bij deze selectie geen aantoonbare hoeveelheid graan. "
                  f"Voor voedselhulp is dit ramptype dus geen goede voorspeller; "
                  f"kijk naar de grafiek hierboven voor het ramptype dat dat wel is.")

    st.markdown(f"""
**1. In het rampjaar zelf:** een {ramp_naam} geeft {beschrijf(klap_verschil, klap_laag, klap_hoog)}.

**2. Een jaar later:** {beschrijf(erna_verschil, erna_laag, erna_hoog)}.
**{ramp['Hersteld'].mean():.0%}** van de rampjaren is binnen een jaar terug op het oude niveau.

**3. Het slechtste herstel in {gekozen_werelddeel}:** {namen} (minstens {minimaal_aantal} rampjaren, zie de kaart).

**Advies:** {advies}
""")

    st.markdown("**Wat dit advies níet kan zeggen**")
    st.markdown("""
- EM-DAT registreert alleen rampen met 10+ doden, 100+ getroffenen, een noodtoestand of een
  hulpverzoek. Rampen in rijke landen staan er vaak niet in en tellen dan als "normaal jaar".
  Het echte verlies is dus eerder groter dan kleiner.
- Productie is per land, een ramp treft een regio. In grote landen verdunt dat het effect.
- Een samenhang is geen oorzaak: in een jaar met een ramp kan ook iets anders gebeuren,
  zoals een conflict (Syrië na 2011) of een prijsval.
- Herstel gaat over hoeveel graan er is, niet over wie het kan betalen.
""")


# ---------------------------------------------------------- over de data ---
with tab_data:
    st.markdown("""
**Bronnen**

- **FAOSTAT**, Crops and livestock products: productie in **ton** per land, gewas en jaar,
  1961-2024. Via de website gedownload in drie delen, samengevoegd met `pd.concat`.
- **EM-DAT**, het internationale rampenregister, 1970-2021, via een mirror op Kaggle.
- **FAOSTAT landcodes** (M49, ISO2, ISO3) en **landmiddelpunten** (GitHub, gavinr)
  om te kunnen koppelen en de kaart te tekenen.

**Hoe de bronnen zijn gekoppeld**

- Op **landnaam** vielen 70 van de 228 EM-DAT-landen weg ("Netherlands (the)", "Turkey",
  "United States of America (the)"). Daarom is gekoppeld op **code**: FAO M49 → ISO3.
- Wat dan nog wegvalt zijn **opgeheven staten** (Sovjet-Unie, Joegoslavië) en
  **eilanden zonder graanteelt** (Singapore, Tuvalu).
- **Ethiopië en Soedan** hebben bij FAO tot 1992 en 2011 een eigen code
  (Ethiopia PDR, Sudan former). Die zijn doorgezet naar ETH en SDN.
- **Namibië** heeft ISO2-code "NA". Pandas leest dat standaard als ontbrekend;
  daarom `keep_default_na=False` bij de landmiddelpunten.
- **Werelddeel** komt uit de kolom Continent van EM-DAT. Een land dat nooit een ramp
  had, heeft geen werelddeel en valt weg zodra je een werelddeel kiest.

**Zelf afgeleide kolommen**

- **Klap_pct**: van het jaar ervóór naar het rampjaar, in procenten.
- **Verandering_pct**: van het jaar ervóór naar het jaar ná de ramp (de definitie uit de opdracht).
- **Hersteld**: True als het jaar ná de ramp minstens zoveel oplevert als het jaar ervóór.
- **Buurjaar**: het jaar direct vóór of ná een rampjaar. Telt standaard niet mee als normaal jaar.
- **Afwijking**: verandering in het rampjaar min de mediane verandering van datzelfde land
  in normale jaren.

**Keuzes bij het opschonen**

- **Lege productiecijfers** zijn weggegooid, niet op 0 gezet: 0 zou een totale misoogst suggereren.
- **Start Year** in plaats van Year: Year is het registratiejaar, niet het jaar van de ramp.
- **Spaties** achter ramptypes ("Extreme temperature ") weggehaald met `.str.strip()`.
- **Meerdere rampen in hetzelfde land en jaar** tellen als één rampjaar.
- **Delen door 0** (productie was het jaar ervóór 0) levert geen verandering op.
""")

    st.markdown("**Rijen na elke stap**")
    st.dataframe(pd.DataFrame({
        "Stap": ["FAO-productie na opschonen", "EM-DAT-rampen",
                 "Land × gewas × jaar na koppelen (1970-2021)", "Waarvan in deze selectie",
                 "Waarvan rampjaren", "Waarvan normale jaren (vergelijkingsgroep)"],
        "Rijen": [len(productie), len(rampen), len(analyse), len(selectie),
                  len(ramp), len(normaal)],
    }), hide_index=True)

    if st.checkbox("Toon de gefilterde data"):
        st.dataframe(selectie)
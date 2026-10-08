"""
SaxSSL Bodenschätzung — Bodeneigenschaften aus Bruker-Alpha-II-MIR-Spektren.

Studierende laden ihre OPUS-Dateien (.0) hoch und erhalten eine Tabelle mit
den Schätzungen der lokalen Bibliothek (SaxSSL + Hannah) und der OSSL (KSSL)
für alle Proben.

Start:
    pip install -r requirements.txt
    streamlit run app.py
"""

from __future__ import annotations

import io
import sys
import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from soilspec.inference.engine import PredictionEngine  # noqa: E402
from soilspec.io import read_opus  # noqa: E402
from tables import ad_summary, average_replicates, label_from_filename, wide_table  # noqa: E402

MANIFEST = HERE / "models" / "library" / "manifest.json"
QUALITY = {"A": "A — gut (RPIQ ≥ 2,5)", "B": "B — brauchbar (≥ 1,9)",
           "C": "C — Screening (≥ 1,4)", "D": "D — unzuverlässig (< 1,4)"}

st.set_page_config(page_title="SaxSSL Bodenschätzung", page_icon="🌱", layout="wide")


@st.cache_resource
def get_engine() -> PredictionEngine:
    return PredictionEngine(MANIFEST)


@st.cache_data(show_spinner=False)
def read_files(payload: tuple[tuple[str, bytes], ...]):
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for name, data in payload:
            p = Path(tmp) / name
            p.write_bytes(data)
            paths.append(p)
        X, wn, ids = read_opus(paths, block="AB")
    return np.asarray(X, float), np.asarray(wn, float), list(ids)


st.title("🌱 SaxSSL Bodenschätzung")
st.caption("Schätzung von Bodeneigenschaften aus MIR-Spektren (Bruker Alpha II, DRIFT). "
           "Lokale Bibliothek: SaxSSL (BDF Sachsen) + Hannah (Auenböden, FIS Boden). "
           "Zum Vergleich: Open Soil Spectral Library (OSSL, KSSL/USA).")

if not MANIFEST.exists():
    st.error(f"Modelle fehlen: `{MANIFEST}`. Repository vollständig klonen (Ordner `models/`).")
    st.stop()
engine = get_engine()
targets = [t["id"] for t in __import__("json").loads(MANIFEST.read_text())["targets"]]
models = engine.list_models()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Einstellungen")
    groups = list(dict.fromkeys(models["group"]))
    sel_groups = st.multiselect("Eigenschaftsgruppen", groups, default=groups)
    peat = st.checkbox("🌿 Torf-/Moorproben (pmird-Modelle)", value=False,
                       help="Zusätzliche Modelle aus der Peatland Mid-Infrared Database "
                            "(Teickner et al.) für organische Proben. Experimentell: pmird-"
                            "Spektren sind überwiegend Transmission (KBr), nicht DRIFT.")
    use_ossl = st.checkbox("OSSL-Vergleich (KSSL, USA)", value=True)
    sources = (["SaxSSL+Hannah"] + (["Torf (pmird)"] if peat else [])
               + (["OSSL-KSSL"] if use_ossl else []))
    show_d = st.checkbox("Auch unzuverlässige Modelle (Klasse D) zeigen", value=False)
    with_iv = st.checkbox("Unsicherheitsintervall anzeigen (~68 %)", value=True)
    avg = st.checkbox("Wiederholungsmessungen je Probe mitteln", value=True,
                      help="Dateien wie 'Probe7-1.0', 'Probe7-2.0' werden zu 'Probe7' gemittelt "
                           "(Bibliotheksspektren sind ebenfalls Mittel aus 4 Messungen).")

# ---------------------------------------------------------------- upload
files = st.file_uploader("OPUS-Dateien (.0, .1, …) hochladen — mehrere auf einmal möglich",
                         accept_multiple_files=True)
if not files:
    st.info("Dateien hochladen, um die Schätzung zu starten. "
            "Unterstützt: Bruker OPUS mit Absorbanzblock (AB).")
    with st.expander("Verfügbare Modelle und Güte"):
        show = models.assign(Güte=models["quality_class"].map(QUALITY))
        st.dataframe(show[["group", "display_name", "unit", "origin", "algo", "R2", "RPIQ",
                           "n_train", "Güte"]], hide_index=True, use_container_width=True)
    st.stop()

try:
    X, wn, ids = read_files(tuple((f.name, f.getvalue()) for f in files))
except Exception as exc:  # noqa: BLE001 — show, don't crash
    st.error(f"Dateien nicht lesbar: {exc}")
    st.stop()

if wn.min() > 650 or wn.max() < 3950:
    st.warning(f"Spektren decken nur {wn.min():.0f}–{wn.max():.0f} cm⁻¹ ab; Modelle brauchen "
               "600–4000 cm⁻¹. Randbereiche werden fortgeschrieben — Ergebnisse unsicher.")

if avg:
    labels = [label_from_filename(f.name) for f in files]
    Xs, names, file_map = average_replicates(X, ids, labels)
else:
    Xs, names, file_map = X, ids, pd.DataFrame({"Datei": ids, "Probe": ids})
st.success(f"{len(files)} Dateien → {len(names)} Proben.")

# ---------------------------------------------------------------- predict
use = models[models["group"].isin(sel_groups) & models["origin"].isin(sources)
             & (models["enabled"] | show_d)]
if use.empty:
    st.warning("Keine Modelle ausgewählt.")
    st.stop()
with st.spinner(f"Schätze {len(use)} Modelle für {len(names)} Proben …"):
    res = engine.predict(Xs, wn, names, model_ids=use["model_id"].tolist(),
                         include_disabled=True)

tab_res, tab_qc, tab_models, tab_help = st.tabs(
    ["📊 Ergebnisse", "🔍 Spektren & Plausibilität", "🧮 Modelle", "❓ Hilfe"])

with tab_res:
    corg = res[(res.target == "corg") & (res.origin == "SaxSSL+Hannah")]
    organic = corg.loc[corg["value"] > 12, "sample_id"].tolist()
    if organic and not peat:
        st.info(f"🌿 {', '.join(organic)}: Corg > 12 % — vermutlich organische Probe/Torf. "
                "Die Mineralboden-Modelle gelten bis ~15 % C; in der Seitenleiste "
                "**Torf-/Moorproben** aktivieren.")
    wide = wide_table(res, targets, with_interval=with_iv)
    st.dataframe(wide, use_container_width=True)
    n_ad = int((res["ad_ok"] == False).sum())  # noqa: E712
    if n_ad:
        st.warning("⚠ = Spektrum liegt außerhalb des Bereichs der Trainingsbibliothek "
                   "(Hotelling-T²). Werte nur mit Vorsicht verwenden.")
    st.caption("Spaltenkopf: Eigenschaft [Einheit] · Modellquelle (Güteklasse). "
               "Wert [untere–obere Grenze] des ~68 %-Intervalls.")

    num = wide_table(res, targets, numeric=True)
    long = res.drop(columns=["ampel", "threshold"], errors="ignore")
    stamp = date.today().isoformat()
    c1, c2 = st.columns(2)
    c1.download_button("⬇️ CSV (Werte)", num.to_csv().encode("utf-8-sig"),
                       f"bodenschaetzung_{stamp}.csv", "text/csv")
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        num.to_excel(xw, sheet_name="Werte")
        wide.to_excel(xw, sheet_name="Werte_mit_Intervall")
        long.to_excel(xw, sheet_name="Details_lang", index=False)
        ad_summary(res).to_excel(xw, sheet_name="Plausibilitaet")
        file_map.to_excel(xw, sheet_name="Dateien", index=False)
        models.to_excel(xw, sheet_name="Modelle", index=False)
    c2.download_button("⬇️ Excel (alle Tabellen)", buf.getvalue(),
                       f"bodenschaetzung_{stamp}.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

with tab_qc:
    import plotly.graph_objects as go
    fig = go.Figure()
    for i, n in enumerate(names):
        fig.add_trace(go.Scatter(x=wn, y=Xs[i], name=n, mode="lines", line={"width": 1}))
    fig.update_layout(xaxis={"title": "Wellenzahl (cm⁻¹)", "autorange": "reversed"},
                      yaxis={"title": "Absorbanz"}, height=420, margin={"t": 20})
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Ähnlichkeit zur Trainingsbibliothek** (T²/T²krit ≤ 1: innerhalb)")
    st.dataframe(ad_summary(res), use_container_width=True)
    with st.expander("Datei → Probe"):
        st.dataframe(file_map, hide_index=True, use_container_width=True)

with tab_models:
    show = models.assign(Güte=models["quality_class"].map(QUALITY))
    st.dataframe(show[["group", "display_name", "unit", "origin", "algo", "R2", "RMSE",
                       "RPD", "RPIQ", "n_train", "Güte", "enabled"]],
                 hide_index=True, use_container_width=True)

with tab_help:
    st.markdown("""
**Was wird geschätzt?** Für jede hochgeladene Probe sagen statistische Modelle
Bodeneigenschaften aus dem MIR-Spektrum voraus. Es sind **Schätzungen**, keine
Laboranalysen.

**Zwei Modellquellen**
- **SaxSSL** (SaxSSL+Hannah): trainiert an ~1650 sächsischen Bodenproben
  (BDF-Dauerbeobachtung + belastete Auenböden), deutsche Labormethoden
  (KA5-Textur, Königswasser, CaCl₂-pH, soliTOC). **Für sächsische Böden erste Wahl.**
- **Torf** (pmird, optional): ~1500 Torfproben aus Mooren weltweit (C, N, C/N,
  Glühverlust). Nur für organische Proben einschalten; experimentell, da die
  Bibliotheksspektren meist in Transmission (KBr) gemessen wurden.
- **OSSL** (KSSL, USA): sehr große US-Bibliothek, USDA-Methoden — u. a. Sand/Schluff-
  Grenze 50 µm statt 63 µm; Werte daher nicht 1:1 vergleichbar. Gut als Gegenprobe.

**Güteklassen** (RPIQ am unabhängigen Testset): A gut · B brauchbar ·
C nur Screening · D unzuverlässig (standardmäßig ausgeblendet).

**Intervall** [von–bis]: ~68 % der Laborwerte liegen erfahrungsgemäß in diesem Bereich.

**⚠ Plausibilität**: Das Spektrum ähnelt keiner Probe der Bibliothek (z. B. anderer
Bodentyp, Torf, Fehlmessung, ungemahlene Probe). Dann Werte nicht verwenden.

**Messhinweise**: Proben luftgetrocknet und fein gemahlen messen (wie die Bibliothek),
4 Wiederholungen je Probe, Dateinamen `Probe-1.0`, `Probe-2.0`, … — sie werden
automatisch gemittelt.
""")

"""
DRIFTS Bodenvariablenschätzung — Bodeneigenschaften aus Bruker-Alpha-II-MIR-Spektren.

OPUS-Dateien (.0) werden hochgeladen und eine Tabelle mit den Schätzungen der lokalen 
Bibliotheken (SaxSSL + SaxTox), der OSSL (KSSL) sowie der Torf-Modelle (pmird) für alle Proben.

Start:
    pip install -r requirements.txt
    streamlit run app.py
"""

from __future__ import annotations

import hashlib
import io
import json
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
from app_tables import (AD_ROW, QUALITY_DOT, ad_summary, average_replicates,  # noqa: E402
                    label_from_filename, result_table)

MANIFEST = HERE / "models" / "library" / "manifest.json"
QUALITY = {"A": "🟢 A — gut (RPIQ ≥ 2,5)", "B": "🟡 B — brauchbar (≥ 1,9)",
           "C": "🟠 C — Screening (≥ 1,4)", "D": "🔴 D — unzuverlässig (< 1,4)"}

CREDITS = ("Entwickelt im [DryWet Soil-Water-Lab](https://www.drywet.de), "
           "TU Bergakademie Freiberg · Code und Modelle: "
           "[github.com/cojacoo/SaxSSL_Models](https://github.com/cojacoo/SaxSSL_Models) (MIT)")

# origin -> (checkbox label, default, help)
SOURCE_UI = {
    "SaxSSL": ("SaxSSL", True, "Sächsische Boden-Dauerbeobachtung (BDF), Archiv + Kampagne 2023."),
    "SaxTox": ("SaxTox", True, "Belastete Auenböden (FIS Boden): Schwermetalle und PAK."),
    "SaxSSL+SaxTox": ("SaxSSL+SaxTox (gemeinsam trainiert)", False,
                      "Ein Modell aus beiden Bibliotheken zusammen; nur für Eigenschaften, "
                      "die in beiden gemessen wurden."),
    "Torf (pmird)": ("🌿 Torf (pmird)", False,
                     "Peatland Mid-Infrared Database (Teickner et al.) für organische Proben. "
                     "Experimentell: pmird-Spektren sind überwiegend Transmission (KBr), nicht DRIFT."),
    "OSSL/KSSL": ("OSSL/KSSL (USA)", True, "Open Soil Spectral Library, KSSL-Ausschnitt, USDA-Methoden."),
}

st.set_page_config(page_title="SaxSSL Bodenschätzung", page_icon="🌱", layout="wide")


@st.cache_resource
def get_engine(manifest_hash: str) -> PredictionEngine:
    # hash in the cache key: Streamlit Cloud keeps the process across pushes,
    # so new models must invalidate the cached engine
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


st.title("𓂃↟𖠰 ོ𓂃 Terra Resiliens Bodenschätzung ↟ ོ☼𓂃")
st.caption("Schätzung von Bodeneigenschaften aus MIR-Spektren (Bruker Alpha II, DRIFT). "
           "Regionale Bibliotheken: SaxSSL (Boden-Dauerbeobachtung Sachsen, Adam et al. 2025) und SaxTox "
           "(Schwermetalle & PAK). Moor-Spezifische Bibliothek pmird (Teickner and Knorr 2025)."
           "Zum Vergleich: Open Soil Spectral Library (OSSL, KSSL/USA, Safanelli et al. 2025).")

if not MANIFEST.exists():
    st.error(f"Modelle fehlen: `{MANIFEST}`. Repository vollständig klonen (Ordner `models/`).")
    st.stop()
engine = get_engine(hashlib.md5(MANIFEST.read_bytes()).hexdigest())
manifest = json.loads(MANIFEST.read_text())
targets = [t["id"] for t in manifest["targets"]]
# pooled models trained on one library only duplicate that library's own model
DUPLICATE = {m["id"] for m in manifest["models"]
             if m.get("model_set") == "mineral" and len(m.get("sources", {})) == 1}
models = engine.list_models()
models = models[~models["model_id"].isin(DUPLICATE)]

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Einstellungen")
    groups = list(dict.fromkeys(models["group"]))
    sel_groups = st.multiselect("Eigenschaftsgruppen", groups, default=groups)
    st.markdown("**Modellquellen**")
    present = set(models["origin"])
    sources = [o for o, (label, on, hlp) in SOURCE_UI.items()
               if o in present and st.checkbox(label, value=on, help=hlp, key=f"src_{o}")]
    min_class = st.select_slider(
        "Mindestgüte der Modelle", options=["A", "B", "C", "D"], value="C",
        format_func=lambda c: f"{QUALITY_DOT[c]} {c}",
        help="Nur Modelle dieser Güteklasse oder besser zeigen. "
             "A gut · B brauchbar · C Screening · D unzuverlässig.")
    with_iv = st.checkbox("Unsicherheitsintervall anzeigen (~68 %)", value=True)
    avg = st.checkbox("Wiederholungsmessungen je Probe mitteln", value=True,
                      help="Dateien wie 'Probe7-1.0', 'Probe7-2.0' werden zu 'Probe7' gemittelt "
                           "(Bibliotheksspektren sind ebenfalls Mittel aus 4 Messungen).")
    st.divider()
    st.caption(CREDITS)

# ---------------------------------------------------------------- upload
files = st.file_uploader("OPUS-Dateien (.0, .1, …) hochladen — mehrere auf einmal möglich",
                         accept_multiple_files=True)
if not files:
    st.info("Dateien hochladen, um die Schätzung zu starten. "
            "Unterstützt: Bruker OPUS mit Absorbanzblock (AB).")
    with st.expander("Verfügbare Modelle und Güte"):
        show = models.assign(Güte=models["quality_class"].map(QUALITY),
                             Quelle=models["origin"])
        st.dataframe(show[["group", "display_name", "unit", "Quelle", "algo", "R2", "RPIQ",
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
             & (models["quality_class"] <= min_class)]   # "A" < "B" < "C" < "D"
if use.empty:
    st.warning("Keine Modelle für diese Auswahl – Eigenschaftsgruppen und Modellquellen "
               "in der Seitenleiste prüfen.")
    st.stop()
bar = st.progress(0.0)
parts = []
for i, m in enumerate(use.itertuples(), 1):     # one model per call -> visible progress
    bar.progress((i - 1) / len(use), text=f"Schätze {m.display_name} · {m.origin} "
                                           f"({i}/{len(use)} Modelle, {len(names)} Proben) …")
    parts.append(engine.predict(Xs, wn, names, model_ids=[m.model_id], include_disabled=True))
bar.empty()
res = pd.concat(parts, ignore_index=True)

tab_res, tab_qc, tab_models, tab_help = st.tabs(
    ["📊 Ergebnisse", "🔍 Spektren & Plausibilität", "🧮 Modelle", "❓ Hilfe"])

with tab_res:
    corg = res[(res.target == "corg") & res.origin.isin(["SaxSSL", "SaxTox", "SaxSSL+SaxTox"])]
    organic = list(dict.fromkeys(corg.loc[corg["value"] > 12, "sample_id"]))
    if organic and "Torf (pmird)" not in sources:
        st.info(f"🌿 {', '.join(organic)}: Corg > 12 % — vermutlich organische Probe/Torf. "
                "Die Mineralboden-Modelle gelten bis ~15 % C; in der Seitenleiste "
                "**෴෴ Torf (pmird) ෴** aktivieren.")
    table = result_table(res, targets, with_interval=with_iv)
    st.dataframe(table, hide_index=True, use_container_width=True,
                 height=min(38 + 35 * len(table), 900),
                 column_config={"Eigenschaft": st.column_config.TextColumn(pinned=True),
                                "Quelle": st.column_config.TextColumn(pinned=True)})
    if (res["ad_ok"] == False).any():  # noqa: E712
        st.warning("⚠ = Spektrum liegt außerhalb der Trainingsbibliothek (Zeile "
                   f"„{AD_ROW}“). Werte nur mit Vorsicht verwenden.")
    st.caption("Güte: " + " · ".join(QUALITY.values()) + ". "
               f"{AD_ROW}: 🟢 ähnlich · 🟠 am Rand · 🔴 fremd (T²/T²krit). "
               + ("Wert [von–bis] = ~68 %-Intervall (siehe Hilfe)." if with_iv else ""))

    num = result_table(res, targets, numeric=True)
    long = res.drop(columns=["ampel", "threshold"], errors="ignore")
    stamp = date.today().isoformat()
    c1, c2 = st.columns(2)
    c1.download_button("⬇ CSV (Werte)", num.to_csv(index=False).encode("utf-8-sig"),
                       f"bodenschaetzung_{stamp}.csv", "text/csv")
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        num.to_excel(xw, sheet_name="Werte", index=False)
        table.to_excel(xw, sheet_name="Werte_mit_Intervall", index=False)
        long.to_excel(xw, sheet_name="Details_lang", index=False)
        ad_summary(res).to_excel(xw, sheet_name="Plausibilitaet")
        file_map.to_excel(xw, sheet_name="Dateien", index=False)
        models.to_excel(xw, sheet_name="Modelle", index=False)
    c2.download_button("⬇ Excel (alle Tabellen)", buf.getvalue(),
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
    show = models.assign(Güte=models["quality_class"].map(QUALITY),
                         Quelle=models["origin"])
    st.dataframe(show[["group", "display_name", "unit", "Quelle", "algo", "R2", "RMSE",
                       "RPD", "RPIQ", "n_train", "Güte"]],
                 hide_index=True, use_container_width=True)

with tab_help:
    st.markdown("""
**Was wird geschätzt?** Statistische Modelle sagen Bodeneigenschaften aus dem
MIR-Spektrum voraus. Es sind **Schätzungen**, keine Laboranalysen.

**Modellquellen** (Spalte *Quelle* zeigt, mit welchen Daten ein Modell trainiert wurde)
- **SaxSSL** – Sächsische Bodenspektralbibliothek: Boden-Dauerbeobachtungsflächen (BDF)
  Sachsen, Archivproben + Kampagne 2023, deutsche Labormethoden (KA5-Textur,
  Königswasser, CaCl₂-pH, soliTOC). **Für sächsische Böden erste Wahl.**
- **SaxTox** – Rückstellproben belasteter Auenböden (FIS Boden Sachsen) mit Schwermetallen
  und PAK; einzige Quelle für BaP und PAK16. Für K und Mg (Königswasser) deutlich besser als SaxSSL.
- **SaxSSL+SaxTox** – optional: ein Modell aus beiden Bibliotheken zusammen, für Eigenschaften,
  die in beiden gemessen wurden.
- **Torf (pmird)** – optional: ~1500 Torfproben weltweit (C, N, C/N, Glühverlust). Nur für
  organische Proben; experimentell, da meist in Transmission (KBr) gemessen.
- **OSSL/KSSL** – große US-Bibliothek, USDA-Methoden (u. a. Sand/Schluff-Grenze 50 µm statt
  63 µm); Werte nicht 1:1 vergleichbar, gut als Gegenprobe.

**Güte** 🟢 A gut · 🟡 B brauchbar · 🟠 C nur Screening · 🔴 D unzuverlässig. Welche Klassen gezeigt
werden, stellt der Schieberegler *Mindestgüte* ein (Standard: bis C).
Grundlage: RPIQ an einem unabhängigen Testset.

**Intervall [von–bis]** Aus den Fehlern des Modells an Proben, die es beim Training *nicht*
gesehen hat (5-fache Kreuzvalidierung und Testset, der ungünstigere Wert): ~68 % der
Laborwerte liegen im Intervall. Es ist eine Eigenschaft von Modell und Daten, keine
pauschale Marge, und es berücksichtigt die **Ähnlichkeit zur Bibliothek**: In der
Kreuzvalidierung wurde gemessen, wie stark der Fehler mit der Unähnlichkeit des Spektrums
wächst; untypische Proben bekommen entsprechend breitere Intervalle (normalisierte
Conformal Prediction, Lei et al. 2018). Bei logarithmisch modellierten Größen (Kohlenstoff,
Stickstoff, Metalle) ist das Intervall relativ und wächst mit dem Wert. Für 🔴-Proben liegt
die Unähnlichkeit jenseits dessen, was kalibriert werden konnte – dort ist auch das
Intervall nur ein Anhaltspunkt. OSSL/KSSL-Intervalle sind konstant breit.

**Ähnlichkeit zur Bibliothek** Hotelling-T² des Spektrums im Hauptkomponentenraum der
Trainingsspektren, geteilt durch das 99. Perzentil der Bibliothek: 🟢 ≤ 1 ähnlich ·
🟠 1–2 am Rand · 🔴 > 2 fremd (anderer Bodentyp, Torf, Fehlmessung, ungemahlene Probe).
Bei 🔴 Werte nicht verwenden.

**Messhinweise** Proben luftgetrocknet und fein gemahlen messen (wie die Bibliothek),
4 Wiederholungen je Probe, Dateinamen `Probe-1.0`, `Probe-2.0`, … – sie werden automatisch
gemittelt.

**Daten und Zitate**
- SaxSSL: Adam, Julich, Benning & Jackisch (2026): Saxon Soil Spectral Library.
  PANGAEA, [doi:10.1594/PANGAEA.984699](https://doi.org/10.1594/PANGAEA.984699)
- pmird: Teickner & Knorr (2026), SOIL 12, 497,
  [doi:10.5194/soil-12-497-2026](https://doi.org/10.5194/soil-12-497-2026)
- OSSL: Open Soil Spectral Library, [soilspectroscopy.org](https://soilspectroscopy.org)

**Über diese App** Entwickelt im **DryWet Soil-Water-Lab** der TU Bergakademie Freiberg
([www.drywet.de](https://www.drywet.de)). Quellcode, Modelle und Anleitung:
[github.com/cojacoo/SaxSSL_Models](https://github.com/cojacoo/SaxSSL_Models) (MIT-Lizenz).
""")

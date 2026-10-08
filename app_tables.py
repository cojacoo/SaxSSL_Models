"""Pure helpers for the student app (no Streamlit, unit-testable)."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ORIGIN_ORDER = {"SaxSSL": 0, "SaxTox": 1, "SaxSSL+SaxTox": 2, "Torf (pmird)": 3, "OSSL/KSSL": 4}
QUALITY_DOT = {"A": "🟢", "B": "🟡", "C": "🟠", "D": "🔴"}
AD_ROW = "Ähnlichkeit zur Bibliothek"


def average_replicates(X: np.ndarray, file_ids: list[str], labels: list[str]
                       ) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """Mean spectrum per sample label. Returns (X_mean, labels, file->label map)."""
    df = pd.DataFrame(X)
    df["_label"] = labels
    g = df.groupby("_label", sort=False)
    Xm = g.mean().to_numpy(float)
    names = list(g.mean().index)
    files = pd.DataFrame({"Datei": file_ids, "Probe": labels})
    return Xm, names, files


def _num(x: float) -> str:
    """3 significant digits, never scientific notation (1490, not 1.49e+03)."""
    return f"{x:.0f}" if abs(x) >= 1000 else f"{x:.3g}"


def _cell(value: float, lo: float, hi: float, ad_ok, with_interval: bool) -> str:
    if not np.isfinite(value):
        return "—"
    txt = _num(value)
    if with_interval and np.isfinite(lo) and np.isfinite(hi):
        txt += f" [{_num(lo)}–{_num(hi)}]"
    if ad_ok is False:
        txt = "⚠ " + txt
    return txt


def ad_flag(ratio) -> str:
    """T²/T²krit (krit = 99. Perzentil der Bibliothek) as traffic light."""
    if ratio is None or not np.isfinite(ratio):
        return "—"
    # ponytail: fixed cut 2× krit for "fremd"; calibrate on known outliers if needed
    dot, word = (("🟢", "ähnlich") if ratio <= 1 else ("🟠", "am Rand") if ratio <= 2
                 else ("🔴", "fremd"))
    return f"{dot} {word} ({ratio:.2f})"


def property_label(row) -> str:
    unit = f" [{row['unit']}]" if row["unit"] and row["unit"] != "-" else ""
    return f"{QUALITY_DOT.get(row['quality_class'], '⚪')} {row['display_name']}{unit}"


def result_table(res: pd.DataFrame, target_order: list[str],
                 with_interval: bool = True, numeric: bool = False) -> pd.DataFrame:
    """Rows = properties (one per model), columns = samples.

    The first rows give, per model set, how similar each spectrum is to its
    training library (column Quelle = model set).
    numeric=True: plain floats (T² ratio in the similarity rows) plus unit/quality
    columns for CSV/Excel; else formatted text with quality dots.
    """
    res = res.copy()
    samples = list(dict.fromkeys(res["sample_id"]))
    rank_t = {t: i for i, t in enumerate(target_order)}
    res["_t"] = res["target"].map(rank_t).fillna(len(rank_t))
    res["_o"] = res["origin"].map(ORIGIN_ORDER).fillna(9)
    res["Quelle"] = res["origin"]
    res = res.sort_values(["_t", "_o"], kind="stable")

    ad = (res.dropna(subset=["t2_ratio"]).groupby(["origin", "sample_id"])["t2_ratio"].first()
             .unstack("sample_id").reindex(columns=samples))
    ad = ad.loc[sorted(ad.index, key=lambda o: ORIGIN_ORDER.get(o, 9))]
    ad_rows = pd.DataFrame({"Eigenschaft": AD_ROW, "Quelle": ad.index})
    ad_vals = ad.reset_index(drop=True) if numeric else ad.map(ad_flag).reset_index(drop=True)

    models = res.drop_duplicates("model_id")
    if numeric:
        res["cell"] = res["value"]
        meta = pd.DataFrame({"Eigenschaft": models["display_name"].values,
                             "Einheit": models["unit"].values, "Quelle": models["Quelle"].values,
                             "Güte": models["quality_class"].values})
        ad_rows = ad_rows.assign(Einheit="T²/T²krit", Güte="")
    else:
        res["cell"] = [_cell(v, lo, hi, a, with_interval)
                       for v, lo, hi, a in zip(res["value"], res["lo"], res["hi"], res["ad_ok"])]
        meta = pd.DataFrame({"Eigenschaft": models.apply(property_label, axis=1).values,
                             "Quelle": models["Quelle"].values})
    vals = (res.pivot(index="model_id", columns="sample_id", values="cell")
               .reindex(index=models["model_id"], columns=samples).reset_index(drop=True))
    top = pd.concat([ad_rows, ad_vals], axis=1)
    body = pd.concat([meta, vals], axis=1)
    return pd.concat([top, body], ignore_index=True)


def ad_summary(res: pd.DataFrame) -> pd.DataFrame:
    """Per sample: does the spectrum resemble the training library (per model set)?"""
    out = (res.dropna(subset=["t2_ratio"])
              .groupby(["sample_id", "origin"])["t2_ratio"].first()
              .unstack("origin"))
    out = out.rename(columns=lambda c: f"T²/T²krit {c}")
    samples = list(dict.fromkeys(res["sample_id"]))
    return out.reindex(samples).round(2)


def label_from_filename(name: str, regex: str | None = None) -> str:
    """Sample label of an OPUS file; replicates share it.

    With a regex: its first group. Default: drop the OPUS extension (.0, .1 …)
    and a replicate marker '-<1-2 digits>' plus anything after it:
    'BDF-LAAB-3_MM400_23-07-11_08-1.0' -> 'BDF-LAAB'
    'Aue_LDAA_M-1-260324.0'            -> 'Aue_LDAA_M'
    'probe17-2.0' -> 'probe17' ; 'Profil_2.0' -> 'Profil_2' (underscore kept)
    """
    stem = Path(name).name
    if regex:
        m = re.search(regex, stem)
        if m:
            return m.group(1)
    stem = re.sub(r"\.\d+$", "", stem)
    m = re.match(r"^(.+?)-\d{1,2}(?:[-_].*)?$", stem)
    return m.group(1) if m else stem

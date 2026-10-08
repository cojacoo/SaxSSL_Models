"""Pure helpers for the student app (no Streamlit, unit-testable)."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ORIGIN_SHORT = {"SaxSSL+Hannah": "SaxSSL", "Torf (pmird)": "Torf", "OSSL-KSSL": "OSSL"}
ORIGIN_ORDER = {"SaxSSL+Hannah": 0, "Torf (pmird)": 1, "OSSL-KSSL": 2}


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


def _cell(value: float, lo: float, hi: float, ad_ok, with_interval: bool) -> str:
    if not np.isfinite(value):
        return "—"
    txt = f"{value:.3g}"
    if with_interval and np.isfinite(lo) and np.isfinite(hi):
        txt += f" [{lo:.3g}–{hi:.3g}]"
    if ad_ok is False:
        txt = "⚠ " + txt
    return txt


def column_label(row) -> str:
    unit = f" [{row['unit']}]" if row["unit"] and row["unit"] != "-" else ""
    src = ORIGIN_SHORT.get(row["origin"], row["origin"])
    return f"{row['display_name']}{unit} · {src} ({row['quality_class']})"


def wide_table(res: pd.DataFrame, target_order: list[str], with_interval: bool = True,
               numeric: bool = False) -> pd.DataFrame:
    """Rows = samples, columns = target × source in config order (local first).

    numeric=True gives plain float values (for CSV/Excel), else formatted text.
    """
    res = res.copy()
    res["col"] = res.apply(column_label, axis=1)
    rank_t = {t: i for i, t in enumerate(target_order)}
    res["_t"] = res["target"].map(rank_t).fillna(len(rank_t))
    res["_o"] = res["origin"].map(ORIGIN_ORDER).fillna(9)
    cols = (res.drop_duplicates("col").sort_values(["_t", "_o"])["col"].tolist())
    if numeric:
        res["cell"] = res["value"]
    else:
        res["cell"] = [
            _cell(v, lo, hi, ad, with_interval)
            for v, lo, hi, ad in zip(res["value"], res["lo"], res["hi"], res["ad_ok"])]
    samples = list(dict.fromkeys(res["sample_id"]))
    wide = res.pivot(index="sample_id", columns="col", values="cell")
    return wide.loc[samples, cols]


def ad_summary(res: pd.DataFrame) -> pd.DataFrame:
    """Per sample: does the spectrum resemble the training library (per source)?"""
    out = (res.dropna(subset=["t2_ratio"])
              .groupby(["sample_id", "origin"])["t2_ratio"].first()
              .unstack("origin"))
    out = out.rename(columns=lambda c: f"T²/T²krit {ORIGIN_SHORT.get(c, c)}")
    samples = list(dict.fromkeys(res["sample_id"]))
    return out.reindex(samples).round(2)


def label_from_filename(name: str, regex: str | None = None) -> str:
    """Sample label of an OPUS file; replicates share it.

    With a regex: its first group. Default: drop the OPUS extension (.0, .1 …)
    and a replicate marker '-<1-2 digits>' plus anything after it:
    'BDF-LAAB-3_MM400_23-07-11_08-1.0' -> 'BDF-LAAB'
    'Hannah_LDAA_M-1-260324.0'         -> 'Hannah_LDAA_M'
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

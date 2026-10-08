"""
PredictionEngine — manifest-driven inference over pretrained model bundles.

Bundles are the plain-dict joblib format used by the Hannah/saxSSL training
scripts and notebooks:

    {'model': <fitted wrapper>, 'preprocessor': <fitted Preprocessor>,
     'wavenumbers': <ascending grid>, 'target': str, 'log_transform': bool, ...}

The manifest (models_manifest.json) adds per-model metadata: display name,
unit, origin, metrics, quality class, conformal quantile, screening threshold
and the applicability-domain (AD) calibration artifact. AD artifacts are
joblib dicts: {'detector': PCOutlierDetector, 'wavenumbers': grid,
'prep_steps': ('snv',), 'n_train': int}.

This module is UI-free: the Streamlit app is a thin layer on top of it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..io.resample import resample_spectra

MANIFEST_VERSION = 1


@dataclass
class ModelEntry:
    """One manifest row; the heavy joblib bundle is loaded lazily."""

    id: str
    target: str
    display_name: str
    unit: str
    group: str
    origin: str
    algo: str
    path: Path
    log_transform: bool
    prep_steps: tuple
    grid: dict
    metrics: dict
    quality_class: str
    conformal: dict | None
    screening_threshold: dict | None
    ad_ref: Path | None
    enabled: bool
    _bundle: dict | None = field(default=None, repr=False)

    @classmethod
    def from_manifest(cls, raw: dict, base_dir: Path) -> "ModelEntry":
        ad = raw.get("ad_ref")
        return cls(
            id=raw["id"],
            target=raw["target"],
            display_name=raw.get("display_name", raw["target"]),
            unit=raw.get("unit", ""),
            group=raw.get("group", ""),
            origin=raw.get("origin", ""),
            algo=raw.get("algo", ""),
            path=base_dir / raw["path"],
            log_transform=bool(raw.get("log_transform", False)),
            prep_steps=tuple(raw.get("prep_steps", ("snv",))),
            grid=raw.get("grid", {}),
            metrics=raw.get("metrics", {}),
            quality_class=raw.get("quality_class", "?"),
            conformal=raw.get("conformal"),
            screening_threshold=raw.get("screening_threshold"),
            ad_ref=(base_dir / ad) if ad else None,
            enabled=bool(raw.get("enabled", True)),
        )

    def load(self) -> dict:
        """joblib.load the bundle, cached on the entry."""
        if self._bundle is None:
            self._bundle = joblib.load(self.path)
        return self._bundle


class PredictionEngine:
    """Load a model manifest and predict soil properties from raw spectra."""

    def __init__(self, manifest_path: str | Path):
        self.manifest_path = Path(manifest_path)
        self.base_dir = self.manifest_path.parent
        with open(self.manifest_path) as fh:
            raw = json.load(fh)
        if raw.get("version") != MANIFEST_VERSION:
            raise ValueError(
                f"Manifest version {raw.get('version')!r} != {MANIFEST_VERSION}"
            )
        self.entries: dict[str, ModelEntry] = {}
        for row in raw["models"]:
            entry = ModelEntry.from_manifest(row, self.base_dir)
            if entry.id in self.entries:
                raise ValueError(f"Duplicate model id {entry.id!r}")
            self.entries[entry.id] = entry
        self._ad_cache: dict[Path, dict] = {}

    # ------------------------------------------------------------------

    def list_models(self, enabled_only: bool = False) -> pd.DataFrame:
        """One row per model for UI tables."""
        rows = []
        for e in self.entries.values():
            if enabled_only and not e.enabled:
                continue
            thr = e.screening_threshold or {}
            rows.append({
                "model_id": e.id,
                "target": e.target,
                "display_name": e.display_name,
                "unit": e.unit,
                "group": e.group,
                "origin": e.origin,
                "algo": e.algo,
                "R2": e.metrics.get("R2"),
                "RMSE": e.metrics.get("RMSE"),
                "RPD": e.metrics.get("RPD"),
                "RPIQ": e.metrics.get("RPIQ"),
                "n_train": e.metrics.get("n_train"),
                "quality_class": e.quality_class,
                "threshold": thr.get("value"),
                "threshold_source": thr.get("source"),
                "enabled": e.enabled,
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------

    def predict(
        self,
        X: np.ndarray,
        wn: np.ndarray,
        sample_ids: list[str] | None = None,
        model_ids: list[str] | None = None,
        include_disabled: bool = False,
    ) -> pd.DataFrame:
        """
        Predict all requested models for all uploaded spectra.

        Parameters
        ----------
        X : (n_samples, n_bands) raw absorbance spectra
        wn : (n_bands,) wavenumbers (ascending or descending)
        sample_ids : names per row; defaults to S1..Sn
        model_ids : subset of manifest ids; defaults to all enabled models

        Returns
        -------
        Tidy DataFrame, one row per sample × model: value, lo, hi,
        quality_class, ad_ok, t2_ratio, ampel, threshold, note.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        n = X.shape[0]
        if sample_ids is None:
            sample_ids = [f"S{i + 1}" for i in range(n)]
        if len(sample_ids) != n:
            raise ValueError("sample_ids length != number of spectra")

        if model_ids is None:
            targets = [e for e in self.entries.values()
                       if e.enabled or include_disabled]
        else:
            missing = [m for m in model_ids if m not in self.entries]
            if missing:
                raise KeyError(f"Unknown model id(s): {missing}")
            targets = [self.entries[m] for m in model_ids]

        X, finite_rows, n_filled = _fill_nan_gaps(X, wn)
        rows = []
        for entry in targets:
            bundle = entry.load()
            grid = np.asarray(bundle["wavenumbers"], dtype=float)
            values = np.full(n, np.nan)
            lo = np.full(n, np.nan)
            hi = np.full(n, np.nan)
            notes = [""] * n

            if finite_rows.any():
                grid_tol = 2.0 * float(np.median(np.abs(np.diff(grid)))) if grid.size > 1 else 0.0
                Xr = resample_spectra(X[finite_rows], wn, grid, coverage_tol=grid_tol)
                Xp = bundle["preprocessor"].transform(Xr)
                pred = np.asarray(bundle["model"].predict(Xp), dtype=float)
                q = (entry.conformal or {}).get("q")
                if entry.log_transform:
                    values[finite_rows] = np.expm1(pred)
                    if q is not None:
                        lo[finite_rows] = np.expm1(pred - q)
                        hi[finite_rows] = np.expm1(pred + q)
                else:
                    values[finite_rows] = pred
                    if q is not None:
                        lo[finite_rows] = pred - q
                        hi[finite_rows] = pred + q

            ad_ok = np.full(n, np.nan)
            t2_ratio = np.full(n, np.nan)
            ad_note = ""
            if entry.ad_ref is not None and finite_rows.any():
                try:
                    ratio = self._ad_ratio(X[finite_rows], wn, entry.ad_ref)
                    t2_ratio[finite_rows] = ratio
                    ad_ok[finite_rows] = (ratio <= 1.0).astype(float)
                except ValueError:
                    ad_note = "AD nicht bewertbar (Gitterabdeckung unzureichend)"

            thr = (entry.screening_threshold or {}).get("value")
            for i in range(n):
                if not finite_rows[i]:
                    notes[i] = "Spektrum zu lückenhaft — keine Vorhersage"
                elif n_filled[i] > 0:
                    notes[i] = f"{n_filled[i]} Bänder interpoliert (Lücken gefüllt)"
                    if ad_note:
                        notes[i] += "; " + ad_note
                elif ad_note:
                    notes[i] = ad_note
                rows.append({
                    "sample_id": sample_ids[i],
                    "model_id": entry.id,
                    "target": entry.target,
                    "display_name": entry.display_name,
                    "unit": entry.unit,
                    "group": entry.group,
                    "origin": entry.origin,
                    "algo": entry.algo,
                    "value": values[i],
                    "lo": lo[i],
                    "hi": hi[i],
                    "quality_class": entry.quality_class,
                    "ad_ok": bool(ad_ok[i]) if np.isfinite(ad_ok[i]) else None,
                    "t2_ratio": t2_ratio[i] if np.isfinite(t2_ratio[i]) else None,
                    "ampel": _ampel(values[i], lo[i], hi[i], thr),
                    "threshold": thr,
                    "note": notes[i],
                })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------

    def _ad_ratio(self, X: np.ndarray, wn: np.ndarray, ad_path: Path) -> np.ndarray:
        """Hotelling T² of samples relative to the AD threshold (ratio > 1 = outside)."""
        if ad_path not in self._ad_cache:
            self._ad_cache[ad_path] = joblib.load(ad_path)
        art = self._ad_cache[ad_path]
        ad_grid = np.asarray(art["wavenumbers"], dtype=float)
        tol = 2.0 * float(np.median(np.abs(np.diff(ad_grid)))) if ad_grid.size > 1 else 0.0
        Xr = resample_spectra(X, wn, ad_grid, coverage_tol=tol)
        Xp = art["preprocessor"].transform(Xr)
        det = art["detector"]
        return det.score_samples(Xp) / det.threshold_


def _fill_nan_gaps(
    X: np.ndarray, wn: np.ndarray, max_missing_frac: float = 0.5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate NaN gaps per spectrum along the wavenumber axis.

    A single missing band (or short run) is filled by linear interpolation
    from its finite neighbours rather than discarding the whole spectrum.
    A row is only marked unusable when more than ``max_missing_frac`` of its
    bands are NaN — too sparse to interpolate meaningfully.

    Returns (X_filled, usable_rows, n_filled) where ``n_filled[i]`` is how
    many bands were interpolated for row i.
    """
    X = np.array(X, dtype=float, copy=True)
    n, b = X.shape
    usable = np.zeros(n, dtype=bool)
    n_filled = np.zeros(n, dtype=int)
    order = np.argsort(wn)          # np.interp needs ascending sample points
    wn_asc = np.asarray(wn, dtype=float)[order]
    for i in range(n):
        row = X[i]
        finite = np.isfinite(row)
        if finite.sum() < b * (1.0 - max_missing_frac):
            continue
        missing = ~finite
        if missing.any():
            rs = row[order]
            fs = np.isfinite(rs)
            rs[~fs] = np.interp(wn_asc[~fs], wn_asc[fs], rs[fs])
            X[i, order] = rs
            n_filled[i] = int(missing.sum())
        usable[i] = True
    return X, usable, n_filled


def _ampel(value: float, lo: float, hi: float, threshold: float | None) -> str | None:
    """Interval-aware traffic light vs a regulatory threshold."""
    if threshold is None or not np.isfinite(value):
        return None
    lo_eff = lo if np.isfinite(lo) else value
    hi_eff = hi if np.isfinite(hi) else value
    if lo_eff > threshold:
        return "rot"
    if hi_eff < threshold:
        return "gruen"
    return "gelb"


def validate_manifest(manifest_path: str | Path) -> list[str]:
    """Return a list of problems (empty = valid). Loads every enabled bundle."""
    problems: list[str] = []
    manifest_path = Path(manifest_path)
    try:
        engine = PredictionEngine(manifest_path)
    except Exception as exc:  # noqa: BLE001 — report, don't crash validation
        return [f"Manifest nicht ladbar: {exc}"]

    for e in engine.entries.values():
        if not e.path.exists():
            problems.append(f"{e.id}: Modelldatei fehlt: {e.path}")
            continue
        if e.ad_ref is not None and not e.ad_ref.exists():
            problems.append(f"{e.id}: AD-Artefakt fehlt: {e.ad_ref}")
        if not e.enabled:
            continue
        try:
            bundle = e.load()
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{e.id}: Bundle nicht ladbar: {exc}")
            continue
        for key in ("model", "preprocessor", "wavenumbers"):
            if key not in bundle:
                problems.append(f"{e.id}: Bundle-Schlüssel fehlt: {key!r}")
        grid = np.asarray(bundle.get("wavenumbers", []), dtype=float)
        if grid.size and e.grid:
            if (abs(grid.min() - e.grid.get("min", grid.min())) > 1.0
                    or abs(grid.max() - e.grid.get("max", grid.max())) > 1.0
                    or (e.grid.get("n") and grid.size != e.grid["n"])):
                problems.append(f"{e.id}: Manifest-Grid ≠ Bundle-Grid")
        if bool(bundle.get("log_transform", False)) != e.log_transform:
            problems.append(f"{e.id}: log_transform Manifest ≠ Bundle")
    return problems

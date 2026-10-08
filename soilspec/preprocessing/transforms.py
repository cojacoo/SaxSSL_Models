"""
Spectral preprocessing transforms.

All functions operate on 2-D arrays (n_samples, n_bands) and return the same shape.
The Preprocessor class chains transforms and is pickle-serializable for model bundles.

References
----------
- SNV: Barnes, R.J., Dhanoa, M.S. & Lister, S.J. (1989) Standard normal variate
  transformation and de-trending of near-infrared diffuse reflectance spectra.
  Appl. Spectrosc. 43(5), 772-777. doi:10.1366/0003702894202201
- MSC: Geladi, P., MacDougall, D. & Martens, H. (1985) Linearization and
  scatter-correction for near-infrared reflectance spectra of meat.
  Appl. Spectrosc. 39(3), 491-500. doi:10.1366/0003702854248656
- Savitzky-Golay: Savitzky, A. & Golay, M.J.E. (1964) Smoothing and
  differentiation of data by simplified least squares procedures.
  Anal. Chem. 36(8), 1627-1639. doi:10.1021/ac60214a047
- SQuaP (Spectral Quality Protocol): Poppiel, R., Paiva, A. & Demattê, J.
  (2022) Bridging the gap between soil spectroscopy and traditional laboratory:
  insights for routine implementation. Geoderma 425, 116029.
  doi:10.1016/j.geoderma.2022.116029; see also Yan, C. (2025) A review on
  spectral data preprocessing techniques for machine learning and quantitative
  analysis. iScience 28, 112759. doi:10.1016/j.isci.2025.112759.
  The implementation here is an explicitly simplified baseline+clip variant.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter as _savgol
from typing import Sequence


# ---------------------------------------------------------------------------
# Functional API
# ---------------------------------------------------------------------------

def snv(X: np.ndarray) -> np.ndarray:
    """Standard Normal Variate: subtract row mean, divide by row std."""
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return (X - mean) / std


def msc(X: np.ndarray, reference: np.ndarray | None = None) -> np.ndarray:
    """
    Multiplicative Scatter Correction.

    Parameters
    ----------
    X : (n_samples, n_bands)
    reference : (n_bands,) ideal reference spectrum.
        Defaults to the column mean of X.
    """
    ref = X.mean(axis=0) if reference is None else reference
    out = np.empty_like(X, dtype=float)
    for i in range(len(X)):
        coef = np.polyfit(ref, X[i], 1)
        out[i] = (X[i] - coef[1]) / coef[0]
    return out


def savgol(
    X: np.ndarray,
    window_length: int = 11,
    polyorder: int = 3,
    deriv: int = 0,
) -> np.ndarray:
    """Savitzky-Golay smoothing / differentiation, applied row-wise."""
    return np.apply_along_axis(
        _savgol, 1, X, window_length=window_length, polyorder=polyorder, deriv=deriv
    )


def derivative(X: np.ndarray, order: int = 1) -> np.ndarray:
    """Finite difference derivative along the spectral axis."""
    return np.diff(X, n=order, axis=1, prepend=np.repeat(X[:, :order], order, axis=1))


def squap(X: np.ndarray) -> np.ndarray:
    """
    Simplified Spectral Quality Protocol (SQuaP).

    Applies baseline correction (subtract row minimum) and clips the
    lowest 5th-percentile of intensities. Full physics-guided SQuaP is
    implemented as an extension of this baseline.

    References
    ----------
    Poppiel et al. (2022) Geoderma 425, 116029, doi:10.1016/j.geoderma.2022.116029;
    Yan (2025) iScience 28, 112759, doi:10.1016/j.isci.2025.112759.
    """
    X_out = X.copy().astype(float)
    baseline = X_out.min(axis=1, keepdims=True)
    X_out -= baseline
    floor = np.percentile(X_out, 5)
    X_out = np.clip(X_out, floor, None)
    return X_out


# ---------------------------------------------------------------------------
# Composable pipeline
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, callable] = {
    "raw": lambda X: X.copy(),
    "snv": snv,
    "msc": msc,
    "savgol": savgol,
    "derivative": derivative,
    "squap": squap,
}


class Preprocessor:
    """
    Chainable, serializable preprocessing pipeline.

    Parameters
    ----------
    steps : sequence of str
        Names from: ``raw``, ``snv``, ``msc``, ``savgol``, ``derivative``, ``squap``.
    kwargs : dict[str, dict]
        Per-step keyword arguments, e.g. ``{"savgol": {"window_length": 15}}``.

    Examples
    --------
    >>> p = Preprocessor(["snv", "savgol"], kwargs={"savgol": {"deriv": 1}})
    >>> X_out = p.fit_transform(X_raw)
    >>> X_new = p.transform(X_new_raw)   # uses cached MSC reference if applicable
    """

    def __init__(
        self,
        steps: Sequence[str] = ("squap",),
        kwargs: dict | None = None,
    ):
        for s in steps:
            if s not in _REGISTRY:
                raise ValueError(f"Unknown preprocessing step '{s}'. Valid: {list(_REGISTRY)}")
        self.steps = list(steps)
        self.kwargs = kwargs or {}
        self._msc_reference: np.ndarray | None = None
        self._fitted = False

    def fit(self, X: np.ndarray) -> "Preprocessor":
        """Learn any data-dependent parameters (MSC reference)."""
        if "msc" in self.steps:
            self._msc_reference = X.mean(axis=0)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply all steps in order."""
        if not self._fitted:
            raise RuntimeError("Call fit() or fit_transform() before transform().")
        out = X
        for step in self.steps:
            kw = self.kwargs.get(step, {})
            if step == "msc":
                out = msc(out, reference=self._msc_reference, **kw)
            else:
                out = _REGISTRY[step](out, **kw)
        return out

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def get_params(self) -> dict:
        return {"steps": self.steps, "kwargs": self.kwargs}

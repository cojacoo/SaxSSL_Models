"""
Resample spectra onto a reference wavenumber grid.

Bruker OPUS files return descending wavenumbers while model grids are stored
ascending; this module normalises orientation before interpolating, so callers
can pass either.
"""

from __future__ import annotations

import numpy as np


def resample_spectra(
    X: np.ndarray,
    wn_from: np.ndarray,
    wn_to: np.ndarray,
    check_coverage: bool = True,
    coverage_tol: float = 0.0,
) -> np.ndarray:
    """
    Interpolate spectra row-wise from ``wn_from`` onto ``wn_to``.

    Parameters
    ----------
    X : (n_samples, n_bands_from)
        Spectra on the source grid. Ascending or descending source grids are
        both accepted (descending input is flipped internally).
    wn_from : (n_bands_from,)
        Source wavenumbers, strictly monotonic.
    wn_to : (n_bands_to,)
        Target wavenumbers, strictly monotonic.
    check_coverage : bool
        If True (default), raise when the target grid extends beyond the
        source range — np.interp would silently clamp to edge values there.
    coverage_tol : float
        Allowed overhang in cm⁻¹ before the coverage check raises. A small
        tolerance (a few cm⁻¹) absorbs instrument edge-range differences,
        where only one or two edge bands get clamped; larger mismatches
        (e.g. a VISNIR grid) still raise.

    Returns
    -------
    (n_samples, n_bands_to) float64
        Spectra on the target grid, oriented like ``wn_to``.
    """
    X = np.asarray(X, dtype=float)
    wn_from = np.asarray(wn_from, dtype=float)
    wn_to = np.asarray(wn_to, dtype=float)

    if X.ndim == 1:
        X = X[None, :]
    if X.shape[1] != wn_from.size:
        raise ValueError(
            f"X has {X.shape[1]} bands but wn_from has {wn_from.size}"
        )

    d_from = np.diff(wn_from)
    if not (np.all(d_from > 0) or np.all(d_from < 0)):
        raise ValueError("wn_from must be strictly monotonic")
    if wn_from[0] > wn_from[-1]:
        wn_from = wn_from[::-1]
        X = X[:, ::-1]

    d_to = np.diff(wn_to)
    if wn_to.size > 1 and not (np.all(d_to > 0) or np.all(d_to < 0)):
        raise ValueError("wn_to must be strictly monotonic")
    flip_out = wn_to.size > 1 and wn_to[0] > wn_to[-1]
    wn_to_asc = wn_to[::-1] if flip_out else wn_to

    tol = max(coverage_tol, 1e-9)
    if check_coverage and (
        wn_to_asc[0] < wn_from[0] - tol or wn_to_asc[-1] > wn_from[-1] + tol
    ):
        raise ValueError(
            f"Target grid {wn_to_asc[0]:.1f}-{wn_to_asc[-1]:.1f} cm⁻¹ exceeds "
            f"source coverage {wn_from[0]:.1f}-{wn_from[-1]:.1f} cm⁻¹"
        )

    out = np.empty((X.shape[0], wn_to_asc.size), dtype=float)
    for i in range(X.shape[0]):
        out[i] = np.interp(wn_to_asc, wn_from, X[i])
    return out[:, ::-1] if flip_out else out

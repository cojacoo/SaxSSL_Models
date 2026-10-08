"""
PCA-based spectral outlier detection via Hotelling T² statistic.

Samples whose Hotelling T² exceeds the F-distribution critical value at
significance level alpha are flagged as outliers.

Reference: Wadoux, A.M.J.-C., Malone, B., Minasny, B., Fajardo, M. &
McBratney, A.B. (2021) Soil Spectral Inference with R. Springer.
doi:10.1007/978-3-030-64896-1
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class PCOutlierDetector:
    """
    Detect spectral outliers using the Hotelling T² statistic on PCA scores.

    Parameters
    ----------
    n_components : int or None
        Number of PCA components to retain.  If None, components explaining
        cumulative variance ≥ ``var_threshold`` are used.
    var_threshold : float
        Cumulative variance threshold when ``n_components`` is None (0.99).
    alpha : float
        Significance level for the F-distribution critical value (0.05).
    scale : bool
        Standardise spectra before PCA (recommended).
    """

    def __init__(
        self,
        n_components: int | None = None,
        var_threshold: float = 0.99,
        alpha: float = 0.05,
        scale: bool = True,
    ):
        self.n_components = n_components
        self.var_threshold = var_threshold
        self.alpha = alpha
        self.scale = scale

        self.scaler_: StandardScaler | None = None
        self.pca_: PCA | None = None
        self.threshold_: float | None = None
        self._n: int | None = None       # training set size
        self._p: int | None = None       # number of PCs used

    def fit(self, X: np.ndarray) -> "PCOutlierDetector":
        """
        Fit PCA on training spectra and compute the T² critical value.

        Parameters
        ----------
        X : (n_samples, n_bands)
        """
        n, _ = X.shape

        if self.scale:
            self.scaler_ = StandardScaler()
            Xs = self.scaler_.fit_transform(X)
        else:
            Xs = X.copy().astype(float)

        # Determine number of components
        if self.n_components is not None:
            p = min(self.n_components, n - 1, Xs.shape[1])
        else:
            pca_full = PCA().fit(Xs)
            cum_var = np.cumsum(pca_full.explained_variance_ratio_)
            p = int(np.searchsorted(cum_var, self.var_threshold) + 1)
            p = max(1, min(p, n - 1))

        self.pca_ = PCA(n_components=p)
        self.pca_.fit(Xs)
        self._n = n
        self._p = p

        # Hotelling T² critical value via F-distribution
        # T² ~ p(n-1)(n+1) / n(n-p) * F(p, n-p, alpha)
        f_crit = stats.f.ppf(1 - self.alpha, dfn=p, dfd=n - p)
        self.threshold_ = (p * (n - 1) * (n + 1)) / (n * (n - p)) * f_crit

        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Return the Hotelling T² statistic for each sample.

        Parameters
        ----------
        X : (n_samples, n_bands)

        Returns
        -------
        t2 : (n_samples,) float — larger = more outlier-like.
        """
        self._check_fitted()
        Xs = self.scaler_.transform(X) if self.scale else X.astype(float)
        scores = self.pca_.transform(Xs)

        # Variances of each PC from training
        var = self.pca_.explained_variance_
        # Mahalanobis distance in PC space = Hotelling T²
        t2 = np.sum(scores ** 2 / var, axis=1)
        return t2

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return a boolean mask: True where a sample is an outlier.

        Parameters
        ----------
        X : (n_samples, n_bands)

        Returns
        -------
        outlier_mask : (n_samples,) bool
        """
        return self.score_samples(X) > self.threshold_

    def flag_outliers(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Return indices of outliers and their T² statistics.

        Returns
        -------
        outlier_idx : int array
        t2_scores   : float array (same length)
        """
        t2 = self.score_samples(X)
        mask = t2 > self.threshold_
        return np.where(mask)[0], t2[mask]

    def summary(self, X: np.ndarray) -> dict:
        """Return a human-readable summary dict."""
        t2 = self.score_samples(X)
        n_out = int((t2 > self.threshold_).sum())
        return {
            "n_samples": len(X),
            "n_components_used": self._p,
            "variance_explained": float(self.pca_.explained_variance_ratio_.sum()),
            "threshold_t2": float(self.threshold_),
            "alpha": self.alpha,
            "n_outliers": n_out,
            "outlier_fraction": float(n_out / len(X)),
            "t2_mean": float(t2.mean()),
            "t2_max": float(t2.max()),
        }

    def _check_fitted(self) -> None:
        if self.pca_ is None:
            raise RuntimeError("Call fit() before scoring.")

"""
Thread-safe model wrappers for XGBoost, LightGBM, and PLS.

Forces single-threaded execution to prevent mutex conflicts when running
alongside PyTorch/MPS on Apple Silicon.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.cross_decomposition import PLSRegression


class ThreadSafeXGBoost:
    _defaults = dict(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.5, reg_lambda=1.0,
        random_state=42, verbosity=0,
        n_jobs=1, nthread=1,
    )

    def __init__(self, **kwargs):
        params = {**self._defaults, **kwargs, "n_jobs": 1, "nthread": 1}
        self.model = XGBRegressor(**params)

    def fit(self, X, y, **kw):
        return self.model.fit(X, y, **kw)

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    def feature_importances(self, importance_type: str = "weight") -> dict:
        return self.model.get_booster().get_score(importance_type=importance_type)

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.model.feature_importances_

    def get_params(self) -> dict:
        return self.model.get_params()


class ThreadSafeLightGBM:
    _defaults = dict(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.5, reg_lambda=1.0,
        random_state=42, verbose=-1,
        n_jobs=1, num_threads=1,
    )

    def __init__(self, **kwargs):
        params = {**self._defaults, **kwargs, "n_jobs": 1, "num_threads": 1}
        self.model = LGBMRegressor(**params)

    def fit(self, X, y, **kw):
        return self.model.fit(X, y, **kw)

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    def feature_importances(self) -> np.ndarray:
        return self.model.feature_importances_

    def get_params(self) -> dict:
        return self.model.get_params()


class ThreadSafePLS:
    _defaults = dict(n_components=5, scale=True, max_iter=500, tol=1e-6)

    def __init__(self, **kwargs):
        params = {**self._defaults, **kwargs}
        self.model = PLSRegression(**params)

    def fit(self, X, y, **kw):
        return self.model.fit(X, y, **kw)

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X).ravel()

    @property
    def x_loadings(self) -> np.ndarray:
        return self.model.x_loadings_

    @property
    def x_weights_(self) -> np.ndarray:
        return self.model.x_weights_

    def get_params(self) -> dict:
        return self.model.get_params()


class CubistWrapper:
    """Sklearn-compatible Cubist wrapper that accepts ndarray input.

    Cubist requires a named-column DataFrame; this wrapper converts internally
    so callers can treat it like any other sklearn estimator.
    """

    _defaults = dict(n_committees=20, random_state=42)

    def __init__(self, **kwargs):
        try:
            from cubist import Cubist
        except ImportError as exc:
            raise ImportError(
                "cubist package required. Install via: pip install cubist"
            ) from exc
        params = {**self._defaults, **kwargs}
        self.model = Cubist(**params)
        self._col_names: list[str] | None = None

    def fit(self, X, y, **kw):
        X = np.asarray(X)
        self._col_names = [f"f{i}" for i in range(X.shape[1])]
        return self.model.fit(pd.DataFrame(X, columns=self._col_names), y, **kw)

    def predict(self, X) -> np.ndarray:
        if self._col_names is None:
            raise RuntimeError("Call fit() before predict().")
        return np.asarray(self.model.predict(
            pd.DataFrame(np.asarray(X), columns=self._col_names)
        ))

    @property
    def feature_importances_(self) -> pd.DataFrame:
        return self.model.feature_importances_

    def get_params(self) -> dict:
        return self.model.get_params()


def get_classical_models(model_configs: dict | None = None) -> dict:
    """
    Return thread-safe classical models including Cubist.

    Parameters
    ----------
    model_configs : dict, optional
        Override params: ``{"PLS": {"n_components": 8}, "Cubist": {"n_committees": 10}, ...}``
    """
    cfg = model_configs or {}
    return {
        "PLS":      ThreadSafePLS(**cfg.get("PLS", {})),
        "XGBoost":  ThreadSafeXGBoost(**cfg.get("XGBoost", {})),
        "LightGBM": ThreadSafeLightGBM(**cfg.get("LightGBM", {})),
        "Cubist":   CubistWrapper(**cfg.get("Cubist", {})),
    }


def get_deep_models(model_configs: dict | None = None) -> dict:
    """
    Return a dict containing CNNRegressor (requires soilspec[deep]).

    Parameters
    ----------
    model_configs : dict, optional
        Override params: ``{"CNN": {"max_epochs": 300, "verbose": True}, ...}``
    """
    try:
        from ..deep.cnn import CNNRegressor
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for deep models. "
            "Install via: pip install soilspec[deep]"
        ) from exc

    cfg = model_configs or {}
    return {
        "CNN": CNNRegressor(**cfg.get("CNN", {})),
    }


def get_threadsafe_models(model_configs: dict | None = None) -> dict:
    """
    Factory that returns all three thread-safe model instances.

    Parameters
    ----------
    model_configs : dict, optional
        Override params per model: ``{"XGBoost": {"n_estimators": 300}, ...}``
    """
    cfg = model_configs or {}
    return {
        "XGBoost":  ThreadSafeXGBoost(**cfg.get("XGBoost", {})),
        "LightGBM": ThreadSafeLightGBM(**cfg.get("LightGBM", {})),
        "PLS":      ThreadSafePLS(**cfg.get("PLS", {})),
    }

"""Model artifact for the global XGBoost calendar/quantile predictor.

A single booster trained with ``reg:quantileerror`` emits several quantiles in
one call, so one global model (with a categorical ``airport`` column) serves
every airport and every horizon. The bundle bakes in the seasonal *profiles*
it was trained with, so serving needs no database access at request time.

Bundles are persisted with joblib and hot-reloaded by the serving app when the
file's mtime changes. ``ConstantBundle`` is a always-available fallback so
``/predict`` still answers before the first real model has been trained.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import features as features_mod
from features import FEATURE_COLUMNS, VALID_AIRPORTS, Profiles

MODEL_FILENAME = "model.joblib"
# p10 / p50 / p90: a point estimate (p50) plus a symmetric uncertainty band.
DEFAULT_QUANTILES = [0.1, 0.5, 0.9]

# Shared booster hyper-parameters. ``reg:quantileerror`` + a vector
# ``quantile_alpha`` (XGBoost >= 2.0) fits all quantiles in a single model.
BASE_PARAMS = {
    "objective": "reg:quantileerror",
    "tree_method": "hist",
    "eta": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "seed": 7,
}


def quantile_column_names(quantile_alpha) -> List[str]:
    """[0.1, 0.5, 0.9] -> ['q10', 'q50', 'q90']."""
    return [f"q{int(round(q * 100))}" for q in quantile_alpha]


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class ModelBundle:
    """A trained global booster plus everything needed to serve it statelessly."""

    booster: object  # xgboost.Booster (kept untyped to avoid an import here)
    feature_names: List[str]
    quantile_alpha: List[float]
    profiles: Dict[str, Profiles]
    trained_at: str
    metrics: dict = field(default_factory=dict)
    best_iteration: Optional[int] = None

    @property
    def quantile_columns(self) -> List[str]:
        return quantile_column_names(self.quantile_alpha)

    def _feature_frame(self, airport: str, timestamps) -> pd.DataFrame:
        profiles = self.profiles.get(airport)
        if profiles is None:
            profiles = features_mod.build_profiles(pd.Series(dtype=float), airport)
        feats = features_mod.compute_features(timestamps, profiles).copy()
        feats["airport"] = pd.Categorical(
            [airport] * len(feats), categories=VALID_AIRPORTS
        )
        return feats

    def predict_frame(self, airport: str, timestamps) -> pd.DataFrame:
        """Return a DataFrame of quantile predictions indexed by ``timestamps``.

        Predictions are clipped to be non-negative and sorted across quantiles
        so the bands never cross (independently-fit quantiles otherwise can).
        """
        import xgboost as xgb

        idx = pd.DatetimeIndex(timestamps)
        cols = self.quantile_columns
        if len(idx) == 0:
            return pd.DataFrame(columns=cols)

        feats = self._feature_frame(airport, idx)
        dmatrix = xgb.DMatrix(feats, enable_categorical=True)
        if self.best_iteration is not None:
            preds = self.booster.predict(
                dmatrix, iteration_range=(0, self.best_iteration + 1)
            )
        else:
            preds = self.booster.predict(dmatrix)

        preds = np.asarray(preds, dtype=float)
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)
        preds = np.maximum(0.0, preds)
        preds = np.sort(preds, axis=1)
        return pd.DataFrame(preds, columns=cols, index=idx)


@dataclass
class ConstantBundle:
    """Fallback used until a real model exists; predicts a flat value."""

    value: float = 5.0
    quantile_alpha: List[float] = field(default_factory=lambda: list(DEFAULT_QUANTILES))
    trained_at: str = field(default_factory=_utcnow_iso)
    metrics: dict = field(default_factory=lambda: {"fallback": True})
    best_iteration: Optional[int] = None

    @property
    def quantile_columns(self) -> List[str]:
        return quantile_column_names(self.quantile_alpha)

    def predict_frame(self, airport: str, timestamps) -> pd.DataFrame:
        idx = pd.DatetimeIndex(timestamps)
        cols = self.quantile_columns
        data = {c: np.full(len(idx), float(self.value)) for c in cols}
        return pd.DataFrame(data, index=idx, columns=cols)


def train_booster(
    x_train,
    y_train,
    x_valid=None,
    y_valid=None,
    quantile_alpha=None,
    num_boost_round: int = 1000,
    early_stopping_rounds: Optional[int] = 50,
    params: Optional[dict] = None,
):
    """Train one multi-quantile booster; returns ``(booster, quantile_alpha, best_iteration)``.

    Falls back to a fixed number of rounds if early stopping is unavailable
    (e.g. no validation split, or the installed XGBoost cannot early-stop on the
    vectorised quantile metric).
    """
    import xgboost as xgb

    quantile_alpha = sorted(quantile_alpha or DEFAULT_QUANTILES)
    train_params = dict(BASE_PARAMS)
    train_params["quantile_alpha"] = quantile_alpha
    if params:
        train_params.update(params)

    dtrain = xgb.DMatrix(x_train, label=np.asarray(y_train, dtype=float), enable_categorical=True)
    evals = [(dtrain, "train")]
    dvalid = None
    if x_valid is not None and len(x_valid):
        dvalid = xgb.DMatrix(
            x_valid, label=np.asarray(y_valid, dtype=float), enable_categorical=True
        )
        evals.append((dvalid, "valid"))

    def _fit(with_early_stop: bool):
        kwargs = {}
        if with_early_stop and dvalid is not None and early_stopping_rounds:
            kwargs["early_stopping_rounds"] = early_stopping_rounds
        return xgb.train(
            train_params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
            verbose_eval=False,
            **kwargs,
        )

    try:
        booster = _fit(with_early_stop=True)
    except xgb.core.XGBoostError:
        booster = _fit(with_early_stop=False)

    best_iteration = getattr(booster, "best_iteration", None)
    return booster, quantile_alpha, best_iteration


def save_bundle(bundle, models_dir: str, filename: str = MODEL_FILENAME) -> str:
    """Atomically persist a bundle so readers never see a half-written file."""
    import joblib

    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, filename)
    fd, tmp = tempfile.mkstemp(dir=models_dir, suffix=".tmp")
    os.close(fd)
    try:
        joblib.dump(bundle, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return path


def load_bundle(models_dir: str, filename: str = MODEL_FILENAME):
    """Load a persisted bundle, or ``None`` if none exists yet."""
    import joblib

    path = os.path.join(models_dir, filename)
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def model_path(models_dir: str, filename: str = MODEL_FILENAME) -> str:
    return os.path.join(models_dir, filename)

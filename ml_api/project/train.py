"""Standalone trainer for the global XGBoost quantile model.

Run out-of-process (cron / k8s CronJob) so the serving replicas never train.
It writes a single ``model.joblib`` bundle plus ``metrics.json`` into
``MODELS_DIR``; the serving app hot-reloads the bundle when its mtime changes.

    python train.py                      # all airports, last TRAIN_DAYS days
    python train.py --airports CPH ARN --train-days 30

Honest metrics come from a time-based holdout (the last ``HOLDOUT_DAYS`` days,
with profiles built only from the earlier data). The shipped model is then
refit on the full window with full-history profiles.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import data as data_mod
import features as features_mod
import model as model_mod

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train")

MODELS_DIR = os.environ.get("MODELS_DIR", "models")
TRAIN_DAYS = int(os.environ.get("TRAIN_DAYS", "365"))
HOLDOUT_DAYS = int(os.environ.get("HOLDOUT_DAYS", "7"))
NUM_BOOST_ROUND = int(os.environ.get("NUM_BOOST_ROUND", "1000"))
EARLY_STOPPING_ROUNDS = int(os.environ.get("EARLY_STOPPING_ROUNDS", "50"))


def _split_histories(histories, holdout_days):
    """Split each airport Series at a single global cutoff -> (train, valid)."""
    latest = None
    for series in histories.values():
        if series is not None and len(series):
            top = series.index.max()
            latest = top if latest is None else max(latest, top)
    if latest is None:
        return {}, {}, None

    cutoff = latest - pd.Timedelta(days=holdout_days)
    train_hist, valid_hist = {}, {}
    for airport, series in histories.items():
        if series is None or len(series) == 0:
            continue
        train_hist[airport] = series[series.index < cutoff]
        valid_hist[airport] = series[series.index >= cutoff]
    return train_hist, valid_hist, cutoff


def _pinball(y, pred, q):
    diff = y - pred
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def _validate(booster, best_iteration, x_valid, y_valid, quantile_alpha):
    """Point + probabilistic metrics of the booster on the holdout frame."""
    import xgboost as xgb

    if x_valid is None or len(x_valid) == 0:
        return {}

    dvalid = xgb.DMatrix(x_valid, enable_categorical=True)
    if best_iteration is not None:
        preds = booster.predict(dvalid, iteration_range=(0, best_iteration + 1))
    else:
        preds = booster.predict(dvalid)
    preds = np.sort(np.maximum(0.0, np.asarray(preds, dtype=float)), axis=1)

    y = np.asarray(y_valid, dtype=float)
    cols = model_mod.quantile_column_names(quantile_alpha)
    median_idx = quantile_alpha.index(0.5) if 0.5 in quantile_alpha else len(quantile_alpha) // 2
    p50 = preds[:, median_idx]

    mae = float(np.mean(np.abs(y - p50)))
    rmse = float(np.sqrt(np.mean((y - p50) ** 2)))
    pinball = {cols[i]: _pinball(y, preds[:, i], q) for i, q in enumerate(quantile_alpha)}
    denom = float(np.mean(np.abs(y))) or 1.0
    wql = float(np.mean(list(pinball.values())) / denom)
    return {
        "MAE": mae,
        "RMSE": rmse,
        "WQL": wql,
        "pinball": pinball,
        "n_valid": int(len(y)),
    }


def train(models_dir=MODELS_DIR, airports=None, train_days=TRAIN_DAYS,
          holdout_days=HOLDOUT_DAYS, num_boost_round=NUM_BOOST_ROUND,
          early_stopping_rounds=EARLY_STOPPING_ROUNDS):
    airports = airports or features_mod.VALID_AIRPORTS
    base_url = data_mod.DEFAULT_BASE_URL
    log.info("Loading %d days of history for %s from %s",
             train_days, ",".join(airports), base_url)
    histories = data_mod.load_histories(base_url, airports=airports, days=train_days)
    total_rows = sum(len(s) for s in histories.values() if s is not None)
    log.info("Loaded %d observations across %d airports", total_rows, len(histories))
    if total_rows == 0:
        raise SystemExit("No data returned from PostgREST; aborting training.")

    # --- Honest holdout metrics (profiles from the train portion only) ---
    train_hist, valid_hist, cutoff = _split_histories(histories, holdout_days)
    train_profiles = {ap: features_mod.build_profiles(s, ap) for ap, s in train_hist.items()}
    x_tr, y_tr = features_mod.build_training_frame(train_hist, train_profiles)
    x_va, y_va = features_mod.build_training_frame(valid_hist, train_profiles)
    log.info("Holdout split at %s: train=%d valid=%d", cutoff, len(x_tr), len(x_va))

    booster, quantile_alpha, best_iteration = model_mod.train_booster(
        x_tr, y_tr, x_va, y_va,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
    )
    metrics = _validate(booster, best_iteration, x_va, y_va, quantile_alpha)
    rounds = (best_iteration + 1) if best_iteration is not None else num_boost_round
    log.info("Holdout metrics: %s (best_iteration=%s)", metrics, best_iteration)

    # --- Final refit on the full window with full-history profiles ---
    full_profiles = {ap: features_mod.build_profiles(s, ap)
                     for ap, s in histories.items() if s is not None and len(s)}
    x_full, y_full = features_mod.build_training_frame(histories, full_profiles)
    final_booster, quantile_alpha, _ = model_mod.train_booster(
        x_full, y_full,
        num_boost_round=rounds,
        early_stopping_rounds=None,
    )

    bundle = model_mod.ModelBundle(
        booster=final_booster,
        feature_names=features_mod.FEATURE_COLUMNS + ["airport"],
        quantile_alpha=quantile_alpha,
        profiles=full_profiles,
        trained_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        metrics=metrics,
        best_iteration=None,  # final model is trained for exactly ``rounds``
    )
    path = model_mod.save_bundle(bundle, models_dir)
    log.info("Saved model bundle -> %s", path)

    meta = {
        "trained_at": bundle.trained_at,
        "train_days": train_days,
        "holdout_days": holdout_days,
        "cutoff": None if cutoff is None else cutoff.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rounds": int(rounds),
        "quantile_alpha": quantile_alpha,
        "airports": list(histories.keys()),
        "n_train_full": int(len(x_full)),
        "metrics": metrics,
    }
    meta_path = os.path.join(models_dir, "metrics.json")
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    log.info("Wrote training metadata -> %s", meta_path)
    return bundle, meta


def _parse_args():
    p = argparse.ArgumentParser(description="Train the global XGBoost quantile model.")
    p.add_argument("--models-dir", default=MODELS_DIR)
    p.add_argument("--airports", nargs="*", default=None)
    p.add_argument("--train-days", type=int, default=TRAIN_DAYS)
    p.add_argument("--holdout-days", type=int, default=HOLDOUT_DAYS)
    p.add_argument("--num-boost-round", type=int, default=NUM_BOOST_ROUND)
    p.add_argument("--early-stopping-rounds", type=int, default=EARLY_STOPPING_ROUNDS)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        models_dir=args.models_dir,
        airports=args.airports,
        train_days=args.train_days,
        holdout_days=args.holdout_days,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
    )

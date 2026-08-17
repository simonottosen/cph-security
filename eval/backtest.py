"""Rolling-origin backtest harness for the waitport queue predictors.

Measures MAE / RMSE per (airport, horizon, model) plus weighted quantile loss
(WQL) for probabilistic models, comparing cheap baselines against the XGBoost
calendar predictor and (when the ``chronos`` package is installed) the batched
Chronos-2 forecaster. Results print as a table and persist under ``eval/results/``
so accuracy can be tracked over time (drift monitoring).

This harness is the accuracy gate for the plan: every model change must lower
MAE/RMSE/WQL versus both the incumbent behaviour and the naive baselines before
it ships.

Usage examples
--------------
    # Fast local smoke test (baselines + XGBoost only; Chronos auto-skipped):
    python eval/backtest.py --airports CPH ARN --backtest-days 7 \
        --origin-every-hours 12 --train-days 45

    # Fuller run (inside the forecast container where chronos is installed):
    python eval/backtest.py --backtest-days 45 --origin-every-hours 6
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML_API_PROJECT = os.path.join(REPO_ROOT, "ml_api", "project")
if ML_API_PROJECT not in sys.path:
    sys.path.insert(0, ML_API_PROJECT)

import features  # noqa: E402  (path injected above; reused by trainer + serving app)
import data  # noqa: E402  (shared PostgREST loader, reused by the trainer)

DEFAULT_BASE_URL = os.environ.get("WAITPORT_API_URL", "https://waitport.com/api/v1/all")
DEFAULT_AIRPORTS = list(features.VALID_AIRPORTS)
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# XGBoost quantile heads (also what B4 will expose from /predict).
XGB_QUANTILES = [0.1, 0.5, 0.9]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def fetch_history(base_url: str, airport: str, start_utc: pd.Timestamp) -> pd.Series:
    """Fetch an airport's queue history (>= start_utc) as a 5-min-gridded Series.

    Thin wrapper over :func:`data.fetch_series` so the harness and the trainer
    share one PostgREST loader (same flooring / de-duplication semantics).
    """
    return data.fetch_series(base_url, airport, start_utc)


def lookup_actuals(series: pd.Series, targets, tolerance_minutes: int = 5) -> pd.Series:
    """Nearest observed queue at each target timestamp, within tolerance."""
    if series.empty:
        return pd.Series(np.nan, index=pd.DatetimeIndex(targets))
    target_idx = pd.DatetimeIndex(targets)
    tol = pd.Timedelta(minutes=tolerance_minutes)
    return series.reindex(target_idx, method="nearest", tolerance=tol)


# ---------------------------------------------------------------------------
# Predictor protocol
# ---------------------------------------------------------------------------
class Predictor:
    """Origin-aware predictor. ``prepare`` does heavy per-origin work once."""

    name = "base"
    probabilistic = False

    def prepare(self, origin: pd.Timestamp, contexts: dict) -> None:
        """``contexts``: airport -> history Series sliced to <= origin."""

    def predict(self, airport: str, origin: pd.Timestamp, targets) -> pd.DataFrame:
        """Return a frame indexed by target timestamp with a ``mean`` column and,
        for probabilistic models, ``q{level}`` columns."""
        raise NotImplementedError


class LastValuePredictor(Predictor):
    name = "last_value"

    def prepare(self, origin, contexts):
        self._last = {ap: (s.iloc[-1] if len(s) else np.nan) for ap, s in contexts.items()}

    def predict(self, airport, origin, targets):
        val = self._last.get(airport, np.nan)
        return pd.DataFrame({"mean": val}, index=pd.DatetimeIndex(targets))


class SeasonalNaivePredictor(Predictor):
    """Same weekday + time-of-day one week earlier."""

    name = "seasonal_naive"

    def prepare(self, origin, contexts):
        self._contexts = contexts

    def predict(self, airport, origin, targets):
        series = self._contexts.get(airport)
        target_idx = pd.DatetimeIndex(targets)
        if series is None or series.empty:
            return pd.DataFrame({"mean": np.nan}, index=target_idx)
        lagged = [t - pd.Timedelta(days=7) for t in target_idx]
        vals = series.reindex(
            pd.DatetimeIndex(lagged), method="nearest", tolerance=pd.Timedelta(minutes=30)
        ).to_numpy()
        # Fall back to the last observed value where last week's slot is missing.
        vals = np.where(np.isnan(vals), series.iloc[-1], vals)
        return pd.DataFrame({"mean": vals}, index=target_idx)


class XGBoostCalendarPredictor(Predictor):
    """Global XGBoost model with categorical airport + quantile heads.

    Refits on all history <= origin at most every ``refit_hours``; the same
    booster then serves every airport at that origin. This is the Phase 2
    (B1/B2/B3/B4) design, validated here before it is wired into serving.
    """

    name = "xgboost"
    probabilistic = True

    def __init__(self, refit_hours: int = 24, train_days: int = 120):
        import xgboost as xgb  # local import so the harness runs without xgboost

        self._xgb = xgb
        self.refit_hours = refit_hours
        self.train_days = train_days
        self._model = None
        self._trained_at = None
        self._feature_names = None
        self._profiles = {}

    def prepare(self, origin, contexts):
        if self._trained_at is not None and (
            origin - self._trained_at < pd.Timedelta(hours=self.refit_hours)
        ):
            # Reuse the current booster; refresh profiles to the new cutoff so
            # feature lookups reflect the latest climatology.
            self._profiles = {
                ap: features.build_profiles(s, ap) for ap, s in contexts.items()
            }
            return

        train_cutoff = origin - pd.Timedelta(days=self.train_days)
        hist_by_ap, prof_by_ap = {}, {}
        for ap, s in contexts.items():
            prof_by_ap[ap] = features.build_profiles(s, ap)
            hist_by_ap[ap] = s[s.index >= train_cutoff]
        self._profiles = prof_by_ap

        x, y = features.build_training_frame(hist_by_ap, prof_by_ap)
        if len(x) < 500:
            self._model = None
            return

        self._feature_names = list(x.columns)
        dtrain = self._xgb.DMatrix(x, label=y.to_numpy(), enable_categorical=True)
        params = {
            "tree_method": "hist",
            "objective": "reg:quantileerror",
            "quantile_alpha": XGB_QUANTILES,
            "eta": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "seed": 7,
        }
        self._model = self._xgb.train(params, dtrain, num_boost_round=300)
        self._trained_at = origin

    def predict(self, airport, origin, targets):
        target_idx = pd.DatetimeIndex(targets)
        if self._model is None or airport not in self._profiles:
            return pd.DataFrame({"mean": np.nan}, index=target_idx)

        feats = features.compute_features(target_idx, self._profiles[airport]).copy()
        feats["airport"] = pd.Categorical([airport] * len(feats), categories=features.VALID_AIRPORTS)
        feats = feats[self._feature_names]
        dmat = self._xgb.DMatrix(feats, enable_categorical=True)
        preds = np.asarray(self._model.predict(dmat))
        if preds.ndim == 1:  # single-quantile fallback
            preds = preds[:, None]

        out = pd.DataFrame(index=target_idx)
        for i, q in enumerate(XGB_QUANTILES):
            col = preds[:, i] if i < preds.shape[1] else preds[:, -1]
            out[f"q{int(q * 100)}"] = np.maximum(0.0, col)
        median_col = f"q{int(0.5 * 100)}"
        out["mean"] = out[median_col] if median_col in out else out.iloc[:, 0]
        return out


class Chronos2Predictor(Predictor):
    """Batched Chronos-2 forecaster (one predict_df with cross_learning=True).

    Imports the forecast service's context helpers so the harness exercises the
    exact serving code path. Skipped automatically when chronos isn't installed.
    """

    name = "chronos2"
    probabilistic = True

    def __init__(self, cross_learning: bool = True):
        self.cross_learning = cross_learning
        self._mod = _load_forecast_module()
        self._pipeline = None
        self._forecasts = {}
        if self._mod is not None and getattr(self._mod, "BaseChronosPipeline", None) is not None:
            self.available = True
        else:
            self.available = False

    def _ensure_pipeline(self):
        if self._pipeline is None:
            self._pipeline = self._mod.get_chronos2_pipeline()
        return self._pipeline

    def prepare(self, origin, contexts):
        if not self.available:
            return
        mod = self._mod
        # Rebuild a raw long frame <= origin so the service helpers can slice it.
        frames = []
        for ap, s in contexts.items():
            if s is None or s.empty:
                continue
            frames.append(pd.DataFrame({"airport": ap, "timestamp": s.index, "queue": s.to_numpy()}))
        if not frames:
            self._forecasts = {}
            return
        df_raw = pd.concat(frames, ignore_index=True)

        contexts_dfs, futures_dfs = [], []
        for ap in contexts:
            ctx, _stats, cov = mod._prepare_airport_context(df_raw, ap)
            if ctx is None:
                continue
            contexts_dfs.append(ctx)
            futures_dfs.append(mod._build_future_covariates(ctx, ap, cov))
        if not contexts_dfs:
            self._forecasts = {}
            return

        context_df = pd.concat(contexts_dfs, ignore_index=True)
        future_df = pd.concat(futures_dfs, ignore_index=True)
        pipeline = self._ensure_pipeline()
        pred_df = pipeline.predict_df(
            context_df,
            future_df=future_df,
            cross_learning=self.cross_learning,
            prediction_length=mod.PREDICTION_LENGTH,
            quantile_levels=mod.QUANTILE_LEVELS,
            id_column="item_id",
            timestamp_column="timestamp",
            target="queue",
        )
        pred_df = pred_df.copy()
        pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"]).dt.tz_localize("UTC")
        self._forecasts = {ap: g.sort_values("timestamp") for ap, g in pred_df.groupby("item_id")}

    def predict(self, airport, origin, targets):
        target_idx = pd.DatetimeIndex(targets)
        fc = self._forecasts.get(airport)
        if fc is None or fc.empty:
            return pd.DataFrame({"mean": np.nan}, index=target_idx)
        indexed = fc.set_index("timestamp")
        # Match the forecast's own grid rather than a hard-coded 5 minutes: the
        # service emits 15-minute steps while targets are evaluated at 5, and a
        # fixed tolerance would silently score unmatched targets as missing data
        # the moment the two grids stopped lining up.
        tolerance = pd.Timedelta(getattr(self._mod, "RESAMPLE_FREQUENCY", "5min")) / 2
        near = indexed.reindex(target_idx, method="nearest", tolerance=tolerance)
        out = pd.DataFrame(index=target_idx)
        out["mean"] = near["mean"].to_numpy() if "mean" in near else np.nan
        for q in (0.1, 0.5, 0.9):
            col = f"{q:.1f}"
            if col in near:
                out[f"q{int(q * 100)}"] = near[col].to_numpy()
        return out


def _load_forecast_module():
    """Load the Chronos forecast service module by path (side-effect free at import)."""
    for rel in ("autogluon/project/app.py", "forecast_api/project/app.py"):
        path = os.path.join(REPO_ROOT, rel)
        if os.path.exists(path):
            try:
                spec = importlib.util.spec_from_file_location("waitport_forecast_app", path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return mod
            except Exception as exc:  # noqa: BLE001
                print(f"[chronos] could not import {rel}: {exc}", file=sys.stderr)
                return None
    return None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def quantile_loss(y: np.ndarray, pred: np.ndarray, q: float) -> np.ndarray:
    diff = y - pred
    return np.maximum(q * diff, (q - 1.0) * diff)


@dataclass
class Accumulator:
    records: list = field(default_factory=list)

    def add(self, model, airport, horizon_h, y, row):
        rec = {"model": model, "airport": airport, "horizon_h": horizon_h,
               "abs_err": abs(y - row["mean"]), "sq_err": (y - row["mean"]) ** 2, "abs_y": abs(y)}
        q_losses = []
        for q in (0.1, 0.5, 0.9):
            col = f"q{int(q * 100)}"
            if col in row and not pd.isna(row[col]):
                q_losses.append(quantile_loss(np.array([y]), np.array([row[col]]), q)[0])
        rec["q_loss"] = float(np.mean(q_losses)) if q_losses else np.nan
        self.records.append(rec)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """MAE / RMSE / WQL per (model, horizon) and an overall roll-up per model."""
    if df.empty:
        return df

    def _agg(group):
        mae = group["abs_err"].mean()
        rmse = np.sqrt(group["sq_err"].mean())
        denom = group["abs_y"].mean()
        wql = (2.0 * group["q_loss"].mean() / denom) if denom and not np.isnan(group["q_loss"].mean()) else np.nan
        return pd.Series({"n": len(group), "MAE": mae, "RMSE": rmse, "WQL": wql})

    per_h = df.groupby(["model", "horizon_h"]).apply(_agg).reset_index()
    overall = df.groupby("model").apply(_agg).reset_index()
    overall["horizon_h"] = "ALL"
    return pd.concat([overall, per_h], ignore_index=True)


def print_table(summary: pd.DataFrame) -> None:
    if summary.empty:
        print("No results (no scored predictions).")
        return
    order = {m: i for i, m in enumerate(
        ["last_value", "seasonal_naive", "xgboost", "chronos2"])}
    summary = summary.sort_values(
        by=["horizon_h", "model"],
        key=lambda s: s.map(lambda v: order.get(v, 99)) if s.name == "model" else s,
    )
    print("\n=== Backtest results (lower is better) ===")
    print(f"{'horizon':>8} {'model':>16} {'n':>6} {'MAE':>8} {'RMSE':>8} {'WQL':>8}")
    for _, r in summary.iterrows():
        wql = "-" if pd.isna(r["WQL"]) else f"{r['WQL']:.3f}"
        h = r["horizon_h"] if r["horizon_h"] == "ALL" else f"{r['horizon_h']}h"
        print(f"{h:>8} {r['model']:>16} {int(r['n']):>6} {r['MAE']:>8.2f} {r['RMSE']:>8.2f} {wql:>8}")


# ---------------------------------------------------------------------------
# Backtest driver
# ---------------------------------------------------------------------------
def run_backtest(args) -> dict:
    horizons = args.horizons
    max_h = max(horizons)
    now = pd.Timestamp.now(tz="UTC").floor(features.RESAMPLE_FREQUENCY)

    # Load enough history: backtest window + training lookback + a week for
    # the seasonal-naive baseline and a day of headroom.
    load_start = now - pd.Timedelta(
        days=args.backtest_days + args.train_days + 8
    )
    print(f"Loading history since {load_start.date()} for {len(args.airports)} airports ...")
    histories = {}
    for ap in args.airports:
        t0 = time.time()
        s = fetch_history(args.base_url, ap, load_start)
        histories[ap] = s
        print(f"  {ap}: {len(s):>7} rows  ({time.time() - t0:.1f}s)")

    # Origins over the backtest window, leaving room for the longest horizon.
    origin_end = now - pd.Timedelta(hours=max_h)
    origin_start = origin_end - pd.Timedelta(days=args.backtest_days)
    origins = pd.date_range(origin_start, origin_end, freq=f"{args.origin_every_hours}h")
    print(f"{len(origins)} origins every {args.origin_every_hours}h "
          f"from {origin_start} to {origin_end}")

    predictors = [LastValuePredictor(), SeasonalNaivePredictor()]
    if not args.no_xgboost:
        try:
            predictors.append(XGBoostCalendarPredictor(
                refit_hours=args.xgb_refit_hours, train_days=args.train_days))
        except Exception as exc:  # noqa: BLE001
            print(f"[xgboost] unavailable, skipping: {exc}", file=sys.stderr)
    if not args.no_chronos:
        chronos = Chronos2Predictor(cross_learning=not args.chronos_no_cross_learning)
        if chronos.available:
            predictors.append(chronos)
        else:
            print("[chronos] package not installed; skipping Chronos-2 predictor.")

    acc = Accumulator()
    for oi, origin in enumerate(origins):
        contexts = {ap: s[s.index <= origin] for ap, s in histories.items()}
        if all(len(c) == 0 for c in contexts.values()):
            continue
        for p in predictors:
            try:
                p.prepare(origin, contexts)
            except Exception as exc:  # noqa: BLE001
                print(f"[{p.name}] prepare failed at {origin}: {exc}", file=sys.stderr)

        for ap in args.airports:
            series = histories[ap]
            if series.empty:
                continue
            targets = [origin + pd.Timedelta(hours=h) for h in horizons]
            actuals = lookup_actuals(series, targets)
            for p in predictors:
                try:
                    preds = p.predict(ap, origin, targets)
                except Exception as exc:  # noqa: BLE001
                    print(f"[{p.name}] predict failed {ap}@{origin}: {exc}", file=sys.stderr)
                    continue
                for h, tgt in zip(horizons, targets):
                    y = actuals.get(tgt, np.nan)
                    if pd.isna(y) or tgt not in preds.index:
                        continue
                    row = preds.loc[tgt]
                    if pd.isna(row.get("mean", np.nan)):
                        continue
                    acc.add(p.name, ap, h, float(y), row)
        if (oi + 1) % 10 == 0:
            print(f"  ... {oi + 1}/{len(origins)} origins")

    df = acc.frame()
    summary = summarize(df)
    print_table(summary)

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "airports": args.airports,
            "backtest_days": args.backtest_days,
            "origin_every_hours": args.origin_every_hours,
            "horizons_h": horizons,
            "train_days": args.train_days,
            "xgb_refit_hours": args.xgb_refit_hours,
            "n_origins": len(origins),
        },
        "models": sorted(df["model"].unique().tolist()) if not df.empty else [],
        "summary": summary.to_dict(orient="records") if not summary.empty else [],
    }
    _persist(result)
    return result


def _persist(result: dict) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    full_path = os.path.join(RESULTS_DIR, f"backtest_{stamp}.json")
    with open(full_path, "w") as fh:
        json.dump(result, fh, indent=2)

    # Append a compact per-model overall roll-up for drift tracking (A2).
    history_path = os.path.join(RESULTS_DIR, "history.jsonl")
    overall = [r for r in result["summary"] if r.get("horizon_h") == "ALL"]
    with open(history_path, "a") as fh:
        fh.write(json.dumps({
            "generated_at": result["generated_at"],
            "config": result["config"],
            "overall": {r["model"]: {"MAE": r["MAE"], "RMSE": r["RMSE"], "WQL": r["WQL"]}
                        for r in overall},
        }) + "\n")
    print(f"\nSaved {full_path}")
    print(f"Appended drift summary to {history_path}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Rolling-origin backtest for waitport predictors")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--airports", nargs="+", default=DEFAULT_AIRPORTS)
    p.add_argument("--backtest-days", type=int, default=30)
    p.add_argument("--origin-every-hours", type=int, default=6)
    p.add_argument("--horizons", nargs="+", type=float,
                   default=[1, 2, 4, 8, 24, 168],
                   help="forecast horizons in hours")
    p.add_argument("--train-days", type=int, default=120,
                   help="XGBoost training lookback per refit")
    p.add_argument("--xgb-refit-hours", type=int, default=24)
    p.add_argument("--no-xgboost", action="store_true")
    p.add_argument("--no-chronos", action="store_true")
    p.add_argument("--chronos-no-cross-learning", action="store_true",
                   help="score Chronos-2 without cross_learning (incumbent behaviour)")
    return p.parse_args(argv)


if __name__ == "__main__":
    run_backtest(parse_args())

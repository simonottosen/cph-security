"""Rolling-origin backtest harness for the waitport queue predictors.

Measures MAE / RMSE per (airport, horizon, model) plus weighted quantile loss
(WQL) for probabilistic models, comparing cheap baselines against the XGBoost
calendar predictor and (when the ``chronos`` package is installed) the batched
Chronos-2 forecaster. Results print as a table and persist under ``eval/results/``
so accuracy can be tracked over time (drift monitoring).

This harness is the accuracy gate for the plan: every model change must lower
MAE/RMSE/WQL versus both the incumbent behaviour and the naive baselines before
it ships.

Reading the output
------------------
Three things are reported that are easy to get wrong by omission:

* **ALL** rolls up only the horizons *every* model reached. Chronos-2 forecasts
  8h ahead while the baselines answer at any horizon, so an unrestricted roll-up
  would reward the short-reach model for skipping the hard horizons.
* **WQL** is averaged over a fixed set of quantiles (``COMMON_QUANTILES``) so it
  means the same thing for a 3-level and a 21-level model; ``WQL_wide`` uses each
  model's own levels and is comparable across runs, not across models.
* **Unscored targets** are listed rather than dropped. A model that cannot reach
  a horizon and a model that returned NaN are different failures, and until this
  existed both looked like a missing table row -- which is how the Chronos arm
  scored nothing at all for two full runs without anyone noticing.

Usage examples
--------------
    # Fast local smoke test (baselines + XGBoost only; Chronos auto-skipped):
    python eval/backtest.py --airports CPH ARN --backtest-days 7 \
        --origin-every-hours 12 --train-days 45 --no-chronos

    # Fuller run (inside the forecast container where chronos is installed).
    # --history-cache is worth it: scoring Chronos needs HISTORY_DAYS of history
    # before the earliest origin so queue_lag_365d resolves, which is ~1.2M rows.
    python eval/backtest.py --backtest-days 45 --origin-every-hours 6 \
        --history-cache /tmp/waitport_history.parquet
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

# The quantiles WQL is computed over for cross-model comparison. Chronos-2 emits
# 21 levels and XGBoost 3; because WQL averages the per-level losses, a 21-level
# number and a 3-level number are not the same quantity and ranking models by
# them would be meaningless. So the headline WQL uses this fixed intersection and
# the wider coverage is reported separately as WQL_wide.
COMMON_QUANTILES = [0.1, 0.5, 0.9]


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

    def reach(self, airport: str, origin: pd.Timestamp):
        """Last timestamp this model can actually forecast, or None for unlimited.

        Chronos-2 emits a fixed number of steps from the end of its context, so
        horizons beyond that are not a weak prediction -- there is no prediction.
        Declaring the reach lets the driver record those targets as out-of-reach
        instead of scoring them as a missing value, which looks identical to a
        broken predictor.
        """
        return None


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
        self.context_skips = {}
        self.lag_365d_fallback_ratio = None
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
            # now=origin, without which the service's staleness gate rejects every
            # historical context and this predictor silently forecasts nothing.
            ctx, stats, cov = mod._prepare_airport_context(df_raw, ap, now=origin)
            if ctx is None:
                self.context_skips[stats.get("status", "unknown")] = (
                    self.context_skips.get(stats.get("status", "unknown"), 0) + 1
                )
                continue
            self._note_lag_resolution(ctx)
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

    def _note_lag_resolution(self, ctx):
        """Record how often queue_lag_365d fell back instead of resolving.

        The lag reindexes the history 365 days before each context row and, when
        that row is absent, cascades silently through the seasonal profile
        (app.py's ``np.where(np.isnan(...))`` chain). So a backtest whose loaded
        history is shorter than the lag produces a covariate that is really just
        queue_avg_dow_time under another name -- with no error and no log line.
        Measuring the collision rate is the only way that shows up.
        """
        if "queue_lag_365d" not in ctx or "queue_avg_dow_time" not in ctx:
            return
        # Exact equality, not isclose: the fallback *assigns* the profile value, so
        # a fallen-back row matches bit for bit while a genuinely resolved one has
        # no reason to.
        ratio = float((ctx["queue_lag_365d"] == ctx["queue_avg_dow_time"]).mean())
        prev = self.lag_365d_fallback_ratio
        self.lag_365d_fallback_ratio = ratio if prev is None else max(prev, ratio)

    def reach(self, airport, origin):
        """Chronos emits PREDICTION_LENGTH steps from the end of its own context.

        Measured from the forecast rather than from the origin, because the
        context ends at the last 15-minute bucket at or before the origin -- so
        the true reach can sit up to one bucket short of origin + 8h.
        """
        fc = self._forecasts.get(airport)
        if fc is None or fc.empty:
            return None
        return fc["timestamp"].max()

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
        # Chronos-2 names its point forecast "predictions"; only look for "mean"
        # after that, then fall back to the median. Reading "mean" alone made every
        # Chronos prediction NaN -- the second reason this arm never scored, and one
        # the staleness gate was hiding, since no context ever got this far.
        for candidate in ("predictions", "mean", "0.5"):
            if candidate in near:
                out["mean"] = near[candidate].to_numpy()
                break
        else:
            raise KeyError(
                f"no point-forecast column in {list(near.columns)[:8]}; "
                "expected one of predictions/mean/0.5"
            )
        # Every level the service publishes, not just the deciles: the tails
        # (0.95/0.99) are where a peak-related covariate would show up, and they
        # cannot be evaluated later from a run that never recorded them. Column
        # names come from the service's own resolver -- Chronos names them by
        # str(q), so formatting 0.05 to one decimal would silently read q10.
        resolve = getattr(self._mod, "_quantile_column_name", None)
        for q in getattr(self._mod, "QUANTILE_LEVELS", COMMON_QUANTILES):
            col = resolve(near, q) if resolve is not None else f"{q:g}"
            if col is not None and col in near:
                out[f"q{int(round(q * 100))}"] = near[col].to_numpy()
        return out


_FORECAST_MODULE = ...  # sentinel: not yet attempted (None is a real "unavailable")


def _load_forecast_module():
    """Load the Chronos forecast service module by path (side-effect free at import).

    Memoised because the driver now needs the service's constants (HISTORY_DAYS)
    to size the history fetch, before any predictor exists.
    """
    global _FORECAST_MODULE
    if _FORECAST_MODULE is not ...:
        return _FORECAST_MODULE
    _FORECAST_MODULE = _load_forecast_module_uncached()
    return _FORECAST_MODULE


def _load_forecast_module_uncached():
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
        # Headline q_loss over the fixed common set, so WQL means the same thing
        # for every model. q_loss_wide uses whatever levels the model published,
        # which is comparable across runs of one model but not across models.
        common, wide = [], []
        for key, value in row.items():
            if not (isinstance(key, str) and key.startswith("q") and key[1:].isdigit()):
                continue
            if pd.isna(value):
                continue
            q = int(key[1:]) / 100.0
            loss = quantile_loss(np.array([y]), np.array([value]), q)[0]
            wide.append(loss)
            if q in COMMON_QUANTILES:
                common.append(loss)
        rec["q_loss"] = float(np.mean(common)) if common else np.nan
        rec["q_loss_wide"] = float(np.mean(wide)) if wide else np.nan
        rec["n_quantiles"] = len(wide)
        self.records.append(rec)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """MAE / RMSE / WQL per (model, horizon) and an overall roll-up per model."""
    if df.empty:
        return df

    def _wql(group, col):
        denom = group["abs_y"].mean()
        mean_loss = group[col].mean() if col in group else np.nan
        if not denom or pd.isna(mean_loss):
            return np.nan
        return 2.0 * mean_loss / denom

    def _agg(group):
        return pd.Series({
            "n": len(group),
            "MAE": group["abs_err"].mean(),
            "RMSE": np.sqrt(group["sq_err"].mean()),
            "WQL": _wql(group, "q_loss"),
            "WQL_wide": _wql(group, "q_loss_wide"),
            # Persisted but not printed: WQL_wide is only comparable between runs
            # that averaged the same number of levels, and this is what says so.
            "n_quantiles": group["n_quantiles"].mean() if "n_quantiles" in group else np.nan,
        })

    # Aggregate over the metric columns only -- passing the grouping columns
    # through .apply is deprecated in pandas 2.2 and removed in 3.
    metric_cols = ["abs_err", "sq_err", "abs_y", "q_loss", "q_loss_wide", "n_quantiles"]
    metric_cols = [c for c in metric_cols if c in df.columns]
    per_h = df.groupby(["model", "horizon_h"])[metric_cols].apply(_agg).reset_index()

    # The ALL roll-up only over horizons every model reached. Chronos stops at 8h
    # while the baselines answer at 168h, so an unrestricted roll-up would credit
    # Chronos for skipping the hardest horizons and read as a large win.
    per_model_h = {m: set(g["horizon_h"]) for m, g in df.groupby("model")}
    common_h = set.intersection(*per_model_h.values()) if per_model_h else set()
    overall = (df[df["horizon_h"].isin(common_h)]
               .groupby("model")[metric_cols].apply(_agg).reset_index())
    overall["horizon_h"] = "ALL"
    overall["all_horizons"] = ",".join(str(h) for h in sorted(common_h))
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
    common = summary.get("all_horizons")
    common = next((v for v in common.dropna()), None) if common is not None else None
    print("\n=== Backtest results (lower is better) ===")
    if common:
        print(f"ALL = roll-up over horizons common to every model: {common}h")
    print("WQL over %s; WQL_wide over each model's own published levels "
          "(comparable across runs, not across models)."
          % ", ".join(str(q) for q in COMMON_QUANTILES))
    print(f"{'horizon':>8} {'model':>16} {'n':>6} {'MAE':>8} {'RMSE':>8} {'WQL':>8} {'WQL_wide':>9}")
    for _, r in summary.iterrows():
        wql = "-" if pd.isna(r["WQL"]) else f"{r['WQL']:.3f}"
        wide = "-" if pd.isna(r.get("WQL_wide", np.nan)) else f"{r['WQL_wide']:.3f}"
        h = r["horizon_h"] if r["horizon_h"] == "ALL" else f"{r['horizon_h']}h"
        print(f"{h:>8} {r['model']:>16} {int(r['n']):>6} {r['MAE']:>8.2f} "
              f"{r['RMSE']:>8.2f} {wql:>8} {wide:>9}")


# ---------------------------------------------------------------------------
# Backtest driver
# ---------------------------------------------------------------------------
def _load_histories(args, load_start: pd.Timestamp, now: pd.Timestamp) -> dict:
    """Fetch each airport's history, optionally via a parquet cache.

    At the widened window a run refetches well over a million rows per airport,
    which makes iterating on the harness itself the slowest part of using it. The
    cache is opt-in and per-airport: an entry is reused only if it starts early
    enough and runs late enough for *this* run, so a shorter cached window
    triggers a refetch rather than quietly truncating the backtest.
    """
    cache_path = getattr(args, "history_cache", None)
    cached = {}
    if cache_path and os.path.exists(cache_path):
        try:
            frame = pd.read_parquet(cache_path)
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
            for ap, g in frame.groupby("airport"):
                cached[ap] = g.set_index("timestamp")["queue"].sort_index()
            print(f"  cache: {cache_path} ({len(frame)} rows, "
                  f"{len(cached)} airports)")
        except Exception as exc:  # noqa: BLE001
            print(f"  cache unreadable, ignoring: {exc}", file=sys.stderr)

    histories, fetched_any = {}, False
    for ap in args.airports:
        s = cached.get(ap)
        fresh_enough = (
            s is not None
            and not s.empty
            and s.index.min() <= load_start + pd.Timedelta(hours=1)
            and s.index.max() >= now - pd.Timedelta(hours=2)
        )
        if fresh_enough:
            histories[ap] = s[s.index >= load_start]
            print(f"  {ap}: {len(histories[ap]):>7} rows  (cached)")
            continue
        t0 = time.time()
        histories[ap] = fetch_history(args.base_url, ap, load_start)
        fetched_any = True
        print(f"  {ap}: {len(histories[ap]):>7} rows  ({time.time() - t0:.1f}s)")

    if cache_path and fetched_any:
        merged = pd.concat(
            [pd.DataFrame({"airport": ap, "timestamp": s.index, "queue": s.to_numpy()})
             for ap, s in histories.items() if not s.empty],
            ignore_index=True,
        ) if any(not s.empty for s in histories.values()) else None
        if merged is not None:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
                merged.to_parquet(cache_path, index=False)
                print(f"  cache written: {cache_path} ({len(merged)} rows)")
            except Exception as exc:  # noqa: BLE001
                print(f"  could not write cache: {exc}", file=sys.stderr)
    return histories


def run_backtest(args) -> dict:
    horizons = args.horizons
    max_h = max(horizons)
    # Place origins on the coarsest grid any model uses. Chronos buckets its
    # context to 15 minutes, so an origin at :05 leaves its context ending at :00
    # and its 8h forecast finishing 5 minutes before the 8h target -- which made
    # the longest horizon unreachable for Chronos at every single origin, purely
    # as an artifact of where the origins were placed.
    origin_grid = max(
        pd.Timedelta(features.RESAMPLE_FREQUENCY),
        pd.Timedelta(getattr(_load_forecast_module(), "RESAMPLE_FREQUENCY", "5min"))
        if not args.no_chronos else pd.Timedelta(0),
    )
    now = pd.Timestamp.now(tz="UTC").floor(origin_grid)

    # Load enough history: backtest window + training lookback + a week for
    # the seasonal-naive baseline and a day of headroom. Chronos additionally
    # needs HISTORY_DAYS before the *earliest* origin, because its covariates
    # reach back queue_lag_365d; short of that the lag falls back silently to the
    # seasonal profile and the run measures a covariate that never resolved.
    baseline_days = args.backtest_days + args.train_days + 8
    chronos_days = baseline_days
    mod = None if args.no_chronos else _load_forecast_module()
    if mod is not None:
        chronos_days = args.backtest_days + int(getattr(mod, "HISTORY_DAYS", 0)) + 1
    load_days = max(baseline_days, chronos_days)
    load_start = now - pd.Timedelta(days=load_days)
    print(f"Loading history since {load_start.date()} ({load_days}d) "
          f"for {len(args.airports)} airports ...")
    histories = _load_histories(args, load_start, now)

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
    out_of_reach = {}
    unscored = {}
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
                reach = p.reach(ap, origin)
                for h, tgt in zip(horizons, targets):
                    y = actuals.get(tgt, np.nan)
                    if pd.isna(y) or tgt not in preds.index:
                        continue
                    if reach is not None and tgt > reach:
                        # Beyond the model's horizon: no prediction exists. Counted
                        # rather than dropped, so "this model cannot answer here"
                        # never again looks the same as "this model returned NaN".
                        out_of_reach[(p.name, h)] = out_of_reach.get((p.name, h), 0) + 1
                        continue
                    row = preds.loc[tgt]
                    if pd.isna(row.get("mean", np.nan)):
                        unscored[(p.name, h)] = unscored.get((p.name, h), 0) + 1
                        continue
                    acc.add(p.name, ap, h, float(y), row)
        if (oi + 1) % 10 == 0:
            print(f"  ... {oi + 1}/{len(origins)} origins")

    df = acc.frame()
    summary = summarize(df)
    print_table(summary)

    # Everything the run could not score, stated out loud. The failure mode this
    # replaces is a model arm that is simply absent from the table with no
    # indication that it was ever meant to be there.
    diagnostics = {
        "load_days": load_days,
        "out_of_reach": {f"{m}@{h}h": n for (m, h), n in sorted(out_of_reach.items())},
        "unscored_nan": {f"{m}@{h}h": n for (m, h), n in sorted(unscored.items())},
    }
    chronos = next((p for p in predictors if p.name == "chronos2"), None)
    if chronos is not None:
        diagnostics["chronos_context_skips"] = dict(chronos.context_skips)
        diagnostics["chronos_lag_365d_fallback_ratio"] = chronos.lag_365d_fallback_ratio
    if diagnostics["out_of_reach"] or diagnostics["unscored_nan"]:
        print("\nUnscored targets:")
        for label, counts in (("out of model reach", diagnostics["out_of_reach"]),
                              ("prediction was NaN", diagnostics["unscored_nan"])):
            for key, n in counts.items():
                print(f"  {key:>22}: {n:>5}  ({label})")
    if chronos is not None:
        skips = diagnostics["chronos_context_skips"]
        if skips:
            print("\nChronos contexts rejected: "
                  + ", ".join(f"{k}={v}" for k, v in sorted(skips.items())))
        ratio = chronos.lag_365d_fallback_ratio
        if ratio is not None:
            # Reported every run, warned on only when the covariate has largely
            # stopped being one. Real year-ago history has gaps -- CPH's is ~8%
            # incomplete -- so warning at a few percent would fire on every honest
            # run and stop being read, which is the failure this exists to avoid.
            print(f"\nqueue_lag_365d fell back to the seasonal profile for "
                  f"{ratio:.1%} of context rows (worst airport).")
            if ratio >= 0.5:
                print(f"WARNING: at {ratio:.0%} fallback, queue_lag_365d is "
                      f"substantially a copy of queue_avg_dow_time.\n"
                      f"         Either the loaded window does not reach 365 days "
                      f"before the earliest origin,\n"
                      f"         or the year-ago history is largely missing. "
                      f"Conclusions about that covariate\n"
                      f"         are unsupported by this run.")

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
        "diagnostics": diagnostics,
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
            # Carried into the drift log so a row whose queue_lag_365d never
            # resolved says so, instead of sitting alongside valid rows.
            "lag_365d_fallback_ratio":
                result.get("diagnostics", {}).get("chronos_lag_365d_fallback_ratio"),
            "overall": {r["model"]: {"MAE": r["MAE"], "RMSE": r["RMSE"],
                                     "WQL": r["WQL"], "WQL_wide": r.get("WQL_wide")}
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
                   default=[1, 2, 4, 8],
                   help="forecast horizons in hours. Default stops at 8h, which is "
                        "Chronos-2's full reach (PREDICTION_LENGTH x 15min); longer "
                        "horizons are scoreable for the baselines but exclude "
                        "Chronos from the ALL roll-up")
    p.add_argument("--history-cache", default=None,
                   help="parquet path to cache fetched history in (opt-in; a run at "
                        "the full Chronos window refetches ~1.2M rows per airport)")
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

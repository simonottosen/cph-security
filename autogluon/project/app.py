"""Chronos-2 forecasting service for airport security queue wait times.

All airports are forecast in a single batched ``predict_df(cross_learning=True)``
call so the model can share signal across related series (helps the low-data
airports most). Forecasts are persisted to disk and reloaded on startup, so
``/forecast`` always serves the last-good result — across restarts and before
the first fresh run of a freshly started worker completes.

Module import is side-effect-free: the backtest harness imports the context
helpers directly. Call :func:`_initialize` (from ``__main__`` or the gunicorn
``post_worker_init`` hook) to load persisted state and start the scheduler.
"""
from __future__ import annotations

import datetime
import importlib
import json
import logging
import os
import platform
import tempfile
import threading
import time

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from flask import Flask, jsonify

_chronos_import_error = None
try:
    from chronos import BaseChronosPipeline
except Exception as exc:  # pragma: no cover - runtime dependency availability check
    BaseChronosPipeline = None
    _chronos_import_error = exc

# Imported lazily so the module (and its context helpers) stay importable in
# environments without apscheduler — e.g. the backtest harness and unit tests.
_apscheduler_import_error = None
try:
    from apscheduler.schedulers.background import BackgroundScheduler
except Exception as exc:  # pragma: no cover - runtime dependency availability check
    BackgroundScheduler = None
    _apscheduler_import_error = exc

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if os.environ.get("CPHAPI_HOST"):
    CPHAPI_HOST = os.environ.get("CPHAPI_HOST")
else:
    # Local testing endpoint
    CPHAPI_HOST = "http://apisix:9080/api/v1/all"


def _detect_default_device_map():
    """Pick a safe default device without requiring a GPU-enabled runtime."""
    if platform.system() == "Darwin":
        return "cpu"

    try:
        torch = importlib.import_module("torch")
    except Exception as exc:  # pragma: no cover - depends on runtime image
        logger.warning("Unable to import torch while selecting device map: %s", exc)
        return "cpu"

    try:
        if torch.cuda.is_available():
            return "cuda"
    except Exception as exc:  # pragma: no cover - depends on runtime image
        logger.warning("Unable to query torch.cuda availability: %s", exc)

    return "cpu"


# Chronos-2 settings (aligned with the Chronos-2 quickstart usage).
CHRONOS2_MODEL_ID = os.environ.get("CHRONOS2_MODEL_ID", "amazon/chronos-2")
CHRONOS2_DEVICE_MAP = os.environ.get("CHRONOS2_DEVICE_MAP")
if CHRONOS2_DEVICE_MAP is None:
    CHRONOS2_DEVICE_MAP = _detect_default_device_map()
PREDICTION_LENGTH = int(os.environ.get("PREDICTION_LENGTH", "96"))
RESAMPLE_FREQUENCY = os.environ.get("RESAMPLE_FREQUENCY", "5min")
CONTEXT_DAYS = int(os.environ.get("CONTEXT_DAYS", "60"))
MAX_FILL_GAP_STEPS = int(os.environ.get("MAX_FILL_GAP_STEPS", "6"))

# How stale the end of the context may be before we decline to forecast at all.
# The horizon is PREDICTION_LENGTH steps from the context end (96 x 5 min = 8 h),
# so at 6 h roughly two usable hours remain -- which is what the site reports.
MAX_CONTEXT_AGE_MINUTES = int(os.environ.get("MAX_CONTEXT_AGE_MINUTES", "360"))

# Cross-series learning: forecast all airports jointly so the model shares
# information across related series. Task-dependent, hence backtest-gated
# (eval/backtest.py can toggle it with --chronos-no-cross-learning).
CROSS_LEARNING = os.environ.get("CROSS_LEARNING", "true").strip().lower() in ("1", "true", "yes", "on")

# Refresh cadence (hours). Shorter than the old 4 h so forecasts stay fresh;
# combined with on-disk persistence this removes stale/empty windows.
FORECAST_INTERVAL_HOURS = float(os.environ.get("FORECAST_INTERVAL_HOURS", "1"))

# Where last-good forecasts are persisted so restarts don't blank /forecast.
FORECAST_DIR = os.environ.get("FORECAST_DIR", os.environ.get("MODELS_DIR", "forecasts"))

FETCH_TIMEOUT = int(os.environ.get("FETCH_TIMEOUT", "60"))

# 21-level quantile grid (was 9). Deciles keep 1-decimal public API keys
# ("0.1".."0.9") for backward compatibility; finer levels use their shortest
# string form ("0.05", "0.15", ...). See _api_quantile_key.
QUANTILE_LEVELS = [
    0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
    0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99,
]

VALID_AIRPORTS = ["AMS", "ARN", "CPH", "DUB", "DUS", "FRA", "IST", "LHR", "EDI", "MUC"]

# Known-future + historical covariates attached to every context/future frame.
COVARIATE_COLUMNS = [
    "hour", "minute", "day_of_week", "is_weekend", "month",
    "tod_sin", "tod_cos", "tow_sin", "tow_cos", "doy_sin", "doy_cos",
    "queue_lag_30d", "queue_lag_365d", "queue_avg_dow_time", "queue_avg_time",
]

_chronos2_pipeline = None


class _NaiveFallbackPipeline:
    """Minimal predict_df-compatible fallback used when Chronos isn't installed.

    Mirrors the real Chronos-2 output schema (a ``predictions`` point column and
    ``str(q)`` quantile columns) so the serving/formatting code path is the same
    whether or not the model is present. Predictions are last-value carry-forward.
    """

    @staticmethod
    def predict_df(
        df,
        future_df=None,
        prediction_length=96,
        quantile_levels=None,
        id_column="item_id",
        timestamp_column="timestamp",
        target="queue",
        **_,
    ):
        if quantile_levels is None:
            quantile_levels = [0.1, 0.5, 0.9]

        if future_df is not None and len(future_df) > 0:
            base = future_df[[id_column, timestamp_column]].copy()
        else:
            out_rows = []
            offset = pd.tseries.frequencies.to_offset(RESAMPLE_FREQUENCY)
            for item_id, grp in df.groupby(id_column):
                grp = grp.sort_values(timestamp_column)
                last_ts = pd.to_datetime(grp[timestamp_column].iloc[-1])
                future_ts = pd.date_range(last_ts + offset, periods=prediction_length, freq=offset)
                out_rows.append(pd.DataFrame({id_column: item_id, timestamp_column: future_ts}))
            base = pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame(columns=[id_column, timestamp_column])

        out = base.copy()
        last_values = (
            df.sort_values(timestamp_column)
            .groupby(id_column)[target]
            .last()
            .to_dict()
        )
        out["predictions"] = out[id_column].map(last_values).astype(float)
        for q in quantile_levels:
            out[str(q)] = out["predictions"]
        return out


def _add_time_covariates(df, timestamp_col="timestamp"):
    """Add deterministic calendar covariates that are known for future timestamps."""
    out = df.copy()
    ts = pd.to_datetime(out[timestamp_col])

    out["hour"] = ts.dt.hour.astype(float)
    out["minute"] = ts.dt.minute.astype(float)
    out["day_of_week"] = ts.dt.dayofweek.astype(float)
    out["is_weekend"] = (ts.dt.dayofweek >= 5).astype(float)
    out["month"] = ts.dt.month.astype(float)

    minute_of_day = (ts.dt.hour * 60 + ts.dt.minute).astype(float)
    week_minute = (ts.dt.dayofweek * 1440 + minute_of_day).astype(float)
    day_of_year = ts.dt.dayofyear.astype(float)

    out["tod_sin"] = np.sin(2 * np.pi * minute_of_day / 1440.0)
    out["tod_cos"] = np.cos(2 * np.pi * minute_of_day / 1440.0)
    out["tow_sin"] = np.sin(2 * np.pi * week_minute / 10080.0)
    out["tow_cos"] = np.cos(2 * np.pi * week_minute / 10080.0)
    out["doy_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    return out


def _build_historical_covariate_tables(history_df):
    """Build lookup tables for historical queue-derived covariates."""
    hist_series = history_df["queue"].copy().sort_index()

    hist_tmp = hist_series.to_frame("queue").reset_index()
    hist_tmp["hour"] = hist_tmp["timestamp"].dt.hour
    hist_tmp["minute"] = hist_tmp["timestamp"].dt.minute
    hist_tmp["day_of_week"] = hist_tmp["timestamp"].dt.dayofweek

    profile_dow_hm = hist_tmp.groupby(["day_of_week", "hour", "minute"])["queue"].mean()
    profile_hm = hist_tmp.groupby(["hour", "minute"])["queue"].mean()
    global_mean = float(hist_series.mean()) if len(hist_series) > 0 else 0.0

    return {
        "hist_series": hist_series,
        "profile_dow_hm": profile_dow_hm,
        "profile_hm": profile_hm,
        "global_mean": global_mean,
    }


def _add_history_queue_covariates(df, cov_tables, timestamp_col="timestamp"):
    """Add lag/profile covariates from historical queue values."""
    out = df.copy()
    ts = pd.to_datetime(out[timestamp_col])
    hist_series = cov_tables["hist_series"]
    profile_dow_hm = cov_tables["profile_dow_hm"]
    profile_hm = cov_tables["profile_hm"]
    global_mean = cov_tables["global_mean"]

    # Exact lag lookups.
    lag_30d = hist_series.reindex(ts - pd.Timedelta(days=30)).to_numpy(dtype=float)
    lag_365d = hist_series.reindex(ts - pd.Timedelta(days=365)).to_numpy(dtype=float)

    # Seasonal profile lookups.
    keys_dow_hm = pd.MultiIndex.from_arrays(
        [ts.dt.dayofweek.to_numpy(), ts.dt.hour.to_numpy(), ts.dt.minute.to_numpy()],
        names=["day_of_week", "hour", "minute"],
    )
    keys_hm = pd.MultiIndex.from_arrays(
        [ts.dt.hour.to_numpy(), ts.dt.minute.to_numpy()],
        names=["hour", "minute"],
    )
    profile_slot = profile_dow_hm.reindex(keys_dow_hm).to_numpy(dtype=float)
    profile_time = profile_hm.reindex(keys_hm).to_numpy(dtype=float)

    # Fill lag covariates with profile-based fallbacks when exact timestamp match is unavailable.
    lag_30d = np.where(np.isnan(lag_30d), profile_slot, lag_30d)
    lag_30d = np.where(np.isnan(lag_30d), profile_time, lag_30d)
    lag_30d = np.where(np.isnan(lag_30d), global_mean, lag_30d)

    lag_365d = np.where(np.isnan(lag_365d), profile_slot, lag_365d)
    lag_365d = np.where(np.isnan(lag_365d), profile_time, lag_365d)
    lag_365d = np.where(np.isnan(lag_365d), global_mean, lag_365d)

    profile_slot = np.where(np.isnan(profile_slot), profile_time, profile_slot)
    profile_slot = np.where(np.isnan(profile_slot), global_mean, profile_slot)
    profile_time = np.where(np.isnan(profile_time), global_mean, profile_time)

    out["queue_lag_30d"] = lag_30d.astype(float)
    out["queue_lag_365d"] = lag_365d.astype(float)
    out["queue_avg_dow_time"] = profile_slot.astype(float)
    out["queue_avg_time"] = profile_time.astype(float)
    return out


def get_chronos2_pipeline():
    """Load Chronos-2 lazily and keep it in memory across retraining cycles."""
    global _chronos2_pipeline

    if _chronos2_pipeline is not None:
        return _chronos2_pipeline

    if BaseChronosPipeline is None:
        logger.warning(
            "Chronos import failed; using naive fallback pipeline. error=%s",
            _chronos_import_error,
        )
        _chronos2_pipeline = _NaiveFallbackPipeline()
        return _chronos2_pipeline

    start = time.time()
    _chronos2_pipeline = BaseChronosPipeline.from_pretrained(
        CHRONOS2_MODEL_ID,
        device_map=CHRONOS2_DEVICE_MAP,
    )
    logger.info(
        "Loaded Chronos-2 pipeline model=%s device_map=%s in %.2fs",
        CHRONOS2_MODEL_ID,
        CHRONOS2_DEVICE_MAP,
        time.time() - start,
    )
    return _chronos2_pipeline


def _single_line_message(code, message):
    return [{
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "message": f"{code}: {message}",
    }]


def _prepare_airport_context(df_raw, code):
    """Prepare univariate context dataframe for Chronos-2 predict_df."""
    df_code = df_raw[df_raw["airport"] == code].copy()
    stats = {
        "raw_rows": int(len(df_code)),
        "valid_rows": 0,
        "rows_needing_fill": 0,
        "rows_not_needing_fill": 0,
        "rows_ffilled": 0,
        "rows_not_ffilled": 0,
        "rows_missing_after_fill": 0,
        "context_rows": 0,
        "window_days": CONTEXT_DAYS,
        "window_start": None,
        "window_end": None,
        "window_grid_rows": 0,
        "window_observed_rows": 0,
        "window_coverage_ratio": 0.0,
        "selected_segment_start": None,
        "selected_segment_end": None,
        "status": "ok",
        "status_message": "",
    }

    if df_code.empty:
        stats["status"] = "no_rows"
        stats["status_message"] = "No rows found for airport"
        return None, stats, None

    df_code["timestamp"] = pd.to_datetime(df_code["timestamp"], utc=True).dt.tz_convert(None)
    df_code["queue"] = pd.to_numeric(df_code["queue"], errors="coerce")
    df_code = df_code.dropna(subset=["timestamp", "queue"])
    stats["valid_rows"] = int(len(df_code))
    if df_code.empty:
        stats["status"] = "no_valid_rows"
        stats["status_message"] = "No valid queue history after cleanup"
        return None, stats, None

    df_code["timestamp"] = df_code["timestamp"].dt.floor(RESAMPLE_FREQUENCY)
    df_code = (
        df_code[["timestamp", "queue"]]
        .groupby("timestamp", as_index=True)
        .mean()
        .sort_index()
    )

    full_history = df_code.copy()
    cov_tables = _build_historical_covariate_tables(full_history)

    # Focus on recent history to make context and fill metrics meaningful for near-term forecasting.
    window_end = df_code.index.max()
    window_start = window_end - pd.Timedelta(days=CONTEXT_DAYS)
    df_code = df_code[df_code.index >= window_start]
    stats["window_start"] = window_start.isoformat()
    stats["window_end"] = window_end.isoformat()
    stats["window_observed_rows"] = int(len(df_code))

    if df_code.empty:
        stats["status"] = "no_recent_rows"
        stats["status_message"] = f"No rows in last {CONTEXT_DAYS} days"
        return None, stats, None

    # Build a strictly regular time grid so Chronos can infer frequency.
    regular_index = pd.date_range(
        start=df_code.index.min(),
        end=df_code.index.max(),
        freq=RESAMPLE_FREQUENCY,
    )
    df_code = df_code.reindex(regular_index)
    df_code.index.name = "timestamp"
    stats["window_grid_rows"] = int(len(df_code))
    missing_before_fill = int(df_code["queue"].isna().sum())
    stats["rows_needing_fill"] = missing_before_fill
    stats["rows_not_needing_fill"] = int(len(df_code) - missing_before_fill)
    stats["rows_ffilled"] = stats["rows_needing_fill"]
    stats["rows_not_ffilled"] = stats["rows_not_needing_fill"]
    if len(df_code) > 0:
        stats["window_coverage_ratio"] = float(stats["rows_not_needing_fill"] / len(df_code))

    # Fill only short gaps; avoid flattening long gaps into stale plateaus.
    df_code["queue"] = df_code["queue"].interpolate(
        method="time",
        limit=MAX_FILL_GAP_STEPS,
        limit_direction="both",
    )
    stats["rows_missing_after_fill"] = int(df_code["queue"].isna().sum())

    # Select the most recent contiguous non-missing segment. It must be the most
    # recent one rather than the longest: the forecast is anchored to the end of
    # the context, so a longer but older segment yields a forecast for a window
    # that has already elapsed -- which every consumer then discards, while the
    # retrain still reports success.
    valid_mask = (~df_code["queue"].isna()).to_numpy()
    seg_start = seg_end = -1
    for i in range(len(valid_mask) - 1, -1, -1):
        if valid_mask[i]:
            if seg_end == -1:
                seg_end = i + 1
            seg_start = i
        elif seg_end != -1:
            break

    if seg_end == -1:
        stats["status"] = "no_contiguous_segment"
        stats["status_message"] = "No contiguous segment available after gap handling"
        return None, stats, None

    df_code = df_code.iloc[seg_start:seg_end].copy()
    stats["selected_segment_start"] = df_code.index.min().isoformat()
    stats["selected_segment_end"] = df_code.index.max().isoformat()

    # Refuse to forecast from a context that has already gone cold: the horizon
    # is anchored to the context end, so most or all of it would lie in the past.
    # Reporting no forecast is honest; emitting an elapsed one is not.
    context_age = pd.Timestamp.now("UTC").tz_localize(None) - df_code.index.max()
    stats["context_age_minutes"] = int(context_age.total_seconds() // 60)
    if context_age > pd.Timedelta(minutes=MAX_CONTEXT_AGE_MINUTES):
        stats["status"] = "stale_history"
        stats["status_message"] = (
            f"Most recent history ends {df_code.index.max().isoformat()}, "
            f"{stats['context_age_minutes']} min ago (limit "
            f"{MAX_CONTEXT_AGE_MINUTES}); refusing to forecast an elapsed window"
        )
        return None, stats, None
    df_code = df_code.reset_index()
    stats["context_rows"] = int(len(df_code))

    if len(df_code) < PREDICTION_LENGTH * 2:
        stats["status"] = "insufficient_history"
        stats["status_message"] = (
            f"Not enough history after preprocessing: {len(df_code)} rows"
        )
        return None, stats, None

    df_code["item_id"] = code
    if getattr(df_code["timestamp"].dt, "tz", None) is not None:
        df_code["timestamp"] = df_code["timestamp"].dt.tz_localize(None)
    df_code = _add_time_covariates(df_code, timestamp_col="timestamp")
    df_code = _add_history_queue_covariates(df_code, cov_tables, timestamp_col="timestamp")
    return df_code, stats, cov_tables


def _build_future_covariates(context_df, code, cov_tables):
    """Build known future covariates for the forecast horizon."""
    offset = pd.tseries.frequencies.to_offset(RESAMPLE_FREQUENCY)
    last_ts = pd.to_datetime(context_df["timestamp"]).max()
    future_timestamps = pd.date_range(
        start=last_ts + offset,
        periods=PREDICTION_LENGTH,
        freq=offset,
    )
    future_df = pd.DataFrame({
        "item_id": code,
        "timestamp": future_timestamps,
    })
    future_df = _add_time_covariates(future_df, timestamp_col="timestamp")
    future_df = _add_history_queue_covariates(future_df, cov_tables, timestamp_col="timestamp")
    return future_df


def _quantile_column_name(pred_df, quantile):
    """Find the column in a predict_df output holding ``quantile``.

    Chronos-2 names quantile columns by ``str(q)`` (e.g. "0.1", "0.05"). We try
    the shortest string forms and the float key, and only fall back to 1-decimal
    formatting for true deciles — otherwise 0.05 would map onto the "0.1" column.
    """
    q = float(quantile)
    columns = pred_df.columns
    candidates = [str(q), f"{q:g}", q]
    if abs(round(q, 1) - q) < 1e-9:
        candidates.append(f"{q:.1f}")
    for c in candidates:
        if c in columns:
            return c
    return None


def _api_quantile_key(quantile):
    """Public JSON key for a quantile level.

    Deciles keep their historical 1-decimal keys ("0.1".."0.9") for backward
    compatibility; finer levels use their shortest string form ("0.05", ...).
    """
    q = float(quantile)
    if abs(round(q, 1) - q) < 1e-9:
        return f"{q:.1f}"
    return f"{q:g}"


def _format_predictions(pred_df):
    """Format a Chronos prediction dataframe into API response records."""
    quantile_cols = {q: _quantile_column_name(pred_df, q) for q in QUANTILE_LEVELS}
    if "predictions" in pred_df.columns:
        mean_col = "predictions"
    elif "mean" in pred_df.columns:
        mean_col = "mean"
    else:
        mean_col = quantile_cols.get(0.5)

    rows = []
    for _, row in pred_df.sort_values("timestamp").iterrows():
        mean_val = row[mean_col] if mean_col is not None else None
        rec = {
            "timestamp": pd.to_datetime(row["timestamp"]).strftime("%Y-%m-%dT%H:%M:%S"),
            "mean": float(mean_val) if mean_val is not None and pd.notna(mean_val) else None,
        }

        for q in QUANTILE_LEVELS:
            col = quantile_cols.get(q)
            if col is not None and pd.notna(row[col]):
                rec[_api_quantile_key(q)] = float(row[col])

        # Frontend Low/High band aliases (waitport airport page).
        if "0.3" in rec:
            rec["q30"] = rec["0.3"]
        if "0.7" in rec:
            rec["q70"] = rec["0.7"]

        rows.append(rec)

    return rows


def _prep_only_metrics(stats):
    """Metrics record for an airport that never reached the model (prep failed)."""
    record = {
        "model": CHRONOS2_MODEL_ID,
        "device_map": CHRONOS2_DEVICE_MAP,
        "cross_learning": CROSS_LEARNING,
        "prediction_length": PREDICTION_LENGTH,
        "total_time_seconds": 0.0,
        "last_trained": datetime.datetime.now().isoformat(),
    }
    record.update(stats or {})
    return record


def _load_raw():
    """Fetch the full raw history frame from PostgREST."""
    response = requests.get(CPHAPI_HOST, timeout=FETCH_TIMEOUT)
    response.raise_for_status()
    return pd.DataFrame(response.json())


def _run_forecast(df_raw):
    """Forecast every airport in one batched cross-learning predict_df call.

    Returns ``(forecasts, metrics)`` dicts keyed by airport code. Airports
    without enough history get an informational message instead of a forecast.
    """
    contexts, futures, prep_stats = [], [], {}
    forecasts, metrics = {}, {}

    for code in VALID_AIRPORTS:
        ctx, stats, cov = _prepare_airport_context(df_raw, code)
        prep_stats[code] = stats
        if ctx is None:
            forecasts[code] = _single_line_message(code, stats["status_message"])
            metrics[code] = _prep_only_metrics(stats)
            continue
        contexts.append(ctx)
        futures.append(_build_future_covariates(ctx, code, cov))

    if not contexts:
        return forecasts, metrics

    context_df = pd.concat(contexts, ignore_index=True)
    future_df = pd.concat(futures, ignore_index=True)

    pipeline = get_chronos2_pipeline()
    start = time.time()
    pred_df = pipeline.predict_df(
        context_df,
        future_df=future_df,
        cross_learning=CROSS_LEARNING,
        prediction_length=PREDICTION_LENGTH,
        quantile_levels=QUANTILE_LEVELS,
        id_column="item_id",
        timestamp_column="timestamp",
        target="queue",
    )
    duration = time.time() - start

    by_item = {code: g for code, g in pred_df.groupby("item_id")} if "item_id" in pred_df.columns else {}
    ctx_lengths = context_df.groupby("item_id").size().to_dict()
    fut_lengths = future_df.groupby("item_id").size().to_dict()
    n_items = len(contexts)

    for ctx in contexts:
        code = ctx["item_id"].iloc[0]
        group = by_item.get(code)
        stats = prep_stats.get(code, {})
        if group is None or len(group) == 0:
            forecasts[code] = _single_line_message(code, "No predictions returned by model")
            metrics[code] = _prep_only_metrics(stats)
            continue

        forecasts[code] = _format_predictions(group)
        record = {
            "model": CHRONOS2_MODEL_ID,
            "device_map": CHRONOS2_DEVICE_MAP,
            "cross_learning": CROSS_LEARNING,
            "batched_items": n_items,
            "history_rows": int(ctx_lengths.get(code, len(ctx))),
            "future_rows": int(fut_lengths.get(code, 0)),
            "prediction_length": PREDICTION_LENGTH,
            "quantile_levels": QUANTILE_LEVELS,
            "covariate_columns": COVARIATE_COLUMNS,
            "total_time_seconds": duration,  # shared across the batched call
            "last_trained": datetime.datetime.now().isoformat(),
        }
        record.update(stats)
        metrics[code] = record

    return forecasts, metrics


# ---------------------------------------------------------------------------
# Forecast persistence (survive restarts; never serve an empty /forecast).
# ---------------------------------------------------------------------------
def _forecast_state_path():
    return os.path.join(FORECAST_DIR, "forecasts.json")


def _persist_forecasts(forecasts, metrics):
    try:
        os.makedirs(FORECAST_DIR, exist_ok=True)
        payload = {
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "predictions": forecasts,
            "metrics": metrics,
        }
        fd, tmp = tempfile.mkstemp(dir=FORECAST_DIR, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as fh:
                json.dump(payload, fh)
            os.replace(tmp, _forecast_state_path())
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
    except Exception as exc:  # noqa: BLE001 - persistence is best-effort
        logger.warning("Could not persist forecasts to %s: %s", FORECAST_DIR, exc)


def _load_persisted_forecasts():
    path = _forecast_state_path()
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load persisted forecasts from %s: %s", path, exc)
        return None


# Latest per-airport forecasts and metrics (rebound atomically by retrain()).
train_metrics = {}
df_preds = {}
_state_lock = threading.Lock()

_scheduler = None
_initialized = False
_init_lock = threading.Lock()

app = Flask(__name__)


def retrain():
    """Refresh all forecasts. Best-effort: on failure, keep the last-good state."""
    global df_preds, train_metrics

    try:
        df_raw = _load_raw()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to load history for retrain; keeping last-good: %s", exc)
        return

    try:
        forecasts, metrics = _run_forecast(df_raw)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Batched forecast failed; keeping last-good: %s", exc)
        return

    with _state_lock:
        df_preds = forecasts
        train_metrics = metrics

    _persist_forecasts(forecasts, metrics)
    n_ok = sum(1 for recs in forecasts.values() if recs and "mean" in recs[0])
    logger.info("Retrain complete: %d/%d airports forecast", n_ok, len(VALID_AIRPORTS))


def _initialize():
    """Load persisted forecasts and start the scheduler. Idempotent."""
    global df_preds, train_metrics, _scheduler, _initialized

    with _init_lock:
        if _initialized:
            return

        payload = _load_persisted_forecasts()
        if payload:
            with _state_lock:
                df_preds = payload.get("predictions", {}) or {}
                train_metrics = payload.get("metrics", {}) or {}
            logger.info("Loaded persisted forecasts for %d airports", len(df_preds))

        if BackgroundScheduler is None:
            logger.warning(
                "apscheduler unavailable (%s); running one-shot retrain without scheduling.",
                _apscheduler_import_error,
            )
            _initialized = True
            retrain()
            return

        scheduler = BackgroundScheduler()
        scheduler.add_job(
            func=retrain,
            trigger="interval",
            hours=FORECAST_INTERVAL_HOURS,
            next_run_time=datetime.datetime.now(),  # kick an immediate, non-blocking refresh
            id="retrain",
            max_instances=1,
            coalesce=True,
        )
        scheduler.start()
        _scheduler = scheduler
        _initialized = True
        logger.info("Scheduler started: retrain every %.2f h", FORECAST_INTERVAL_HOURS)


@app.route('/forecast/<airport>', methods=['GET'])
def get_forecast(airport):
    code = airport.upper()
    if code not in VALID_AIRPORTS:
        return jsonify({'error': f'Invalid airport code: {airport}'}), 404
    with _state_lock:
        preds = df_preds.get(code, [])
    return jsonify({'predictions': preds})


@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Return latest forecasting metrics."""
    with _state_lock:
        return jsonify(dict(train_metrics))


@app.route('/health', methods=['GET'])
def health():
    with _state_lock:
        n = len(df_preds)
    return jsonify({
        "status": "ok",
        "airports_loaded": n,
        "chronos_available": BaseChronosPipeline is not None,
        "cross_learning": CROSS_LEARNING,
        "prediction_length": PREDICTION_LENGTH,
        "quantile_levels": QUANTILE_LEVELS,
    })


if __name__ == '__main__':
    _initialize()
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", "5000")))

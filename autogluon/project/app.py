import datetime
import logging
import multiprocessing as mp
import os
import platform
import time

import numpy as np
import pandas as pd
import requests
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv
from flask import Flask, jsonify

try:
    from chronos import BaseChronosPipeline
except Exception:  # pragma: no cover - runtime dependency availability check
    BaseChronosPipeline = None

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if os.environ.get("CPHAPI_HOST"):
    CPHAPI_HOST = os.environ.get("CPHAPI_HOST")
else:
    # Local testing endpoint
    CPHAPI_HOST = "http://apisix:9080/api/v1/all"

# Chronos-2 settings (aligned with the Chronos-2 quickstart usage).
CHRONOS2_MODEL_ID = os.environ.get("CHRONOS2_MODEL_ID", "amazon/chronos-2")
# Apple Silicon default: CPU (safe baseline). You can override with CHRONOS2_DEVICE_MAP=mps if supported.
CHRONOS2_DEVICE_MAP = os.environ.get("CHRONOS2_DEVICE_MAP")
if CHRONOS2_DEVICE_MAP is None:
    CHRONOS2_DEVICE_MAP = "cpu" if platform.system() == "Darwin" else "cuda"
PREDICTION_LENGTH = int(os.environ.get("PREDICTION_LENGTH", "96"))
RESAMPLE_FREQUENCY = os.environ.get("RESAMPLE_FREQUENCY", "5min")
CONTEXT_DAYS = int(os.environ.get("CONTEXT_DAYS", "60"))
MAX_FILL_GAP_STEPS = int(os.environ.get("MAX_FILL_GAP_STEPS", "6"))
QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

VALID_AIRPORTS = ["AMS", "ARN", "CPH", "DUB", "DUS", "FRA", "IST", "LHR", "EDI", "MUC"]

_chronos2_pipeline = None


class _NaiveFallbackPipeline:
    """Minimal predict_df-compatible fallback used when Chronos isn't installed."""

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
            for item_id, grp in df.groupby(id_column):
                grp = grp.sort_values(timestamp_column)
                last_ts = pd.to_datetime(grp[timestamp_column].iloc[-1])
                offset = pd.tseries.frequencies.to_offset(RESAMPLE_FREQUENCY)
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
        out["mean"] = out[id_column].map(last_values).astype(float)
        for q in quantile_levels:
            out[f"{q:.1f}"] = out["mean"]
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
            "Chronos package not available; using naive fallback pipeline."
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

    # Select the longest contiguous non-missing segment in the recent window.
    valid_mask = (~df_code["queue"].isna()).to_numpy()
    best_start = best_end = -1
    current_start = -1
    for i, is_valid in enumerate(valid_mask):
        if is_valid and current_start == -1:
            current_start = i
        if (not is_valid) and current_start != -1:
            if (i - current_start) > (best_end - best_start):
                best_start, best_end = current_start, i
            current_start = -1
    if current_start != -1 and (len(valid_mask) - current_start) > (best_end - best_start):
        best_start, best_end = current_start, len(valid_mask)

    if best_start == -1:
        stats["status"] = "no_contiguous_segment"
        stats["status_message"] = "No contiguous segment available after gap handling"
        return None, stats, None

    df_code = df_code.iloc[best_start:best_end].copy()
    stats["selected_segment_start"] = df_code.index.min().isoformat()
    stats["selected_segment_end"] = df_code.index.max().isoformat()
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
    candidates = [f"{quantile:.1f}", quantile, str(quantile)]
    for c in candidates:
        if c in pred_df.columns:
            return c
    return None


def _format_predictions(pred_df):
    """Format Chronos prediction dataframe into API response records."""
    quantile_cols = {
        q: _quantile_column_name(pred_df, q)
        for q in QUANTILE_LEVELS
    }
    mean_col = "mean" if "mean" in pred_df.columns else quantile_cols.get(0.5)

    rows = []
    for _, row in pred_df.sort_values("timestamp").iterrows():
        rec = {
            "timestamp": pd.to_datetime(row["timestamp"]).strftime("%Y-%m-%dT%H:%M:%S"),
            "mean": float(row[mean_col]) if mean_col is not None else None,
        }

        for q in QUANTILE_LEVELS:
            col = quantile_cols.get(q)
            if col is not None:
                rec[f"{q:.1f}"] = float(row[col])

        if "0.3" in rec:
            rec["q30"] = rec["0.3"]
        if "0.7" in rec:
            rec["q70"] = rec["0.7"]

        rows.append(rec)

    return rows


def _forecast_airport(code, df_raw, pipeline):
    """Forecast for a single airport code using Chronos-2."""
    context_df, prep_stats, cov_tables = _prepare_airport_context(df_raw, code)
    if context_df is None:
        pred_records = _single_line_message(code, prep_stats["status_message"])
        metrics = {
            "model": CHRONOS2_MODEL_ID,
            "device_map": CHRONOS2_DEVICE_MAP,
            "prediction_length": PREDICTION_LENGTH,
            "total_time_seconds": 0.0,
            "last_trained": datetime.datetime.now().isoformat(),
            **prep_stats,
        }
        return code, pred_records, metrics

    future_df = _build_future_covariates(context_df, code, cov_tables)
    start = time.time()
    pred_df = pipeline.predict_df(
        context_df,
        future_df=future_df,
        prediction_length=PREDICTION_LENGTH,
        quantile_levels=QUANTILE_LEVELS,
        id_column="item_id",
        timestamp_column="timestamp",
        target="queue",
    )
    duration = time.time() - start

    if "item_id" in pred_df.columns:
        pred_df = pred_df[pred_df["item_id"] == code].copy()

    pred_records = _format_predictions(pred_df)

    metrics = {
        "model": CHRONOS2_MODEL_ID,
        "device_map": CHRONOS2_DEVICE_MAP,
        "history_rows": int(len(context_df)),
        "future_rows": int(len(future_df)),
        "prediction_length": PREDICTION_LENGTH,
        "covariate_columns": [
            "hour",
            "minute",
            "day_of_week",
            "is_weekend",
            "month",
            "tod_sin",
            "tod_cos",
            "tow_sin",
            "tow_cos",
            "doy_sin",
            "doy_cos",
            "queue_lag_30d",
            "queue_lag_365d",
            "queue_avg_dow_time",
            "queue_avg_time",
        ],
        "total_time_seconds": duration,
        "last_trained": datetime.datetime.now().isoformat(),
        **prep_stats,
    }
    return code, pred_records, metrics


def _train_airport(code, df_raw):
    """Backward-compatible training/forecast entrypoint used by retrain and tests."""
    pipeline = get_chronos2_pipeline()
    return _forecast_airport(code, df_raw, pipeline)


# Placeholder for latest forecasting metrics
train_metrics = {}

# Containers for per-airport forecasts and metrics
df_preds = {}

app = Flask(__name__)


def retrain():
    global df_preds, train_metrics
    df_preds.clear()
    train_metrics.clear()

    response = requests.get(CPHAPI_HOST, timeout=60)
    response.raise_for_status()
    df_raw = pd.DataFrame(response.json())

    for code in VALID_AIRPORTS:
        try:
            code, pred_records, metrics = _train_airport(code, df_raw)
            df_preds[code] = pred_records
            train_metrics[code] = metrics
        except Exception as e:
            logger.exception("Error forecasting airport %s: %s", code, e)
            df_preds[code] = _single_line_message(code, f"Forecast failed: {e}")
            train_metrics[code] = {
                "model": CHRONOS2_MODEL_ID,
                "device_map": CHRONOS2_DEVICE_MAP,
                "prediction_length": PREDICTION_LENGTH,
                "total_time_seconds": 0.0,
                "last_trained": datetime.datetime.now().isoformat(),
                "status": "error",
                "status_message": str(e),
            }


@app.route('/forecast/<airport>', methods=['GET'])
def get_forecast(airport):
    code = airport.upper()
    if code not in VALID_AIRPORTS:
        return jsonify({'error': f'Invalid airport code: {airport}'}), 404
    return jsonify({'predictions': df_preds.get(code, [])})


@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Return latest forecasting metrics."""
    return jsonify(train_metrics)


if __name__ == '__main__':
    mp.freeze_support()  # Good practice on Windows; harmless on *nix
    retrain()            # First full forecasting pass

    scheduler = BackgroundScheduler()
    scheduler.add_job(func=retrain, trigger='interval', hours=4)
    scheduler.start()

    app.run(host='0.0.0.0', port="5000")

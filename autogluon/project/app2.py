"""
app2.py

Side-by-side comparison service:
- Runs the existing AutoGluon Chronos-Bolt baseline (as in autogluon/project/app.py)
  and a Chronos-2 zero-shot inference pipeline for each airport.
- Exposes the same endpoints as the original service but returns both predictions
  and simple metrics for comparison.

Notes:
- Chronos-2 is loaded lazily inside worker processes to avoid cross-process issues.
- Chronos-2 device selection can be controlled via CHRONOS2_DEVICE env:
    - "auto" (default): use CUDA if torch.cuda.is_available() else "cpu"
    - "cuda" or "cpu": force that device
- This file intentionally does not modify the original app.py; it is a parallel
  service for evaluation purposes.
"""

import os
import time
import datetime
import logging
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import requests
from flask import Flask, jsonify

load_dotenv()

# Optional Chronos type imports (safe): try to import for typing/inspection.
# If the chronos package isn't installed, this will be False and the runtime
# will continue to operate (Chronos-2 paths are already guarded).
try:
    from chronos import BaseChronosPipeline, Chronos2Pipeline  # type: ignore
    _CHRONOS_MODULE_AVAILABLE = True
except Exception:
    _CHRONOS_MODULE_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Configuration
MODEL_PATH = "models/ts_predictor"  # AutoGluon baseline persists here (same as app.py)
CPHAPI_HOST = os.environ.get("CPHAPI_HOST", "http://apisix:9080/api/v1/all")
PREDICTION_LENGTH = 96  # 96 * 5min = 8 hours ahead (same horizon as original app.py)
FREQ = "5min"

# Chronos-2 settings via environment
_CHRONOS2_S3 = os.environ.get("CHRONOS2_S3", "s3://autogluon/chronos-2/")
_CHRONOS2_DEVICE = os.environ.get("CHRONOS2_DEVICE", "auto")  # auto|cuda|cpu

# Enable/availability flags for Chronos-2
# CHRONOS2_ENABLED: "auto" (default) will enable if the 'chronos' package is importable,
# "1"/"true"/"yes" forces enabled, "0"/"false"/"no" forces disabled.
import importlib.util
_CHRONOS2_ENABLED = os.environ.get("CHRONOS2_ENABLED", "auto").lower()

def chronos2_is_enabled() -> bool:
    """Decide whether Chronos-2 paths should run in this process."""
    if _CHRONOS2_ENABLED in ("0", "false", "off", "no"):
        return False
    if _CHRONOS2_ENABLED in ("1", "true", "on", "yes"):
        return True
    # auto: check if module is importable
    return importlib.util.find_spec("chronos") is not None

# Airports / holiday map copied from original app.py for consistent behavior
import holidays
from dateutil.easter import easter

airport_holiday_map = {
    'AMS': holidays.Netherlands(),
    'ARN': holidays.Sweden(),
    'CPH': holidays.Denmark(),
    'DUB': holidays.Ireland(),
    'DUS': holidays.Germany(),
    'FRA': holidays.Germany(),
    'IST': holidays.Turkey(),
    'LHR': holidays.UnitedKingdom(),
    'EDI': holidays.UnitedKingdom(),
    'MUC': holidays.Germany(),
}
VALID_AIRPORTS = list(airport_holiday_map.keys())

# Global containers (in-memory) for predictions and metrics
autogluon_preds = {}
chronos2_preds = {}
comparison_metrics = {}

# Flask app
app = Flask(__name__)

# Chronos-2 lazy loader (per-process)
_CHRONOS_PIPELINE = None

def _select_chronos2_device():
    """Resolve device_map for Chronos-2 based on CHRONOS2_DEVICE env and availability."""
    device_pref = _CHRONOS2_DEVICE.lower()
    if device_pref == "cpu":
        return "cpu"
    if device_pref == "cuda":
        return "cuda"
    # auto: try to detect CUDA
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

def get_chronos2_pipeline():
    """Lazily load and cache Chronos-2 BaseChronosPipeline in the current process.

    Robust behavior:
    - Honor CHRONOS2_ENABLED (auto|1|0)
    - Try multiple model sources (CHRONOS2_S3, CHRONOS2_MODEL_ID)
    - On safetensors/header errors, retry once with force_download=True to force a re-download.
    """
    global _CHRONOS_PIPELINE
    if _CHRONOS_PIPELINE is not None:
        return _CHRONOS_PIPELINE

    # Respect the enable flag / availability
    if not chronos2_is_enabled():
        logger.info("Chronos-2 is disabled or not installed in this environment.")
        raise RuntimeError("Chronos-2 disabled or not available")

    # Prepare model source list (try in order)
    sources = []
    if _CHRONOS2_S3:
        sources.append(_CHRONOS2_S3)
    env_model_id = os.environ.get("CHRONOS2_MODEL_ID")
    if env_model_id:
        sources.append(env_model_id)
    # Allow a default HF id via env if project maintainers want to set it; otherwise rely on provided S3
    default_hf = os.environ.get("CHRONOS2_DEFAULT_HF_ID")
    if default_hf:
        sources.append(default_hf)

    # Device selection
    device = _select_chronos2_device()

    # Try to import BaseChronosPipeline once
    try:
        from chronos import BaseChronosPipeline
    except Exception as e:
        logger.exception(f"chronos package import failed: {e}")
        raise

    # Helper to detect safetensors header/deserialization issues
    def _is_safetensor_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        if "safetensor" in msg or "safetensors" in msg or "incomplete metadata" in msg or "file not fully covered" in msg:
            return True
        # try matching known safetensors_rust exception class if available
        try:
            import safetensors_rust
            return isinstance(exc, getattr(safetensors_rust, "SafetensorError", Exception))
        except Exception:
            return False

    last_exc = None
    for src in sources:
        try:
            logger.info(f"Attempting to load Chronos-2 from source: {src} (device={device})")
            # Normal attempt
            try:
                _CHRONOS_PIPELINE = BaseChronosPipeline.from_pretrained(src, device_map=device)
            except TypeError:
                # Some versions may not accept device_map keyword; fallback to positional or no device_map
                _CHRONOS_PIPELINE = BaseChronosPipeline.from_pretrained(src)
            logger.info(f"Loaded Chronos-2 pipeline from {src}")
            return _CHRONOS_PIPELINE
        except Exception as e:
            logger.warning(f"Failed to load Chronos-2 from {src}: {e}")
            last_exc = e
            # If it's a safetensors/header issue, try a forced re-download once
            if _is_safetensor_error(e):
                logger.info(f"Detected safetensors/header issue when loading {src}; retrying with force_download=True")
                try:
                    try:
                        _CHRONOS_PIPELINE = BaseChronosPipeline.from_pretrained(src, device_map=device, force_download=True)
                    except TypeError:
                        _CHRONOS_PIPELINE = BaseChronosPipeline.from_pretrained(src, force_download=True)
                    logger.info(f"Successfully re-downloaded and loaded Chronos-2 from {src}")
                    return _CHRONOS_PIPELINE
                except Exception as e2:
                    logger.warning(f"Forced re-download also failed for {src}: {e2}")
                    last_exc = e2
                    # continue to next source
                    continue
            # Otherwise try next source
            continue

    # If we reach here, nothing worked
    logger.exception(f"Unable to load Chronos-2 pipeline from any configured source. Last error: {last_exc}")
    raise RuntimeError(f"Failed to load Chronos-2 pipeline: {last_exc}")

def _rename_quantile_cols_and_format(df):
    """Normalize prediction DataFrame to records with timestamp, mean, q30, q70.
    Accepts a pandas DataFrame with a DatetimeIndex or a 'timestamp' column.
    Expects columns like 'mean', 0.3, 0.7 or '0.3', '0.7' or 'predictions'.
    """
    if df is None or df.empty:
        return []
    out = df.reset_index()
    # Normalize timestamp column name
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
    elif "index" in out.columns:
        out = out.rename(columns={"index": "timestamp"})
        out["timestamp"] = pd.to_datetime(out["timestamp"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
    else:
        # If index is DatetimeIndex
        try:
            out.index.name = "timestamp"
            out = out.reset_index()
            out["timestamp"] = pd.to_datetime(out["timestamp"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
        except Exception:
            pass

    # Some APIs return a 'predictions' column for point estimate; accept that as mean
    if "predictions" in out.columns and "mean" not in out.columns:
        out = out.rename(columns={"predictions": "mean"})

    # Find quantile columns and map 0.3->q30, 0.7->q70 (also accept strings)
    rename_map = {}
    for c in list(out.columns):
        try:
            f = float(c)
            if int(round(f*100)) == 30:
                rename_map[c] = "q30"
            elif int(round(f*100)) == 70:
                rename_map[c] = "q70"
        except Exception:
            # handle strings like "0.3"
            if str(c) in {"0.3", "0.30"}:
                rename_map[c] = "q30"
            if str(c) in {"0.7", "0.70"}:
                rename_map[c] = "q70"
    out = out.rename(columns=rename_map)

    # Ensure mean present, otherwise try to compute it from quantiles or fallbacks
    if "mean" not in out.columns:
        if "q30" in out.columns and "q70" in out.columns:
            out["mean"] = ((out["q30"].astype(float) + out["q70"].astype(float)) / 2.0)
        else:
            # last resort: try first numeric column
            numeric_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
            if numeric_cols:
                out["mean"] = out[numeric_cols[0]]

    # Keep only timestamp, mean, q30, q70 (if present)
    keep = ["timestamp", "mean"]
    if "q30" in out.columns:
        keep.append("q30")
    if "q70" in out.columns:
        keep.append("q70")
    out = out[keep]

    # Fill missing quantiles with mean (conservative)
    if "q30" not in out.columns:
        out["q30"] = out["mean"]
    if "q70" not in out.columns:
        out["q70"] = out["mean"]

    # Convert to python types
    records = []
    for _, row in out.iterrows():
        records.append({
            "timestamp": row["timestamp"],
            "mean": float(row["mean"]) if not pd.isna(row["mean"]) else None,
            "q30": float(row["q30"]) if not pd.isna(row["q30"]) else None,
            "q70": float(row["q70"]) if not pd.isna(row["q70"]) else None,
        })
    return records

def _train_and_predict_for_airport(code, df_raw):
    """Worker function executed in a separate process.
    Returns: (code, autogluon_records, chronos2_records, metrics)
    """
    try:
        from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
        import numpy as _np
        # Copy & basic cleaning consistent with app.py
        df_code = df_raw[df_raw["airport"] == code].copy()
        if df_code.empty:
            return code, [], [], {"error": "no_data"}

        df_code["timestamp"] = pd.to_datetime(df_code["timestamp"], errors="coerce")
        df_code["queue"] = pd.to_numeric(df_code["queue"], errors="coerce")
        df_code = df_code.dropna(subset=["timestamp", "queue"]).set_index("timestamp")
        df_code = df_code[["queue"]].groupby(level=0).mean()

        # Resample to 5-minute grid and forward-fill (same as original)
        tmp_resampled = df_code.resample(FREQ, origin="start_day").ffill()
        tmp_resampled.index.name = None
        df_resampled = tmp_resampled.reset_index().rename(columns={"index": "timestamp"})

        # Build many of the same features used by the original app to keep baseline parity
        hol = airport_holiday_map.get(code, holidays.country_holidays("DK"))
        df_features = df_resampled.copy()
        df_features["hour_of_day"] = df_features["timestamp"].dt.hour
        df_features["day_of_week"] = df_features["timestamp"].dt.dayofweek
        df_features["lag_1_queue"] = df_features["queue"].shift(1).fillna(0)
        df_features["is_holiday"] = df_features["timestamp"].dt.date.apply(lambda d: 1 if d in hol else 0)
        df_features["rolling_mean_1h"] = df_features["queue"].rolling(window=4, min_periods=1).mean()
        df_features["queue_last_week"] = df_features["queue"].shift(7 * 24 * 4)
        df_features["month"] = df_features["timestamp"].dt.month
        df_features["year"] = df_features["timestamp"].dt.year
        df_features["lag_2_queue"] = df_features["queue"].shift(2).fillna(0)
        df_features["lag_3_queue"] = df_features["queue"].shift(3).fillna(0)
        df_features["lag_4_queue"] = df_features["queue"].shift(4).fillna(0)
        df_features["is_weekend"] = (df_features["timestamp"].dt.dayofweek >= 5).astype(int)
        df_features["is_business_day"] = ((df_features["timestamp"].dt.dayofweek < 5) & (df_features["is_holiday"] == 0)).astype(int)
        df_features["rolling_std_1h"] = df_features["queue"].rolling(window=4, min_periods=1).std()
        df_features["quarter"] = df_features["timestamp"].dt.quarter
        df_features["is_winter"] = df_features["month"].isin([12, 1, 2]).astype(int)
        df_features["is_spring"] = df_features["month"].isin([3, 4, 5]).astype(int)
        df_features["is_summer"] = df_features["month"].isin([6, 7, 8]).astype(int)
        df_features["is_autumn"] = df_features["month"].isin([9, 10, 11]).astype(int)
        holiday_dates = sorted(hol)
        def days_to_next_holiday(date):
            future = [h for h in holiday_dates if h >= date]; return (future[0] - date).days if future else _np.nan
        def days_since_last_holiday(date):
            past = [h for h in holiday_dates if h <= date]; return (date - past[-1]).days if past else _np.nan
        df_features["days_to_next_holiday"] = df_features["timestamp"].dt.date.apply(days_to_next_holiday)
        df_features["days_since_last_holiday"] = df_features["timestamp"].dt.date.apply(days_since_last_holiday)
        def is_school_holiday(date):
            year = date.year
            periods = [
                (datetime.date(year-1,12,20), datetime.date(year,1,3)),
                (easter(year)-datetime.timedelta(days=3), easter(year)+datetime.timedelta(days=1)),
                (datetime.date(year,7,1), datetime.date(year,8,31)),
                (datetime.date.fromisocalendar(year,42,1), datetime.date.fromisocalendar(year,42,5)),
                (datetime.date.fromisocalendar(year,7,1), datetime.date.fromisocalendar(year,7,5))
            ]
            return any(start<=date<=end for start,end in periods)
        df_features["is_school_holiday"] = df_features["timestamp"].dt.date.apply(lambda d: 1 if is_school_holiday(d) else 0)
        df_features["hour_sin"] = np.sin(2*np.pi*df_features["hour_of_day"]/24)
        df_features["hour_cos"] = np.cos(2*np.pi*df_features["hour_of_day"]/24)
        df_features["dow_sin"]  = np.sin(2*np.pi*df_features["day_of_week"]/7)
        df_features["dow_cos"]  = np.cos(2*np.pi*df_features["day_of_week"]/7)
        df_features["doy"]      = df_features["timestamp"].dt.dayofyear
        df_features["doy_sin"]  = np.sin(2*np.pi*df_features["doy"]/365)
        df_features["doy_cos"]  = np.cos(2*np.pi*df_features["doy"]/365)
        df_features["diff_1_queue"] = df_features["queue"] - df_features["lag_1_queue"]
        df_features["diff_2_queue"] = df_features["diff_1_queue"].diff().fillna(0)
        for window,label in [(2,'30min'),(4,'1h'),(16,'4h'),(96,'24h')]:
            df_features[f'rolling_mean_{label}']=df_features['queue'].rolling(window=window,min_periods=1).mean()
            df_features[f'rolling_std_{label}']=df_features['queue'].rolling(window=window,min_periods=1).std()
            df_features[f'rolling_max_{label}']=df_features['queue'].rolling(window=window,min_periods=1).max()
            df_features[f'rolling_min_{label}']=df_features['queue'].rolling(window=window,min_periods=1).min()
        df_features['expanding_mean']=df_features['queue'].expanding(min_periods=1).mean()
        df_features['expanding_std']=df_features['queue'].expanding(min_periods=1).std().fillna(0)
        for k in [1,2,3]:
            df_features[f'year_fft{k}_sin']=np.sin(2*np.pi*k*df_features['doy']/365)
            df_features[f'year_fft{k}_cos']=np.cos(2*np.pi*k*df_features['doy']/365)
        df_features = df_features.dropna()

        # Prepare AutoGluon TimeSeriesDataFrame for baseline
        df_features['item_id'] = code
        df_features['timestamp'] = df_features['timestamp'].dt.tz_localize(None)
        ts_df = TimeSeriesDataFrame.from_data_frame(df_features, id_column='item_id', timestamp_column='timestamp')

        # Train baseline Chronos (bolt_base) similarly to original app.py
        start_train = time.time()
        airport_path = os.path.join(MODEL_PATH, code)
        os.makedirs(airport_path, exist_ok=True)
        try:
            predictor = TimeSeriesPredictor(
                path=airport_path,
                prediction_length=PREDICTION_LENGTH,
                target='queue',
                freq=FREQ,
                verbosity=0,
                log_to_file=False
            ).fit(
                train_data=ts_df,
                hyperparameters={'Chronos': [{'model_path': 'bolt_base'}]},
                enable_ensemble=False,
                time_limit=300
            )
            predictor.persist()
            train_duration = time.time() - start_train
        except Exception as e:
            logger.exception(f"AutoGluon training failed for {code}: {e}")
            train_duration = 0.0
            predictor = None

        # Autogluon predictions (if predictor available)
        try:
            if predictor is not None:
                preds_ag = predictor.predict(ts_df)
                records_ag = _rename_quantile_cols_and_format(preds_ag.loc[code] if code in preds_ag.index else preds_ag)
            else:
                records_ag = []
        except Exception as e:
            logger.exception(f"AutoGluon prediction failed for {code}: {e}")
            records_ag = []

        # Chronos-2 zero-shot inference
        # Skip if Chronos-2 is explicitly disabled or not installed
        if not chronos2_is_enabled():
            logger.info(f"Chronos-2 not enabled/available — skipping Chronos-2 inference for {code}")
            chronos_duration = 0.0
            records_ch = []
        else:
            try:
                pipeline = get_chronos2_pipeline()
                # Prepare long-format dataframe expected by Chronos-2 pipeline
                chronos_df = df_resampled.copy()
                chronos_df["item_id"] = code
                chronos_df["timestamp"] = chronos_df["timestamp"].dt.tz_localize(None)
                # Chronos-2 expects target column name -- use 'queue' as target
                # predict_df will return a dataframe with index item_id and timestamp and columns 'predictions' and quantiles
                start_inf = time.time()
                pred_df_ch = pipeline.predict_df(
                    chronos_df,
                    prediction_length=PREDICTION_LENGTH,
                    quantile_levels=[0.3, 0.7],
                    id_column='item_id',
                    timestamp_column='timestamp',
                    target='queue',
                )
                chronos_duration = time.time() - start_inf
                # Extract predictions for this airport
                try:
                    # pipeline.predict_df often returns long-format with one row per timestamp per series; select by item_id
                    recs_ch = pred_df_ch.query(f"item_id == @code and target_name == 'queue'").copy()
                    # Some versions simply return ts with index = item_id; handle both
                    if recs_ch.empty:
                        # If not long format, attempt selection using .loc
                        try:
                            recs_ch = pred_df_ch.loc[code]
                        except Exception:
                            recs_ch = pred_df_ch
                except Exception:
                    recs_ch = pred_df_ch
                records_ch = _rename_quantile_cols_and_format(recs_ch)
            except Exception as e:
                logger.exception(f"Chronos-2 inference failed for {code}: {e}")
                chronos_duration = 0.0
                records_ch = []

        # Clean up AutoGluon predictor in this worker
        try:
            del predictor
        except Exception:
            pass
        gc.collect()

        metrics = {
            "airport": code,
            "autogluon": {
                "training_time_seconds": round(train_duration, 2),
                "model_persisted_to": airport_path,
                "trained": bool(train_duration > 0),
            },
            "chronos2": {
                "inference_time_seconds": round(chronos_duration, 3) if 'chronos_duration' in locals() else None,
                "device": _select_chronos2_device(),
                "pipeline_source": _CHRONOS2_S3,
                "inference_performed": len(records_ch) > 0,
            },
            "context_rows": int(len(df_resampled)),
            "last_trained": datetime.datetime.now().isoformat(),
        }
        return code, records_ag, records_ch, metrics
    except Exception as e:
        logger.exception(f"Worker failed for {code}: {e}")
        return code, [], [], {"error": str(e)}

def retrain():
    """Main retraining/repredict routine called periodically or at startup."""
    global autogluon_preds, chronos2_preds, comparison_metrics
    autogluon_preds.clear()
    chronos2_preds.clear()
    comparison_metrics.clear()

    # Try to load data from configured API host; fall back to local all.json when unavailable.
    try:
        response = requests.get(CPHAPI_HOST, timeout=10)
        response.raise_for_status()
        df_raw = pd.DataFrame(response.json())
    except Exception as e:
        logger.warning(f"Failed to fetch data from {CPHAPI_HOST}: {e}. Attempting to load local all.json")
        local_path = os.environ.get("LOCAL_JSON_PATH") or os.path.join(os.path.dirname(__file__), "all.json")
        if os.path.exists(local_path):
            try:
                df_raw = pd.read_json(local_path)
            except Exception as e2:
                logger.exception(f"Failed to load local JSON at {local_path}: {e2}")
                raise
        else:
            logger.exception(f"Local file not found at {local_path}; cannot proceed")
            raise

    # Submit tasks for each airport in VALID_AIRPORTS
    with ProcessPoolExecutor(max_workers=min(len(VALID_AIRPORTS), os.cpu_count() or 1)) as executor:
        futures = {executor.submit(_train_and_predict_for_airport, code, df_raw): code for code in VALID_AIRPORTS}
        for future in as_completed(futures):
            code = futures[future]
            try:
                code, ag_recs, ch_recs, metrics = future.result()
                autogluon_preds[code] = ag_recs
                chronos2_preds[code] = ch_recs
                comparison_metrics[code] = metrics
            except Exception as e:
                logger.error(f"Error processing airport {code}: {e}")
                autogluon_preds[code] = []
                chronos2_preds[code] = []
                comparison_metrics[code] = {"error": str(e)}

# Flask endpoints

@app.route("/forecast/<airport>", methods=["GET"])
def get_forecast(airport):
    code = airport.upper()
    if code not in VALID_AIRPORTS:
        return jsonify({"error": f"Invalid airport code: {airport}"}), 404
    return jsonify({
        "predictions": {
            "autogluon": autogluon_preds.get(code, []),
            "chronos2": chronos2_preds.get(code, []),
        }
    })

@app.route("/metrics", methods=["GET"])
def get_metrics():
    """Return latest training + inference metrics for each airport."""
    return jsonify(comparison_metrics)

if __name__ == "__main__":
    # Run one initial pass then start scheduler (lightweight)
    import multiprocessing as mp
    from apscheduler.schedulers.background import BackgroundScheduler

    mp.freeze_support()
    retrain()

    scheduler = BackgroundScheduler()
    # Keep same cadence as original: every 4 hours
    scheduler.add_job(func=retrain, trigger="interval", hours=4)
    scheduler.start()

    # Expose Flask app
    app.run(host="0.0.0.0")

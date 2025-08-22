import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import holidays
import datetime
from dateutil.easter import easter
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from flask import Flask, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
import datetime
import os
import time
import gc
import requests
import logging
import platform
import json
import multiprocessing as mp
from dotenv import load_dotenv
from concurrent.futures import ProcessPoolExecutor, as_completed

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directory where the persisted model will be saved
MODEL_PATH = 'models/ts_predictor_week'
if os.environ.get("CPHAPI_HOST"):
    CPHAPI_HOST = os.environ.get("CPHAPI_HOST")
else:
    CPHAPI_HOST = "http://apisix:9080/api/v1/all"

# When running on macOS prefer the local all.json file in this package directory
LOCAL_JSON_PATH = None
if platform.system() == "Darwin":
    LOCAL_JSON_PATH = os.path.join(os.path.dirname(__file__), "all.json")

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

# Valid airports for reference
VALID_AIRPORTS = list(airport_holiday_map.keys())

def _next_monday_midnight(from_dt=None):
    """Return the next Monday at 00:00 (if today is Monday, return next week's Monday)."""
    if from_dt is None:
        from_dt = datetime.datetime.now()
    days_ahead = (7 - from_dt.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    next_monday = (from_dt + datetime.timedelta(days=days_ahead)).replace(hour=0, minute=0, second=0, microsecond=0)
    return next_monday

def _train_airport(code, df_raw):
    """Train and forecast for a single airport code; returns code, predictions, actuals, and metrics.
    This variant produces a 7-day forecast at 15-minute frequency (672 points) starting at the upcoming Monday 00:00.
    """
    try:
        # Make copy of raw data for this airport
        df_code = df_raw[df_raw['airport'] == code].copy()
        if df_code.empty:
            return code, [], {'error': 'no_data'}

        df_code['timestamp'] = pd.to_datetime(df_code['timestamp'])
        # Data Wrangling
        df_code = df_code.set_index('timestamp')
        df_code['queue'] = pd.to_numeric(df_code['queue'], errors='coerce')
        df_code = df_code.dropna(subset=['queue'])
        df_code = df_code[['queue']].groupby(level=0).mean()

        # Resample to 15-minute frequency and forward-fill
        tmp_resampled = df_code.resample('15min', origin='start_day').ffill()
        tmp_resampled.index.name = None

        # Determine prediction anchor: upcoming Monday 00:00
        next_monday = _next_monday_midnight()
        cutoff_ts = next_monday - datetime.timedelta(minutes=15)  # last observed timestamp should be Sunday 23:45

        # If our data ends before cutoff_ts, extend by forward-filling up to cutoff_ts so predictions always start at Monday 00:00
        start_idx = tmp_resampled.index.min()
        if pd.isna(start_idx):
            return code, [], {'error': 'no_valid_timestamps'}

        # Normalize index and cutoff to naive UTC to avoid tz-aware/naive mismatches when creating the date range
        idx_tz = getattr(tmp_resampled.index, "tz", None)
        if idx_tz is not None:
            # Convert the underlying index to UTC and drop timezone information
            tmp_resampled.index = tmp_resampled.index.tz_convert("UTC").tz_localize(None)
            start_idx = tmp_resampled.index.min()

        def _to_naive_utc(ts):
            ts = pd.Timestamp(ts)
            if ts.tz is None:
                return ts
            py = ts.to_pydatetime()
            py_utc = py.astimezone(datetime.timezone.utc)
            return pd.Timestamp(py_utc.replace(tzinfo=None))

        cutoff_naive = _to_naive_utc(cutoff_ts)
        full_idx = pd.date_range(start=start_idx, end=cutoff_naive, freq='15min')
        tmp_resampled = tmp_resampled.reindex(full_idx).ffill()

        # Minimal series for Chronos-only training.
        # Use week-ago filling to avoid creating a long flat forward-fill tail.
        df_resampled = tmp_resampled.reset_index().rename(columns={'index': 'timestamp'})

        # Attempt to fill missing values by copying the value from one week earlier when available,
        # then interpolate short gaps, and finally do a conservative ffill/bfill.
        tmp = tmp_resampled.copy()
        week_ago_idx = tmp.index - pd.Timedelta(days=7)
        week_vals = tmp.reindex(week_ago_idx)['queue'].values
        tmp['queue'] = tmp['queue'].fillna(pd.Series(week_vals, index=tmp.index))

        # Interpolate short gaps (up to 1 day by default)
        tmp['queue'] = tmp['queue'].interpolate(method='time', limit=96)

        # Final conservative fills for any remaining missing data
        tmp['queue'] = tmp['queue'].ffill().bfill()

        # Prepare a minimal dataframe with only timestamp, queue and item_id for Chronos
        df_resampled = tmp.reset_index().rename(columns={'index': 'timestamp'})
        df_resampled['item_id'] = code
        # Ensure timestamps are naive (no tz) for AutoGluon compatibility
        df_resampled['timestamp'] = df_resampled['timestamp'].dt.tz_localize(None)
        ts_df = TimeSeriesDataFrame.from_data_frame(df_resampled[['item_id', 'timestamp', 'queue']], id_column='item_id', timestamp_column='timestamp')

        prediction_length = 7 * 24 * 4  # 672

        # Train Chronos bolt_base only. Disable validation windows to lower minimum required history.
        start = time.time()
        airport_path = os.path.join(MODEL_PATH, code)
        os.makedirs(airport_path, exist_ok=True)
        try:
            predictor = TimeSeriesPredictor(
                path=airport_path,
                prediction_length=prediction_length,
                target='queue',
                freq='15min',
                verbosity=0,
                log_to_file=False
            ).fit(
                train_data=ts_df,
                hyperparameters={'Chronos': [{'model_path': 'bolt_base'}]},
                enable_ensemble=False,
                time_limit=300,
                num_val_windows=0
            )
            duration = time.time() - start
            predictor.persist()
        except Exception as e:
            # If training fails, fall back to a weekly-copy forecast (prefer repeating last-week pattern to a flat constant).
            logger.exception(f"AutoGluon training failed for {code}, falling back to weekly-copy: {e}")
            pred_idx = pd.date_range(start=next_monday, periods=prediction_length, freq='15min')
            records = []
            for ts in pred_idx:
                week_ago_ts = ts - pd.Timedelta(days=7)
                if week_ago_ts in tmp.index and not pd.isna(tmp.loc[week_ago_ts, 'queue']):
                    val = float(tmp.loc[week_ago_ts, 'queue'])
                else:
                    # nearest previous filled value
                    val = float(tmp['queue'].ffill().reindex([week_ago_ts]).iloc[0]) if not pd.isna(tmp['queue'].ffill().reindex([week_ago_ts]).iloc[0]) else 0.0
                records.append({'timestamp': ts.strftime('%Y-%m-%dT%H:%M:%S'), 'mean': val, 'q30': val, 'q70': val})
            return code, records, {
                'total_time_seconds': 0,
                'last_trained': datetime.datetime.now().isoformat(),
                'model_performance': [],
                'prediction_start': next_monday.isoformat(),
                'prediction_length': prediction_length,
                'fallback': 'weekly_copy',
                'error': str(e)
            }

        # Metrics
        summary = predictor.fit_summary(); perf = summary.get('model_performance'); perf_records = []
        if isinstance(perf, dict):
            for name, metrics in perf.items():
                if isinstance(metrics, dict): perf_records.append({'model': name, **metrics})
                elif isinstance(metrics, (float, int, np.floating, np.integer)): perf_records.append({'model': name, 'metric_value': float(metrics)})
                else: perf_records.append({'model': name, 'raw_metrics': metrics})
        elif isinstance(perf, (float, int, np.floating, np.integer)): perf_records = [{'metric_value': float(perf)}]
        else:
            try:
                perf_records = perf.reset_index().to_dict(orient='records')
            except Exception:
                perf_records = [{'raw_performance': perf}]

        # Prepare input for prediction: use df_features up to cutoff_ts so predictions start at next_monday
        df_pred_input = df_features[df_features['timestamp'] <= pd.Timestamp(cutoff_ts)].copy()
        if df_pred_input.empty:
            # As a fallback, use the last required_min_obs rows (should not happen because we reindexed earlier)
            df_pred_input = df_features.tail(required_min_obs).copy()

        ts_df_pred = TimeSeriesDataFrame.from_data_frame(df_pred_input, id_column='item_id', timestamp_column='timestamp')

        # Forecast 7 days (672 × 15‑min) ahead from the upcoming Monday 00:00
        preds = predictor.predict(ts_df_pred)
        # Extract predictions for this code
        # preds is a TimeSeriesDataFrame with index item_id; select safely
        try:
            y_pred = preds.loc[code]
        except Exception:
            # If selection fails, attempt to convert predictions to dataframe and filter
            y_pred = preds

        df_pred_code = y_pred.reset_index().rename(
            columns={'timestamp': 'timestamp', 'mean': 'mean', '0.3': 'q30', '0.7': 'q70'}
        )
        # Ensure timestamps are ISO formatted
        df_pred_code['timestamp'] = df_pred_code['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S')

        # Free AutoGluon resources held in this worker
        del predictor
        gc.collect()

        return code, df_pred_code.to_dict(orient='records'), {
            'total_time_seconds': duration,
            'last_trained': datetime.datetime.now().isoformat(),
            'model_performance': perf_records,
            'prediction_start': next_monday.isoformat(),
            'prediction_length': prediction_length
        }
    except Exception as e:
        logger.exception(f"Training failed for {code}: {e}")
        return code, [], {'error': str(e)}

# Placeholder for latest training metrics
train_metrics = {}

# Containers for per-airport forecasts and metrics
df_preds = {}

app = Flask(__name__)

def retrain():
    global df_preds, train_metrics
    # Clear previous run data
    df_preds.clear()
    train_metrics.clear()
    # Load data from API (or local all.json when running on macOS)
    if LOCAL_JSON_PATH and os.path.exists(LOCAL_JSON_PATH):
        with open(LOCAL_JSON_PATH, 'r') as f:
            data = json.load(f)
        df_raw = pd.DataFrame(data)
    else:
        response = requests.get(CPHAPI_HOST)
        response.raise_for_status()
        df_raw = pd.DataFrame(response.json())
    # Parallel training for each airport
    with ProcessPoolExecutor(max_workers=min(len(VALID_AIRPORTS), os.cpu_count() or 1)) as executor:
        futures = {executor.submit(_train_airport, code, df_raw): code for code in VALID_AIRPORTS}
        for future in as_completed(futures):
            code = futures[future]
            try:
                code, pred_records, metrics = future.result()
                df_preds[code] = pred_records
                train_metrics[code] = metrics
            except Exception as e:
                logger.error(f"Error training airport {code}: {e}")

# Scheduler will be started from the main guard below to avoid multiprocessing‑spawn recursion

# Dynamic per-airport forecast endpoint
@app.route('/forecast/<airport>', methods=['GET'])
def get_forecast(airport):
    code = airport.upper()
    if code not in VALID_AIRPORTS:
        return jsonify({'error': f'Invalid airport code: {airport}'}), 404
    return jsonify({
        'predictions': df_preds.get(code, [])
    })


# Endpoint to expose latest training metrics
@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Return latest training performance metrics."""
    return jsonify(train_metrics)

if __name__ == '__main__':
    mp.freeze_support()  # Good practice on Windows; harmless on *nix
    retrain()            # First full training pass

    scheduler = BackgroundScheduler()
    scheduler.add_job(func=retrain, trigger='interval', hours=4)
    scheduler.start()

    app.run(host='0.0.0.0')

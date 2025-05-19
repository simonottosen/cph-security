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
import requests
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directory where the persisted model will be saved
MODEL_PATH = 'models/ts_predictor'
if os.environ.get("CPHAPI_HOST"):
    CPHAPI_HOST = os.environ.get("CPHAPI_HOST")
else:
    CPHAPI_HOST = "http://apisix:9080/api/v1/all"

airport_holiday_map = {
    'AMS': holidays.Netherlands(),
    'ARN': holidays.Sweden(),
    'CPH': holidays.Denmark(),
    'DUB': holidays.Ireland(),
    'DUS': holidays.Germany(),
    'FRA': holidays.Germany(),
    'IST': holidays.Turkey(),
    'LHR': holidays.UnitedKingdom(),
    'MUC': holidays.Germany(),
    'OSL': holidays.Norway(),
}

# Valid airports for reference
VALID_AIRPORTS = list(airport_holiday_map.keys())


# ------------------------------------------------------------------------------
# Utility: Get Data from Local Database (Full) - For reference
# ------------------------------------------------------------------------------
def get_data_via_local_database() -> pd.DataFrame:
    """
    Pulls *all* data for each airport. This can be quite large/slow.
    """
    start_time_load_data = time.time()
    
    # List of airports
    airports = ["CPH", "OSL", "ARN", "DUS", "FRA", "MUC", "LHR", "AMS", "DUB", "IST"]
    
    dfs = []
    for airport_code in airports:
        url = (
            f"{CPHAPI_HOST}"
            f"?order=id.desc&select=id,queue,timestamp,airport"
            f"&airport=eq.{airport_code}"
        )
        df_ap = pd.read_json(url)
        dfs.append(df_ap)
    
    df = pd.concat(dfs, ignore_index=True)
    print("Loaded ALL data successfully from local database in %.2f seconds " % (time.time() - start_time_load_data))
    return df


# Placeholder for latest training metrics
train_metrics = {}

# Containers for per-airport forecasts and metrics
df_preds = {}
df_actuals = {}

app = Flask(__name__)

def retrain():
    global df_preds, df_actuals, train_metrics
    # Clear previous run data
    df_preds.clear()
    df_actuals.clear()
    train_metrics.clear()
    # Load data from API
    response = requests.get(CPHAPI_HOST)
    response.raise_for_status()
    df_raw = pd.DataFrame(response.json())
    # Train separate model and generate forecasts for each airport
    for code in VALID_AIRPORTS:
        # Filter raw data for this airport
        df_code = df_raw[df_raw['airport'] == code].copy()
        df_code['timestamp'] = pd.to_datetime(df_code['timestamp'])

        # Data Wrangling
        df_code = df_code.set_index('timestamp')
        df_code['queue'] = pd.to_numeric(df_code['queue'], errors='coerce')
        df_code = df_code.dropna(subset=['queue'])
        df_code = df_code[['queue']].groupby(level=0).mean()

        # Resample to 15-minute intervals and forward fill
        df_resampled = df_code.resample('15min', origin='start_day').ffill().reset_index()

        # Holiday feature for this airport
        hol = airport_holiday_map.get(code, holidays.country_holidays('DK'))

        # Feature engineering
        df_features = df_resampled.copy()
        df_features['hour_of_day'] = df_features['timestamp'].dt.hour
        df_features['day_of_week'] = df_features['timestamp'].dt.dayofweek
        df_features['lag_1_queue'] = df_features['queue'].shift(1).fillna(0)
        df_features['is_holiday'] = df_features['timestamp'].dt.date.apply(lambda d: 1 if d in hol else 0)
        df_features['rolling_mean_1h'] = df_features['queue'].rolling(window=4, min_periods=1).mean()
        df_features['queue_last_week'] = df_features['queue'].shift(7 * 24 * 4)
        df_features['month'] = df_features['timestamp'].dt.month
        df_features['year'] = df_features['timestamp'].dt.year
        df_features = df_features.dropna()

        # Prepare for AutoGluon
        df_features['item_id'] = code
        df_features['timestamp'] = df_features['timestamp'].dt.tz_localize(None)
        ts_df = TimeSeriesDataFrame.from_data_frame(
            df_features, id_column='item_id', timestamp_column='timestamp'
        )
        prediction_length = 96
        train_data, test_data = ts_df.train_test_split(prediction_length=prediction_length)

        # Train and persist model
        start = time.time()
        airport_model_path = os.path.join(MODEL_PATH, code)
        os.makedirs(airport_model_path, exist_ok=True)
        predictor = TimeSeriesPredictor(
            path=airport_model_path,
            prediction_length=prediction_length,
            target='queue',
            freq='15min'
        ).fit(
            train_data=train_data,
            hyperparameters={'Chronos': [{'model_path': 'bolt_base'}]},
            enable_ensemble=False,
            time_limit=300
        )
        duration = time.time() - start
        predictor.persist()

        # Capture training metrics for this airport
        summary = predictor.fit_summary()
        perf = summary.get('model_performance')
        perf_records = []
        if isinstance(perf, dict):
            for name, metrics in perf.items():
                if isinstance(metrics, dict):
                    record = {'model': name, **metrics}
                elif isinstance(metrics, (float, int, np.floating, np.integer)):
                    record = {'model': name, 'metric_value': float(metrics)}
                else:
                    record = {'model': name, 'raw_metrics': metrics}
                perf_records.append(record)
        elif isinstance(perf, (float, int, np.floating, np.integer)):
            perf_records = [{'metric_value': float(perf)}]
        else:
            try:
                perf_df = perf.reset_index()
                perf_records = perf_df.to_dict(orient='records')
            except Exception:
                perf_records = [{'raw_performance': perf}]
        train_metrics[code] = {
            'total_time_seconds': duration,
            'last_trained': datetime.datetime.now().isoformat(),
            'model_performance': perf_records
        }

        # Generate forecasts for this airport
        preds = predictor.predict(train_data)
        y_pred = preds.loc[code]
        y_test = test_data.loc[code]['queue'].loc[y_pred.index]
        df_pred_code = y_pred.reset_index().rename(
            columns={'timestamp': 'timestamp', 'mean': 'mean', '0.3': 'q30', '0.7': 'q70'}
        )
        df_pred_code['timestamp'] = df_pred_code['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S')
        df_actual_code = y_test.reset_index().rename(columns={'queue': 'actual'})
        df_actual_code['timestamp'] = df_actual_code['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S')
        df_actual_code['smoothed'] = df_actual_code['actual'].rolling(window=3, center=True).mean()

        df_preds[code] = df_pred_code.to_dict(orient='records')
        df_actuals[code] = df_actual_code.to_dict(orient='records')

# Initial training and scheduler setup
retrain()
scheduler = BackgroundScheduler()
scheduler.add_job(func=retrain, trigger='interval', hours=4, next_run_time=datetime.datetime.now())
scheduler.start()

# Dynamic per-airport forecast endpoint
@app.route('/forecast/<airport>', methods=['GET'])
def get_forecast(airport):
    code = airport.upper()
    if code not in VALID_AIRPORTS:
        return jsonify({'error': f'Invalid airport code: {airport}'}), 404
    return jsonify({
        'predictions': df_preds.get(code, []),
        'actual_smoothed': df_actuals.get(code, [])
    })


# Endpoint to expose latest training metrics
@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Return latest training performance metrics."""
    return jsonify(train_metrics)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

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
import multiprocessing as mp
from dotenv import load_dotenv
from concurrent.futures import ProcessPoolExecutor, as_completed

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
    'EDI': holidays.UnitedKingdom(),
    'MUC': holidays.Germany(),
    'OSL': holidays.Norway(),
}

# Valid airports for reference
VALID_AIRPORTS = list(airport_holiday_map.keys())

def _train_airport(code, df_raw):
    """Train and forecast for a single airport code; returns code, predictions, actuals, and metrics."""
    # Make copy of raw data for this airport
    df_code = df_raw[df_raw['airport'] == code].copy()
    df_code['timestamp'] = pd.to_datetime(df_code['timestamp'])
    # Data Wrangling
    df_code = df_code.set_index('timestamp')
    df_code['queue'] = pd.to_numeric(df_code['queue'], errors='coerce')
    df_code = df_code.dropna(subset=['queue'])
    df_code = df_code[['queue']].groupby(level=0).mean()
    # Resample and reset index without duplicate timestamp conflict
    tmp_resampled = df_code.resample('5min', origin='start_day').ffill()
    tmp_resampled.index.name = None
    df_resampled = tmp_resampled.reset_index().rename(columns={'index': 'timestamp'})
    # Holidays
    hol = airport_holiday_map.get(code, holidays.country_holidays('DK'))
    # Feature engineering (copy from existing block)
    df_features = df_resampled.copy()
    df_features['hour_of_day'] = df_features['timestamp'].dt.hour
    df_features['day_of_week'] = df_features['timestamp'].dt.dayofweek
    df_features['lag_1_queue'] = df_features['queue'].shift(1).fillna(0)
    df_features['is_holiday'] = df_features['timestamp'].dt.date.apply(lambda d: 1 if d in hol else 0)
    df_features['rolling_mean_1h'] = df_features['queue'].rolling(window=4, min_periods=1).mean()
    df_features['queue_last_week'] = df_features['queue'].shift(7 * 24 * 4)
    df_features['month'] = df_features['timestamp'].dt.month
    df_features['year'] = df_features['timestamp'].dt.year
    df_features['lag_2_queue'] = df_features['queue'].shift(2).fillna(0)
    df_features['lag_3_queue'] = df_features['queue'].shift(3).fillna(0)
    df_features['lag_4_queue'] = df_features['queue'].shift(4).fillna(0)
    df_features['is_weekend'] = (df_features['timestamp'].dt.dayofweek >= 5).astype(int)
    df_features['is_business_day'] = ((df_features['timestamp'].dt.dayofweek < 5) & (df_features['is_holiday']==0)).astype(int)
    df_features['rolling_std_1h'] = df_features['queue'].rolling(window=4, min_periods=1).std()
    df_features['quarter'] = df_features['timestamp'].dt.quarter
    df_features['is_winter'] = df_features['month'].isin([12,1,2]).astype(int)
    df_features['is_spring'] = df_features['month'].isin([3,4,5]).astype(int)
    df_features['is_summer'] = df_features['month'].isin([6,7,8]).astype(int)
    df_features['is_autumn'] = df_features['month'].isin([9,10,11]).astype(int)
    holiday_dates = sorted(hol)
    def days_to_next_holiday(date):
        future = [h for h in holiday_dates if h>=date]; return (future[0]-date).days if future else np.nan
    def days_since_last_holiday(date):
        past = [h for h in holiday_dates if h<=date]; return (date-past[-1]).days if past else np.nan
    df_features['days_to_next_holiday'] = df_features['timestamp'].dt.date.apply(days_to_next_holiday)
    df_features['days_since_last_holiday'] = df_features['timestamp'].dt.date.apply(days_since_last_holiday)
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
    df_features['is_school_holiday'] = df_features['timestamp'].dt.date.apply(lambda d: 1 if is_school_holiday(d) else 0)
    def get_school_periods(year):
        return [
            (datetime.date(year-1,12,20), datetime.date(year,1,3)),
            (easter(year)-datetime.timedelta(days=3), easter(year)+datetime.timedelta(days=1)),
            (datetime.date(year,7,1), datetime.date(year,8,31)),
            (datetime.date.fromisocalendar(year,42,1), datetime.date.fromisocalendar(year,42,5)),
            (datetime.date.fromisocalendar(year,7,1), datetime.date.fromisocalendar(year,7,5))
        ]
    def days_to_next_school_holiday(date):
        periods = get_school_periods(date.year)+get_school_periods(date.year+1)
        starts = sorted(start for start,_ in periods)
        future = [s for s in starts if s>=date]; return (future[0]-date).days if future else np.nan
    def days_since_last_school_holiday(date):
        periods = get_school_periods(date.year)+get_school_periods(date.year-1)
        ends = sorted(end for _,end in periods)
        past = [e for e in ends if e<=date]; return (date-past[-1]).days if past else np.nan
    df_features['days_to_next_school_holiday'] = df_features['timestamp'].dt.date.apply(days_to_next_school_holiday)
    df_features['days_since_last_school_holiday'] = df_features['timestamp'].dt.date.apply(days_since_last_school_holiday)
    df_features['hour_sin'] = np.sin(2*np.pi*df_features['hour_of_day']/24)
    df_features['hour_cos'] = np.cos(2*np.pi*df_features['hour_of_day']/24)
    df_features['dow_sin']  = np.sin(2*np.pi*df_features['day_of_week']/7)
    df_features['dow_cos']  = np.cos(2*np.pi*df_features['day_of_week']/7)
    df_features['doy']      = df_features['timestamp'].dt.dayofyear
    df_features['doy_sin']  = np.sin(2*np.pi*df_features['doy']/365)
    df_features['doy_cos']  = np.cos(2*np.pi*df_features['doy']/365)
    df_features['diff_1_queue'] = df_features['queue']-df_features['lag_1_queue']
    df_features['diff_2_queue'] = df_features['diff_1_queue'].diff().fillna(0)
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
    # Prepare data for AutoGluon
    df_features['item_id'] = code
    df_features['timestamp'] = df_features['timestamp'].dt.tz_localize(None)
    ts_df = TimeSeriesDataFrame.from_data_frame(df_features, id_column='item_id', timestamp_column='timestamp')
    prediction_length=96
    train_data, test_data = ts_df.train_test_split(prediction_length=prediction_length)
    # Train model
    start=time.time()
    airport_path=os.path.join(MODEL_PATH,code)
    os.makedirs(airport_path,exist_ok=True)
    predictor=TimeSeriesPredictor(path=airport_path,prediction_length=prediction_length,target='queue',freq='5min',verbosity=0,log_to_file=False).fit(train_data=train_data,hyperparameters={'Chronos': [{'model_path': 'bolt_base'}]},enable_ensemble=False,time_limit=300)
    duration=time.time()-start
    predictor.persist()
    # Metrics
    summary=predictor.fit_summary(); perf=summary.get('model_performance'); perf_records=[]
    if isinstance(perf,dict):
        for name,metrics in perf.items():
            if isinstance(metrics,dict): perf_records.append({'model':name,**metrics})
            elif isinstance(metrics,(float,int,np.floating,np.integer)): perf_records.append({'model':name,'metric_value':float(metrics)})
            else: perf_records.append({'model':name,'raw_metrics':metrics})
    elif isinstance(perf,(float,int,np.floating,np.integer)): perf_records=[{'metric_value':float(perf)}]
    else:
        try: perf_records=perf.reset_index().to_dict(orient='records')
        except Exception: perf_records=[{'raw_performance':perf}]
    # Forecast 6 hours (24 × 15‑min) ahead from the latest timestamp
    preds = predictor.predict(ts_df)
    y_pred = preds.loc[code]

    df_pred_code = y_pred.reset_index().rename(
        columns={'timestamp': 'timestamp', 'mean': 'mean', '0.3': 'q30', '0.7': 'q70'}
    )
    df_pred_code['timestamp'] = df_pred_code['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S')

    # Free AutoGluon resources held in this worker
    del predictor
    gc.collect()

    return code, df_pred_code.to_dict(orient='records'), {
        'total_time_seconds': duration,
        'last_trained': datetime.datetime.now().isoformat(),
        'model_performance': perf_records
    }

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
    # Load data from API
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

import os
import time
import logging
import joblib
import numpy as np
import pandas as pd
import holidays
from flask import Flask, request, jsonify
from flask_caching import Cache
from flask_crontab import Crontab

from datetime import datetime, timedelta, timezone
from urllib.parse import quote
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
from supabase import create_client, Client
from dotenv import load_dotenv
from sklearn.base import BaseEstimator, RegressorMixin

load_dotenv()

# ------------------------------------------------------------------------------
# Logging Setup
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Environment & Global Variables
# ------------------------------------------------------------------------------
UseSupabase = 0  # If 1 then Yes, if 0 then No. Do not use Supabase for production
if os.environ.get("CPHAPI_HOST"):
    CPHAPI_HOST = os.environ.get("CPHAPI_HOST")
else:
    CPHAPI_HOST = "http://apisix:9080/api/v1/all"

# A dictionary mapping each airport code to its local holiday calendar
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

# Models directory (persist models across restarts). Can be overridden with MODELS_DIR env var.
MODELS_DIR = os.environ.get("MODELS_DIR", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ------------------------------------------------------------------------------
# A Fallback Regressor that Always Returns 5
# ------------------------------------------------------------------------------
class FallbackRegressor(BaseEstimator, RegressorMixin):
    """A fallback regressor that always returns 5."""
    def __init__(self, constant=5):
        self.constant = constant

    def fit(self, X, y=None):
        return self  # No actual fitting

    def predict(self, X):
        return np.full(shape=(len(X),), fill_value=self.constant)

# ------------------------------------------------------------------------------
# Utility: Get Data from Supabase (Full)
# ------------------------------------------------------------------------------
def get_data_via_supabase_client() -> pd.DataFrame:
    """
    Fetch data from Supabase. Adjust table/columns/filters as needed.
    """
    supabase_url = os.environ.get("SUPABASE_URL", "")
    supabase_key = os.environ.get("SUPABASE_KEY", "")
    if not supabase_url or not supabase_key:
        raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY environment variables.")
    
    supabase: Client = create_client(supabase_url, supabase_key)
    
    response = (
        supabase
        .table("waitingtime")
        .select("id,queue,timestamp,airport")
        .neq("airport", "BER")   # Exclude BER
        .neq("airport", "OSL")   # Exclude OSL
        .order("id", desc=True)
        .execute()
    )
    df = pd.DataFrame(response.data)
    return df

# ------------------------------------------------------------------------------
# Utility: Get Data from Local Database (Full) - For reference
# ------------------------------------------------------------------------------
def get_data_via_local_database() -> pd.DataFrame:
    """
    Pulls *all* data for each airport. This can be quite large/slow.
    """
    start_time_load_data = time.time()
    
    # List of airports
    airports = ["CPH", "ARN", "DUS", "FRA", "MUC", "LHR", "AMS", "DUB", "IST", "EDI"]
    
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
    logger.info("Loaded ALL data successfully from local database in %.2f seconds", time.time() - start_time_load_data)
    return df

# ------------------------------------------------------------------------------
# NEW Utility: Get Data from Local Database (Light) - Only last 7 days
# ------------------------------------------------------------------------------
def get_data_via_local_database_light() -> pd.DataFrame:
    """
    Pull only the last 7 days of data for each airport. 
    Since entries occur every 5 minutes, you get ~2016 rows/airport/week,
    for a total ~20k rows across 10 airports, which is much faster to load.
    Adjust the time window if needed.
    """
    start_time_load_data = time.time()
    
    # How many days of data for "light" fetch
    num_days = 7
    
    # Calculate the start time in UTC (e.g., 7 days ago)
    end_time_utc = datetime.now(timezone.utc)
    start_time_utc = end_time_utc - timedelta(days=num_days)
    # Use an explicit Z timezone marker and URL-encode the timestamp for the query
    start_time_str = start_time_utc.isoformat().replace("+00:00", "Z")
    start_time_param = quote(start_time_str, safe="")

    airports = ["CPH", "ARN", "DUS", "FRA", "MUC", "LHR", "AMS", "DUB", "IST", "EDI"]
    
    dfs = []
    for airport_code in airports:
        url = (
            f"{CPHAPI_HOST}"
            f"?order=id.desc"
            f"&select=id,queue,timestamp,airport"
            f"&airport=eq.{airport_code}"
            f"&timestamp=gte.{start_time_param}"
        )
        df_ap = pd.read_json(url)
        dfs.append(df_ap)
    
    df = pd.concat(dfs, ignore_index=True)
    logger.info("Loaded LIGHT data (last 7 days) from local database in %.2f seconds", time.time() - start_time_load_data)
    return df

# ------------------------------------------------------------------------------
# Flask App Setup
# ------------------------------------------------------------------------------
app = Flask(__name__)
cache = Cache(app, config={'CACHE_TYPE': 'simple', 'CACHE_DEFAULT_TIMEOUT': 300})  # default TTL: 5 minutes
# crontab/training removed from the API process; use a separate trainer service (CronJob) instead.

# ------------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------------
def add_holiday_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized Holiday feature: sets 1 if the timestamp's date is a local holiday
    for the airport indicated in the row. This avoids row-wise apply().
    """
    if df.empty:
        df['Holiday'] = 0
        return df

    # Dates as datetime.date
    date_index = pd.Index(df.index.date)
    holiday_flags = np.zeros(len(df), dtype=np.int8)

    for ap, cal in airport_holiday_map.items():
        if ap in df.columns:
            # boolean mask where this airport is active for the row
            ap_mask = df[ap].to_numpy(dtype=bool)
            # create set of holiday dates for fast membership testing
            hol_dates = set(cal.keys())
            # vectorized date membership
            date_mask = np.array([d in hol_dates for d in date_index], dtype=bool)
            combined = ap_mask & date_mask
            holiday_flags = holiday_flags | combined.astype(np.int8)

    df['Holiday'] = holiday_flags
    return df

def add_lag_features(df: pd.DataFrame, lags=(1,2,3,6,12)) -> pd.DataFrame:
    """
    Create per-airport lag features named '{airport}_lag_{L}' where L is number of rows back.
    This assumes the index is time-ordered.
    """
    for airport_code in VALID_AIRPORTS:
        if airport_code not in df.columns:
            continue
        airport_data = df[df[airport_code] == 1].copy()
        if airport_data.empty:
            # still create columns to keep a consistent schema
            for L in lags:
                df[f'{airport_code}_lag_{L}'] = 0.0
            continue

        airport_data = airport_data.sort_index()
        for L in lags:
            df.loc[airport_data.index, f'{airport_code}_lag_{L}'] = airport_data['queue'].shift(L)

    # Fill NaNs produced by shifting with 0.0 (training will drop initial NaNs if needed)
    lag_cols = [c for c in df.columns if c.endswith(tuple([f'_lag_{L}' for L in lags]))]
    for col in lag_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)
    return df

def preprocess_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Centralized preprocessing shared by get_data() and get_data_light():
      - timezone conversion
      - indexing and dedup/tiebreaker
      - time features
      - airport one-hot
      - rolling features per airport
      - holiday feature
      - cyclical features and is_weekend
      - lag features (per airport)
      - basic cleaning (id drop, queue numeric & NaN drop)
    """
    if dataframe.empty:
        return pd.DataFrame()

    # Convert timestamps from UTC to Europe/Copenhagen and set index
    StartTime = pd.to_datetime(dataframe["timestamp"], utc=True)
    StartTime = StartTime.dt.tz_convert('Europe/Copenhagen')
    dataframe = dataframe.assign(timestamp=StartTime)

    df = dataframe.set_index("timestamp").sort_index()
    df.index = df.index + pd.to_timedelta(df.groupby(level=0).cumcount(), unit='ns')

    # Basic time features
    df['year'] = df.index.year
    df['hour'] = df.index.hour
    df['day'] = df.index.day
    df['month'] = df.index.month
    df['weekday'] = df.index.weekday

    # airport one-hot
    if 'airport' in df.columns:
        df_airport = pd.get_dummies(df['airport'])
        df = pd.concat([df, df_airport], axis=1).drop(columns=['airport'])

    # rolling features per airport
    for airport_code in VALID_AIRPORTS:
        if airport_code in df.columns:
            _compute_rolling_features(df, airport_code)

    # holiday
    df = add_holiday_feature(df)

    # cyclical time features and weekend flag
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin']  = np.sin(2 * np.pi * df['weekday'] / 7)
    df['dow_cos']  = np.cos(2 * np.pi * df['weekday'] / 7)
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)

    # lag features per airport
    df = add_lag_features(df)

    # housekeeping
    if 'id' in df.columns:
        df.drop(['id'], axis=1, inplace=True)

    df['queue'] = pd.to_numeric(df['queue'], errors='coerce')
    df['queue'] = df['queue'].apply(lambda x: np.nan if x > 1e10 else x)
    df.dropna(subset=['queue'], inplace=True)

    return df

def _compute_rolling_features(df: pd.DataFrame, airport_code: str) -> None:
    """
    For a given airport_code, create two new columns in df:
      - '{airport_code}_rolling_24h': mean queue of the last 24 hours
      - '{airport_code}_rolling_7d':  mean queue of the last 7 days
    Each row's rolling average is based on that airport's data up to that row's timestamp index.
    """
    airport_data = df[df[airport_code] == 1].copy()
    if airport_data.empty:
        return

    # Sort by time to ensure proper rolling
    airport_data.sort_index(inplace=True)

    airport_data['rolling_24h'] = (
        airport_data['queue']
            .rolling('24h', min_periods=1)
            .mean()
    )
    airport_data['rolling_7d'] = (
        airport_data['queue']
            .rolling('7d', min_periods=1)
            .mean()
    )

    df.loc[airport_data.index, f'{airport_code}_rolling_24h'] = airport_data['rolling_24h']
    df.loc[airport_data.index, f'{airport_code}_rolling_7d'] = airport_data['rolling_7d']

# ------------------------------------------------------------------------------
# Data Fetching with Caching
# ------------------------------------------------------------------------------
@cache.memoize()
def get_data():
    """
    Fetch the *full* dataset from Supabase (or local API) and preprocess.
    """
    start_time = time.time()
    logger.info("Fetching full dataset...")

    try:
        if UseSupabase == 1:
            dataframe = get_data_via_supabase_client()
        else:
            dataframe = get_data_via_local_database()
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return pd.DataFrame()

    # Centralized preprocessing
    df = preprocess_dataframe(dataframe)

    logger.info("Fetched & preprocessed FULL dataset in %.2f seconds", time.time() - start_time)
    return df

@cache.memoize()
def get_data_light():
    """
    Fetch a *light* dataset (e.g., last 7 days).
    Preprocess similarly (convert timestamps, add time features, etc.).
    """
    start_time = time.time()
    logger.info("Fetching light dataset...")

    try:
        if UseSupabase == 1:
            # If you prefer a light approach for Supabase, you can create a similar function
            # that filters data by time or row count from Supabase. For now, fallback:
            dataframe = get_data_via_supabase_client()
        else:
            dataframe = get_data_via_local_database_light()
    except Exception as e:
        logger.error(f"Error fetching light data: {e}")
        return pd.DataFrame()

    # Centralized preprocessing
    df = preprocess_dataframe(dataframe)

    logger.info("Fetched & preprocessed LIGHT dataset in %.2f seconds", time.time() - start_time)
    return df

# ------------------------------------------------------------------------------
# Separate Model Training
# ------------------------------------------------------------------------------
def train_models():
    """
    Train a separate XGBoost model for each airport in VALID_AIRPORTS.
    If no data or training fails, store a FallbackRegressor for that airport.
    """
    start_time = time.time()
    logger.info("Starting model training for each airport...")

    df = get_data()
    if df.empty:
        logger.warning("No data returned. Saving fallback models for all airports.")
        for ap in VALID_AIRPORTS:
            fallback = FallbackRegressor(constant=5)
            joblib.dump(fallback, os.path.join(MODELS_DIR, f'trained_model_{ap}.joblib'))
        return

    df.sort_index(inplace=True)
    df = df.reindex(sorted(df.columns), axis=1)

    for airport_code in VALID_AIRPORTS:
        if airport_code not in df.columns:
            logger.warning(f"No column for {airport_code}. Saving fallback model.")
            fallback = FallbackRegressor()
            joblib.dump(fallback, os.path.join(MODELS_DIR, f'trained_model_{airport_code}.joblib'))
            continue

        airport_df = df[df[airport_code] == 1]
        if airport_df.empty:
            logger.warning(f"No data for airport {airport_code}, saving fallback.")
            fallback = FallbackRegressor()
            joblib.dump(fallback, os.path.join(MODELS_DIR, f'trained_model_{airport_code}.joblib'))
            continue

        # Ensure rolling columns exist
        rolling_cols = [f'{airport_code}_rolling_24h', f'{airport_code}_rolling_7d']
        for rc in rolling_cols:
            if rc not in airport_df.columns:
                airport_df[rc] = 0.0

        cutoff = int(len(airport_df) * 0.9)
        train_df = airport_df.iloc[:cutoff]
        test_df  = airport_df.iloc[cutoff:]
        logger.info(f"{airport_code} - train size: {len(train_df)}, test size: {len(test_df)}")

        if len(train_df) < 10:
            logger.warning(f"Not enough training rows for {airport_code}, using fallback.")
            fallback = FallbackRegressor()
            joblib.dump(fallback, os.path.join(MODELS_DIR, f'trained_model_{airport_code}.joblib'))
            continue

        try:
            X_train = train_df.drop('queue', axis=1)
            y_train = train_df['queue']
            X_test  = test_df.drop('queue', axis=1)
            y_test  = test_df['queue']

            # Drop airport one-hot columns for per-airport training (they are constant)
            airport_cols = [ap for ap in VALID_AIRPORTS if ap in X_train.columns]
            X_train = X_train.drop(columns=airport_cols, errors='ignore')
            X_test = X_test.drop(columns=airport_cols, errors='ignore')

            # Ensure lag columns exist
            lag_cols = [f'{airport_code}_lag_{L}' for L in (1,2,3,6,12)]
            for lc in lag_cols:
                if lc not in X_train.columns:
                    X_train[lc] = 0.0
                    X_test[lc] = 0.0

            model = xgb.XGBRegressor(
                objective='reg:squarederror',
                tree_method='hist',
                n_estimators=1000,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                random_state=7,
            )
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                early_stopping_rounds=50,
                verbose=False
            )

            if not X_test.empty:
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                rmse = mean_squared_error(y_test, y_pred, squared=False)
                mae = mean_absolute_error(y_test, y_pred)
                logger.info(f"{airport_code} - MAE: {mae:.2f} RMSE: {rmse:.2f} MSE: {mse:.2f}")

            save_path = os.path.join(MODELS_DIR, f'trained_model_{airport_code}.joblib')
            logger.info(f"Saving model to {save_path}")
            joblib.dump(model, save_path)
            if os.path.exists(save_path):
                logger.info(f"Model for {airport_code} saved at {save_path}.")
            else:
                logger.error(f"Model file for {airport_code} was not found after save attempt: {save_path}")
        except Exception as e:
            logger.error(f"Error training {airport_code}, using fallback. Error: {e}")
            fallback = FallbackRegressor()
            joblib.dump(fallback, os.path.join(MODELS_DIR, f'trained_model_{airport_code}.joblib'))

    logger.info("Finished training in %.2f seconds", time.time() - start_time)

def load_model_for_airport(airport_code: str):
    """
    Load model from 'trained_model_{airport_code}.joblib'.
    If file not found or error, return FallbackRegressor(5).
    Models should be produced by an external trainer and persisted to storage.
    """
    try:
        model = joblib.load(os.path.join(MODELS_DIR, f"trained_model_{airport_code}.joblib"))
        logger.info(f"Loaded model for {airport_code}.")
        return model
    except Exception as e:
        logger.error(f"Could not load model for {airport_code}, using fallback. Error: {e}")
        return FallbackRegressor(constant=5)

# ------------------------------------------------------------------------------
# Prediction Logic
# ------------------------------------------------------------------------------
def predict_queue(timestamp_df: pd.DataFrame) -> float:
    """
    Predict queue length for a single row input with ['timestamp', 'airport'].
    If no model/data, returns fallback=5.
    """
    airport = timestamp_df["airport"].iloc[0]
    model = load_model_for_airport(airport)

    # 1) Convert timestamp to datetime + localize
    modeldatetime = pd.to_datetime(timestamp_df["timestamp"], utc=True)
    modeldatetime = modeldatetime.dt.tz_convert('Europe/Copenhagen')

    timestamp_df["timestamp"] = modeldatetime
    timestamp_df = timestamp_df.set_index(timestamp_df.timestamp)

    # 2) Date/time features
    timestamp_df['year'] = timestamp_df.index.year
    timestamp_df['hour'] = timestamp_df.index.hour
    timestamp_df['day'] = timestamp_df.index.day
    timestamp_df['month'] = timestamp_df.index.month
    timestamp_df['weekday'] = timestamp_df.index.weekday

    # cyclical encoding for hour and day-of-week and weekend flag
    timestamp_df['hour_sin'] = np.sin(2 * np.pi * timestamp_df['hour'] / 24)
    timestamp_df['hour_cos'] = np.cos(2 * np.pi * timestamp_df['hour'] / 24)
    timestamp_df['dow_sin']  = np.sin(2 * np.pi * timestamp_df['weekday'] / 7)
    timestamp_df['dow_cos']  = np.cos(2 * np.pi * timestamp_df['weekday'] / 7)
    timestamp_df['is_weekend'] = (timestamp_df['weekday'] >= 5).astype(int)

    # 3) One-hot
    for ap in VALID_AIRPORTS:
        timestamp_df[ap] = 1 if ap == airport else 0

    # 4) Rolling features from LIGHT data + short lags
    df_light = get_data_light()
    lag_list = (1,2,3,6,12)
    if df_light.empty:
        rolling_24h_val = 0.0
        rolling_7d_val  = 0.0
        lag_vals = {L: 0.0 for L in lag_list}
    else:
        cutoff_time = modeldatetime.iloc[0]
        airport_df = df_light[(df_light[airport] == 1) & (df_light.index <= cutoff_time)].copy()
        if airport_df.empty:
            rolling_24h_val = 0.0
            rolling_7d_val  = 0.0
            lag_vals = {L: 0.0 for L in lag_list}
        else:
            airport_df.sort_index(inplace=True)
            airport_df['rolling_24h'] = airport_df['queue'].rolling('24h', min_periods=1).mean()
            airport_df['rolling_7d']  = airport_df['queue'].rolling('7d',  min_periods=1).mean()

            last_row = airport_df.iloc[-1]
            rolling_24h_val = last_row['rolling_24h'] if not pd.isna(last_row['rolling_24h']) else 0.0
            rolling_7d_val  = last_row['rolling_7d']  if not pd.isna(last_row['rolling_7d']) else 0.0

            # compute simple lag values from the most recent rows (lag=1 is the last observed queue)
            lag_vals = {}
            for L in lag_list:
                if len(airport_df) >= L:
                    lag_vals[L] = airport_df['queue'].iloc[-L]
                else:
                    lag_vals[L] = 0.0

    # set the airport's rolling features
    timestamp_df[f'{airport}_rolling_24h'] = rolling_24h_val
    timestamp_df[f'{airport}_rolling_7d']  = rolling_7d_val

    # For other airports, set rolling to 0
    for ap in VALID_AIRPORTS:
        if ap != airport:
            timestamp_df[f'{ap}_rolling_24h'] = 0.0
            timestamp_df[f'{ap}_rolling_7d']  = 0.0

    # set lag features for this airport and zeros for others
    for L, val in lag_vals.items():
        timestamp_df[f'{airport}_lag_{L}'] = val
    for ap in VALID_AIRPORTS:
        if ap != airport:
            for L in lag_list:
                timestamp_df[f'{ap}_lag_{L}'] = 0.0

    # 5) Holiday feature
    timestamp_df = add_holiday_feature(timestamp_df)

    # 6) Drop 'timestamp' & 'airport'
    for col in ['timestamp', 'airport']:
        if col in timestamp_df.columns:
            timestamp_df.drop(col, axis=1, inplace=True)

    # 7) Sort columns
    timestamp_df = timestamp_df.reindex(sorted(timestamp_df.columns), axis=1)

    # 8) Predict
    prediction = model.predict(timestamp_df)
    # Do not bias predictions with an arbitrary offset. Round to nearest integer minute.
    return int(round(prediction[0]))

# ------------------------------------------------------------------------------
# Flask Routes
# ------------------------------------------------------------------------------
@app.route('/predict')
def make_prediction():
    """
    Usage: /predict?airport=ARN&timestamp=YYYY-MM-DDTHH:MM (UTC)
    Returns JSON {'predicted_queue_length_minutes': X}
    """
    start_time = time.time()
    logger.info("Received request to predict queue.")

    input_date_str = request.args.get('timestamp')
    airport_code   = request.args.get('airport')

    if not input_date_str and not airport_code:
        return jsonify({
            'error': 'Missing "airport" and "timestamp". e.g., /predict?airport=ARN&timestamp=YYYY-MM-DDTHH:MM'
        }), 400

    if not input_date_str:
        return jsonify({
            'error': 'Missing "timestamp". e.g., /predict?airport=ARN&timestamp=YYYY-MM-DDTHH:MM'
        }), 400

    if not airport_code:
        return jsonify({
            'error': 'Missing "airport". e.g., /predict?airport=ARN&timestamp=YYYY-MM-DDTHH:MM'
        }), 400

    airport_code = airport_code.upper()
    if airport_code not in VALID_AIRPORTS:
        return jsonify({
            'error': f'Invalid airport code "{airport_code}". Valid codes: {",".join(VALID_AIRPORTS)}.'
        }), 400

    try:
        input_date = pd.to_datetime(input_date_str, utc=True)
    except (ValueError, TypeError):
        return jsonify({'error': 'Invalid "timestamp". Expected YYYY-MM-DDTHH:MM (UTC)'}), 400

    input_data = pd.DataFrame({'timestamp': [input_date], 'airport': [airport_code]})
    predicted = predict_queue(input_data)

    response = {'predicted_queue_length_minutes': predicted}
    logger.info("Prediction completed in %.2f seconds", time.time() - start_time)
    return jsonify(response)

# ------------------------------------------------------------------------------
# Crontab Job for Retraining (always enabled) + startup model ensure
# ------------------------------------------------------------------------------
def ensure_models_on_startup():
    """
    Ensure models exist on disk. If any are missing, perform initial training.
    This guarantees the first requests after a fresh deploy won't hit fallbacks.
    """
    missing = []
    for ap in VALID_AIRPORTS:
        path = os.path.join(MODELS_DIR, f"trained_model_{ap}.joblib")
        if not os.path.exists(path):
            missing.append(ap)

    if missing:
        logger.info(f"Models missing for {len(missing)} airports: {','.join(missing)}. Running initial training.")
        train_models()
    else:
        logger.info("All models present on startup.")

crontab = Crontab(app)

@crontab.job(minute=0, hour=0)
def scheduled_training():
    """
    Daily re-training (midnight). Runs inside the app process.
    """
    logger.info(f'Crontab triggered: retraining models at {datetime.now()}')
    train_models()

# ------------------------------------------------------------------------------
# On Startup: Train once (optional)
# ------------------------------------------------------------------------------
with app.app_context():
    logger.info("Initializing application...")
    # Ensure models exist on startup (creates them if missing)
    ensure_models_on_startup()

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')

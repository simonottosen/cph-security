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
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
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
    CPHAPI_HOST = "https://waitport.com/api/v1/all"

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
    'MUC': holidays.Germany(),
    'OSL': holidays.Norway(),
}

# Valid airports for reference
VALID_AIRPORTS = list(airport_holiday_map.keys())

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
# Utility: Get Data from Supabase
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
        .neq("airport", "BER")  # Exclude BER
        .order("id", desc=True)
        .execute()
    )
    df = pd.DataFrame(response.data)
    return df

def get_data_via_local_database() -> pd.DataFrame:
    start_time_load_data = time.time()
    newmodeldata_url = (
        str(CPHAPI_HOST)
        + "?order=id.desc&limit=100000&select=id,queue,timestamp,airport&airport=not.eq.BER"
    )
    df = pd.read_json(newmodeldata_url)
    print("Loaded data successfully in %.2f seconds " % (time.time() - start_time_load_data))
    return df

# ------------------------------------------------------------------------------
# Flask App Setup
# ------------------------------------------------------------------------------
app = Flask(__name__)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})
crontab = Crontab(app)

# ------------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------------
def add_holiday_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'Holiday' feature (1/0) indicating if a day is a local holiday for the
    airport corresponding to each row.

    Since the DataFrame is one-hot encoded for airports (e.g., 'AMS'=1, 'CPH'=1),
    we check for each airport code which is 1 and use that country's holiday set.
    """
    def is_local_holiday(row):
        # row.name is the datetime index for that row
        for ap, holiday_obj in airport_holiday_map.items():
            if ap in row.index and row[ap] == 1:
                dt = row.name.date()
                return int(dt in holiday_obj)
        return 0

    df['Holiday'] = df.apply(is_local_holiday, axis=1)
    return df

def _compute_rolling_features(df: pd.DataFrame, airport_code: str) -> None:
    """
    For a given airport_code, create two new columns in df:
      - '{airport_code}_rolling_24h': mean queue of the last 24 hours
      - '{airport_code}_rolling_7d':  mean queue of the last 7 days
    Each row's rolling average is based on that airport's data up to that row's index timestamp.
    """
    # Filter only rows for this airport
    airport_data = df[df[airport_code] == 1].copy()
    if airport_data.empty:
        return  # no data to process

    # Sort by time to ensure proper rolling
    airport_data.sort_index(inplace=True)

    # Rolling 24h average
    airport_data['rolling_24h'] = (
        airport_data['queue']
            .rolling('24h', min_periods=1)
            .mean()
    )
    # Rolling 7d average
    airport_data['rolling_7d'] = (
        airport_data['queue']
            .rolling('7d', min_periods=1)
            .mean()
    )

    # Assign back to main DF
    df.loc[airport_data.index, f'{airport_code}_rolling_24h'] = airport_data['rolling_24h']
    df.loc[airport_data.index, f'{airport_code}_rolling_7d']  = airport_data['rolling_7d']

# ------------------------------------------------------------------------------
# Data Fetching with Caching
# ------------------------------------------------------------------------------
@cache.memoize()
def get_data():
    """
    Fetch large dataset from Supabase (or API) and preprocess.
    Localize from UTC to Europe/Copenhagen time zone.
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
        return pd.DataFrame()  # Return empty DF on error

    # Convert timestamps from UTC to Europe/Copenhagen
    # 1) parse as UTC
    StartTime = pd.to_datetime(dataframe["timestamp"], utc=True)
    # 2) convert to local time zone
    StartTime = StartTime.dt.tz_convert('Europe/Copenhagen')

    dataframe["timestamp"] = StartTime
    
    # Set index
    df = dataframe.set_index("timestamp")

    # -------------------------------------------------------------------------
    # Option 4 fix: Sort by the timestamp index and offset duplicates by 1ns
    # -------------------------------------------------------------------------
    df.sort_index(inplace=True)
    df.index = df.index + pd.to_timedelta(df.groupby(level=0).cumcount(), unit='ns')

    # Basic time features
    df['year'] = df.index.year
    df['hour'] = df.index.hour
    df['day'] = df.index.day
    df['month'] = df.index.month
    df['weekday'] = df.index.weekday

    # One-hot encode airport
    df_airport = pd.get_dummies(df['airport'])
    df = pd.concat([df, df_airport], axis=1).drop(columns=['airport'])

    # Compute rolling features for each known airport
    for airport_code in VALID_AIRPORTS:
        if airport_code in df.columns:
            _compute_rolling_features(df, airport_code)

    # Add holiday feature
    df = add_holiday_feature(df)

    # Remove 'id' if present
    if 'id' in df.columns:
        df.drop(['id'], axis=1, inplace=True)

    # Clip extremely large queue values
    df['queue'] = df['queue'].apply(lambda x: np.nan if x > 1e10 else x)
    df.dropna(subset=['queue'], inplace=True)

    logger.info("Fetched and preprocessed full dataset in %.2f seconds", time.time() - start_time)
    return df

@cache.memoize()
def get_data_light():
    """
    Fetch a smaller dataset. Same transformations as get_data().
    Localize from UTC to Europe/Copenhagen time zone.
    """
    start_time = time.time()
    logger.info("Fetching light dataset...")

    try:
        if UseSupabase == 1:
            dataframe = get_data_via_supabase_client()
        else:
            dataframe = get_data_via_local_database()
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return pd.DataFrame()  # Return empty DF on error

    StartTime = pd.to_datetime(dataframe["timestamp"], utc=True)
    StartTime = StartTime.dt.tz_convert('Europe/Copenhagen')
    dataframe["timestamp"] = StartTime
    
    df = dataframe.set_index("timestamp")

    # -------------------------------------------------------------------------
    # Option 4 fix: Sort by the timestamp index and offset duplicates by 1ns
    # -------------------------------------------------------------------------
    df.sort_index(inplace=True)
    df.index = df.index + pd.to_timedelta(df.groupby(level=0).cumcount(), unit='ns')

    df['year'] = df.index.year
    df['hour'] = df.index.hour
    df['day'] = df.index.day
    df['month'] = df.index.month
    df['weekday'] = df.index.weekday

    df_airport = pd.get_dummies(df['airport'])
    df = pd.concat([df, df_airport], axis=1).drop(columns=['airport'])

    for airport_code in VALID_AIRPORTS:
        if airport_code in df.columns:
            _compute_rolling_features(df, airport_code)

    df = add_holiday_feature(df)

    if 'id' in df.columns:
        df.drop(['id'], axis=1, inplace=True)

    df['queue'] = df['queue'].apply(lambda x: np.nan if x > 1e10 else x)
    df.dropna(subset=['queue'], inplace=True)

    logger.info("Fetched and preprocessed light dataset in %.2f seconds", time.time() - start_time)
    return df

# ------------------------------------------------------------------------------
# Separate Model Training
# ------------------------------------------------------------------------------
def train_models():
    """
    Train a separate XGBoost model for each airport in VALID_AIRPORTS.
    If no data or training fails for an airport, store a FallbackRegressor that returns 5.
    """
    start_time = time.time()
    logger.info("Starting model training for each airport...")

    df = get_data()
    if df.empty:
        logger.warning("No data returned at all. Saving fallback models for all airports.")
        for ap in VALID_AIRPORTS:
            fallback = FallbackRegressor(constant=5)
            joblib.dump(fallback, f'trained_model_{ap}.joblib')
        return

    # Sort rows and columns so XGBoost sees consistent ordering
    df.sort_index(inplace=True)
    df = df.reindex(sorted(df.columns), axis=1)

    for airport_code in VALID_AIRPORTS:
        if airport_code not in df.columns:
            logger.warning(f"No column for {airport_code}. Saving fallback model.")
            fallback = FallbackRegressor()
            joblib.dump(fallback, f'trained_model_{airport_code}.joblib')
            continue

        airport_df = df[df[airport_code] == 1]
        if airport_df.empty:
            logger.warning(f"No data for airport {airport_code}, saving fallback.")
            fallback = FallbackRegressor()
            joblib.dump(fallback, f'trained_model_{airport_code}.joblib')
            continue

        # Ensure rolling columns exist
        rolling_cols = [f'{airport_code}_rolling_24h', f'{airport_code}_rolling_7d']
        for rc in rolling_cols:
            if rc not in airport_df.columns:
                airport_df[rc] = 0.0

        cutoff = int(len(airport_df) * 0.9)
        train_df = airport_df.iloc[:cutoff]
        test_df  = airport_df.iloc[cutoff:]
        # Log how many rows are in the training set vs test set
        logger.info(f"{airport_code} - training set size: {len(train_df)}, test set size: {len(test_df)}")

        if len(train_df) < 10:
            logger.warning(f"Not enough training rows ({len(train_df)}) for {airport_code}, using fallback.")
            fallback = FallbackRegressor()
            joblib.dump(fallback, f'trained_model_{airport_code}.joblib')
            continue

        try:
            X_train = train_df.drop('queue', axis=1)
            y_train = train_df['queue']
            X_test  = test_df.drop('queue', axis=1)
            y_test  = test_df['queue']

            model = xgb.XGBRegressor(
                objective='reg:squarederror',
                random_state=7
            )
            model.fit(X_train, y_train)

            # Evaluate
            if not X_test.empty:
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                logger.info(f"{airport_code} - Holdout set MSE: {mse:.2f}")

            joblib.dump(model, f'trained_model_{airport_code}.joblib')
            logger.info(f"Model for {airport_code} saved.")
        except Exception as e:
            logger.error(f"Error training {airport_code}, using fallback. Error: {e}")
            fallback = FallbackRegressor()
            joblib.dump(fallback, f'trained_model_{airport_code}.joblib')

    logger.info("Finished training separate models in %.2f seconds", time.time() - start_time)

def load_model_for_airport(airport_code: str):
    """
    Load model from 'trained_model_{airport_code}.joblib'.
    If file not found or an error occurs, return FallbackRegressor (always returns 5).
    """
    try:
        model = joblib.load(f"trained_model_{airport_code}.joblib")
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
    Given a single-row DataFrame with columns ['timestamp', 'airport'],
    return a queue length prediction. If no model or data, returns fallback (5).

    Steps:
    1) Convert from string to datetime (UTC), then tz_convert to Europe/Copenhagen.
    2) Add date/time features (year, hour, etc.).
    3) One-hot encode the 'airport' by setting the corresponding column=1.
    4) For rolling features, look up historical data up to that point in get_data_light().
    5) Add 'Holiday' feature for the correct airport.
    6) Drop 'timestamp' & 'airport' columns (they're not numeric).
    7) Sort columns so they match training.
    8) Call model.predict() on the single row.
    """
    airport = timestamp_df["airport"].iloc[0]
    model = load_model_for_airport(airport)

    # 1) Convert timestamp to datetime + localize
    modeldatetime = pd.to_datetime(timestamp_df["timestamp"], utc=True)
    modeldatetime = modeldatetime.dt.tz_convert('Europe/Copenhagen')

    timestamp_df["timestamp"] = modeldatetime
    timestamp_df = timestamp_df.set_index(timestamp_df.timestamp)

    # 2) Basic date/time features
    timestamp_df['year'] = timestamp_df.index.year
    timestamp_df['hour'] = timestamp_df.index.hour
    timestamp_df['day'] = timestamp_df.index.day
    timestamp_df['month'] = timestamp_df.index.month
    timestamp_df['weekday'] = timestamp_df.index.weekday

    # 3) One-hot for all valid airports
    for ap in VALID_AIRPORTS:
        timestamp_df[ap] = 1 if ap == airport else 0

    # 4) Retrieve rolling features from historical data
    df_light = get_data_light()
    if df_light.empty:
        rolling_24h_val = 0.0
        rolling_7d_val  = 0.0
    else:
        cutoff_time = modeldatetime.iloc[0]
        airport_df = df_light[(df_light[airport] == 1) & (df_light.index <= cutoff_time)].copy()
        if airport_df.empty:
            rolling_24h_val = 0.0
            rolling_7d_val  = 0.0
        else:
            airport_df.sort_index(inplace=True)
            airport_df['rolling_24h'] = airport_df['queue'].rolling('24h', min_periods=1).mean()
            airport_df['rolling_7d']  = airport_df['queue'].rolling('7d',  min_periods=1).mean()
            last_row = airport_df.iloc[-1]
            rolling_24h_val = last_row['rolling_24h'] if not pd.isna(last_row['rolling_24h']) else 0.0
            rolling_7d_val  = last_row['rolling_7d']  if not pd.isna(last_row['rolling_7d'])  else 0.0

    timestamp_df[f'{airport}_rolling_24h'] = rolling_24h_val
    timestamp_df[f'{airport}_rolling_7d']  = rolling_7d_val

    # For other airports, set them to 0
    for ap in VALID_AIRPORTS:
        if ap != airport:
            timestamp_df[f'{ap}_rolling_24h'] = 0.0
            timestamp_df[f'{ap}_rolling_7d']  = 0.0

    # 5) Add local holiday feature
    timestamp_df = add_holiday_feature(timestamp_df)

    # 6) Drop 'timestamp' & 'airport' so XGBoost doesn't see non-numeric dtypes
    drop_cols = ['timestamp', 'airport']
    for col in drop_cols:
        if col in timestamp_df.columns:
            timestamp_df.drop(col, axis=1, inplace=True)

    # 7) Sort columns in alphabetical order to match training
    timestamp_df = timestamp_df.reindex(sorted(timestamp_df.columns), axis=1)

    # 8) Predict
    prediction = model.predict(timestamp_df)
    prediction = prediction + 1
    return round(prediction[0])

# ------------------------------------------------------------------------------
# Flask Routes
# ------------------------------------------------------------------------------
@app.route('/predict')
def make_prediction():
    """
    /predict?airport=ARN&timestamp=YYYY-MM-DDTHH:MM (UTC time)
    Returns JSON with 'predicted_queue_length_minutes'
    """
    start_time = time.time()
    logger.info("Received request to predict queue.")

    input_date_str = request.args.get('timestamp')
    airport_code   = request.args.get('airport')

    if not input_date_str and not airport_code:
        return jsonify({
            'error': 'Missing "airport" and "timestamp". Usage: /predict?airport=ARN&timestamp=YYYY-MM-DDTHH:MM'
        }), 400

    if not input_date_str:
        return jsonify({
            'error': 'Missing "timestamp". Usage: /predict?airport=ARN&timestamp=YYYY-MM-DDTHH:MM'
        }), 400

    if not airport_code:
        return jsonify({
            'error': 'Missing "airport". Usage: /predict?airport=ARN&timestamp=YYYY-MM-DDTHH:MM'
        }), 400

    airport_code = airport_code.upper()
    if airport_code not in VALID_AIRPORTS:
        return jsonify({
            'error': f'Invalid airport code "{airport_code}". Valid codes are {",".join(VALID_AIRPORTS)}.'
        }), 400

    try:
        # interpret the user input as UTC date/time
        input_date = pd.to_datetime(input_date_str, utc=True)
    except (ValueError, TypeError):
        return jsonify({'error': 'Invalid "timestamp" format. Expected YYYY-MM-DDTHH:MM (UTC)'}), 400

    input_data = pd.DataFrame({'timestamp': [input_date], 'airport': [airport_code]})
    predicted = predict_queue(input_data)

    response = {'predicted_queue_length_minutes': predicted}
    logger.info("Prediction completed in %.2f seconds", time.time() - start_time)
    return jsonify(response)

# ------------------------------------------------------------------------------
# Crontab Job for Retraining
# ------------------------------------------------------------------------------
@crontab.job(minute=0, hour=0)
def scheduled_training():
    """
    Daily re-training (midnight). Customize as needed.
    """
    logger.info(f'Crontab triggered: retraining models at {datetime.now()}')
    train_models()

# ------------------------------------------------------------------------------
# On Startup: Train once (optional)
# ------------------------------------------------------------------------------
with app.app_context():
    logger.info("Initializing application...")
    # Train all models once on startup
    train_models()

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
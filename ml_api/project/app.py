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
if os.environ.get("CPHAPI_HOST"):
    CPHAPI_HOST = os.environ.get("CPHAPI_HOST")
else:
    CPHAPI_HOST = "https://waitport.com/api/v1/all"

# Denmark holidays for holiday feature
dk_holidays = holidays.Denmark()

# Valid airports for reference
VALID_AIRPORTS = ['AMS', 'ARN', 'CPH', 'DUB', 'DUS', 'FRA', 'IST', 'LHR', 'MUC', 'OSL']

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
    Add a 'Holiday' feature (1/0) indicating if a day is a holiday in Denmark,
    without flattening the index to midnight.
    """
    df['Holiday'] = df.index.to_series().apply(lambda dt: int(dt.date() in dk_holidays))
    return df

def _compute_rolling_averages(df: pd.DataFrame, airport_code: str, now: datetime) -> None:
    """
    For a given airport_code, compute 'yesterday_average_queue' and 'lastweek_average_queue'.
    Assign them as single values for all rows of that airport. If no data, default to 0.
    """
    airport_data = df[df[airport_code] == 1]
    
    # YESTERDAY AVERAGE
    yesterday = now - timedelta(days=1)
    mask_yesterday = (
        (airport_data['year'] == yesterday.year) &
        (airport_data['month'] == yesterday.month) &
        (airport_data['day'] == yesterday.day) &
        (airport_data['hour'] >= 7) & (airport_data['hour'] <= 22)
    )
    yest_avg = airport_data[mask_yesterday]['queue'].mean()
    if pd.isna(yest_avg):
        yest_avg = 0  # fallback
    
    df.loc[df[airport_code] == 1, 'yesterday_average_queue'] = yest_avg

    # LAST WEEK AVERAGE (rolling 7 days, hours 7-22)
    week_ago = now - pd.Timedelta(days=7)
    mask_week = (
        (df.index >= week_ago) & 
        (df.index <= now) &
        (df['hour'] >= 7) & (df['hour'] <= 22) &
        (df[airport_code] == 1)
    )
    # Attempt a rolling(24) on queue
    rolling_series = df[mask_week]['queue'].reset_index(drop=True).rolling(24).mean()
    if len(rolling_series) == 0:
        lastweek_avg = 0
    else:
        lastweek_avg = rolling_series.iloc[-1]
        if pd.isna(lastweek_avg):
            lastweek_avg = 0
    
    df.loc[df[airport_code] == 1, 'lastweek_average_queue'] = lastweek_avg

def fetch_data(url: str) -> pd.DataFrame:
    """
    Common function for reading JSON from a given URL into a DataFrame.
    """
    return pd.read_json(url)

# ------------------------------------------------------------------------------
# Data Fetching with Caching
# ------------------------------------------------------------------------------
@cache.memoize()
def get_data():
    """
    Fetch large dataset from Supabase (or API) and preprocess.
    """
    start_time = time.time()
    logger.info("Fetching full dataset...")

    now = datetime.now()

    try:
        dataframe = get_data_via_supabase_client()
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return pd.DataFrame()  # Return empty DF on error

    # Convert timestamp to naive local (offset +2 hours)
    StartTime = pd.to_datetime(dataframe["timestamp"])
    StartTime = StartTime.apply(lambda t: t.replace(tzinfo=None))
    StartTime = StartTime + pd.DateOffset(hours=2)
    dataframe["timestamp"] = StartTime
    
    # Set index
    df = dataframe.set_index("timestamp")
    
    # Add date/time features
    df['year'] = df.index.year
    df['hour'] = df.index.hour
    df['day'] = df.index.day
    df['month'] = df.index.month
    df['weekday'] = df.index.weekday

    # One-hot encode airport
    df_airport = pd.get_dummies(df['airport'])
    df = pd.concat([df, df_airport], axis=1).drop(columns=['airport'])

    # Compute rolling averages for each known airport
    for airport_code in VALID_AIRPORTS:
        if airport_code in df.columns:
            _compute_rolling_averages(df, airport_code, now)

    # Holiday feature
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
    """
    start_time = time.time()
    logger.info("Fetching light dataset...")

    now = datetime.now()
    try:
        dataframe = get_data_via_supabase_client()
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return pd.DataFrame()  # Return empty DF on error

    StartTime = pd.to_datetime(dataframe["timestamp"])
    StartTime = StartTime.apply(lambda t: t.replace(tzinfo=None))
    StartTime = StartTime + pd.DateOffset(hours=2)
    dataframe["timestamp"] = StartTime
    
    df = dataframe.set_index("timestamp")
    
    df['year'] = df.index.year
    df['hour'] = df.index.hour
    df['day'] = df.index.day
    df['month'] = df.index.month
    df['weekday'] = df.index.weekday

    df_airport = pd.get_dummies(df['airport'])
    df = pd.concat([df, df_airport], axis=1).drop(columns=['airport'])

    for airport_code in VALID_AIRPORTS:
        if airport_code in df.columns:
            _compute_rolling_averages(df, airport_code, now)

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

    df = get_data()  # full dataset
    if df.empty:
        logger.warning("No data returned at all. Saving fallback models for all airports.")
        for ap in VALID_AIRPORTS:
            fallback = FallbackRegressor(constant=5)
            joblib.dump(fallback, f'trained_model_{ap}.joblib')
        return

    # Sort by index (time)
    df.sort_index(inplace=True)

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

        cutoff = int(len(airport_df) * 0.8)
        train_df = airport_df.iloc[:cutoff]
        test_df  = airport_df.iloc[cutoff:]

        if len(train_df) < 10:  # or any minimum you want
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

            # Evaluate quickly
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
    """
    airport = timestamp_df["airport"].iloc[0]
    model = load_model_for_airport(airport)

    # Convert timestamp
    modeldatetime = pd.to_datetime(timestamp_df["timestamp"])
    timestamp_df["timestamp"] = modeldatetime
    timestamp_df = timestamp_df.set_index(timestamp_df.timestamp)

    # Add date/time features
    timestamp_df['year'] = timestamp_df.index.year
    timestamp_df['hour'] = timestamp_df.index.hour
    timestamp_df['day'] = timestamp_df.index.day
    timestamp_df['month'] = timestamp_df.index.month
    timestamp_df['weekday'] = timestamp_df.index.weekday

    # One-hot for all valid airports
    for ap in VALID_AIRPORTS:
        timestamp_df[ap] = 1 if ap == airport else 0

    # Get yesterday/lastweek averages from the light DF
    df_light = get_data_light()
    if df_light.empty:
        # Fallback to 0
        timestamp_df['yesterday_average_queue'] = 0
        timestamp_df['lastweek_average_queue'] = 0
    else:
        # Sort for newest row
        df_light["date"] = pd.to_datetime(df_light[["year", "month", "day", "hour"]])
        df_sorted = df_light.sort_values("date", ascending=False)
        # Filter for this airport
        newest_rows = df_sorted[df_sorted[airport] == 1]
        if newest_rows.empty:
            yest_avg = 0
            lastweek_avg = 0
        else:
            newest_row = newest_rows.iloc[0]
            yest_avg = newest_row.get("yesterday_average_queue", 0)
            lastweek_avg = newest_row.get("lastweek_average_queue", 0)
            if pd.isna(yest_avg):
                yest_avg = 0
            if pd.isna(lastweek_avg):
                lastweek_avg = 0

        timestamp_df['yesterday_average_queue'] = yest_avg
        timestamp_df['lastweek_average_queue'] = lastweek_avg

    timestamp_df = add_holiday_feature(timestamp_df)

    # Clean up columns not used by model
    drop_cols = ['timestamp', 'airport', 'date']
    for c in drop_cols:
        if c in timestamp_df.columns:
            timestamp_df.drop(c, axis=1, inplace=True)

    # Predict
    prediction = model.predict(timestamp_df)
    # If you strictly want the fallback=5 as final, consider removing this +1 offset
    prediction = prediction + 1
    return round(prediction[0])

# ------------------------------------------------------------------------------
# Flask Routes
# ------------------------------------------------------------------------------
@app.route('/predict')
def make_prediction():
    """
    /predict?airport=ARN&timestamp=YYYY-MM-DDTHH:MM
    Returns JSON with 'predicted_queue_length_minutes'
    """
    start_time = time.time()
    logger.info("Received request to predict queue.")

    input_date_str = request.args.get('timestamp')
    airport_code   = request.args.get('airport')

    if not input_date_str and not airport_code:
        return jsonify({'error': 'Missing "airport" and "timestamp". Usage: /predict?airport=ARN&timestamp=YYYY-MM-DDTHH:MM'}), 400

    if not input_date_str:
        return jsonify({'error': 'Missing "timestamp". Usage: /predict?airport=ARN&timestamp=YYYY-MM-DDTHH:MM'}), 400

    if not airport_code:
        return jsonify({'error': 'Missing "airport". Usage: /predict?airport=ARN&timestamp=YYYY-MM-DDTHH:MM'}), 400

    airport_code = airport_code.upper()
    if airport_code not in VALID_AIRPORTS:
        return jsonify({'error': f'Invalid airport code "{airport_code}". Valid codes are {",".join(VALID_AIRPORTS)}.'}), 400

    try:
        input_date = pd.to_datetime(input_date_str)
    except (ValueError, TypeError):
        return jsonify({'error': 'Invalid "timestamp" format. Expected YYYY-MM-DDTHH:MM'}), 400

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
    # In production, you'd typically use gunicorn or uwsgi, e.g.:
    #   gunicorn -w 2 -b 0.0.0.0:5000 app2:app
    app.run(debug=False, host='0.0.0.0')
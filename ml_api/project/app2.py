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

def get_data_via_supabase_client(limit=100000) -> pd.DataFrame:
    supabase_url = os.environ.get("SUPABASE_URL", "")
    supabase_key = os.environ.get("SUPABASE_KEY", "")
    if not supabase_url or not supabase_key:
        raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY environment variables.")
    
    supabase: Client = create_client(supabase_url, supabase_key)
    
    # For example, query the 'all' table and exclude 'BER' in 'airport'
    # Adjust table name, column names, or filters as needed
    response = (
        supabase
        .table("waitingtime")
        .select("id,queue,timestamp,airport")
        .neq("airport", "BER")
        .order("id", desc=True)
        .limit(limit)
        .execute()
    )
    # response.data is a list of dicts
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
    Add a 'Holiday' feature (1/0) indicating if a day is a holiday in Denmark.
    """
    # Convert the DataFrame index to daily resolution to match with holidays
    df.index = pd.to_datetime(df.index.date)
    df['Holiday'] = df.index.map(lambda x: int(x in dk_holidays))
    return df

def _compute_rolling_averages(df: pd.DataFrame, airport_code: str, now: datetime) -> None:
    """
    For a given airport_code, compute the 'yesterday_average_queue' and 
    'lastweek_average_queue' for the entire df in-place. If either is NaN, default to 0. 
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
        yest_avg = 0  # default to 0 if no data
    df.loc[df[airport_code] == 1, 'yesterday_average_queue'] = yest_avg

    # LAST WEEK AVERAGE
    # We'll do a 7-day window from 'now' minus 7 days, only for hours 7-22
    week_ago = now - pd.Timedelta(days=7)
    mask_week = (
        (df.index >= week_ago) & 
        (df.index <= now) &
        (df['hour'] >= 7) & (df['hour'] <= 22) &
        (df[airport_code] == 1)
    )
    # Compute rolling(24) on queue if possible
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
    Common function for reading data from the API into a DataFrame.
    """
    return pd.read_json(url)

# ------------------------------------------------------------------------------
# Data Fetching with Caching
# ------------------------------------------------------------------------------
@cache.memoize()
def get_data():
    """
    Fetch large dataset from API and preprocess.
    """
    start_time = time.time()
    logger.info("Fetching full dataset...")

    now = datetime.now()
    #newmodeldata_url = f"{CPHAPI_HOST}?order=id.desc&limit=100000&select=id,queue,timestamp,airport&airport=not.eq.BER"
    try:
       # dataframe = fetch_data(newmodeldata_url)
        dataframe = get_data_via_supabase_client(limit=100000)
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return pd.DataFrame()  # Return empty df on error

    # Convert timestamp to local naive datetime, then add 2 hours offset
    StartTime = pd.to_datetime(dataframe["timestamp"])
    StartTime = StartTime.apply(lambda t: t.replace(tzinfo=None))
    StartTime = StartTime + pd.DateOffset(hours=2)
    dataframe["timestamp"] = StartTime
    
    # Set index, drop raw timestamp column
    df = dataframe.set_index(dataframe.timestamp)
    df.drop('timestamp', axis=1, inplace=True)

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

    # Remove 'id' since it's not useful
    if 'id' in df.columns:
        df.drop(['id'], axis=1, inplace=True)
    
    # Clip or remove extremely large values in queue
    # E.g., set to NaN if above 1e10, then drop them
    df['queue'] = df['queue'].apply(lambda x: np.nan if x > 1e10 else x)
    df.dropna(subset=['queue'], inplace=True)

    logger.info("Fetched and preprocessed full dataset in %.2f seconds", time.time() - start_time)
    return df


@cache.memoize()
def get_data_light():
    """
    Fetch smaller (light) dataset from API for quick usage in predictions.
    """
    start_time = time.time()
    logger.info("Fetching light dataset...")

    now = datetime.now()
    newmodeldata_url = f"{CPHAPI_HOST}?select=id,queue,timestamp,airport&airport=not.eq.BER&order=id.desc&limit=20000"
    try:
        dataframe = get_data_via_supabase_client(limit=100000)
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return pd.DataFrame()  # Return empty df on error

    # Convert timestamp to local naive datetime, then add 2 hours offset
    StartTime = pd.to_datetime(dataframe["timestamp"])
    StartTime = StartTime.apply(lambda t: t.replace(tzinfo=None))
    StartTime = StartTime + pd.DateOffset(hours=2)
    dataframe["timestamp"] = StartTime
    
    # Set index, drop raw timestamp column
    df = dataframe.set_index(dataframe.timestamp)
    df.drop('timestamp', axis=1, inplace=True)

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

    # Remove 'id' since it's not useful
    if 'id' in df.columns:
        df.drop(['id'], axis=1, inplace=True)

    # Clip or remove extremely large values in queue
    df['queue'] = df['queue'].apply(lambda x: np.nan if x > 1e10 else x)
    df.dropna(subset=['queue'], inplace=True)

    logger.info("Fetched and preprocessed light dataset in %.2f seconds", time.time() - start_time)
    return df

# ------------------------------------------------------------------------------
# Model Training & Loading
# ------------------------------------------------------------------------------

def train_model():
    """
    Train an XGBoost model on the full dataset using a time-series split.
    Saves the model to 'trained_model.joblib'.
    """
    start_time = time.time()
    logger.info("Starting model training...")

    df = get_data()
    if df.empty:
        logger.error("No data returned for training. Training aborted.")
        return None

    # Ensure df is sorted by index (time)
    df_sorted = df.sort_index()
    # Time-based split: first 80% as train, last 20% as test
    cutoff = int(len(df_sorted) * 0.8)
    train_df = df_sorted.iloc[:cutoff]
    test_df  = df_sorted.iloc[cutoff:]

    X_train = train_df.drop('queue', axis=1)
    y_train = train_df['queue']
    X_test  = test_df.drop('queue', axis=1)
    y_test  = test_df['queue']

    # Define the model
    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=7)

    # Fit the model
    model.fit(X_train, y_train)

    # Evaluate quickly on holdout set
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    logger.info(f"Holdout set MSE: {mse:.2f}")

    # Save trained model
    joblib.dump(model, 'trained_model.joblib')
    logger.info("Model trained and saved to 'trained_model.joblib' in %.2f seconds", time.time() - start_time)
    return model

def load_model():
    """
    Load model from joblib file. Return None if file not found or load error.
    """
    try:
        model = joblib.load('trained_model.joblib')
        logger.info("Loaded model from 'trained_model.joblib'.")
        return model
    except FileNotFoundError:
        logger.error("Model file 'trained_model.joblib' not found.")
        return None
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

# ------------------------------------------------------------------------------
# Prediction Logic
# ------------------------------------------------------------------------------

def predict_queue(timestamp_df: pd.DataFrame) -> float:
    """
    Given a DataFrame with columns ['timestamp', 'airport'] (single row),
    generate a queue length prediction using the trained XGBoost model.
    """
    # Load or reload the model
    model = load_model()
    if model is None:
        # Indicate to caller that model is not available
        return None

    # Extract row data
    airport = timestamp_df["airport"].iloc[0]
    modeldatetime = pd.to_datetime(timestamp_df["timestamp"])
    timestamp_df["timestamp"] = modeldatetime
    timestamp_df = timestamp_df.set_index(timestamp_df.timestamp)

    # Add time features
    timestamp_df['year'] = timestamp_df.index.year
    timestamp_df['hour'] = timestamp_df.index.hour
    timestamp_df['day'] = timestamp_df.index.day
    timestamp_df['month'] = timestamp_df.index.month
    timestamp_df['weekday'] = timestamp_df.index.weekday

    # One-hot columns for airport
    for ap in VALID_AIRPORTS:
        timestamp_df[ap] = 1 if ap == airport else 0

    # For yesterday/lastweek columns, we take them from the newest row in the light DF
    df_light = get_data_light()
    if df_light.empty:
        # If we cannot retrieve any data, set them to 0 as fallback
        timestamp_df['yesterday_average_queue'] = 0
        timestamp_df['lastweek_average_queue'] = 0
    else:
        # Sort by date
        df_light["date"] = pd.to_datetime(df_light[["year", "month", "day", "hour"]])
        df_sorted = df_light.sort_values("date", ascending=False)

        # Attempt to find the newest row for our target airport
        # If none found, default to 0
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

    # Add holiday feature
    timestamp_df = add_holiday_feature(timestamp_df)

    # Drop extraneous columns for model input
    if 'timestamp' in timestamp_df.columns:
        timestamp_df.drop('timestamp', axis=1, inplace=True)
    if 'airport' in timestamp_df.columns:
        timestamp_df.drop('airport', axis=1, inplace=True)
    if 'date' in timestamp_df.columns:
        timestamp_df.drop('date', axis=1, inplace=True)

    # Predict
    prediction = model.predict(timestamp_df)
    # Possibly offset the prediction (why +1? domain knowledge or correction)
    # Then round
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

    # Validate timestamp
    try:
        input_date = pd.to_datetime(input_date_str)
    except (ValueError, TypeError):
        return jsonify({'error': 'Invalid "timestamp" format. Expected YYYY-MM-DDTHH:MM'}), 400

    # Prepare input
    input_data = pd.DataFrame({'timestamp': [input_date], 'airport': [airport_code]})

    # Call prediction
    predicted = predict_queue(input_data)
    if predicted is None:
        # Means model wasn't loaded properly
        return jsonify({'error': 'Model not available. Please try again later.'}), 500

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
    logger.info(f'Crontab triggered: retraining model at {datetime.now()}')
    train_model()

# ------------------------------------------------------------------------------
# On Startup: Train once (optional) or just load
# ------------------------------------------------------------------------------
with app.app_context():
    logger.info("Initializing application...")
    # Optional: Train the model once on startup
    # If you prefer not to train on every startup, comment this out:
    train_model()

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    # In production, use a WSGI server like gunicorn or uwsgi, e.g.:
    #   gunicorn -w 2 -b 0.0.0.0:5000 your_script:app
    app.run(debug=False, host='0.0.0.0')
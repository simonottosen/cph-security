from flask import Flask, request, jsonify
import pandas as pd
import urllib.request
import json
import holidays
from lightgbm import LGBMRegressor
from datetime import datetime, timedelta
from flask_caching import Cache
from flask_crontab import Crontab
import os
import time
import joblib
import numpy as np

if os.environ.get("CPHAPI_HOST"):
    CPHAPI_HOST = os.environ.get("CPHAPI_HOST")
else:
    CPHAPI_HOST = "https://waitport.com/api/v1/all"



# Create Flask app instance
app = Flask(__name__)





cache = Cache(app, config={'CACHE_TYPE': 'simple'})

# Adding Denmark holidays to be used in add_holiday_features function
dk_holidays = holidays.Denmark()


def add_holiday_feature(df):
    '''This function gets a DataFrame and returns one with an additional column with 1/0 
    value indicating whether the day is a holiday in Denmark'''

    # Turning date format into datetime64 for easier indexing and formatting
    df.index = pd.to_datetime(df.index.date)

    # Adding a Holiday feature column indicating if the date is a holiday in Denmark
    df['Holiday'] = df.index.map(lambda x: int(x in dk_holidays))

    return df


@cache.memoize()
def get_data():
    start_time_load_data = time.time()
    '''
    This function fetches the external dataset from API endpoint.
    '''
    now = datetime.now()
    newmodeldata_url = (str(CPHAPI_HOST) + str("?select=id,queue,timestamp,airport"))
    dataframe = pd.read_json(newmodeldata_url)
    print("Loaded data successfully in %.2f seconds " % (time.time() - start_time_load_data))
    StartTime = dataframe["timestamp"]
    StartTime = pd.to_datetime(StartTime)
    StartTime = StartTime.apply(lambda t: t.replace(tzinfo=None))
    StartTime = StartTime + pd.DateOffset(hours=2)
    dataframe["timestamp"] = StartTime
    df = dataframe.set_index(dataframe.timestamp)
    df.drop('timestamp', axis=1, inplace=True)
    df['year'] = df.index.year
    df['hour'] = df.index.hour
    df['day'] = df.index.day
    df['month'] = df.index.month
    df['weekday'] = df.index.weekday
    df_airport = pd.get_dummies(df['airport'])
    df_test = pd.concat([df, df_airport], axis=1)
    df = df_test
    df = df.drop(columns=['airport'])
    for airport_code in ['AMS', 'ARN', 'BER', 'CPH', 'DUB', 'DUS', 'FRA', 'OSL']:
        airport_data = df[df[airport_code] == 1]
        
        yesterday = now - timedelta(days=1)
        yesterday_data = airport_data[(airport_data['year'] == yesterday.year) &
                                      (airport_data['month'] == yesterday.month) &
                                      (airport_data['day'] == yesterday.day)]
        
        yesterday_data_between_7_and_22 = yesterday_data[(yesterday_data['hour'] >= 7) & 
                                                         (yesterday_data['hour'] <= 22)]
        yesterday_average_queue = yesterday_data_between_7_and_22['queue'].mean()
        
        df.loc[df[airport_code] == 1, 'yesterday_average_queue'] = yesterday_average_queue
        
        now = pd.Timestamp.now().floor('H')
        week_ago = now - pd.Timedelta(days=7)
        mask = (df.index >= week_ago) & (df.index <= now)
        mask &= (df.index.hour >= 7) & (df.index.hour <= 22)
        mask &= (df[airport_code] == 1)
        lastweek_average_queue = df[mask]['queue'].reset_index(drop=True).rolling(24).mean().iloc[-1]
        df.loc[df[airport_code] == 1, 'lastweek_average_queue'] = lastweek_average_queue


        
    # Adding Holiday features to dataframe
    df = add_holiday_feature(df)

    df.drop(['id'], axis=1, inplace=True)
    print("Returned data successfully in %.2f seconds " % (time.time() - start_time_load_data))
    return df

@cache.memoize()
def get_data_light():
    start_time_load_data = time.time()
    '''
    This function fetches the external dataset from API endpoint.
    '''
    now = datetime.now()
    newmodeldata_url = (str(CPHAPI_HOST) + str("?select=id,queue,timestamp,airport&order=id.desc&limit=20000"))
    dataframe = pd.read_json(newmodeldata_url)
    print("Loaded light dataset successfully in %.2f seconds " % (time.time() - start_time_load_data))
    StartTime = dataframe["timestamp"]
    StartTime = pd.to_datetime(StartTime)
    StartTime = StartTime.apply(lambda t: t.replace(tzinfo=None))
    StartTime = StartTime + pd.DateOffset(hours=2)
    dataframe["timestamp"] = StartTime
    df = dataframe.set_index(dataframe.timestamp)
    df.drop('timestamp', axis=1, inplace=True)
    df['year'] = df.index.year
    df['hour'] = df.index.hour
    df['day'] = df.index.day
    df['month'] = df.index.month
    df['weekday'] = df.index.weekday
    df_airport = pd.get_dummies(df['airport'])
    df_test = pd.concat([df, df_airport], axis=1)
    df = df_test
    df = df.drop(columns=['airport'])
    for airport_code in ['AMS', 'ARN', 'BER', 'CPH', 'DUB', 'DUS', 'FRA', 'OSL']:
        airport_data = df[df[airport_code] == 1]
        
        yesterday = now - timedelta(days=1)
        yesterday_data = airport_data[(airport_data['year'] == yesterday.year) &
                                      (airport_data['month'] == yesterday.month) &
                                      (airport_data['day'] == yesterday.day)]
        
        yesterday_data_between_7_and_22 = yesterday_data[(yesterday_data['hour'] >= 7) & 
                                                         (yesterday_data['hour'] <= 22)]
        yesterday_average_queue = yesterday_data_between_7_and_22['queue'].mean()
        
        df.loc[df[airport_code] == 1, 'yesterday_average_queue'] = yesterday_average_queue
        
        now = pd.Timestamp.now().floor('H')
        week_ago = now - pd.Timedelta(days=7)
        mask = (df.index >= week_ago) & (df.index <= now)
        mask &= (df.index.hour >= 7) & (df.index.hour <= 22)
        mask &= (df[airport_code] == 1)
        lastweek_average_queue = df[mask]['queue'].reset_index(drop=True).rolling(24).mean().iloc[-1]
        df.loc[df[airport_code] == 1, 'lastweek_average_queue'] = lastweek_average_queue


        
    # Adding Holiday features to dataframe
    df = add_holiday_feature(df)

    df.drop(['id'], axis=1, inplace=True)
    print("Returned light dataset successfully in %.2f seconds " % (time.time() - start_time_load_data))
    return df


#@cache.memoize()
def train_model():
    start_time_train_model = time.time()
    print("Started model training")
    # Fetching external dataset from API endpoint to train the model
    df = get_data()

    X = df.drop('queue', axis=1)
    y = df['queue']
    X_train = X
    y_train = y
    model = LGBMRegressor(random_state=42)  # Using LightGBM model to train on data
    model.fit(X_train, y_train)
    print("Trained model succesfully in %.2f seconds " % (time.time() - start_time_train_model))
    
    
    # Save trained model to disk
    joblib.dump(model, 'trained_model.joblib')
    print("Saved trained model to disk")
    
    return model



def predict_queue(timestamp):
    '''
    This function takes input timestamp and predicts the queue length based on a pre-trained LightGBM model.
    '''
    now = datetime.now()
    # Manipulating input data to get features out of it
    #print(timestamp)
    airport = timestamp["airport"].iloc[0]
    modeldatetime = timestamp["timestamp"]
    modeldatetime = pd.to_datetime(modeldatetime)
    timestamp["timestamp"] = modeldatetime
    timestamp = timestamp.set_index(timestamp.timestamp)
    timestamp['year'] = timestamp.index.year
    timestamp['hour'] = timestamp.index.hour
    timestamp['day'] = timestamp.index.day
    timestamp['month'] = timestamp.index.month
    timestamp['weekday'] = timestamp.index.weekday

    airport_dict = {
        "ARN": [1, 0, 0, 0, 0, 0, 0, 0],
        "BER": [0, 1, 0, 0, 0, 0, 0, 0],
        "CPH": [0, 0, 1, 0, 0, 0, 0, 0],
        "DUS": [0, 0, 0, 1, 0, 0, 0, 0],
        "FRA": [0, 0, 0, 0, 1, 0, 0, 0],
        "OSL": [0, 0, 0, 0, 0, 1, 0, 0],
        "AMS": [0, 0, 0, 0, 0, 0, 1, 0],
        "DUB": [0, 0, 0, 0, 0, 0, 0, 1]
    }

    if airport in airport_dict:
        timestamp[['ARN', 'BER', 'CPH', 'DUS', 'FRA', 'OSL', 'AMS', 'DUB']] = airport_dict[airport]
    
    df = get_data_light()    
    # Create a new column "date" by concatenating year, month, day, and hour
    df["date"] = pd.to_datetime(df[["year", "month", "day", "hour"]])

    # Sort the DataFrame by the "date" column in descending order
    df_sorted = df.sort_values("date", ascending=False)

    # Find the first row where CPH is True
    newest_row = df_sorted[df_sorted[airport] == True].iloc[0]

    timestamp['yesterday_average_queue'] = newest_row["yesterday_average_queue"]
    timestamp['lastweek_average_queue'] = newest_row["lastweek_average_queue"]
    
    # Apply add_holiday_feature to add a column indicating whether the time is a holiday or not.
    
    timestamp = add_holiday_feature(timestamp)
    timestamp.drop('timestamp', axis=1, inplace=True)
    timestamp = timestamp.drop(columns=['airport'])
    model = load_model()
    predict = model.predict(timestamp)
    predict = predict * 1.33
    return round(predict[0])


@app.route('/predict')
def make_prediction():
    start_time_predict = time.time()
    '''
    This route provides interface to take timestamp as parameter and invoke predict_queue method
    '''
    print("Received request to predict queue")
    input_date_str = request.args.get('timestamp')
    airport_code = request.args.get('airport')
    valid_airports = ['ARN', 'BER', 'CPH', 'DUS', 'FRA', 'OSL', 'AMS', 'DUB']
    
    if not input_date_str and not airport_code:
        return jsonify({'error': 'Missing "airport" and "timestamp" parameters. Usage: /predict?airport=ARN&timestamp=YYYY-MM-DDTHH:MM'}), 400

    if not input_date_str:
        return jsonify({'error': 'Missing "timestamp" parameter. Usage: /predict?airport=ARN&timestamp=YYYY-MM-DDTHH:MM'}), 400

    if not airport_code:
        return jsonify({'error': 'Missing "airport" parameter. Usage: /predict?airport=ARN&timestamp=YYYY-MM-DDTHH:MM'}), 400

    if airport_code.upper() not in valid_airports:
        return jsonify({'error': f'Invalid airport code "{airport_code}". Valid airport codes are {",".join(valid_airports)}.'}), 400

    try:
        input_date = pd.to_datetime(input_date_str)
    except (ValueError, TypeError):
        return jsonify({'error': 'Invalid "timestamp" parameter format. Required format is YYYY-MM-DDTHH:MM. Usage: /predict?airport=ARN&timestamp=YYYY-MM-DDTHH:MM'}), 400
    airport_code = airport_code.upper()
    input_data = pd.DataFrame({'timestamp': [input_date], 'airport': [airport_code]})
    output = {'predicted_queue_length_minutes': predict_queue(input_data)}
    print("Predicted successfully in %.2f seconds " % (time.time() - start_time_predict))
    return jsonify(output)


def load_model():
    try:
        # Load trained model from disk
        model = joblib.load('trained_model.joblib')
        print("Loaded trained model from disk")
    except FileNotFoundError:
        print("Error: Saved model file not found")
        model = None
    except Exception as e:
        print(f"Error loading saved model file: {e}")
        model = None
    return model

with app.app_context():
    print("Loading Waitport application")
    train_model()


crontab = Crontab(app)
@crontab.job(minute=0, hour=0)
def train():
    global model
    print(f'Training model due to crontab... ({datetime.datetime.now()})')
    model = train_model()

# Main section to be executed after importing module.
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')

#Example usage
#$ curl http://localhost:5000/predict?timestamp=2023-03-07T15:34:56
#{"predicted_queue_length_minutes": 25}

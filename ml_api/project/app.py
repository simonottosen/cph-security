from flask import Flask, request, jsonify
import pandas as pd
import urllib.request
import json
import holidays
from lightgbm import LGBMRegressor
from datetime import datetime
from flask_caching import Cache
from flask_crontab import Crontab
import os

if os.environ.get("CPHAPI_HOST"):
    CPHAPI_HOST = os.environ.get("CPHAPI_HOST")
else:
    CPHAPI_HOST = "http://cphapi.simonottosen.dk"



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
    '''
    This function fetches the external dataset from API endpoint.
    '''
    newmodeldata_url = (str(CPHAPI_HOST) + str("/waitingtime?select=id,queue,timestamp,airport"))
    data = urllib.request.urlopen(newmodeldata_url).read()
    output = json.loads(data)
    dataframe = pd.DataFrame(output)
    StartTime = dataframe["timestamp"]
    StartTime = pd.to_datetime(StartTime)
    StartTime = StartTime.apply(lambda t: t.replace(tzinfo=None))
    StartTime = StartTime + pd.DateOffset(hours=2)
    dataframe["timestamp"] = StartTime
    df = dataframe.set_index(dataframe.timestamp)
    df.drop('timestamp', axis=1, inplace=True)
    df['hour'] = df.index.hour
    df['day'] = df.index.day
    df['month'] = df.index.month
    df['weekday'] = df.index.weekday
    df_airport = pd.get_dummies(df['airport'])
    df_test = pd.concat([df, df_airport], axis=1)
    df = df_test
    df = df.drop(columns=['airport'])

    # Adding Holiday features to dataframe
    df = add_holiday_feature(df)

    df.drop(['id'], axis=1, inplace=True)

    return df

@cache.memoize()
def train_model():

    # Fetching external dataset from API endpoint to train the model
    df = get_data()

    X = df.drop('queue', axis=1)
    y = df['queue']
    X_train = X.iloc[:]
    y_train = y.iloc[:]
    model = LGBMRegressor(random_state=42)  # Using LightGBM model to train on data
    model.fit(X_train, y_train)
    return model


def predict_queue(timestamp):
    '''
    This function takes input timestamp and predicts the queue length based on a pre-trained LightGBM model.
    '''

    # Manipulating input data to get features out of it
    print(timestamp)
    airport = timestamp["airport"].iloc[0]
    modeldatetime = timestamp["timestamp"]
    modeldatetime = pd.to_datetime(modeldatetime)
    timestamp["timestamp"] = modeldatetime
    timestamp = timestamp.set_index(timestamp.timestamp)
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

    # Apply add_holiday_feature to add a column indicating whether the time is a holiday or not.
    timestamp = add_holiday_feature(timestamp)

    timestamp.drop('timestamp', axis=1, inplace=True)
    timestamp = timestamp.drop(columns=['airport'])
    predict = model.predict(timestamp)
    predict = predict * 1.33
    return round(predict[0])


@app.route('/predict')
def make_prediction():
    '''
    This route provides interface to take timestamp as parameter and invoke predict_queue method
    '''
    input_date_str = request.args.get('timestamp')
    airport_code = request.args.get('airport')
    airport_code = airport_code.upper()
    valid_airports = ['ARN', 'BER', 'CPH', 'DUS', 'FRA', 'OSL', 'AMS', 'DUB']
    if not input_date_str:
        return jsonify({'error': 'Missing "timestamp" parameter. Usage: /predict?airport=ARN&timestamp=YYYY-MM-DDTHH:MM'}), 400

    if not airport_code:
        return jsonify({'error': 'Missing "airport" parameter. Usage: /predict?airport=ARN&timestamp=YYYY-MM-DDTHH:MM'}), 400

    if airport_code not in valid_airports:
        return jsonify({'error': f'Invalid airport code "{airport_code}". Valid airport codes are {",".join(valid_airports)}.'}), 400

    try:
        input_date = pd.to_datetime(input_date_str)
    except (ValueError, TypeError):
        return jsonify({'error': 'Invalid "timestamp" parameter format. Required format is YYYY-MM-DDTHH:MM. Usage: /predict?airport=ARN&timestamp=YYYY-MM-DDTHH:MM'}), 400

    input_data = pd.DataFrame({'timestamp': [input_date], 'airport': [airport_code]})
    output = {'predicted_queue_length_minutes': predict_queue(input_data)}
    return jsonify(output)

model = None
def load_model():
    global model
    model = train_model()
load_model()

crontab = Crontab(app)
@crontab.job(minute=0, hour=0)
def train():
    print('Training model...')
    train_model()

# Main section to be executed after importing module.
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')

#Example usage
#$ curl http://localhost:5000/predict?timestamp=2023-03-07T15:34:56
#{"predicted_queue_length_minutes": 25}

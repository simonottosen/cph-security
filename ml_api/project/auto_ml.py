import pandas as pd
import urllib.request
import json
import holidays
from lightgbm import LGBMRegressor
from datetime import datetime
from flask_caching import Cache
from flask_crontab import Crontab
import os
import tensorflow as tf
import autokeras as ak
import numpy as np

CPHAPI_HOST = "https://cphapi.simonottosen.dk"



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

def get_data():

    '''
    This function fetches the external dataset from API endpoint.
    '''
    hour = 60*60
    day = 24*hour
    year = (365.2425)*day
    newmodeldata_url = (str(CPHAPI_HOST) + str("/waitingtime?select=id,queue,timestamp&airport=eq.CPH"))
    data = urllib.request.urlopen(newmodeldata_url).read()
    output = json.loads(data)
    dataframe = pd.DataFrame(output)
    StartTime = dataframe["timestamp"]
    StartTime = pd.to_datetime(StartTime)
    StartTime = StartTime.apply(lambda t: t.replace(tzinfo=None))
    StartTime = StartTime + pd.DateOffset(hours=2)
    dataframe["timestamp"] = StartTime
    timestamp_s = StartTime.map(pd.Timestamp.timestamp)
    df = dataframe.set_index(dataframe.timestamp)
    df.drop('timestamp', axis=1, inplace=True)
    df['hour'] = df.index.hour
    df['day'] = df.index.day
    df['month'] = df.index.month
    df['weekday'] = df.index.weekday
    df['Hour sin'] = np.sin(timestamp_s * (2 * np.pi / hour))
    df['Hour cos'] = np.cos(timestamp_s * (2 * np.pi / hour))
    df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
    df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
    df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

    # Adding Holiday features to dataframe
    df = add_holiday_feature(df)

    df.drop(['id'], axis=1, inplace=True)

    return df


df = get_data()
dataset = df 




val_split = int(len(dataset) * 0.7)
data_train = dataset[:val_split]
validation_data = dataset[val_split:]

data_x = data_train[
    [
        "hour",
        "day",
        "month",
        "weekday",
        "Holiday",
    ]
].astype("float64")

data_x_val = validation_data[
    [
        "hour",
        "day",
        "month",
        "weekday",
        "Holiday",
    ]
].astype("float64")

# Data with train data and the unseen data from subsequent time steps.
data_x_test = dataset[
    [
        "hour",
        "day",
        "month",
        "weekday",
        "Holiday",
    ]
].astype("float64")

data_y = data_train["queue"].astype("float64")

data_y_val = validation_data["queue"].astype("float64")



predict_from = 1
predict_until = 10
lookback = 3
clf = ak.TimeseriesForecaster(
    lookback=lookback,
    predict_from=predict_from,
    predict_until=predict_until,
    max_trials=1,
    objective="val_loss",
)
# # Train the TimeSeriesForecaster with train data
# clf.fit(
#     x=data_x,
#     y=data_y,
#     validation_data=(data_x_val, data_y_val),
#     epochs=10,
# )
# Predict with the best model(includes original training data).

#predictions = clf.predict(data_x_test)
#print(predictions.shape)
# # Evaluate the best model with testing data.
#print(clf.evaluate(data_x_val, data_y_val))

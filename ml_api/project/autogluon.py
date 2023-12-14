import pandas as pd
import urllib.request
import json
import holidays
from lightgbm import LGBMRegressor
from datetime import datetime, timedelta
import os
import time
import joblib
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

dk_holidays = holidays.Denmark()
CPHAPI_HOST = "https://waitport.com/api/v1/all?limit=1000"


def add_holiday_feature(df):
    '''This function gets a DataFrame and returns one with an additional column with 1/0 
    value indicating whether the day is a holiday in Denmark'''

    # Turning date format into datetime64 for easier indexing and formatting
    df.index = pd.to_datetime(df.index.date)

    # Adding a Holiday feature column indicating if the date is a holiday in Denmark
    df['Holiday'] = df.index.map(lambda x: int(x in dk_holidays))

    return df



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

    print("Returned data successfully in %.2f seconds " % (time.time() - start_time_load_data))
    return df


def get_data_light():
    start_time_load_data = time.time()
    '''
    This function fetches the external dataset from API endpoint.
    '''
    now = datetime.now()
    newmodeldata_url = (str(CPHAPI_HOST) + str("?select=id,queue,timestamp,airport&order=id.desc&limit=16500"))
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
    yesterday = now - timedelta(days=1)
    yesterday_year = yesterday.year
    yesterday_month = yesterday.month
    yesterday_day = yesterday.day

    now = pd.Timestamp.now().floor('H')
    week_ago = now - pd.Timedelta(days=7)

    # Precompute the mask for the entire DataFrame
    df_mask = (df.index >= week_ago) & (df.index <= now)
    df_mask &= (df.index.hour >= 7) & (df.index.hour <= 22)

    for airport_code in ['AMS', 'ARN', 'BER', 'CPH', 'DUB', 'DUS', 'FRA', 'OSL']:
        airport_mask = df[airport_code] == 1
        print (airport_code)

        # Filter airport data based on airport mask
        airport_data = df[airport_mask]

        # Filter based on the 'yesterday' timestamp
        yesterday_data = airport_data[(airport_data['year'] == yesterday_year) &
                                      (airport_data['month'] == yesterday_month) &
                                      (airport_data['day'] == yesterday_day)]

        # Filter based on time range
        yesterday_data_between_7_and_22 = yesterday_data[(yesterday_data['hour'] >= 7) & 
                                                         (yesterday_data['hour'] <= 22)]

        # Compute yesterday's average queue
        yesterday_average_queue = yesterday_data_between_7_and_22['queue'].mean()

        # Update 'yesterday_average_queue' column
        df.loc[airport_mask, 'yesterday_average_queue'] = yesterday_average_queue

        # Compute last week's average queue
        lastweek_average_queue = df.loc[airport_mask & df_mask, 'queue'].rolling(24).mean().iloc[-1]

        # Update 'lastweek_average_queue' column
        df.loc[airport_mask, 'lastweek_average_queue'] = lastweek_average_queue


        
    # Adding Holiday features to dataframe
    df = add_holiday_feature(df)

    print("Returned light dataset successfully in %.2f seconds " % (time.time() - start_time_load_data))
    return df


df = get_data()

train_data = TimeSeriesDataFrame.from_data_frame(
    df,
    id_column="id",
    timestamp_column="timestamp"
)
train_data.head()


predictor = TimeSeriesPredictor(
    prediction_length=48,
    path="autogluon-predict",
    target="queue",
    eval_metric="MASE",
    ignore_time_index=True,
)

predictor.fit(
    train_data,
    presets="medium_quality",
    time_limit=60,
)

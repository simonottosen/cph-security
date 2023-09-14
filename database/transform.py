import json
from datetime import datetime
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pandas import Series
import os
import urllib.request
from pandas import json_normalize
import xgboost
from autots import AutoTS, load_hourly

data = urllib.request.urlopen("https://waitport.com/api/v1/all?airport=eq.DUB").read()
output = json.loads(data)
dataframe = pd.DataFrame(output)
dataframe['id'] = range(1, len(dataframe) + 1)

frame = dataframe

def align_with_five_minute(frame):
    frame['timestamp'] = pd.to_datetime(frame['timestamp'])
    frame.set_index('timestamp', inplace=True)
    frame.index = frame.index.round('5min')
    frame.reset_index(inplace=True)
    return frame

def fill_blank_values(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    # drop duplicate indices (timestamps)
    df = df.loc[~df.index.duplicated(keep='first')]
    
    # resample to 5 minutes and forward fill missing values
    df_resampled = df.resample('1H').ffill()
    df_resampled['id'] = range(1, len(df_resampled) + 1)
    df_resampled = df_resampled.dropna()
    return df_resampled


frame = dataframe
frame2 = align_with_five_minute(frame)
df = fill_blank_values(frame2)

df = df.drop(['id', 'airport'], axis=1)

long = False

model = AutoTS(
    forecast_length=21,
    frequency='infer',
    prediction_interval=0.9,
    ensemble='auto',
    model_list="fast",  # "superfast", "default", "fast_parallel"
    transformer_list="fast",  # "superfast",
    drop_most_recent=1,
    max_generations=1,
    num_validations=2,
    validation_method="backwards"
)
model = model.fit(
    df,)

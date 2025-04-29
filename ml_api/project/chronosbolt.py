# %%

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import holidays
import datetime
from dateutil.easter import easter

from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

# Load the data
df_raw = pd.read_json('cph.json')
df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])

# Data Wrangling
df_raw = df_raw.set_index('timestamp')

# Convert 'queue' to numeric, coercing errors
df_raw['queue'] = pd.to_numeric(df_raw['queue'], errors='coerce')

# Remove rows with non-numeric 'queue' values
df_raw = df_raw.dropna(subset=['queue'])

# Remove irrelevant columns
df_raw = df_raw.drop(columns=['id', 'airport'])

# Group by timestamp and take the mean of the queue length
df_raw = df_raw.groupby(level=0).mean()

# Resample to 15-minute intervals and forward fill
df_resampled = df_raw.resample('15min', origin='start_day').ffill()

# Reset the index
df_resampled = df_resampled.reset_index()

# Generate a list of holidays (example: Denmark holidays)
dk_holidays = holidays.country_holidays('DK')

# Create 'is_holiday' feature: 1 if the date is a holiday, else 0
df_features = df_resampled.copy()
df_features['hour_of_day'] = df_features['timestamp'].dt.hour
df_features['day_of_week'] = df_features['timestamp'].dt.dayofweek
df_features['lag_1_queue'] = df_features['queue'].shift(1).fillna(0)

df_features['is_holiday'] = df_features['timestamp'].dt.date.apply(lambda date: 1 if date in dk_holidays else 0)

# Additional features: rolling average and time-shifted features
# Rolling 1-hour average of queue (4 intervals of 15min)
df_features['rolling_mean_1h'] = df_features['queue'].rolling(window=4, min_periods=1).mean()
# Queue at same time last week (7 days * 24 hours * 4 intervals)
df_features['queue_last_week'] = df_features['queue'].shift(7 * 24 * 4)
# Month and year features
df_features['month'] = df_features['timestamp'].dt.month
df_features['year'] = df_features['timestamp'].dt.year

# Additional lag features
df_features['lag_2_queue'] = df_features['queue'].shift(2).fillna(0)
df_features['lag_3_queue'] = df_features['queue'].shift(3).fillna(0)
df_features['lag_4_queue'] = df_features['queue'].shift(4).fillna(0)

# Weekend and business day flags
df_features['is_weekend'] = (df_features['timestamp'].dt.dayofweek >= 5).astype(int)
df_features['is_business_day'] = ((df_features['timestamp'].dt.dayofweek < 5) & (df_features['is_holiday'] == 0)).astype(int)

# Rolling standard deviation (1-hour)
df_features['rolling_std_1h'] = df_features['queue'].rolling(window=4, min_periods=1).std()

# Seasonal indicators
df_features['quarter'] = df_features['timestamp'].dt.quarter
df_features['is_winter'] = df_features['month'].isin([12, 1, 2]).astype(int)
df_features['is_spring'] = df_features['month'].isin([3, 4, 5]).astype(int)
df_features['is_summer'] = df_features['month'].isin([6, 7, 8]).astype(int)
df_features['is_autumn'] = df_features['month'].isin([9, 10, 11]).astype(int)

# Holiday proximity features
holiday_dates = sorted(dk_holidays)
def days_to_next_holiday(date):
    future = [h for h in holiday_dates if h >= date]
    return (future[0] - date).days if future else np.nan
def days_since_last_holiday(date):
    past = [h for h in holiday_dates if h <= date]
    return (date - past[-1]).days if past else np.nan
df_features['days_to_next_holiday'] = df_features['timestamp'].dt.date.apply(days_to_next_holiday)
df_features['days_since_last_holiday'] = df_features['timestamp'].dt.date.apply(days_since_last_holiday)

# School holiday periods
def is_school_holiday(date):
    year = date.year
    # Christmas break (Dec 20 of previous year to Jan 3 of current year)
    periods = [
        (datetime.date(year-1, 12, 20), datetime.date(year, 1, 3)),
        # Easter break (Maundy Thursday to Easter Monday)
        (easter(year) - datetime.timedelta(days=3), easter(year) + datetime.timedelta(days=1)),
        # Summer break (July 1 to August 31)
        (datetime.date(year, 7, 1), datetime.date(year, 8, 31)),
        # Autumn break (week 42 Monday to Friday)
        (datetime.date.fromisocalendar(year, 42, 1), datetime.date.fromisocalendar(year, 42, 5)),
        # Winter break (week 7 Monday to Friday)
        (datetime.date.fromisocalendar(year, 7, 1), datetime.date.fromisocalendar(year, 7, 5))
    ]
    return any(start <= date <= end for start, end in periods)
df_features['is_school_holiday'] = df_features['timestamp'].dt.date.apply(lambda d: 1 if is_school_holiday(d) else 0)
# School holiday proximity features
def get_school_periods(year):
    return [
        (datetime.date(year-1, 12, 20), datetime.date(year, 1, 3)),
        (easter(year) - datetime.timedelta(days=3), easter(year) + datetime.timedelta(days=1)),
        (datetime.date(year, 7, 1), datetime.date(year, 8, 31)),
        (datetime.date.fromisocalendar(year, 42, 1), datetime.date.fromisocalendar(year, 42, 5)),
        (datetime.date.fromisocalendar(year, 7, 1), datetime.date.fromisocalendar(year, 7, 5))
    ]

def days_to_next_school_holiday(date):
    periods = get_school_periods(date.year) + get_school_periods(date.year + 1)
    starts = sorted(start for start, _ in periods)
    future = [s for s in starts if s >= date]
    return (future[0] - date).days if future else np.nan

def days_since_last_school_holiday(date):
    periods = get_school_periods(date.year) + get_school_periods(date.year - 1)
    ends = sorted(end for _, end in periods)
    past = [e for e in ends if e <= date]
    return (date - past[-1]).days if past else np.nan

df_features['days_to_next_school_holiday'] = df_features['timestamp'].dt.date.apply(days_to_next_school_holiday)
df_features['days_since_last_school_holiday'] = df_features['timestamp'].dt.date.apply(days_since_last_school_holiday)

# Cyclical time encodings
df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour_of_day'] / 24)
df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour_of_day'] / 24)
df_features['dow_sin']  = np.sin(2 * np.pi * df_features['day_of_week']  / 7)
df_features['dow_cos']  = np.cos(2 * np.pi * df_features['day_of_week']  / 7)
df_features['doy']      = df_features['timestamp'].dt.dayofyear
df_features['doy_sin']  = np.sin(2 * np.pi * df_features['doy'] / 365)
df_features['doy_cos']  = np.cos(2 * np.pi * df_features['doy'] / 365)

# Higher-order difference features
df_features['diff_1_queue'] = df_features['queue'] - df_features['lag_1_queue']
df_features['diff_2_queue'] = df_features['diff_1_queue'].diff().fillna(0)

# Expanded rolling statistics for multiple horizons
for window, label in [(2, '30min'), (4, '1h'), (16, '4h'), (96, '24h')]:
    df_features[f'rolling_mean_{label}'] = df_features['queue'].rolling(window=window, min_periods=1).mean()
    df_features[f'rolling_std_{label}']  = df_features['queue'].rolling(window=window, min_periods=1).std()
    df_features[f'rolling_max_{label}']  = df_features['queue'].rolling(window=window, min_periods=1).max()
    df_features[f'rolling_min_{label}']  = df_features['queue'].rolling(window=window, min_periods=1).min()

# Expanding (cumulative) features
df_features['expanding_mean'] = df_features['queue'].expanding(min_periods=1).mean()
df_features['expanding_std']  = df_features['queue'].expanding(min_periods=1).std().fillna(0)

# Fourier terms for yearly seasonality (first three harmonics)
for k in [1, 2, 3]:
    df_features[f'year_fft{k}_sin'] = np.sin(2 * np.pi * k * df_features['doy'] / 365)
    df_features[f'year_fft{k}_cos'] = np.cos(2 * np.pi * k * df_features['doy'] / 365)

# Preserve a full-feature set for imputations
df_features_full = df_features.copy()
# Remove rows with missing feature values (e.g. from shifts and rolling windows)
df_features = df_features_full.dropna()

# %%

# Model training
# After dropping NaNs, features and target align
X = df_features[['hour_of_day', 'day_of_week', 'lag_1_queue', 
                 'is_holiday', 'rolling_mean_1h', 'queue_last_week', 
                 'month', 'year']]
y = df_features['queue']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Data Preparation for Missing Values (If any)
df_missing_indices = df_resampled[df_resampled['queue'].isnull()].index
if not df_missing_indices.empty:
    X_missing = df_features_full.loc[df_missing_indices, ['hour_of_day', 'day_of_week', 'lag_1_queue', 'is_holiday', 'rolling_mean_1h', 'queue_last_week', 'month', 'year']]
    X_missing.fillna(0, inplace=True)
    y_missing_pred = model.predict(X_missing)
    df_resampled.loc[df_missing_indices, 'queue'] = y_missing_pred

# Create a combined dataframe with missing value imputations and additional features
df_combined = df_features.copy()
# Assign queue values corresponding to df_features rows to avoid length mismatch
df_combined['queue'] = df_resampled.loc[df_features.index, 'queue'].values

# Print the head of the combined dataframe
print(df_combined.tail())
# Assign a constant item_id for single-series forecasting
df_combined['item_id'] = 'queue_series'
# Remove timezone for AutoGluon compatibility
df_combined['timestamp'] = df_combined['timestamp'].dt.tz_localize(None)

# Chronos-Bolt forecasting with AutoGluon
ts_df = TimeSeriesDataFrame.from_data_frame(
    df_combined,
    id_column='item_id',
    timestamp_column='timestamp'
)
# Train/test split for forecasting
prediction_length = 96  # number of periods to forecast (4 intervals = 1 hour ahead)
train_data, test_data = ts_df.train_test_split(prediction_length=prediction_length)
# Initialize and run Chronos-Bolt (Base) in zero-shot mode
predictor = TimeSeriesPredictor(
    path='chronos_bolt_base',
    prediction_length=prediction_length,
    target='queue',
    freq='15min'
).fit(
    train_data=train_data,
    hyperparameters={'Chronos': [{'model_path': 'bolt_base'}]},
    enable_ensemble=False,
    time_limit=300
)
# Generate and display forecasts
predictions = predictor.predict(train_data)
print(predictions.head())

# %%

# Visualization: show last day of actuals and compare with predicted horizon
# Plotly-based interactive plot of forecast vs. actual
import plotly.graph_objects as go

# Interactive plot of forecast vs. actual
item_id = 'queue_series'
# Extract forecast and restrict actuals to prediction window
y_pred = predictions.loc[item_id]
y_test = test_data.loc[item_id]['queue'].loc[y_pred.index]

# Prepare DataFrames for Plotly
df_pred = y_pred.reset_index().rename(columns={'timestamp':'timestamp','mean':'mean','0.3':'q30','0.7':'q70'})
df_actual = y_test.reset_index().rename(columns={'queue':'actual'})
# Smooth actual future series with a 3-point centered rolling window
df_actual['smoothed'] = df_actual['actual'].rolling(window=3, center=True).mean()

# Build interactive figure
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_pred['timestamp'], y=df_pred['mean'], mode='lines', name='Mean forecast'))
fig.add_trace(go.Scatter(x=df_pred['timestamp'], y=df_pred['q30'], mode='lines', line=dict(width=0), showlegend=False))
fig.add_trace(go.Scatter(x=df_pred['timestamp'], y=df_pred['q70'], mode='lines', fill='tonexty', line=dict(width=0), name='30%-70% interval'))
# Plot smoothed actual future series
fig.add_trace(go.Scatter(
    x=df_actual['timestamp'],
    y=df_actual['smoothed'],
    mode='lines',
    name='Actual future (smoothed)'
))

# Layout and display
fig.update_layout(
    title='Forecast Horizon with Prediction Intervals',
    xaxis_title='Timestamp',
    yaxis_title='Queue Length',
    template='plotly_white'
)
fig.show()
# %%

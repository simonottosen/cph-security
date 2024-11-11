import json
import pandas as pd
import urllib.request
from autots import AutoTS
import holidays
from datetime import datetime, timedelta
import time  # Needed for time.time()
import pytz  # Import for timezone handling

# Start timing the data loading process
start_time_load_data = time.time()

# Step 1: Read data from the API
newmodeldata_url = "https://waitport.com/api/v1/all?airport=eq.DUB"
dataframe = pd.read_json(newmodeldata_url)
print("Done step 1")

# Step 2: Add holidays and create new columns

# Convert 'timestamp' to datetime with UTC timezone
dataframe['timestamp'] = pd.to_datetime(dataframe['timestamp'], utc=True)

# Convert timestamps to Europe/Dublin timezone
dataframe['timestamp'] = dataframe['timestamp'].dt.tz_convert('Europe/Dublin')

# Remove timezone information if you prefer naive datetime objects
dataframe['timestamp'] = dataframe['timestamp'].dt.tz_localize(None)

# Set 'timestamp' as the index
dataframe.set_index('timestamp', inplace=True)

# Create a copy to work with
df = dataframe.copy()

# Extract date and time features from the index
df['year'] = df.index.year
df['month'] = df.index.month
df['day'] = df.index.day
df['hour'] = df.index.hour
df['weekday'] = df.index.weekday

# Create dummy variables for 'airport' (though it's only 'DUB' in this case)
df_airport = pd.get_dummies(df['airport'])
df = pd.concat([df, df_airport], axis=1)
df.drop(columns=['airport'], inplace=True)

# Define Irish holidays
ie_holidays = holidays.Ireland()

# Add 'Holiday' column: 1 if the date is a holiday, 0 otherwise
df['Holiday'] = df.index.normalize().isin(ie_holidays).astype(int)

# Ensure 'date' column exists for grouping
df['date'] = df.index.normalize()

# Filter data between 7 AM and 10 PM
df_between_7_and_22 = df[(df['hour'] >= 7) & (df['hour'] <= 22)]

# Compute average queue per date
avg_queue_per_date = df_between_7_and_22.groupby('date')['queue'].mean().reset_index()
avg_queue_per_date.columns = ['date', 'average_queue']

# Create a mapping from date to average_queue
avg_queue_dict = avg_queue_per_date.set_index('date')['average_queue'].to_dict()

# Map the previous day's average queue to each timestamp
df['date_minus_1'] = df['date'] - pd.Timedelta(days=1)
df['yesterday_average_queue'] = df['date_minus_1'].map(avg_queue_dict)

# Map the last week's average queue (7 days ago) to each timestamp
df['date_minus_7'] = df['date'] - pd.Timedelta(days=7)
df['lastweek_average_queue'] = df['date_minus_7'].map(avg_queue_dict)

# Remove 'id' column if it exists
if 'id' in df.columns:
    df.drop(['id'], axis=1, inplace=True)

# Drop temporary date columns if not needed
df.drop(['date', 'date_minus_1', 'date_minus_7'], axis=1, inplace=True)

print("Returned data successfully in %.2f seconds " % (time.time() - start_time_load_data))
print("Done step 2")

# Step 3: The 'timestamp' is already set as the index; no need to set it again
print("Done step 3")

# Step 4: Remove duplicate timestamps if any
df = df[~df.index.duplicated(keep='first')]
print("Done step 4")

# Step 5: Resample the data to every 5 minutes and forward-fill missing values
df_resampled = df.resample('5 min').ffill()  # '5T' is the alias for 5 minutes
print("Done step 5")

# Step 6: Drop unnecessary columns if they exist
columns_to_drop = [col for col in ['id', 'airport', 'DUB'] if col in df_resampled.columns]
df_resampled.drop(columns=columns_to_drop, inplace=True, errors='ignore')
df_resampled.dropna(subset=['queue'], inplace=True)
df_resampled.dropna(subset=['queue', 'lastweek_average_queue', 'yesterday_average_queue'], inplace=True)
int_columns = ['queue', 'year', 'month', 'day', 'hour', 'weekday', 'Holiday']

# Convert specified columns to integers
for col in int_columns:
    df_resampled[col] = df_resampled[col].astype(int)

# Round average queue columns to nearest integer and convert to int
df_resampled['yesterday_average_queue'] = df_resampled['yesterday_average_queue'].round().astype(int)
df_resampled['lastweek_average_queue'] = df_resampled['lastweek_average_queue'].round().astype(int)


print("Done step 6")

# Now, df_resampled is your final DataFrame ready for modeling

# Step 6: Fit the model using AutoTS
model = AutoTS(
    forecast_length=21,
    frequency='infer',
    prediction_interval=0.9,
    ensemble='auto',
    model_list="fast_parallel",  # Options: "superfast", "default", "fast_parallel"
    transformer_list="superfast",  # Options: "superfast"
    drop_most_recent=1,
    max_generations=1,
    num_validations=2,
    no_negatives=True,
    constraint=2.0,
    validation_method="similarity"
)

model = model.fit(df_resampled)
print("Done step 6")

import json
import urllib.request
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import urllib.error

try:
    data = urllib.request.urlopen("https://waitport.com/api/v1/all").read()
    output = json.loads(data)
except urllib.error.URLError as e:
    print(f"Failed to retrieve data: {e}")
    # Handle the error or exit the script
else:
    dataframe = pd.DataFrame(output)
    # Continue with the rest of your code

dataframe['id'] = range(1, len(dataframe) + 1)
data = dataframe

# Convert 'timestamp' to datetime and round to nearest 5 minutes
data['timestamp'] = pd.to_datetime(data['timestamp'])
data['rounded_timestamp'] = data['timestamp'].dt.round('5min')

# Drop duplicates based on airport code and rounded timestamp
data_no_duplicates = data.drop_duplicates(subset=['airport', 'rounded_timestamp'])

# Define unique airports
unique_airports = data_no_duplicates['airport'].unique()

# Feature engineering: using hour of the day as a feature
data['hour'] = data['timestamp'].dt.hour

# Splitting the data into features (X) and target (y)
X = data[['hour', 'airport']]  # Using 'hour' and 'airport' as features
y = data['queue']

# Encoding categorical data (airport codes)
X = pd.get_dummies(X, columns=['airport'])

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Function to align features
def align_features(df, model_columns):
    """ Add missing dummy columns and align the order of all columns to match the model's training data """
    # Add missing columns with default value of 0
    missing_cols = set(model_columns) - set(df.columns)
    for c in missing_cols:
        df[c] = 0
    
    # Reorder columns to match the order used in the model's training data
    df = df[model_columns]
    return df

# Creating a DataFrame for each airport with a complete time range
predicted_data = pd.DataFrame()

for airport in unique_airports:
    # Selecting data for the current airport
    airport_data = data_no_duplicates[data_no_duplicates['airport'] == airport].copy()

    # Reindexing with the complete time range
    reindexed_data = airport_data.set_index('rounded_timestamp').reindex(all_timestamps).reset_index()
    reindexed_data['airport'] = airport

    # Prepare data for prediction (filling missing 'queue' values)
    reindexed_data['hour'] = reindexed_data['index'].dt.hour
    features = pd.get_dummies(reindexed_data[['hour', 'airport']], columns=['airport'])
    features = align_features(features, X_train.columns)  # Align feature columns and order

    # Predict missing 'queue' values
    missing_indices = reindexed_data['queue'].isnull()
    if missing_indices.any():
        reindexed_data.loc[missing_indices, 'queue'] = model.predict(features[missing_indices])

    # Append to the final DataFrame
    predicted_data = pd.concat([predicted_data, reindexed_data], ignore_index=True)

# Sorting and cleaning up the final DataFrame
predicted_data = predicted_data.sort_values(by=['airport', 'index'])
predicted_data.rename(columns={'index': 'timestamp'}, inplace=True)

# Displaying a sample of the final DataFrame
predicted_data_sample = predicted_data.sample(10)  # Display a sample for verification

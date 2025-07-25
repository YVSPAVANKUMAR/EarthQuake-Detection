import pandas as pd
import numpy as np
import datetime
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# Load data
data = pd.read_csv("database.csv")
data = data[['Date', 'Time', 'Latitude', 'Longitude', 'Depth', 'Magnitude']]

# Generate timestamp feature
timestamp = []
for d, t in zip(data['Date'], data['Time']):
    try:
        ts = datetime.datetime.strptime(d+' '+t, '%m/%d/%Y %H:%M:%S')
        timestamp.append(time.mktime(ts.timetuple()))
    except (ValueError, OverflowError):
        timestamp.append(0)

data['Timestamp'] = pd.Series(timestamp)
final_data = data.drop(['Date', 'Time'], axis=1)
final_data = final_data[final_data['Timestamp'] != 0]

# Features and targets
X = final_data[['Timestamp', 'Latitude', 'Longitude']]
y = final_data[['Magnitude', 'Depth']]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train, y_train)

# Save model
joblib.dump(regressor, 'earthquake_rf_model.joblib')
print("Model trained and saved as earthquake_rf_model.joblib")

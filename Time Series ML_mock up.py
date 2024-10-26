import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math

PATH = r"D:\Didit Wahyu Pradipta\Kerjaan\SESMUBI_DWP\Final Project\Pseudo Code"

df = pd.read_excel(os.path.join(PATH, 'kalsel_infl.xlsx'))

df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'], format='%Y-%b')
df.set_index('Date', inplace=True)  # Set Date as index
df = df.dropna(subset=['yoy'])

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(df[['yoy']])
data_scaled = scaler.transform(df[['yoy']])

# Prepare data for LSTM
sequence_length = 48
X, y = [], []
for i in range(len(data_scaled) - sequence_length):
    X.append(data_scaled[i:i + sequence_length])
    y.append(data_scaled[i + sequence_length])

X, y = np.array(X), np.array(y)

# Split the data into training and testing sets
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


# Create the LSTM model
model = Sequential()
model.add(LSTM(105, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(105, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train the model with early stopping
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model.fit(X_train, y_train, epochs=55, batch_size=16, validation_data=(X_test, y_test), callbacks=[early_stopping], verbose=1)

##########################################################################

# Make predictions for the training and testing sets
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Inverse transform to get actual values
y_train_actual = scaler.inverse_transform(y_train)
y_train_pred_actual = scaler.inverse_transform(y_train_pred)
y_test_actual = scaler.inverse_transform(y_test)
y_test_pred_actual = scaler.inverse_transform(y_test_pred)

# Create DataFrame for plotting
df_train = df.iloc[sequence_length:sequence_length + len(y_train_actual)].copy()
df_test = df.iloc[sequence_length + len(y_train_actual):].copy()

# Calculate performance metrics for training set
train_rmse = np.sqrt(mean_squared_error(y_train_actual, y_train_pred_actual))
train_mae = mean_absolute_error(y_train_actual, y_train_pred_actual)
train_r2 = r2_score(y_train_actual, y_train_pred_actual)

# Calculate performance metrics for testing set
test_rmse = np.sqrt(mean_squared_error(y_test_actual, y_test_pred_actual))
test_mae = mean_absolute_error(y_test_actual, y_test_pred_actual)
test_r2 = r2_score(y_test_actual, y_test_pred_actual)

# Store performance metrics in a DataFrame
performance_metrics = pd.DataFrame({
    'Metric': ['RMSE', 'MAE', 'R2 Score'],
    'Train': [train_rmse, train_mae, train_r2],
    'Test': [test_rmse, test_mae, test_r2]
})

print(performance_metrics)

# Plot the results
plt.figure(figsize=(14, 6))
plt.plot(df_train.index, y_train_actual, label='Train Data Actual')
plt.plot(df_train.index, y_train_pred_actual, label='Train Data Prediction', linestyle='--')
plt.plot(df_test.index, y_test_actual, label='Test Data Actual')
plt.plot(df_test.index, y_test_pred_actual, label='Test Data Prediction', linestyle='--')
plt.title('Inflation Forecast - Train and Test Data')
plt.xlabel('Date')
plt.ylabel('Inflation (yoy)')
plt.legend()
plt.show()

##########################################################################

# Forecast future values until December 2024
forecast_steps = 15  # Forecasting for 3 months (October, November, December 2024)
last_sequence = data_scaled[-sequence_length:]
future_predictions = []

for _ in range(forecast_steps):
    next_pred = model.predict(last_sequence.reshape(1, sequence_length, 1))
    future_predictions.append(next_pred[0, 0])
    last_sequence = np.append(last_sequence[1:], next_pred, axis=0)

# Inverse transform future predictions
future_predictions_actual = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Create a DataFrame for future predictions
forecast_dates = pd.date_range(start='2024-10-01', periods=forecast_steps, freq='MS')
forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecasted Inflation': future_predictions_actual.flatten()})

print(forecast_df)

# Plot the results
plt.figure(figsize=(14, 6))
plt.plot(df_train.index, y_train_actual, label='Train Data Actual')
plt.plot(df_train.index, y_train_pred_actual, label='Train Data Prediction', linestyle='--')
plt.plot(df_test.index, y_test_actual, label='Test Data Actual')
plt.plot(df_test.index, y_test_pred_actual, label='Test Data Prediction', linestyle='--')
plt.plot(forecast_df['Date'], forecast_df['Forecasted Inflation'], label='Forecasted Inflation (Oct-Dec 2024)', linestyle='--')
plt.title('Inflation Forecast - Train, Test, and Future Predictions')
plt.xlabel('Date')
plt.ylabel('Inflation (yoy)')
plt.legend()
plt.show()

##########################################################################

fig, ax = plt.subplots(figsize=(6, 2))  # Adjust the size as needed
ax.axis('off')  # Turn off the axis

# Create the table
table = ax.table(cellText=performance_metrics.values, 
                 colLabels=performance_metrics.columns, 
                 cellLoc='center', 
                 loc='center')

# Format the table to make it look better
table.auto_set_font_size(False)
table.set_fontsize(10)
table.auto_set_column_width(col=list(range(len(performance_metrics.columns))))

# Define the file path for saving the image
file_path = os.path.join(PATH, 'performance_metrics.png')

# Save the table as a PNG image
plt.savefig(file_path, bbox_inches='tight', dpi=300)
plt.show()


########## XGBoost ############
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

df = pd.read_excel(os.path.join(PATH, 'kalsel_infl.xlsx'))

df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'], format='%Y-%b')
df.set_index('Date', inplace=True)  # Set Date as index
df = df.dropna(subset=['yoy'])

# Feature engineering - create lag features
for lag in range(1, 13):
    df[f'lag_{lag}'] = df['yoy'].shift(lag)


# Prepare the data for training and testing
X = df.drop(columns=['yoy', 'Year', 'Month'])
y = df['yoy']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Scale the features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the XGBoost model
model = XGBRegressor(objective='reg:squarederror', n_estimators=150, learning_rate=0.1, max_depth=5)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Calculate performance metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Store performance metrics and display as a table
performance_metrics = pd.DataFrame({
    'Metric': ['RMSE', 'MAE', 'R2 Score'],
    'Value': [rmse, mae, r2]
})

print(performance_metrics)

# Plot the results
plt.figure(figsize=(14, 6))
plt.plot(y_test.index, y_test, label='Actual YoY Inflation')
plt.plot(y_test.index, y_pred, label='Predicted YoY Inflation', linestyle='--')
plt.title('Year-on-Year Inflation Forecast vs Actual (Test Set)')
plt.xlabel('Date')
plt.ylabel('Inflation (YoY)')
plt.legend()
plt.show()

# Forecast future values until December 2024
future_months = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), end='2024-12-01', freq='MS')
future_lags = df.iloc[-12:]['yoy'].values.reshape(1, -1)  # Last 12 months of 'yoy' values for lag features
future_predictions = []

for _ in range(len(future_months)):
    future_scaled = scaler.transform(np.pad(future_lags, ((0, 0), (X_train.shape[1] - future_lags.shape[1], 0)), 'constant'))
    next_pred = model.predict(future_scaled)[0]
    future_predictions.append(next_pred)
    future_lags = np.append(future_lags[:, 1:], next_pred).reshape(1, -1)

# Create a DataFrame for future predictions
future_df = pd.DataFrame({'Date': future_months, 'Forecasted YoY Inflation': future_predictions})
future_df.set_index('Date', inplace=True)

# Plot the forecast
plt.figure(figsize=(14, 6))
plt.plot(df.index, df['yoy'], label='Historical YoY Inflation')
plt.plot(future_df.index, future_df['Forecasted YoY Inflation'], label='Forecasted YoY Inflation', linestyle='--')
plt.title('Year-on-Year Inflation Forecast until December 2024')
plt.xlabel('Date')
plt.ylabel('Inflation (YoY)')
plt.legend()
plt.show()


####### FB Prophet #############

from prophet import Prophet

df = pd.read_excel(os.path.join(PATH, 'kalsel_infl.xlsx'))

df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'], format='%Y-%b')
df.set_index('Date', inplace=True)  # Set Date as index
df = df.dropna(subset=['yoy'])

# Prepare data for Prophet
prophet_df = df.reset_index()[['Date', 'yoy']]
prophet_df.rename(columns={'Date': 'ds', 'yoy': 'y'}, inplace=True)

# Create and fit the Prophet model
model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False, changepoint_prior_scale=0.05)
model.fit(prophet_df)

# Make future dataframe for forecasting
future = model.make_future_dataframe(periods=12, freq='M')
forecast = model.predict(future)

# Plot the forecast
fig = model.plot(forecast)
plt.title('Year-on-Year Inflation Forecast using FB Prophet')
plt.xlabel('Date')
plt.ylabel('Inflation (YoY)')
plt.show()

# Plot the forecast components
fig2 = model.plot_components(forecast)
plt.show()

# Display forecasted values
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(12)


### LSTM Forecast Prices ###

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.layers import Dropout

# Load the dataset
PATH = r"D:\Didit Wahyu Pradipta\Kerjaan\SESMUBI_DWP\Final Project\ML Model"
file_name = "rea_holdings_share_prices.xlsx"
full_path = os.path.join(PATH, file_name)

# Read the dataset
df = pd.read_excel(full_path, header=1)

# Use only the 'Close' column for prediction
close_data = df[['Close']]

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_data)

# Prepare the data for LSTM
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

# Define the time step for LSTM input
TIME_STEP = 100  # Use last 60 days to predict the next value

# Split the data into training and testing datasets
train_size = int(len(scaled_data) * 0.8)
test_size = len(scaled_data) - train_size
train_data, test_data = scaled_data[0:train_size, :], scaled_data[train_size:len(scaled_data), :]

# Create the training and testing datasets
X_train, y_train = create_dataset(train_data, TIME_STEP)
X_test, y_test = create_dataset(test_data, TIME_STEP)

# Reshape input to be [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build the LSTM model
model = Sequential()
model.add(LSTM(115, return_sequences=True, input_shape=(TIME_STEP, 1)))
model.add(LSTM(115, return_sequences=False))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_absolute_error')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Predict the prices on the training dataset
train_predictions = model.predict(X_train)

# Predict the prices on the testing dataset
test_predictions = model.predict(X_test)

# Inverse transform the predictions to get the actual values
train_predictions = scaler.inverse_transform(train_predictions)
y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
test_predictions = scaler.inverse_transform(test_predictions)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate and print the Mean Squared Error
mse = mean_squared_error(y_test_actual, test_predictions)
print(f"Mean Squared Error: {mse}") 

# Plot the results with Date on the x-axis
dates_test = df['Date'].iloc[train_size + TIME_STEP + 1:train_size + TIME_STEP + 1 + len(y_test_actual)]
plt.figure(figsize=(12, 6))
plt.plot(dates_test, y_test_actual, label='Actual Prices')
plt.plot(dates_test, test_predictions, label='Predicted Prices')
plt.xlabel('Date')
plt.ylabel('CPO Close Price')
plt.title('CPO Close Price Prediction')
plt.legend()
plt.show()

# Forecast future values (e.g., till Dec 2024) ----> Continue to run line 387 instead
num_future_predictions = (pd.Timestamp('2024-12-31') - pd.to_datetime(df['Date'].iloc[-1])).days
last_sequence = scaled_data[-TIME_STEP:]
future_predictions = []

for _ in range(num_future_predictions):
    # Predict the next value
    next_prediction = model.predict(last_sequence.reshape(1, TIME_STEP, 1))[0, 0]
    # Append the prediction to the future predictions list
    future_predictions.append(next_prediction)
    # Update the last sequence to include the new prediction
    last_sequence = np.append(last_sequence[1:], next_prediction).reshape(TIME_STEP, 1)

# Inverse transform the future predictions
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Combine actual, predicted, and future predictions for plotting
train_dates = df['Date'].iloc[TIME_STEP:TIME_STEP + len(y_train_actual)]
test_dates = df['Date'].iloc[train_size + TIME_STEP + 1:train_size + TIME_STEP + 1 + len(y_test_actual)]
future_dates = pd.date_range(df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=num_future_predictions)

plt.figure(figsize=(12, 6))
plt.plot(train_dates, y_train_actual, label='Train Data Actual')
plt.plot(train_dates, train_predictions, label='Train Data Prediction')
plt.plot(test_dates, y_test_actual, label='Test Data Actual')
plt.plot(test_dates, test_predictions, label='Test Data Prediction')
plt.xlabel('Date')
plt.ylabel('CPO Close Price')
plt.title('CPO Close Price Prediction (Train and Test)')
plt.legend()
plt.show()

# Plot test data actual and test data prediction until Dec 31st, 2024
plt.figure(figsize=(12, 6))
plt.plot(test_dates, y_test_actual, label='Test Data Actual')
plt.plot(test_dates, test_predictions, label='Test Data Prediction')
plt.plot(future_dates, future_predictions, label='Future Predictions (till Dec 2024)')
plt.xlabel('Date')
plt.ylabel('CPO Close Price')
plt.title('CPO Close Price Test Data and Future Forecast (till Dec 2024)')
plt.legend()
plt.show()

### add random noise in the prediction ###
import random

# Forecast future values (e.g., till Dec 2024)
num_future_predictions = (pd.Timestamp('2024-12-31') - pd.to_datetime(df['Date'].iloc[-1])).days
last_sequence = scaled_data[-TIME_STEP:]
future_predictions = []

for _ in range(num_future_predictions):
    # Predict the next value
    next_prediction = model.predict(last_sequence.reshape(1, TIME_STEP, 1))[0, 0]
    # Add random noise to the prediction
    next_prediction += random.uniform(-0.011, 0.011)  # Adjust the range as needed
    # Append the prediction to the future predictions list
    future_predictions.append(next_prediction)
    # Update the last sequence to include the new prediction
    last_sequence = np.append(last_sequence[1:], next_prediction).reshape(TIME_STEP, 1)
    
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Combine actual, predicted, and future predictions for plotting
train_dates = df['Date'].iloc[TIME_STEP:TIME_STEP + len(y_train_actual)]
test_dates = df['Date'].iloc[train_size + TIME_STEP + 1:train_size + TIME_STEP + 1 + len(y_test_actual)]
future_dates = pd.date_range(df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=num_future_predictions)

plt.figure(figsize=(12, 6))
plt.plot(train_dates, y_train_actual, label='Train Data Actual')
plt.plot(train_dates, train_predictions, label='Train Data Prediction')
plt.plot(test_dates, y_test_actual, label='Test Data Actual')
plt.plot(test_dates, test_predictions, label='Test Data Prediction')
plt.xlabel('Date')
plt.ylabel('CPO Close Price')
plt.title('CPO Close Price Prediction (Train and Test)')
plt.legend()
plt.show()

# Plot test data actual and test data prediction until Dec 31st, 2024
plt.figure(figsize=(12, 6))
plt.plot(test_dates, y_test_actual, label='Test Data Actual')
plt.plot(test_dates, test_predictions, label='Test Data Prediction')
plt.plot(future_dates, future_predictions, label='Future Predictions (till Dec 2024)')
plt.xlabel('Date')
plt.ylabel('CPO Close Price')
plt.title('CPO Close Price Test Data and Future Forecast (till Dec 2024)')
plt.legend()
plt.show()

# Create a DataFrame to show future predictions with corresponding dates
future_predictions_df = pd.DataFrame({'Date': future_dates, 'Predicted Close Price': future_predictions.flatten()})
print("Future Predictions (till Dec 2024):")
print(future_predictions_df)
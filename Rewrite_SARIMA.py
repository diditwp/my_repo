# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 17:38:45 2024

@author: Didit Wahyu Pradipta
"""

import pandas as pd
import numpy as np
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from math import sqrt
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import os
import warnings
warnings.filterwarnings("ignore")

PATH = r"D:\Didit Wahyu Pradipta\Kerjaan\KPw Kalsel\Data Inflasi"

df = pd.read_excel(os.path.join(PATH, 'bjm_inflyoy.xlsx'))

df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'], format='%Y-%b')
df.set_index('Date', inplace=True)  # Set Date as index

# Step 1: Check for stationarity and apply differencing if necessary
stationary_test = adfuller(df['Inflation'])

print('ADF Statistic:', stationary_test[0])
print('p-value:', stationary_test[1])
print('Critical Values:')
for key, value in stationary_test[4].items():
    print(f'   {key}: {value}')

if stationary_test[1] < 0.05:
    print("The time series is stationary.")
else:
    print("The time series is non-stationary.")
    
# Plot ACF and PACF
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# Create the ACF plot
plot_acf(df['Inflation'], ax=axes[0])
axes[0].set_title('Autocorrelation Function (ACF)')

# Create the PACF plot
plot_pacf(df['Inflation'], ax=axes[1])
axes[1].set_title('Partial Autocorrelation Function (PACF)')

# Split data into training and test sets
train_end_date = '2021-12-01'
train_data = df[:train_end_date]
test_data = df[train_end_date:]

# Step 2: Use auto_arima to find the best parameters for SARIMA
model = auto_arima(train_data['Inflation'], 
                   seasonal=True, 
                   m=12, 
                   trace=True, 
                   error_action='ignore', 
                   suppress_warnings=True,
                   stepwise=True,
                   n_jobs=-1)

print(f"Best order: {model.order}")
print(f"Best seasonal order: {model.seasonal_order}")

# Step 3: Fit the best model on the entire training data
final_model = SARIMAX(train_data['Inflation'], 
                      order=model.order, 
                      seasonal_order=model.seasonal_order)
results = final_model.fit()

# Step 4: Forecasting from October 2024 to December 2024
forecast_steps = 3  # Forecasting for 6 months (October to December 2024)
forecast = results.get_forecast(steps=forecast_steps)
forecast_mean = forecast.predicted_mean

# Create a DataFrame for future predictions
forecast_dates = pd.date_range(start='2024-10-01', periods=forecast_steps, freq='M')
forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecasted Inflation': forecast_mean})
print(forecast_df)
################################################################
# Step 5: Predict values for training and test sets
y_train_pred = results.fittedvalues

test_predictions = results.get_prediction(start=test_data.index[0], end=test_data.index[-1]).predicted_mean
y_test_actual = test_data['Inflation']

# Step 6: Calculate adjusted R-squared for training and test predictions
n_train = len(train_data)
n_test = len(test_data)
p = len(model.order) + len(model.seasonal_order) - 1

# Training adjusted R-squared
rss_train = np.sum((train_data['Inflation'] - y_train_pred) ** 2)
tss_train = np.sum((train_data['Inflation'] - np.mean(train_data['Inflation'])) ** 2)
r_squared_train = 1 - (rss_train / tss_train)
adjusted_r_squared_train = 1 - ((1 - r_squared_train) * (n_train - 1) / (n_train - p - 1))
print(f"Adjusted R-squared (Train): {adjusted_r_squared_train}")

# Testing adjusted R-squared
rss_test = np.sum((y_test_actual - test_predictions) ** 2)
tss_test = np.sum((y_test_actual - np.mean(y_test_actual)) ** 2)
r_squared_test = 1 - (rss_test / tss_test)
adjusted_r_squared_test = 1 - ((1 - r_squared_test) * (n_test - 1) / (n_test - p - 1))
print(f"Adjusted R-squared (Test): {adjusted_r_squared_test}")

# Calculate performance metrics for training set
train_rmse = np.sqrt(mean_squared_error(train_data['Inflation'], y_train_pred))
train_mae = mean_absolute_error(train_data['Inflation'], y_train_pred)
train_r2 = r2_score(train_data['Inflation'], y_train_pred)

# Calculate performance metrics for testing set
test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_predictions))
test_mae = mean_absolute_error(y_test_actual, test_predictions)
test_r2 = r2_score(y_test_actual, test_predictions)

# Store performance metrics in a DataFrame
performance_metrics = pd.DataFrame({
    'Metric': ['RMSE', 'MAE', 'R2 Score'],
    'Train': [train_rmse, train_mae, train_r2],
    'Test': [test_rmse, test_mae, test_r2]
})

print(performance_metrics)

# Step 6: Plot the results with 5 lines
plt.figure(figsize=(12, 6))

# Plot train data actual and prediction
plt.plot(train_data.index, train_data['Inflation'], label='Train Data Actual')
plt.plot(train_data.index, y_train_pred, label='Train Data Prediction', linestyle='--')

# Plot test data actual and prediction
plt.plot(test_data.index, y_test_actual, label='Test Data Actual')
plt.plot(test_data.index, test_predictions, label='Test Data Prediction', linestyle='--')

# Plot forecasted values
plt.plot(forecast_df['Date'], forecast_df['Forecasted Inflation'], label='Forecasted Inflation (Oct-Dec 2024)', linestyle='--')

plt.title('Inflation Forecast - Train, Test, and Future Predictions')
plt.xlabel('Date')
plt.ylabel('Inflation (yoy)')
plt.legend()
plt.show()

################################################################

# Print the forecasted values for each month
forecast_dates = pd.date_range(start='2024-10-01', periods=forecast_steps, freq='M')
forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecasted Inflation': forecast_mean})
for index, row in forecast_df.iterrows():
    print(f"Forecasted inflation for {row['Date'].strftime('%B %Y')}: {row['Forecasted Inflation']}")

# Step 5: Evaluate the model on the test data (Jan 2000 to June 2024)
# Predict the values in the test set
test_predictions = results.get_prediction(start=test_data.index[0], end=test_data.index[-1]).predicted_mean
test_actuals = df['Inflation'].loc[test_predictions.index]

# Calculate evaluation metrics
mse = mean_squared_error(test_actuals, test_predictions)
rmse = sqrt(mse)
mae = mean_absolute_error(test_actuals, test_predictions)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
print('AIC: ', results.aic)

# Calculate adjusted R-squared
n = len(test_actuals)
p = len(model.order) + len(model.seasonal_order) - 1
rss = np.sum((test_actuals - test_predictions) ** 2)
tss = np.sum((test_actuals - np.mean(test_actuals)) ** 2)
r_squared = 1 - (rss / tss)
adjusted_r_squared = -1*(1 - ((1 - r_squared) * (n - 1) / (n - p - 1)))

print(f"R-squared: {r_squared}")
print(f"Adjusted R-squared: {adjusted_r_squared}")

# Generate dates from October 2024 to December 2024
forecast_dates = pd.date_range(start='2024-10-01', periods=3, freq='MS')

actual_2024 = df['Inflation']['2024-01-01':'2024-09-30']
# Plot actual inflation and forecasted inflation for 2024
plt.figure(figsize=(12, 6))
plt.plot(actual_2024.index, actual_2024, label='Actual Inflation')

# Plot forecasted inflation with manual dates
plt.plot(forecast_dates, forecast_mean, label='Forecasted Inflation', linestyle='--')

# Annotate data points
for i in range(len(actual_2024)):
    plt.annotate(f'{actual_2024.iloc[i]:.2f}', (actual_2024.index[i], actual_2024.iloc[i]), textcoords="offset points", xytext=(0,10), ha='center')
for i in range(len(forecast_mean)):
    plt.annotate(f'{forecast_mean.iloc[i]:.2f}', (forecast_dates[i], forecast_mean.iloc[i]), textcoords="offset points", xytext=(0,10), ha='center')

plt.title('Actual and Forecasted Inflation for 2024')
plt.xlabel('Date')
plt.ylabel('Inflation')
plt.legend()
plt.show()

# Actual Inflation vs Forecast
plt.figure(figsize=(12, 6))

# Filter the date range for actual inflation data (January 2023 to June 2024)
df_filtered = df.loc['2022-01-01':'2024-09-30']

plt.plot(df_filtered.index, df_filtered['Inflation'].loc['2022-01-01':'2024-09-30'], label='Actual Inflation')
plt.plot(df_filtered.index, test_predictions.loc['2022-01-01':'2024-09-30'], label='SARIMA Forecast')

plt.title('Actual and Model Forecasted Inflation')
plt.xlabel('Date')
plt.ylabel('Inflation')
plt.legend()

plt.show()


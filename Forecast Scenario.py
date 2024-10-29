# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 11:16:22 2024

@author: Didit Wahyu Pradipta
"""
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.utils import resample
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

PATH = r"D:\Didit Wahyu Pradipta\Kerjaan\KPw Kalsel\Data Inflasi"

df = pd.read_excel(os.path.join(PATH, 'bjm_inflyoy.xlsx'))


df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'], format='%Y-%b')
df.set_index('Date', inplace=True)  

######################### Plotting the Rolling Mean
start_date = '2018-01-01'
end_date = '2021-06-01'
filtered_df = df.loc[start_date:end_date]

fig, ax = plt.subplots(figsize=(8, 3.5))

# Plot the original inflation rates
filtered_df['Inflation'].plot(ax=ax, label='Actual')

# Plot the rolling mean of the inflation rates
filtered_df['Inflation'].rolling(window=12).mean().plot(ax=ax, label='Rolling Mean', 
                                                        linestyle='-', color='red', linewidth=1)

ax.set_ylabel('Inflation(%yoy)')
ax.set_xlabel('Date')
ax.set_title('Inflation Rates in Banjarmasin (Jan 2020 - Jun 2024)')
ax.legend()

plt.show()

#########################

# H-P Filter
cycle, trend = sm.tsa.filters.hpfilter(df['Inflation'], 129600)
infl_decomp = df[['Inflation']]
infl_decomp["cycle"] = cycle
infl_decomp["trend"] = trend

fig, ax = plt.subplots()
infl_decomp[["Inflation", "trend"]]["2000-01-01":].plot(ax=ax, fontsize=8)
plt.show()

########################

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

# Historical data for bootstrapping (2015-2023)
historical_data = df['Inflation'].loc['2018-01-01':'2021-12-01']

# Create bootstrapped samples
def bootstrap_samples(data, n_samples=100000):
    bootstrapped_means = []
    for i in range(n_samples):
        sample = resample(data)
        bootstrapped_means.append(np.mean(sample))
    return bootstrapped_means

# Create bootstrapped samples for October
oct_data = historical_data[historical_data.index.month == 10]
bootstrapped_oct = bootstrap_samples(oct_data)

# Plot the distribution of bootstrapped values
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(bootstrapped_oct, bins=30, edgecolor='k', alpha=0.7)
ax.axvline(np.mean(bootstrapped_oct), color='r', linestyle='dashed', linewidth=1)
ax.set_title('Distribution of Bootstrapped Inflation Values for October')
ax.set_xlabel('Inflation (%)')
ax.set_ylabel('Frequency')

plt.show()

print('mean: ', np.mean(bootstrapped_oct))
print('SD: ', np.std(bootstrapped_oct))
0
# Step 4: Comparison of the three approaches

# Forecast results for July
arima_forecast = 3.271911
LSTM = 2.558021
XG_Boost = 3.819170
Random_Forest = 2.440868
historical_forecast = 4.5


# Monte Carlo simulation function
def monte_carlo_simulation(forecast, historical_distribution, n_simulations=10000):
    results = []
    for _ in range(n_simulations):
        simulated_value = np.random.choice(historical_distribution)
        results.append(simulated_value)
    results = np.array(results)
    accuracy_prob = np.mean(np.abs(results - forecast) < 0.5)  # Define accuracy as within 0.5pp
    return accuracy_prob

# Monte Carlo simulation for each forecast
arima_prob = monte_carlo_simulation(arima_forecast, bootstrapped_oct)
LSTM_prob = monte_carlo_simulation(LSTM, bootstrapped_oct)
XG_Boost_prob = monte_carlo_simulation(XG_Boost, bootstrapped_oct)
Random_Forest_prob = monte_carlo_simulation(Random_Forest, bootstrapped_oct)
historical_prob = monte_carlo_simulation(historical_forecast, bootstrapped_oct)

print(f"SARIMA Forecast Probability: {arima_prob}")
print(f"LSTM Forecast Probability: {LSTM_prob}")
print(f"XGBoost Forecast Probability: {XG_Boost_prob}")
print(f"Random Forest Forecast Probability: {Random_Forest_prob}")
print(f"Historical Forecast Probability: {historical_prob}")

# Visualize the probabilities
labels = ['SARIMA', 'LSTM', 'XGBoost', 'Random Forest','Historical']
probabilities = [arima_prob, LSTM_prob, XG_Boost_prob, Random_Forest_prob, historical_prob]

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(labels, probabilities, color=['blue', 'green', 'red', 'purple', 'yellow'])
ax.set_ylabel('Probability of Accurate Forecast')
ax.set_title('Comparison of Forecasting Methods for October')
ax.set_ylim(0, max(probabilities) + 0.1)

for i, prob in enumerate(probabilities):
    ax.text(i, prob + 0.001, f'{prob:.4f}', ha='center', va='bottom')

plt.show()

# Choose the best model based on probability
best_model = labels[np.argmax(probabilities)]
print(f"The best model based on the probability of accurate forecast is: {best_model}")
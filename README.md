# Multivariate Time Series Forecasting

## Overview
Multivariate Time Series Forecasting is preferable when the variables have dependencies or interactions with one another. The goal is to capture these interdependencies to make accurate predictions for each variable over a future time period. In this project, I focus on forecasting stock prices using multivariate time series models.

## Process Followed

### 1. Identifying Relevant Variables
- I determine which variables are relevant to the forecast.
- Besides the target variable (closing price), I identify other influencing factors such as opening price, high, low, volume, etc.

### 2. Data Collection
- I gather historical stock market data for multiple stocks.
- I ensure that the data covers a sufficient time period to capture trends, seasonality, and cyclic behavior.

### 3. Data Preprocessing
- I handle missing values, outliers, and anomalies using imputation, data smoothing, or anomaly detection techniques.
- I convert the `Date` column to a datetime type for time series analysis.
- I check how many unique stocks (Tickers) are present and their respective data points.
- I resample the data to a consistent time frequency if necessary (e.g., daily, weekly) based on forecasting goals.

```python
stocks_data['Date'] = pd.to_datetime(stocks_data['Date'])
missing_values = stocks_data.isnull().sum()
unique_stocks = stocks_data['Ticker'].value_counts()
print(missing_values)
```

- The dataset contains data for four unique stocks: Apple (AAPL), Microsoft (MSFT), Netflix (NFLX), and Google (GOOG), each with 62 data points.

### 4. Exploratory Data Analysis (EDA)
- I check the time range of the dataset.
- I visualize the closing price trends for each stock.

```python
time_range = stocks_data['Date'].min(), stocks_data['Date'].max()
print(time_range)
```

- Time range: `2023-02-07` to `2023-05-05` (approximately 3 months).
- Visualizing the closing price trends:

**![Closing Price Trends](https://github.com/Sourabh1710/Multivariate-Time-Series-Forecasting/blob/main/images/Closing%20Price%20Trends%20for%20AAPL%2C%20MSFT%2C%20NFLX%2C%20GOOG.png)**

### 5. Model Selection and Data Preparation
- Since my dataset is multivariate, a Vector AutoRegression (VAR) model is selected.
- Before applying VAR, I ensure that each time series is stationary.
- I use the Augmented Dickey-Fuller (ADF) test to check for stationarity.

```python
from statsmodels.tsa.stattools import adfuller

def adf_test(series):
    result = adfuller(series)
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')

for stock in ['AAPL', 'MSFT', 'NFLX', 'GOOG']:
    print(f"ADF Test for {stock} Closing Price:")
    adf_test(stocks_data[stocks_data['Ticker'] == stock]['Close'])
```

**ADF Test Results:**
- **AAPL**: p-value = 0.927 (Non-stationary)
- **MSFT**: p-value = 0.944 (Non-stationary)
- **NFLX**: p-value = 0.225 (Non-stationary)
- **GOOG**: p-value = 0.567 (Non-stationary)

### 6. Making the Data Stationary
Since all series are non-stationary, I apply differencing to transform the data.

```python
stocks_data['Close_Diff'] = stocks_data.groupby('Ticker')['Close'].diff()
```

After differencing, the ADF test confirms that all series are now stationary.

### 7. Model Training
- I train a VAR model using the stationary data.
- I forecast future stock prices for the next 5 days.

```python
from statsmodels.tsa.api import VAR

model = VAR(stocks_data[['Close_Diff']].dropna())
model_fitted = model.fit()
forecast = model_fitted.forecast(model_fitted.y, steps=5)
```

- I convert the forecasted differenced values back to the original scale.

### 8. Visualizing the Forecasted Prices
- I plot historical closing prices along with the forecasted prices.

**![Forecasted Prices](https://github.com/Sourabh1710/Multivariate-Time-Series-Forecasting/blob/main/images/Historical%20and%20Forecasted%20Closing%20Prices.png)**

## Summary
- This project demonstrates Multivariate Time Series Forecasting using Python.
- I explored the dataset, performed EDA, and ensured data stationarity before applying the VAR model.
- The forecasted values were plotted alongside historical stock prices to visualize trends.

## Author
Sourabh Sonker <br>
Data Scientist


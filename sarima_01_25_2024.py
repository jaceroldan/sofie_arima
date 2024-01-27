import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_predict
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

df1 = pd.read_csv('2024-passengers-p1-embarking.csv', parse_dates=['Quarter'], index_col = ['Quarter'])
df2 = pd.read_csv('2024-passengers-p3-embarking.csv', parse_dates=['Quarter'], index_col = ['Quarter'])
df3 = pd.read_csv('2024-passengers-p1-disembarking.csv', parse_dates=['Quarter'], index_col = ['Quarter'])
df4 = pd.read_csv('2024-passengers-p3-disembarking.csv', parse_dates=['Quarter'], index_col = ['Quarter'])
df5 = pd.read_csv('2024-shipcall-p1.csv', parse_dates=['Quarter'], index_col = ['Quarter'])
df6 = pd.read_csv('2024-shipcall-p3.csv', parse_dates=['Quarter'], index_col = ['Quarter'])

# df.head()
# plt.xlabel('Quarter')
# plt.ylabel('Number passengers')
# plt.plot(df)

# rolling_mean = df.rolling(window = 12).mean()
# rolling_std = df.rolling(window = 12).std()
# plt.plot(df, color = 'blue', label = 'Original')
# plt.plot(rolling_mean, color = 'red', label = 'Rolling Mean')
# plt.plot(rolling_std, color = 'black', label = 'Rolling Std')
# plt.legend(loc = 'best')
# plt.title('Rolling Mean & Rolling Standard Deviation')
# plt.show()


# result = adfuller(df['Passengers'])
# print('ADF Statistic: {}'.format(result[0]))
# print('p-value: {}'.format(result[1]))
# print('Critical Values:')
# for key, value in result[4].items():
#     print('\t{}: {}'.format(key, value))

# df_log = np.log(df)
# plt.plot(df_log)

def get_stationarity(timeseries):
    
    # rolling statistics
    rolling_mean = timeseries.rolling(window=12).mean()
    rolling_std = timeseries.rolling(window=12).std()
    
    # rolling statistics plot
    original = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolling_mean, color='red', label='Rolling Mean')
    std = plt.plot(rolling_std, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    # Dickey–Fuller test:
    result = adfuller(timeseries['Passengers'])
    print('ADF Statistic: {}'.format(result[0]))
    print('p-value: {}'.format(result[1]))
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))


# rolling_mean = df_log.rolling(window=12).mean()
# df_log_minus_mean = df_log - rolling_mean
# df_log_minus_mean.dropna(inplace=True)
# get_stationarity(df_log_minus_mean)

# rolling_mean_exp_decay = df_log.ewm(halflife=12, min_periods=0, adjust=True).mean()
# df_log_exp_decay = df_log - rolling_mean_exp_decay
# df_log_exp_decay.dropna(inplace=True)
# get_stationarity(df_log_exp_decay)

# df_log_shift = df_log - df_log.shift()
# df_log_shift.dropna(inplace=True)
# get_stationarity(df_log_shift)

# print(df_log)
# decomposition = seasonal_decompose(df_log) 
# model = ARIMA(df_log, order=(2,1,2))
# results = model.fit()
# print(results.fittedvalues)
# plt.plot(df_log_shift)
# plt.plot(results.fittedvalues, color='red')

# predictions_ARIMA_diff = pd.Series(results.fittedvalues, copy=True)
# predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
# predictions_ARIMA_log = pd.Series(df_log['Passengers'].iloc[0], index=df_log.index)
# predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)
# predictions_ARIMA = np.exp(predictions_ARIMA_log)
# plt.plot(df)
# plt.plot(predictions_ARIMA)

# ppredict = plot_predict(results,1,120)
# print(np.exp(df_log))
# print(ppredict)
title1 = 'Pier 1 - Embarking - Forecast (SARIMA)'
title2 = 'Pier 3 - Embarking - Forecast (SARIMA)'
title3 = 'Pier 1 - Disembarking - Forecast (SARIMA)'
title4 = 'Pier 3 - Disembarking - Forecast (SARIMA)'
title5 = 'Pier 1 - Shipcall - Forecast (SARIMA)'
title6 = 'Pier 3 - Shipcall - Forecast (SARIMA)'

# print('Pier 1 - Embarking - Forecast')
# print('Pier 3 - Embarking - Forecast')
# print('Pier 1 - Disembarking - Forecast')
# print('Pier 3 - Disembarking - Forecast')

# print(np.exp(results.forecast(steps=40)))
# plt.title('Pier 1 - Embarking')
# plt.title('Pier 3 - Embarking')
# plt.title('Pier 1 - Disembarking')
# plt.title('Pier 3 - Disembarking')
# plt.title('Pier 1 - Shipcall')
# plt.title('Pier 3 - Shipcall')
# plt.show()


from statsmodels.tsa.statespace.sarimax import SARIMAX

def fit_sarima_and_predict(df, order, seasonal_order, forecast_steps, title, save=True):
    """
    Fit a SARIMA model to a time series and make predictions.

    Parameters:
    - time_series (pd.Series): The time series data.
    - order (tuple): The order of the non-seasonal ARIMA components (p, d, q).
    - seasonal_order (tuple): The order of the seasonal ARIMA components (P, D, Q, s).
    - forecast_steps (int): The number of steps to forecast into the future.

    Returns:
    - pd.DataFrame: A DataFrame with the original time series and the forecasted values.
    """

    # Fit SARIMA model
    # print('time series: ', time_series['Quarter'])
    time_series = df['Passengers']
    model = SARIMAX(time_series, order=order, seasonal_order=seasonal_order)
    results = model.fit(disp=False)

    # Specify the frequency explicitly
    original_frequency = time_series.index.freq
    forecast_frequency = 'QS'  # Adjust the frequency here

    # Make predictions
    forecast = results.get_forecast(steps=forecast_steps, freq=forecast_frequency)
    forecast_index = pd.date_range(time_series.index[-1], periods=forecast_steps + 1, freq=forecast_frequency)[1:]
    forecast_df = pd.DataFrame({'original': time_series, 'forecast': forecast.predicted_mean}, index=forecast_index)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(time_series, label='Original')
    plt.plot(forecast_df['forecast'], label='Forecast', color='red', linestyle='dashed')
    plt.title('SARIMA Forecast: ' + title)
    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.legend()
    if save:
        plt.savefig('data'+title+'.png')
    else:
        plt.show()

    forecast_df['original'] = forecast_df.index
    forecast_df.to_csv('data'+title+'.csv', sep=',', index=False, encoding='utf-8')

    return forecast_df

fit_sarima_and_predict(df1, (2,1,2), (1,1,1,4), 120, title1)
fit_sarima_and_predict(df2, (2,1,2), (1,1,1,4), 120, title2)
fit_sarima_and_predict(df3, (2,1,2), (1,1,1,4), 120, title3)
fit_sarima_and_predict(df4, (2,1,2), (1,1,1,4), 120, title4)
fit_sarima_and_predict(df5, (2,1,2), (1,1,1,4), 120, title5)
fit_sarima_and_predict(df6, (2,1,2), (1,1,1,4), 120, title6)

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_predict
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# df = pd.read_csv('2024-passengers-p1-embarking.csv', parse_dates=['Quarter'], index_col = ['Quarter'])
# df = pd.read_csv('2024-passengers-p3-embarking.csv', parse_dates=['Quarter'], index_col = ['Quarter'])
# df = pd.read_csv('2024-passengers-p1-disembarking.csv', parse_dates=['Quarter'], index_col = ['Quarter'])
df = pd.read_csv('2024-passengers-p3-disembarking.csv', parse_dates=['Quarter'], index_col = ['Quarter'])

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

df_log = np.log(df)
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


rolling_mean = df_log.rolling(window=12).mean()
# df_log_minus_mean = df_log - rolling_mean
# df_log_minus_mean.dropna(inplace=True)
# get_stationarity(df_log_minus_mean)

rolling_mean_exp_decay = df_log.ewm(halflife=12, min_periods=0, adjust=True).mean()
# df_log_exp_decay = df_log - rolling_mean_exp_decay
# df_log_exp_decay.dropna(inplace=True)
# get_stationarity(df_log_exp_decay)

df_log_shift = df_log - df_log.shift()
# df_log_shift.dropna(inplace=True)
# get_stationarity(df_log_shift)

# print(df_log)
decomposition = seasonal_decompose(df_log) 
model = ARIMA(df_log, order=(2,1,2))
results = model.fit()
# print(results.fittedvalues)
# plt.plot(df_log_shift)
# plt.plot(results.fittedvalues, color='red')

predictions_ARIMA_diff = pd.Series(results.fittedvalues, copy=True)
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
predictions_ARIMA_log = pd.Series(df_log['Passengers'].iloc[0], index=df_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)
predictions_ARIMA = np.exp(predictions_ARIMA_log)
# plt.plot(df)
# plt.plot(predictions_ARIMA)

ppredict = plot_predict(results,1,120)
print(np.exp(df_log))
print(ppredict)
print('Pier 3 - Disembarking - Forecast')
print(np.exp(results.forecast(steps=40)))
plt.title('Pier 3 - Disembarking')
plt.show()

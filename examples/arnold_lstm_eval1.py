# !pip install -r https://gist.githubusercontent.com/stancsz/c9fb51930b4ad40e5d13e502deaebaec/raw/281fad49b22327d0c717f0784a72b16a08f1bbd8/requirements.txt
# to minify  pyminifier arnold_lstm_eval1.py >> temp.py
# source ref https://www.kaggle.com/saurabhshahane/stock-prices-predictions-eda-lstm-deepexploration#Predictions
# -*- coding: utf-8 -*-
"""
#### Libraries
"""
import os
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns
import plotly.figure_factory as ff
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
import calendar
import re
import time
import finnhub
from see_rnn import *


def get_exclude_column_names(days):
    exclude_column_names = []
    for i in days:
        exclude_column_names.append('next_' + str(i) + '_high')
    return exclude_column_names


def rename_reference_df_column_names(df, suffix):
    """
    example usage
        df_dji = rename_reference_df_column_names(df_dji, "_dji")
    ['c', 'h', 'l', 'o', 's', 't', 'v', 'wma'] -> ['c_dji', 'h_dji', 'l_dji', 'o_dji', 's_dji', 't_dji', 'v_dji', 'wma_dji']
    :param df: df to operate
    :param suffix: suffix to add to append
    :return:
    """
    old_names = df.columns
    # print(old_names)
    new_names = [s + suffix for s in old_names]
    # print(new_names)
    df.columns = new_names
    return df


def merge_dfs(df_left, df_right, col, suffix):
    """
    merge two df based on col & suffix
    :param df_left: left df, original stock like `GME`
    :param df_right: right df, some indexes like `^DJI`
    :param col: column to join two dfs
    :param suffix: in this case we've set it to _dji or _(ticker)
    :return:
    """
    # print(pd.merge(left=df_left, right=df_right, left_on='t', right_on='t' + suffix).head())
    df_merged = pd.merge(left=df_left, right=df_right, left_on=col, right_on=col + suffix)
    return df_merged


def generate_time_fields(df_dji):
    df_dji['ts'] = pd.to_datetime(df_dji['t'], unit='s')
    df_dji['date'] = pd.to_datetime(df_dji['t'], unit='s').dt.date


def get_numerical_df(df):
    return df.select_dtypes(include=np.number).reindex()


def x_days_ago(x):
    current_time = calendar.timegm(time.gmtime())  # seconds since Unix epoch
    x_days_ago = current_time - days(x)
    return x_days_ago;


def days(x):
    return x * 24 * 60 * 60


def get_stock_data(stock_name, days, period, finnhub_api_key):
    '''
      Returns a dictionary of timestamps, closing prices and status response for the given stock symbol

        Parameters:
                stock_name (string): symbol of the stock
                days (int): Number of days to collect data for (going back from today)
                period (string): Options are by minute: ('1' , '5', '15', '30', '60') and by day/week/month ('D', 'W', 'M')

        Returns:
                c: List of close prices for returned candles.
                t: List of timestamp for returned candles.
                s: Status of the response, either 'ok' or 'no_data'.

    '''
    current_time = calendar.timegm(time.gmtime())  # seconds since Unix epoch
    finnhub_client = finnhub.Client(api_key=finnhub_api_key)
    res = finnhub_client.technical_indicator(symbol=stock_name, resolution=period, _from=x_days_ago(days),
                                             to=current_time, indicator='wma',
                                             indicator_fields={"timeperiod": 3})  # wma = weighted moving average

    if res['s'] == 'ok':
        return res
    else:
        return 'no data found for given parameters!'


def get_sentiment_data(stock_name):
    '''
    Returns company's news sentiment and statistics in a dictionary format.

      Parameters:
              stock_name (string): symbol of the stock

      Returns:
              buzz: Statistics of company news in the past week.
                    Includes articlesInLastWeek, buzz and weeklyAverage

              companyNewsScore:   News score.
              sectorAverageBullishPercent:  Sector average bullish percent.
              sectorAverageNewsScore:   Sectore average score.
              sentiment:  News sentiment. Includes bearishPercent and bullishPercent
              symbol:     company's symbol (redundant)
    '''

    finnhub_client = finnhub.Client(api_key="c10t49748v6o1us2neqg")
    return finnhub_client.news_sentiment(stock_name)


def get_day_of_week(date):
    """gets day of week from a timestamp object
    Params:
      data: pandas.TimeStamp object to convert
    Returns:
      day of week <int>: 0 for monday, 4 for friday
    """
    return date.dayofweek


def get_day_of_year(date):
    """gets day of year from a timestamp object
    Params:
      data: pandas.TimeStamp object to convert
    Returns:
      day of month <int>: 1 for the Jan 1st, 365 (genearlly) for Dec. 31st, etc.
    """
    return date.dayofyear


def is_quarter_end(date):
    """returns true if day is end of the quarter
    Params:
      data: pandas.TimeStamp object to convert
    Returns:
      True if date is quarter end, else False
    """
    return date.is_quarter_end


def get_calendar_features(df):
    df['day_of_week'] = df.apply(lambda x: get_day_of_week(x.ts), axis=1)
    df['day_of_year'] = df.apply(lambda x: get_day_of_year(x.ts), axis=1)
    df['is_quarter_end'] = df.apply(lambda x: is_quarter_end(x.ts), axis=1)
    ## TODO: Convert is_quarter_end with to_categorical

    return df


def get_moving_average(df, col='c', window=5):
    """Add moving average to df with moving average
    Weights are based on (1/n)/sum(weights) where n = distance from current prediction from 1/1 to 1/window.
    Eg., yesterday's moving average for a 5 day window is weighted as: 5/(5+4+3+2+1) = 0.33333
    Params:
      df: df to add col to
      col: column name to find average of
      window: how many time steps to consider
    Returns:
      Updated df with new column name = "<window>_wma" where window = number of time steps, eg., 100_wma
    """
    # TODO: Fix trailing and leading NaN
    # weights = np.arange(window, 0, -1)
    weights = np.arange(1, window + 1, 1)
    sum_weights = np.sum(weights)
    col_name = str(window) + "wma"

    df[col_name] = (
        df[col].rolling(window=window, center=False).apply(lambda x: np.sum(weights * x) / sum_weights, raw=False))
    return df


def get_vwap(row):
    """To be used in lambda function for row-wise calc
    """
    average_price = (row.h + row.l + row.c) / 3
    wvap = average_price / row.v
    return wvap


def add_vwap_col(df):
    """Add volume weighted average price to df
    Params:
      df: df to add vwap to. Expeting col names "h", "l", "c" for calc
    Returns
      df with new col named "vwap"
     """
    df['vwap'] = df.apply(lambda row: get_vwap(row), axis=1)
    return df


def get_high(df_new, days):
    for i in days:
        df_new['next_' + str(i) + '_high'] = df_new['h'].rolling(window=i).max().shift(-i).fillna(0)
        df_new['next_' + str(i) + '_low'] = df_new['l'].rolling(window=i).min().shift(-i).fillna(0)
    return df_new


def get_low(df_new, days):
    for i in days:
        df_new['last_' + str(i) + '_high'] = df_new['h'].rolling(window=i).max().shift(i).fillna(0)
        df_new['last_' + str(i) + '_low'] = df_new['l'].rolling(window=i).min().shift(i).fillna(0)
    return df_new


def get_enriched_stock_data(df, stock_name, days, period, finnhub_api_key):
    """
    combines old df with new data from api call and then merge them together

    :param df: old df
    :param stock_name: new ticker to call api
    :param days: same as df's days ago
    :param period: only supports date for now: D
    :param finnhub_api_key: api key to make calls
    :return:
    """
    # index_sym="^DJI"
    suffix = "_" + re.sub('[^A-Za-z0-9]+', '', stock_name.lower())
    df_dji = pd.DataFrame.from_dict(get_stock_data(stock_name, days, period, finnhub_api_key))
    generate_time_fields(df_dji)
    df_dji = rename_reference_df_column_names(df_dji, suffix)
    df_merged = merge_dfs(df, df_dji, 'date', suffix)
    return df_merged


finnhub_token = 'c1c3f6v48v6sp0s58o20'
stock_sym = 'GME'
days_ago = 250
start = '2020-12-01'
metrics_interested = ['next_1_high', 'next_1_low', 'next_3_high', 'next_3_low', 'next_7_high', 'next_7_low']

# get df from finnhub
df = pd.DataFrame.from_dict(get_stock_data(stock_sym, days_ago, 'D', finnhub_token))
generate_time_fields(df)
time.sleep(0.2)

# get df with added enriched data, right now only supports daily value
df = get_enriched_stock_data(df, "^DJI", days_ago, 'D', finnhub_token)
print(df)
print(df.head())

# plot something
# fig = get_candlestick_plot(df)
# fig.show()

# feature engineering, calendar and ma, vwap
df_proc = get_calendar_features(df)
df_proc = get_moving_average(df)
df_proc.fillna(method='backfill')
df_proc = add_vwap_col(df)

# feature engineering, get high and get low
days = [1, 3, 5, 7]
df = get_high(df, days)
df = get_low(df, days)
print(df.head)
print(df.shape)

# Stage prep data
fdir, fname = 'temp', 'interm_data.csv'
if not os.path.exists(fdir):
    os.mkdir(fdir)
df.to_csv(os.path.join(fdir, fname))

# df = pd.read_pickle('/kaggle/input/bse-stocks-data-15-minute-interval-historical/SIEMENS-15minute-Hist')
# df = pd.DataFrame(df)
# df['date'] = df['date'].apply(pd.to_datetime)
df.set_index('date', inplace=True)

print(df.columns)

df = df.rename(columns={"o": "open",
                        "h": "high",
                        "l": "low",
                        "c": "close",
                        "v": "volume"})

print(df.columns)


def plot_ohlc_table():
    global fig
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(['date', 'open', 'high', 'low', 'close', 'volume']),
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[df.index, df.open, df.high, df.low, df.close, df.volume],
                   fill_color='lavender',
                   align='left'))
    ])
    fig.show()


def OHLC_Line_Plots():
    global fig
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    from plotly.graph_objs import Line
    fig = make_subplots(rows=4, cols=1, subplot_titles=('Open', 'High', 'Low', 'Close'))
    fig.add_trace(
        Line(x=df.index, y=df.open),
        row=1, col=1
    )
    fig.add_trace(
        Line(x=df.index, y=df.high),
        row=2, col=1
    )
    fig.add_trace(
        Line(x=df.index, y=df.low),
        row=3, col=1
    )
    fig.add_trace(
        go.Line(x=df.index, y=df.close),
        row=4, col=1
    )
    fig.update_layout(height=1400, width=1000, title_text="OHLC Line Plots")
    fig.show()
    return go


def visualize_pattern():
    global fig
    # only first 5000 values are taken because it was looking very crowded
    result = seasonal_decompose(df.close.head(5000), model='additive', period=30)
    fig = go.Figure()
    fig = result.plot()
    fig.set_size_inches(20, 19)


def plot_candle_sticks():
    global fig
    #### We can see 5 different types of candlesticks below.
    """
    
        open_data = [33.0, 33.3, 33.5, 33.0, 34.1]
        high_data = [33.1, 33.3, 33.6, 33.2, 34.8]
        low_data = [32.7, 32.7, 32.8, 32.6, 32.8]
        close_data = [33.0, 32.9, 33.3, 33.1, 33.1]
        dates = [datetime(year=2013, month=10, day=10),
                 datetime(year=2013, month=11, day=10),
                 datetime(year=2013, month=12, day=10),
                 datetime(year=2014, month=1, day=10),
                 datetime(year=2014, month=2, day=10)]
    
        fig = go.Figure(data=[go.Candlestick(x=dates,
                               open=open_data, high=high_data,
                               low=low_data, close=close_data,
                       increasing_line_color= 'green', decreasing_line_color= 'red')])
    
        fig.show()
    
        """  # Candlestick chart for Siemens"""
    import plotly.graph_objects as go
    import pandas as pd
    from datetime import datetime
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                                         open=df['open'],
                                         high=df['high'],
                                         low=df['low'],
                                         close=df['close'])])
    fig.show()
    return go, pd


def visualize_result():
    global fig
    print('The Mean Squared Error is',
          mean_squared_error(valid_df[metric].values, valid_df[metric + '_predictions'].values))
    """# Let's have a closer look"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=valid_df.index, y=valid_df[metric],
                             mode='lines',
                             name='Test'))
    fig.add_trace(go.Scatter(x=valid_df.index, y=valid_df[metric + '_predictions'],
                             mode='lines',
                             name='Predicted'))
    """### From this visualization we can conclude that LSTM worked well. It can be improved though!"""
    grads1 = get_gradients(lstm_model, 1, x_train_data, y_train_data)
    grads2 = get_gradients(lstm_model, 2, x_train_data, y_train_data)
    output = get_outputs(lstm_model, 1, x_train_data)
    """# A Visual Exploration of LSTM's
    
    
    ## Layer 1 Visualization
    Layer1 plots are as follows
    1. 1D with 10 gradients
    2. 1D with 500 gradients
    3. 2D with 500 gradients
    3. 1D with all gradients
    """
    features_1D(grads1[:10], n_rows=2)
    features_1D(grads1[:500], n_rows=2)
    features_2D(grads1[:500])
    features_1D(grads1, n_rows=2)
    """## Layer2 Visualization
    Layer2 plots are as follows
    1. 1D with 500 gradients
    2. 2D with 500 gradients
    3. 1D with all gradients
    4. 2D with all gradients
    """
    features_1D(grads2[:500], n_rows=2)
    features_2D(grads2[:500], n_rows=2)
    features_1D(grads2, n_rows=2)
    """## Output Layer Visualization"""
    features_1D(output, n_rows=2)
    features_2D(output, n_rows=2)
    """## LSTM Histograms"""
    rnn_histogram(lstm_model, 'lstm', equate_axes=False)
    """## HEATMAP"""
    rnn_heatmap(lstm_model, 'lstm')

result_df=pd.DataFrame()

for metric in metrics_interested:
    """## Dataset"""
    # plot_ohlc()

    """### So basically this dataset contains 6 different features i.e. date, open, high, low, close, volume
    
    ### Date - This contains date + time at the instant of trade
    
    ### Open - Open is the price when the stock began
    
    ### High - Maximum price at the given time period
    
    ### Low - Minimum price at the given time period
    
    ### Close - Price at which stock ended
    
    ### Volume - It is the total amount of trading activity
    
    ### Incase of our data the time period is 15 minutes
    """

    # go = OHLC_Line_Plots()

    """## Visualizing Patterns in the Data"""

    # visualize_pattern()

    # """## Candlestick

    # * Candlestick charts are used by traders to determine possible price movement based on past patterns.

    # * Candlesticks are useful when trading as they show four price points (open, close, high, and low) throughout the period of time the trader specifies.

    # * Many algorithms are based on the same price information shown in candlestick charts.

    # * Trading is often dictated by emotion, which can be read in candlestick charts.

    # <img src="https://alpari.com/storage/inline-images/Forex%20candlestick%20patterns%20and%20how%20to%20use%20them%20-%201_0.png">

    # ## Sample Candlesticks

    # go, pd = plot_candle_sticks()

    """# Creating Train Test Data"""

    new_df = pd.DataFrame()
    new_df = df[metric]
    new_df.index = df.index

    scaler = MinMaxScaler(feature_range=(0, 1))
    final_dataset = new_df.values

    test_train_split = int(final_dataset.shape[0] -3 )

    train_data = final_dataset[0:test_train_split, ]
    valid_data = final_dataset[test_train_split:, ]

    train_df = pd.DataFrame()
    valid_df = pd.DataFrame()
    train_df[metric] = train_data
    train_df.index = new_df[0:test_train_split].index
    valid_df[metric] = valid_data
    valid_df.index = new_df[test_train_split:].index

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(final_dataset.reshape(-1, 1))

    x_train_data, y_train_data = [], []

    for i in range(60, len(train_data)):
        x_train_data.append(scaled_data[i - 60:i, 0])
        y_train_data.append(scaled_data[i, 0])

    x_train_data, y_train_data = np.array(x_train_data), np.array(y_train_data)

    x_train_data = np.reshape(x_train_data, (x_train_data.shape[0], x_train_data.shape[1], 1))

    """# Long Short Term Memory Networks(LSTM)
    
    Do you think about everything from scratch. No. You perform the actions based on your past memory. For example if you are reading a newspaper, you understand words because in your past you have read them and they are stored in your memory. If you encounter a new word then it gets stored in your memory newly. So the question is Do you want your model to process everything from scratch? Or you want to make it more intelligent by creating a memory space. Thats when LSTM comes into the game. LSTM which is long short term memory is the type of RNN which can hold memory for longer period of time. They are a good fit for time series preditiction, or forecasting problems.
    
    <img src="https://miro.medium.com/max/3000/1*laH0_xXEkFE0lKJu54gkFQ.png">
    
    Image Taken From [Here](https://medium.com/mlreview/understanding-lstm-and-its-diagrams-37e2f46f1714)
    """

    lstm_model = Sequential()
    lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train_data.shape[1], 1)))
    lstm_model.add(LSTM(units=50))
    lstm_model.add(Dense(1))

    inputs_data = new_df[len(new_df) - len(valid_data) - 60:].values
    inputs_data = inputs_data.reshape(-1, 1)
    inputs_data = scaler.transform(inputs_data)

    lstm_model.compile(loss='mean_squared_error', optimizer='adam')
    lstm_model.fit(x_train_data, y_train_data, epochs=30, batch_size=7, verbose=1)

    lstm_model.summary()

    X_test = []
    for i in range(60, inputs_data.shape[0]):
        X_test.append(inputs_data[i - 60:i, 0])
    X_test = np.array(X_test)

    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_metric = lstm_model.predict(X_test)
    predicted_metric = scaler.inverse_transform(predicted_metric)

    valid_df[metric+'_predictions'] = predicted_metric

    """# Predictions"""

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_df.index, y=train_df[metric],
                             mode='lines',
                             name='Siemens Train Data'))
    fig.add_trace(go.Scatter(x=valid_df.index, y=valid_df[metric],
                             mode='lines',
                             name='Siemens Valid Data'))
    fig.add_trace(go.Scatter(x=valid_df.index, y=valid_df[metric+'_predictions'],
                             mode='lines',
                             name='Prediction'))

    from sklearn.metrics import mean_squared_error
    from see_rnn import *

    # visualize_result()

    print(valid_df[metric+'_predictions'])
    result_df[stock_sym+'_'+metric+'_predictions']= valid_df[metric+'_predictions']

print(result_df)


# grid search for this is needed https://machinelearningmastery.com/how-to-grid-search-deep-learning-models-for-time-series-forecasting/
from datetime import datetime
# sudo apt-get install python3-tk
import matplotlib.pyplot as plt
import numpy as np
from pandas_datareader import DataReader

stock_sym='GME'
days_ago=120
start='2020-12-01'
# the next line can be High, Low, Open, Close, Volume, Adj Close
metrics_interested=['next_3_high', 'next_3_low']


# Get the stock quote
df = DataReader(stock_sym, data_source='yahoo', start=start, end=datetime.now())
# Show teh data
print(df.head())


# import code that's from deeptendies project

import calendar
import time
import finnhub
import pandas as pd
import plotly.graph_objects as go


def x_days_ago(x):
    current_time = calendar.timegm(time.gmtime())  # seconds since Unix epoch
    x_days_ago = current_time - days(x)
    return x_days_ago;


def days(x):
    return x * 24 * 60 * 60


def get_stock_data(stock_name, days, period):
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
    finnhub_client = finnhub.Client(api_key="c10t49748v6o1us2neqg")
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

# Get Daily Stock Data for AAPL for the last 30 days
df = pd.DataFrame.from_dict(get_stock_data(stock_sym, days_ago, 'D'))
df['t'] = pd.to_datetime(df['t'], unit = 's')
# dropping entries where weighted moving average is 0
# df = df.drop(df[df.wma==0].index)
df

def get_candlestick_plot(df):
  """Gets candle stick plot from finnhub dataframe assuming typical column header names
  Params:
    df to plot from finnhub
  Returns:
    plotly graph objects figure
  """
  df['t'] = pd.to_datetime(df['t'], unit = 's')
  fig = go.Figure(data=[go.Candlestick(x=df['t'],
  open=df['o'],
  high=df['h'],
  low=df['l'],
  close=df['c'])])
  return fig

# TODO: save fig and move from local env to mounted drive
fig = get_candlestick_plot(df)
fig.show()


def get_line_plot(df, title = "Price vs. Date", x_step = 5):
  """Gets line plot for a standard finnhub df
  Params:
    df: df to plot
    title: name of plot
    x_step: number of time steps to print on x axis (ie., x_steps per tick). Note that buisness days only ploted!
  Returns:
    plt.fig instance
  """
  fig, ax = plt.subplots(figsize=(24,18))
  fig = plt.plot(range(df.shape[0]),(df['l']+df['h'])/2.0)
  plt.xticks(range(0,df.shape[0],x_step),df['t'].loc[::x_step],rotation=45)
  plt.xlabel('Date',fontsize=18)
  plt.ylabel('Mid Price',fontsize=18)
  plt.title(title)
  return fig

# fig = get_line_plot(df, title="Apple Price vs. Date")
# plt.draw()




def plt_visual_raw(stock_sym, metric_interested, df):
    plt.figure(figsize=(16, 8))
    plt.title(stock_sym + " " + metric_interested + ' Price History')
    plt.plot(df[metric_interested])
    plt.xlabel('Date', fontsize=18)
    plt.ylabel(stock_sym + " " + metric_interested + ' Price USD ($)', fontsize=18)
    plt.show()


"""Date features"""


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
    df['day_of_week'] = df.apply(lambda x: get_day_of_week(x.t), axis=1)
    df['day_of_year'] = df.apply(lambda x: get_day_of_year(x.t), axis=1)
    df['is_quarter_end'] = df.apply(lambda x: is_quarter_end(x.t), axis=1)
    ## TODO: Convert is_quarter_end with to_categorical

    return df


df_test = get_calendar_features(df)
df_test

"""Weighted Moving Average"""

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
  weights = np.arange(1, window+1, 1)
  sum_weights = np.sum(weights)
  col_name = str(window) + "wma"

  df[col_name] = (df[col].rolling(window=window, center=False).apply(lambda x: np.sum(weights*x) / sum_weights, raw=False))
  return df

df_test = get_moving_average(df)
df_test

df_test.fillna(method='backfill')

""" Volume weighted average price
See: https://www.investopedia.com/terms/v/vwap.asp#:~:text=The%20volume%20weighted%20average%20price%20(VWAP)%20is%20a%20trading%20benchmark,and%20value%20of%20a%20security.
"""

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

df_test = add_vwap_col(df)
df_test

days=[1,3,5,7]

def get_high(df_new, days):
    for i in days:
        df_new['next_'+str(i)+'_high']=df_new['h'].rolling(window=i).max().shift(-i).fillna(0)
        df_new['next_'+str(i)+'_low']=df_new['l'].rolling(window=i).min().shift(-i).fillna(0)


def get_low(df_new, days):
    for i in days:
        df_new['last_'+str(i)+'_high']=df_new['h'].rolling(window=i).max().shift(i).fillna(0)
        df_new['last_'+str(i)+'_low']=df_new['l'].rolling(window=i).min().shift(i).fillna(0)


def get_exclude_column_names(days):
    exclude_column_names=[]
    for i in days:
        exclude_column_names.append('next_'+str(i)+'_high')
    return exclude_column_names



get_high(df, days)
get_low(df, days)


# print(df.head)
# print(df.shape)







for metric_interested in metrics_interested:
    # metric_interested = 'next_3_low'

    df[df[metric_interested].eq(0)] = np.nan





    # plt_visual_raw(stock_sym, metric_interested, df)

    # Create a new dataframe with only the 'Close column
    data = df.filter([metric_interested])
    # Convert the dataframe to a numpy array
    dataset = data.values
    # Get the number of rows to train the model on
    training_data_len = int(np.ceil( len(dataset) * .95 ))

    # print("training_data_len: %s" %training_data_len )


    #scaling
    # Scale the data
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)
    # scaled_data
    # Create the training data set
    # Create the scaled training data set
    train_data = scaled_data[0:int(training_data_len), :]
    # Split the data into x_train and y_train data sets
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60:i, 0])
        y_train.append(train_data[i, 0])
        if i <= 61:
            print(x_train)
            print(y_train)
            print()

    # Convert the x_train and y_train to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape the data
    x_train = np.atleast_2d(x_train) # experimenting to solve the tuple index out of range issue

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    # print("x_train.shape:")
    # print(x_train.shape)


    #LSTM
    from keras.models import Sequential
    from keras.layers import Dense, LSTM, Masking

    # Build the LSTM model
    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=(x_train.shape[1], 1))) # handle nans https://stackoverflow.com/questions/52570199/multivariate-lstm-with-missing-values
    model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(x_train, y_train, batch_size=1, epochs=20)



    # Test
    # Create the testing data set
    # Create a new array containing scaled values from index 1543 to 2002

    # training and validating
    test_data = scaled_data[training_data_len - 60:, :]

    # Create the data sets x_test and y_test
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])

    # Convert the data to a numpy array
    x_test = np.array(x_test)

    # Reshape the data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Get the models predicted price values
    predictions = model.predict(x_test)

    print(predictions)

    predictions = scaler.inverse_transform(predictions)

    print(predictions)


    # Get the root mean squared error (RMSE)
    rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
    print("rmse %s" %rmse)




    ## Plot the data Again
    # Plot the data
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions


    def plot_predicted():
        # Visualize the data
        plt.figure(figsize=(16, 8))
        plt.title(metric_interested + ' Model')
        plt.xlabel('Date', fontsize=18)
        plt.ylabel(metric_interested + ' Price USD ($)', fontsize=18)
        plt.plot(train[metric_interested])
        plt.plot(valid[[metric_interested, 'Predictions']])
        plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
        plt.show()


    # plot_predicted()

    print(valid)
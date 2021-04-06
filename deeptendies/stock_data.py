"""Class to query finnhub API, preprocess data, feature engineer, and store relevent attributes

Part of the deeptendies stock sequence modelling package.

  Typical usage: 
  foo = StockData("GME")
  df = foo.get_df()
  df = foo.engineer_features()
  df = foo.get_cleaned_data()
"""
import re
import finnhub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf 
from sklearn.preprocessing import MinMaxScaler
import calendar
from datetime import datetime, timedelta, timezone
import time
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
import yaml 
import os
from deeptendies.window_norm_timeseries_generator import WindowNormTimeseriesGenerator

# from deeptendies.utils import generate_time_fields, rename_reference_df_column_names, merge_dfs


class StockData(): 
  """Class for storing data on a given ticker. 

  Combines api call, preprocessing, and feature engineering into one, deep fucking value package. 
  """

  def __init__(self, ticker, days=7300, period='D', api_key=None): 
    """Instantiates new StockData object.
    
      Args: 
        stock_name (string): symbol of the stock
        days (int): Number of days to collect data for (going back from today)
        period (string): Options are by minute: ('1' , '5', '15', '30', '60') and by day/week/month ('D', 'W', 'M')
    """
    self.ticker = ticker
    if api_key == None: 
      self.api_key = self.get_api_key("secrets.yaml")
    else: 
      self.api_key = api_key
    self.df = self.get_stock_data(ticker, days, period)

  def get_df(self): 
    return self.df

  def drop_cols(self, cols=[None]): 
    """Drop requested cols from self.df
    """
    for col in cols: 
      self.df = self.df.drop(columns=cols)

  def get_cleaned_data(self,  categorical_cols=["is_quarter_end"], index_col="t", drop_cols=['s', 'wma'], imputation_strategy='drop'):
    """Converts dataframe to np.arr tensor ready to use in tf functions
    Params: 
      df: Pandas df to convert, expecting typical finnhub formated col names
      categorical_cols: column names to convert to categorical data
      index_col: column name to reindex to
      drop_cols:  column names to drop. 
      imputation_strategy: method  for imputation {‘backfill’, ‘bfill’, ‘pad’, ‘ffill’, drop}. drop means drop the rows. 
    Returns:
      numpy array 
    """
    df = self.df
    df = df.drop(columns=drop_cols)
    df = df.set_index(keys=index_col)
    for col in categorical_cols: 
      # NOTE: this should be a applymap for speed, but this is fine for short amounts of columns
      df[col] = df[col].astype('int32')
    if imputation_strategy == 'drop': 
      df=df.dropna()
    elif imputation_strategy==None: 
      return df
    else: 
      df = df.fillna(method=imputation_strategy)
      ## TODO: Split into X,y?
    return df


  def reorder_cols(self, cols=[]):
    self.df = self.df.loc[:, cols]
    return self.df

  def engineer_features(self, windows=[100, 50, 20], days=[1, 3, 5, 7]): 
    """Perform all feature engineering steps
      Args: 
        windows <list <int>>: windows to consider for mwa and mvwap
        days <list <int>>: Days to consider for high/low prices. 
    """
    self.df = self.get_calendar_features(self.df)
    for window in windows: 
      self.df = self.get_moving_average(self.df, col='c', window=window)
      self.df = self.add_mvwap_col(self.df, window=window)
    self.df = self.get_high(self.df, days)
    self.df = self.get_low(self.df, days)

  def get_api_key(self, fname): 
    """OS agnostic api key fetcher. 
    Place secrets.yaml in outer dir
    """
    fpath = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", fname))
    with open(fpath) as f: 
        credentials = yaml.safe_load(f)
        return credentials['finnhub-apikey']
    
  def x_days_ago(self, x):
      current_time = calendar.timegm(time.gmtime())  # seconds since Unix epoch
      x_days_ago = current_time - self.days(x)
      return x_days_ago

  def days(self, x):
      return x * 24 * 60 * 60


  def get_stock_data(self, stock_name, days, period):
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

      ## TODO: Delete imported WMA as we calculate manually below?
      current_time = calendar.timegm(time.gmtime())  # seconds since Unix epoch
      finnhub_client = finnhub.Client(api_key=self.api_key)
      res = finnhub_client.technical_indicator(symbol=stock_name, resolution=period, _from=self.x_days_ago(days),
                                              to=current_time, indicator='wma',
                                              indicator_fields={"timeperiod": 3})  # wma = weighted moving average
      if res['s'] == 'ok':
        df = pd.DataFrame.from_dict(res)
        df['t'] = pd.to_datetime(df['t'], unit='s')
        return df
      else:
          raise Exception(f'No data found for ticker = {stock_name}, days = {days}, period = {period}! Aborting StockData init')


  def get_sentiment_data(self, stock_name):
    ## TODO: Not required?
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


  def get_day_of_week(self, date):
      """gets day of week from a timestamp object
      Params:
        data: pandas.TimeStamp object to convert
      Returns:
        day of week <int>: 0 for monday, 4 for friday
      """
      return date.dayofweek


  def get_day_of_year(self, date):
      """gets day of year from a timestamp object
      Params:
        data: pandas.TimeStamp object to convert
      Returns:
        day of month <int>: 1 for the Jan 1st, 365 (genearlly) for Dec. 31st, etc.
      """
      return date.dayofyear


  def is_quarter_end(self, date):
      """returns true if day is end of the quarter
      Params:
        data: pandas.TimeStamp object to convert
      Returns:
        True if date is quarter end, else False
      """
      return date.is_quarter_end


  def get_calendar_features(self, df):
      df['day_of_week'] = df.apply(lambda x: self.get_day_of_week(x.t), axis=1)
      df['day_of_year'] = df.apply(lambda x: self.get_day_of_year(x.t), axis=1)
      df['is_quarter_end'] = df.apply(lambda x: self.is_quarter_end(x.t), axis=1)
      # TODO: Convert is_quarter_end with to_categorical?

      return df


  def get_moving_average(self, df, col='c', window=5):
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
    # TODO: Fix trailing NaN
    weights = np.arange(1, window+1, 1)
    sum_weights = np.sum(weights)
    col_name = str(window) + "wma"

    df[col_name] = (df[col].rolling(window=window, center=False).apply(lambda x: np.sum(weights*x) / sum_weights, raw=False))
    return df


  def add_mvwap_col(self, df, window=100): 
    """Add moving volume weighted average price to df between window periods 
    Params: 
      df: df to add vwap to. Expecting col names "h", "l", "c" for calc
      window <int>: window to consider (days)
    Returns 
      df with new col named "vwap"
    """
    avgPriceVol = df.apply(lambda x: (x.c + x.h + x.l) * x.v/3, axis=1)
    col_name = str(window) + "mvwap"
    df[col_name] = avgPriceVol.rolling(window, center=False, min_periods=1).sum() / df.v.rolling(window, center=False, min_periods=1).sum()
    return df


  def get_high(self, df_new, days):
      for i in days:
          df_new['next_'+str(i)+'_high']=df_new['h'].rolling(window=i).max().shift(-i).fillna(np.nan) #TODO: Try out other imputation methods?
          df_new['next_'+str(i)+'_low']=df_new['l'].rolling(window=i).min().shift(-i).fillna(np.nan) 
      return df_new


  def get_low(self, df_new, days):
      for i in days:
          df_new['last_'+str(i)+'_high']=df_new['h'].rolling(window=i).max().shift(i).fillna(np.nan)
          df_new['last_'+str(i)+'_low']=df_new['l'].rolling(window=i).min().shift(i).fillna(np.nan)
      return df_new


  def get_enriched_stock_data(self, df, stock_name, days, period, finnhub_api_key):
    # TODO: Incorporate into class. Talk to stan about goals here (reimport, re-feature engineer?) Not tested. 
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
      suffix="_"+re.sub('[^A-Za-z0-9]+', '', stock_name.lower())
      df_dji = pd.DataFrame.from_dict(self.get_stock_data(stock_name, days, period, finnhub_api_key))
      generate_time_fields(df_dji)
      df_dji = rename_reference_df_column_names(df_dji, suffix)
      df_merged = merge_dfs(df, df_dji, 'date', suffix)
      return df_merged

  def get_closing_correlation_heatmap(self, tickers=None, days=365, period = 'D'): 
    dfs= {}
    for ticker in tickers: 
      dfs[ticker] = pd.DataFrame.from_dict(self.get_stock_data(ticker, days, period))
    merged_df = self.df.c.to_frame()
    for ticker in dfs: 
      merged_df = merged_df.join(dfs[ticker].c, lsuffix="-" + ticker )
    merged_df = merged_df.rename(columns={"c": "c-" + self.ticker})
    static_corrs = merged_df.corr(method='spearman')
    ax = sns.heatmap(static_corrs)
    return ax


  def get_scaled_df(self, df, test_percentage=0.3, ignore_cols=['day_of_week', 'day_of_year', 'is_quarter_end'], target_col = 'c'):
      train_idx, _  = StockData.get_train_test_split(df, test_percentage)
      scaler = MinMaxScaler()
      target_scaler = MinMaxScaler()
      cols = list(set(df.columns) - set(ignore_cols))
      train_df = df[:train_idx]
      target_scaler.fit(train_df[target_col].to_numpy().reshape(-1,1))
      scaler.fit(train_df[cols])
      df[cols] = scaler.transform(df[cols])
      self.scaler = scaler
      self.target_scaler = target_scaler
      return df

  def inverse_transform_target_vector(self, y): 
      if type(y) == pd.core.series.Series: 
          y = y.to_numpy()
      if len(y.shape) == 0 : 
          y = y.reshape(1,-1)
      elif len(y.shape) == 1: 
          y = y.reshape(-1, 1)
      return self.target_scaler.inverse_transform(y)
    
  @staticmethod
  def get_train_test_split(df, test_percentage=0.3): 
    """Helper to get test_train percentages
    """
    train_idx = np.int(len(df)*(1-test_percentage))
    return train_idx, len(df) - train_idx

  @staticmethod
  def get_timeseries_generators(df, test_percentage=0.3, target_col="c", length = 100, batch_size=1, windowed_norm = False, min_max_scaler=False): 
    """Get train/test generators

      Similar to: https://jackdry.com/using-an-lstm-based-model-to-predict-stock-returns

      Args: 
        df = df to get generators for, typically self.df
        test_percentage = percentage of set to hold back 
        target_col = name of column for target
        length = rolling window length to consider (eg., 100 days)
        batch_size = batch size for generator
        windowed_norm = if True, return generators with window wise normalization
        min_max_scaler = if true && window_norm, return min_max scaled window wise normalization. If false, return standard scaler normalization. 
      
      Returns: 
        (trainGen, testGen) = tf.keras.preprocessing.sequence.TimeseriesGenerator objects, one for training, one for testing. 
    """
    train_idx, test_idx = StockData.get_train_test_split(df, test_percentage)
    if windowed_norm: 
      trainGen = WindowNormTimeseriesGenerator(
        df.values, 
        df[target_col].values,
        length=length, 
        batch_size=batch_size,
        start_index =0,
        end_index = train_idx-1
      )

      testGen = WindowNormTimeseriesGenerator(
          df.values, 
          df[target_col].values,
          length=length, 
          batch_size=batch_size,
          start_index =train_idx
      )
    else: 
      trainGen = tf.keras.preprocessing.sequence.TimeseriesGenerator(
          df.values, 
          df[target_col].values,
          length=length, 
          batch_size=batch_size,
          start_index =0,
          end_index = train_idx-1
      )
      testGen = tf.keras.preprocessing.sequence.TimeseriesGenerator(
          df.values, 
          df[target_col].values,
          length=length, 
          batch_size=batch_size,
          start_index =train_idx
      )
    return trainGen, testGen

  def get_line_plot(self, df=None, title = "Closing Price vs. Date", x_step = 365, plot_features=True):
      """Gets line plot for a standard finnhub df
      Params: 
      df: df to plot
      title: name of plot
      x_step: number of time steps to print on x axis (ie., x_steps per tick). Note that buisness days only ploted! 
      plot_features <bool>: true if you want to see the mva and mvwap overlaid
      Returns:
      plt.fig instance
      """
      if df == None: 
        df = self.df
      fig, ax = plt.subplots(figsize=(24,18))
      ax.plot(range(df.shape[0]),(df['c']), linewidth=5.0, label="Close", c='black')
      if plot_features: 
          features = ['100wma', '100mvwap', '50wma', '50mvwap', '20wma', '20mvwap']
          colors = ['firebrick', 'navy', 'red', 'blue', 'salmon', 'cornflowerblue']
          for feature, color in zip(features, colors): 
              ax.plot(range(df.shape[0]), df[feature].values, label=feature, c=color)
      plt.xticks(range(0,df.shape[0],x_step),df.index[::x_step],rotation=45)
      plt.xlabel('Date',fontsize=24)
      plt.ylabel('Mid Price',fontsize=24)
      plt.title(title, fontsize=36)
      leg = plt.legend()
      return fig



if __name__ == '__main__':
  test = StockData('GME') 
  test.engineer_features()
  test.get_cleaned_data()
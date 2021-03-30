"""Class to query finnhub API, preprocess data, feature engineer, and store relevent attributes

Part of the deeptendies stock sequence modelling package.

  Typical usage: 
  foo = StockData("GME")
  df = foo.getDf()
  etc
"""
import re
import finnhub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

# from deeptendies.utils import generate_time_fields, rename_reference_df_column_names, merge_dfs


class StockData(): 
  """Class for storing data on a given ticker. 

  Combines api call, preprocessing, and feature engineering into one, deep fucking value package. 
  """

  def __init__(self, ticker, days=7300, period='D'): 
    """Instantiates new StockData object.
    
      Args: 
        stock_name (string): symbol of the stock
        days (int): Number of days to collect data for (going back from today)
        period (string): Options are by minute: ('1' , '5', '15', '30', '60') and by day/week/month ('D', 'W', 'M')
    """
    self.api_key = self.get_api_key("secrets.yaml")
    self.df = self.get_stock_data(ticker, days, period)
    self.df = self.get_calendar_features(self.df)
    

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
          raise Exception(f'No data found for ticker {stock_name}, days {days}, period {period}! Aborting StockData init')


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
      # print(df.iloc[0])
      # print(self.get_day_of_week())
      df['day_of_week'] = df.apply(lambda x: self.get_day_of_week(x.t), axis=1)
      df['day_of_year'] = df.apply(lambda x: self.get_day_of_year(x.t), axis=1)
      df['is_quarter_end'] = df.apply(lambda x: self.is_quarter_end(x.t), axis=1)
      # TODO: Convert is_quarter_end with to_categorical

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
    # TODO: Fix trailing and leading NaN
    # weights = np.arange(window, 0, -1)
    weights = np.arange(1, window+1, 1)
    sum_weights = np.sum(weights)
    col_name = str(window) + "wma"

    df[col_name] = (df[col].rolling(window=window, center=False).apply(lambda x: np.sum(weights*x) / sum_weights, raw=False))
    return df


  def get_vwap(self, row):
    """To be used in lambda function for row-wise calc
    """
    average_price = (row.h + row.l + row.c) / 3
    wvap = average_price / row.v
    return wvap


  def add_vwap_col(self, df):
    """Add volume weighted average price to df
    Params:
      df: df to add vwap to. Expeting col names "h", "l", "c" for calc
    Returns
      df with new col named "vwap"
    """
    df['vwap'] = df.apply(lambda row: get_vwap(row), axis=1)
    return df


  def get_high(self, df_new, days):
      for i in days:
          df_new['next_'+str(i)+'_high']=df_new['h'].rolling(window=i).max().shift(-i).fillna(0)
          df_new['next_'+str(i)+'_low']=df_new['l'].rolling(window=i).min().shift(-i).fillna(0)
      return df_new


  def get_low(self, df_new, days):
      for i in days:
          df_new['last_'+str(i)+'_high']=df_new['h'].rolling(window=i).max().shift(i).fillna(0)
          df_new['last_'+str(i)+'_low']=df_new['l'].rolling(window=i).min().shift(i).fillna(0)
      return df_new


  def get_enriched_stock_data(self, df, stock_name, days, period, finnhub_api_key):
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
      df_dji = pd.DataFrame.from_dict(get_stock_data(stock_name, days, period, finnhub_api_key))
      generate_time_fields(df_dji)
      df_dji = rename_reference_df_column_names(df_dji, suffix)
      df_merged = merge_dfs(df, df_dji, 'date', suffix)
      return df_merged


if __name__ == '__main__':
  test = StockData('GME') 
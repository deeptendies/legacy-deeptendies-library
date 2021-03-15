import calendar
import time
import finnhub
import pandas as pd
import plotly.graph_objects as go

def x_days_ago(x):
    x_days_ago = current_time - days(x)
    return x_days_ago;

def days(x):
    return x*24*60*60

finnhub_client = finnhub.Client(api_key="c10t49748v6o1us2neqg")
current_time = calendar.timegm(time.gmtime()) #seconds since Unix epoch
# Stock candles Days
days_30 = finnhub_client.stock_candles('JPM', 'D', x_days_ago(30), current_time)
# Stock candles Minutes
days_1_min = finnhub_client.stock_candles('JPM', '1', x_days_ago(1), current_time)
# Really good stuffs
rsi_window_size = 3
advanced = finnhub_client.technical_indicator(symbol="AAPL", resolution='D', _from=x_days_ago(30), to=current_time, indicator='wma')
print(advanced)
print(finnhub_client.news_sentiment('AAPL'))
print(finnhub_client.aggregate_indicator('AAPL', 'W'))
df = pd.DataFrame.from_dict(advanced)
df['t'] = pd.to_datetime(df['t'], unit = 's')

print(df.head())

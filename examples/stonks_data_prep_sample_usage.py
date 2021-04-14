# sudo apt-get install python3-tk
import os
import yaml
from deeptendies.stonks import *
import pandas as pd
# base configs
from deeptendies.stonks import get_enriched_stock_data
from deeptendies.utils import generate_time_fields


# just an example, use generated key from https://finnhub.io/dashboard
# finnhub_token = "c1c318v48v6sp0s58ffg"

credentials='/home/stan/github/mltrade/secrets.yaml'
# load secrets from yaml example:
with open(credentials) as credentials:
    credentials = yaml.safe_load(credentials)
    finnhub_token=credentials['finnhub-apikey']

stock_sym='GME'
days_ago=250
start='2020-12-01'
metrics_interested=['next_3_high', 'next_3_low']


# get df from finnhub
df = pd.DataFrame.from_dict(get_stock_data(stock_sym, days_ago, 'D', finnhub_token))
generate_time_fields(df)
time.sleep(0.2)

# get df with added enriched data, right now only supports daily value
df= get_enriched_stock_data(df, "^DJI", days_ago, 'D', finnhub_token)
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
days=[1,3,5,7]
df = get_high(df, days)
df = get_low(df, days)
print(df.head)
print(df.shape)


# Stage prep data
fdir, fname='temp','interm_data.csv'
if not os.path.exists(fdir):
    os.mkdir(fdir)
df.to_csv(os.path.join(fdir, fname))

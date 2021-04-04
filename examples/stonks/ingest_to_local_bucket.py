import os
import pathlib
from datetime import date

from deeptendies.stock_data import StockData

today = date.today()

# get api keys from here https://finnhub.io/dashboard
# if dev or deployment, save a `secrets.yaml` in work dir
# secrets.yaml example: https://raw.githubusercontent.com/deeptendies/deeptendies/master/secrets.yaml.example
ticker = 'GME'
days = 180

stonk = StockData(ticker, days=days, api_key="c1h8m1n48v6t9ghtpkh0")
stonk.df


def df_to_filesys_operator(df, path, fname):
    if not os.path.exists(path):
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    df.to_csv(os.path.join(path, fname))
    return


bucket = 'bucket=filesys'
topic = 'topic=' + ticker
processed_at = 'processed_at=' + today.strftime("%Y-%m-%d")
version = 'raw'
version = 'version=' + version
filename = ticker + "_" + str(days) + '.csv'

# raw
path = os.path.join(
    bucket,
    topic,
    version,
    processed_at
)

df_to_filesys_operator(stonk.df,
                       path,
                       filename)

version = 'feature_engineered'
version = 'version=' + version
# engineer features
stonk.engineer_features()
path = os.path.join(
    bucket,
    topic,
    version,
    processed_at
)
df_to_filesys_operator(stonk.df,
                       path,
                       filename)

# clean dataframe
drop_cols = ['next_1_high', 'next_1_low', 'next_3_high', 'next_3_low', 'next_5_high',
             'next_5_low', 'next_7_high', 'next_7_low', 'last_1_high', 'last_1_low',
             'last_3_high', 'last_3_low', 'last_5_high', 'last_5_low', 'last_7_high',
             'last_7_low', "s", "wma"]
df = stonk.get_cleaned_data(drop_cols=drop_cols)

version = 'cleaned'
version = 'version=' + version
path = os.path.join(
    bucket,
    topic,
    version,
    processed_at
)
df_to_filesys_operator(df,
                       path,
                       filename)

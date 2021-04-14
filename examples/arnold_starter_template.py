import pandas as pd

# to get started developing your own arnold, clone this starter file, rename
# it as a new file, following mike's naming convention arnold_<model name>_<extra tag>.py
file = '../bucket=fs/topic=AAPL/version=demo/processed_at=2021-04-10/AAPL_2011-04-01_to_2021-04-01.csv'
# ,High,Low,Open,Close,Volume,Adj Close
df = pd.read_csv(file, header=0)

yahoo_to_finnhub_header_map = {"Date": 't',
                               "High": 'h',
                               "Low": 'l',
                               "Close": 'c',
                               "Open": 'o',
                               "Volume": 'v'}  # adj close not used in finnhub
df = df.rename(columns=yahoo_to_finnhub_header_map).set_index('t')
print(df.head())

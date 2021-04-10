import os
import pathlib
import random

import numpy as np
from datetime import date
from pandas_datareader import data
from deeptendies.utils.local_bucket import save_data

# ingest data for final report

today = date.today()
start_date = '2011-04-01'
end_date = '2021-04-01'

stonks = ["TSLA", "AAPL", "MSFT", "BA", "KO"]



yahoo_to_finnhub_header_map = {"Date": 't',
                               "High": 'h',
                               "Low": 'l',
                               "Close": 'c',
                               "Open": 'o',
                               "Volume": 'v'}

# load data with yahoo / data reader
for ticker in stonks:
    print(ticker, start_date, end_date)
    candle_stick_data = data.DataReader(ticker, 'yahoo', start_date, end_date)
    save_data(dataframe=candle_stick_data,
              bucket='fs',
              topic=ticker,
              version='yahoo',
              suffix=f"{start_date}_to_{end_date}",
              path="/home/stan/github/deeptendies")

    candle_stick_data = candle_stick_data.reset_index().rename(columns=yahoo_to_finnhub_header_map)

    print(candle_stick_data)
    save_data(dataframe=candle_stick_data,
              bucket='fs',
              topic=ticker,
              version='finnhub',
              suffix=f"{start_date}_to_{end_date}",
              path="/home/stan/github/deeptendies")
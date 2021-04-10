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

# load data with yahoo / data reader
for ticker in stonks:
    print(ticker, start_date, end_date)
    candle_stick_data = data.DataReader(ticker, 'yahoo', start_date, end_date)
    print(candle_stick_data)
    save_data(dataframe=candle_stick_data,
              bucket='fs',
              topic=ticker,
              version='demo',
              suffix=f"{start_date}_to_{end_date}",
              path="/home/stan/github/deeptendies")
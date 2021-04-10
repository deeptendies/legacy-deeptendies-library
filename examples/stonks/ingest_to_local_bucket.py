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




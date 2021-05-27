from dtlite.data import ingest_yahoo_finance
from dtlite.model import train_and_predict


def run_forecast(stock='gme',
                 start='2021-01-01',
                 end='2021-01-30',
                 forecast_days=5,
                 *args, **kwargs):
    df = ingest_yahoo_finance(
        stock='gme',
        start='2021-05-10',
        end='2021-05-26',
        forecast_days=5,
        filter=['High']
    )
    forecast = train_and_predict(data=df)
    print(forecast)


if __name__ == '__main__':
    run_forecast()
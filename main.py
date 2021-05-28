from dtlite.data import ingest_yahoo_finance
from dtlite.model import train_and_predict


def run_forecast(stock='gme',
                 start='2021-05-10',
                 end='2021-05-27',
                 forecast_days=5,
                 *args, **kwargs):
    df = ingest_yahoo_finance(
        stock=stock,
        start=start,
        end=end,
        forecast_days=forecast_days,
        filter=['High']
    )
    forecast = train_and_predict(data=df)
    # print(forecast)
    return forecast


if __name__ == '__main__':
    stocks=['gme', 'amc', 'cour', 'sret']
    for i in stocks:
        fc=run_forecast(i)
        print(i, fc)
from pandas_datareader import data


def ingest_yahoo_finance(
        stock='gme',
        start='2021-01-01',
        end='2021-01-30',
        *args, **kwargs):
    """
    :param stock:
    :param start:
    :param end:
    :param args:
    :param kwargs:
        filter: from High, Low, Open, Close, Volume, Adj Close
    :return:
    """

    stock_data = data.DataReader(stock,
                                 start=start,
                                 end=end,
                                 data_source='yahoo')
    if kwargs.get('filter') is not None:
        stock_data = stock_data[kwargs.get('filter')]
    return stock_data

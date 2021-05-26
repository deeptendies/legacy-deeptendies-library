import unittest

from dtlite.data import ingest_yahoo_finance


class TestDTLite(unittest.TestCase):

    def test_ingest(self):
        data = ingest_yahoo_finance(
            stock='gme',
            start='2021-05-1',
            end='2021-05-26',
            filter=['High', 'Low']
        )
        print(data)
        self.assertEqual(2, len(data.columns))



if __name__ == '__main__':
    unittest.main()
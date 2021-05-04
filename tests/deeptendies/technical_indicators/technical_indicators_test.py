import unittest
class TestStringMethods(unittest.TestCase):
    def test_technical_indicator(self):
        import pandas as pd
        import pandas_ta as ta
        self.chdir_root()
        file = "bucket=fs/topic=AAPL/version=yahoo/processed_at=2021-04-10/AAPL_2011-04-01_to_2021-04-01.csv"
        df = pd.DataFrame()  # Empty DataFrame
        df = pd.read_csv(file, sep=",")
        df.set_index(pd.DatetimeIndex(df["Date"]), inplace=True)
        df.ta.log_return(cumulative=True, append=True)
        df.ta.percent_return(cumulative=True, append=True)
        print(df.head())
        df.ta.indicators()
        # self.assertEqual(1,1)

    def chdir_root(self):
        """
        helper function, not a test
        :return:
        """
        import os
        path = os.path.dirname(os.path.abspath(__file__))
        file_to_search = "setup.py"
        while file_to_search not in os.listdir(path):
            path = os.path.dirname(path)
        os.chdir(path)

if __name__ == '__main__':
    unittest.main()

"""shell
pip install autokeras
"""
import os

from deeptendies.utils import get_numerical_df

"""
To make this tutorial easy to follow, we use the UCI Airquality dataset, and try to
forecast the AH value at the different timesteps. Some basic preprocessing has also
been performed on the dataset as it required cleanup.
## A Simple Example
The first step is to prepare your data. Here we use the [UCI Airquality dataset]
(https://archive.ics.uci.edu/ml/datasets/Air+Quality) as an example.
"""

import pandas as pd
import tensorflow as tf



fdir, fname = 'temp', 'interm_data.csv'
file = os.path.join(fdir, fname)

# load dataset
df = pd.read_csv(file, header=0, index_col=0)
dataset = get_numerical_df(df)
# print(dataset.columns)
dataset = dataset.dropna()

def df_except_columns(df_columns, exceptions):
    df_columns=list(df_columns)
    for dfc in df_columns:
        change = True
        for exc in exceptions:
            if exc not in dfc:
                change = False
        if change:
            df_columns.remove(dfc)
    return df_columns

# print(df_except_columns(dataset.columns,["high","next"]))

val_split = int(len(dataset) * 0.7)
data_train = dataset[:val_split]
validation_data = dataset[val_split:]

prep_x_cols = df_except_columns(dataset.columns,["high","next"])
prep_x_cols = df_except_columns(prep_x_cols,["low","next"])
data_x = data_train[prep_x_cols].astype('float64')

data_x_val = validation_data[prep_x_cols].astype(
                                  'float64')

# Data with train data and the unseen data from subsequent time steps.
data_x_test = dataset[prep_x_cols].astype('float64')

data_y = data_train['next_3_high'].astype('float64')

data_y_val = validation_data['next_3_high'].astype('float64')

print(data_x.shape)  # (6549, 12)
print(data_y.shape)  # (6549,)

"""
The second step is to run the [TimeSeriesForecaster](/time_series_forecaster).
As a quick demo, we set epochs to 10.
You can also leave the epochs unspecified for an adaptive number of epochs.
"""
import autokeras as ak

predict_from = 1
predict_until = 10
lookback = 3
clf = ak.TimeseriesForecaster(
    lookback=lookback,
    predict_from=predict_from,
    predict_until=predict_until,
    max_trials=1,
    objective='val_loss'
)
# Train the TimeSeriesForecaster with train data
clf.fit(x=data_x, y=data_y, validation_data=(data_x_val, data_y_val), batch_size=32,
        epochs=10)
# Predict with the best model(includes original training data).
predictions = clf.predict(data_x_test)
print(predictions.shape)
# Evaluate the best model with testing data.
print(clf.evaluate(data_x_val, data_y_val))
from datetime import date

import pandas as pd
# get ingested data
from deeptendies.utils import get_numerical_df

today = date.today()
file = '../stonks/bucket=filesys/topic=GME/version=raw/processed_at=2021-04-04/GME_180.csv'
df = pd.read_csv(file, header=0, index_col=0)
print(df.columns)
dataset = get_numerical_df(df)[-60:] # TODO: figure out what to tune this to

val_split = int(len(dataset) * 0.7) #TODO: figure out what's a better way to split this
data_train = dataset[:val_split]
validation_data = dataset[val_split:]

feature_cols = ['o', 'h', 'l', 'v', 'c']

data_x = data_train[feature_cols].astype('float64')

data_x_val = validation_data[feature_cols].astype(
    'float64')

# Data with train data and the unseen data from subsequent time steps.
data_x_test = dataset[feature_cols].astype('float64')

target_col = 'c'

data_y = data_train[target_col].astype('float64')

data_y_val = validation_data[target_col].astype('float64')

print(f"data_x.shape: {data_x.shape}, data_y.shape: {data_y.shape}")
print(f"data_x_val.shape: {data_x_val.shape}, data_y_val.shape: {data_y_val.shape}, data_x_test.shape: {data_x_test.shape}")

"""
The second step is to run the [TimeSeriesForecaster](/time_series_forecaster).
As a quick demo, we set epochs to 10.
You can also leave the epochs unspecified for an adaptive number of epochs.
"""
import autokeras as ak

predict_from = 1
predict_until = 5
lookback = 5
clf = ak.TimeseriesForecaster(
    lookback=lookback,
    predict_from=predict_from,
    predict_until=predict_until,
    max_trials=10,
    objective='val_loss'
)
# Train the TimeSeriesForecaster with train data
clf.fit(x=data_x, y=data_y, validation_data=(data_x_val, data_y_val), batch_size=32,
        epochs=25)
# Predict with the best model(includes original training data).
predictions = clf.predict(data_x_test)
print(predictions.shape)
# Evaluate the best model with testing data.
print(clf.evaluate(data_x_val, data_y_val))

data_x_test
print(predictions)
print(data_y_val)


# TODO: make a dataframe, satge the prediction data and save it to bucket
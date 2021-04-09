import os
import pathlib
import time
from datetime import date

import pandas as pd
# get ingested data
from deeptendies import utils

# load data
from deeptendies.utils.local_bucket import save_data

today = date.today()
file = '../stonks/bucket=filesys/topic=GME/version=raw/processed_at=2021-04-04/GME_180.csv'
df = pd.read_csv(file, header=0, index_col=0)
print(df.columns)

# setup jobs
param_grid = {
    "data_lookbacks": [75, 100, 125],
    "feature_columns": [['h', 'l'], ['h', 'l', 'v'], ['o', 'h', 'l', 'v', 'c']],
    "lookbacks": [1, 3, 5]
}


def runner(data_lookback, feature_column, lookback, stats):
    dataset = df[-data_lookback:]  # TODO: figure out what to tune this to
    val_split = int(len(dataset) * 0.7)  # TODO: figure out what's a better way to split this
    data_train = dataset[:val_split]
    validation_data = dataset[val_split:]

    cols = feature_column  # ['o', 'h', 'l', 'v', 'c'] #TODO: grid search this also?

    data_x = data_train[cols].astype('float64')

    data_x_val = validation_data[cols].astype(
        'float64')

    # Data with train data and the unseen data from subsequent time steps.
    data_x_test = dataset[cols].astype('float64')

    target_col = 'c'

    data_y = data_train[target_col].astype('float64')

    data_y_val = validation_data[target_col].astype('float64')

    print(data_x.shape)
    print(data_y.shape)

    """
    The second step is to run the [TimeSeriesForecaster](/time_series_forecaster).
    As a quick demo, we set epochs to 10.
    You can also leave the epochs unspecified for an adaptive number of epochs.
    """
    import autokeras as ak

    predict_from = 1
    predict_until = 5
    lookback = lookback

    path = "fs::models"
    path = os.path.join(path, time.strftime("%Y-%m-%d_%H:%M"))
    if not os.path.exists(path):
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    clf = ak.TimeseriesForecaster(
        lookback=lookback,
        predict_from=predict_from,
        predict_until=predict_until,
        max_trials=1,
        objective='val_loss',
        directory=path
    )
    # Train the TimeSeriesForecaster with train data
    clf.fit(x=data_x, y=data_y, validation_data=(data_x_val, data_y_val), batch_size=32,
            epochs=1)
    # Predict with the best model(includes original training data).
    predictions = clf.predict(data_x_test)
    print(predictions.shape)
    # Evaluate the best model with testing data.
    metrics = clf.evaluate(data_x_val, data_y_val)
    data_x_test
    print(predictions)
    print(data_y_val)
    stats.append([data_lookback, feature_column, lookback, path, metrics, predictions, data_y_val])


# TODO: make a dataframe, satge the prediction data and save it to bucket

stats = []
for data_lookback in param_grid['data_lookbacks']:
    for feature_column in param_grid['feature_columns']:
        for lookback in param_grid['lookbacks']:
            print(data_lookback, feature_column, lookback)
            try:
                runner(data_lookback, feature_column, lookback, stats)
            except Exception as e:
                stats.append([data_lookback, feature_column, lookback, '', '', "(error): " + str(e), '(job failed)'])
                pass
            df = pd.DataFrame(stats,
                              columns=['data_lookback', 'feature_column', 'lookback', 'path', 'metrics', 'predictions',
                                       'data_y_val'])
            save_data(dataframe=df, bucket='fs::stats', topic="gridsearch_report", version='demo',
                      suffix=f"{time.strftime('%Y-%m-%d_%H:%M')}")

print("autokeras grid search is finished")
exit()

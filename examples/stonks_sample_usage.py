# sudo apt-get install python3-tk
from deeptendies.plots import *
from deeptendies.stonks import *



import pandas as pd

# base configs
from deeptendies.utils import rename_reference_df_column_names

stock_sym='GME'
days_ago=250
start='2020-12-01'
metrics_interested=['next_3_high', 'next_3_low']
finnhub_token = "c10t49748v6o1us2neqg"


# get df from finnhub
df = pd.DataFrame.from_dict(get_stock_data(stock_sym, days_ago, 'D', finnhub_token))
df['t'] = pd.to_datetime(df['t'], unit = 's')
time.sleep(0.2)


# get dji index
df_dji = pd.DataFrame.from_dict(get_stock_data("^DJI", days_ago, 'D', finnhub_token))
df_dji['t'] = pd.to_datetime(df_dji['t'], unit ='s')


print(df_dji.columns)
df_dji = rename_reference_df_column_names(df_dji, "_dji")


exit()

# plot something
fig = get_candlestick_plot(df)
fig.show()

# feature engineering, calendar and ma, vwap
df_proc = get_calendar_features(df)
df_proc = get_moving_average(df)
df_proc.fillna(method='backfill')
df_proc = add_vwap_col(df)

# feature engineering, get high and get low
days=[1,3,5,7]
df_new = get_high(df, days)
df_new = get_low(df, days)
print(df.head)
print(df.shape)


for metric_interested in metrics_interested:
    # metric_interested = 'next_3_low'
    df[df[metric_interested].eq(0)] = np.nan
    # plt_visual_raw(stock_sym, metric_interested, df)
    # Create a new dataframe with only the 'Close column
    data = df.filter([metric_interested])
    # Convert the dataframe to a numpy array
    dataset = data.values
    # Get the number of rows to train the model on
    training_data_len = int(np.ceil( len(dataset) * .95 ))
    # print("training_data_len: %s" %training_data_len )

    #scaling
    # Scale the data
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)
    # scaled_data
    # Create the training data set
    # Create the scaled training data set
    train_data = scaled_data[0:int(training_data_len), :]
    # Split the data into x_train and y_train data sets
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60:i, 0])
        y_train.append(train_data[i, 0])
        if i <= 61:
            print(x_train)
            print(y_train)
            print()

    # Convert the x_train and y_train to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape the data
    x_train = np.atleast_2d(x_train) # experimenting to solve the tuple index out of range issue

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    # print("x_train.shape:")
    # print(x_train.shape)


    #LSTM
    from keras.models import Sequential
    from keras.layers import Dense, LSTM, Masking

    # Build the LSTM model
    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=(x_train.shape[1], 1))) # handle nans https://stackoverflow.com/questions/52570199/multivariate-lstm-with-missing-values
    model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(x_train, y_train, batch_size=1, epochs=20)

    # Test
    # Create the testing data set
    # Create a new array containing scaled values from index 1543 to 2002

    # training and validating
    test_data = scaled_data[training_data_len - 60:, :]

    # Create the data sets x_test and y_test
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])

    # Convert the data to a numpy array
    x_test = np.array(x_test)

    # Reshape the data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Get the models predicted price values
    predictions = model.predict(x_test)

    print(predictions)

    predictions = scaler.inverse_transform(predictions)

    print(predictions)

    # Get the root mean squared error (RMSE)
    rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
    print("rmse %s" %rmse)

    ## Plot the data Again
    # Plot the data
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions

    # plot_predicted()

    print(valid)
import pandas as pd
import numpy as np


file='../temp/interm_data.csv'
df = pd.read_csv(file, header=0, index_col=0)


stock_sym='GME'
days_ago=250
start='2020-12-01'
metrics_interested=['next_3_high', 'next_3_low']


for metric_interested in metrics_interested:

    # metric_interested = 'next_3_low'
    df[df[metric_interested].eq(0)] = np.nan
    df.dropna() # drop na so it doesn't skew with the training / testing


    # plt_visual_raw(stock_sym, metric_interested, df)
    # Create a new dataframe with only the 'Close column
    # data = df.filter([metric_interested])
    xlist = ['h', 'l', 'v']
    ylist = [metric_interested]
    xdata=df.filter(xlist)
    ydata=df.filter(ylist)


    # Convert the dataframe to a numpy array
    xdataset = xdata.values
    ydataset = ydata.values
    # Get the number of rows to train the model on
    xtraining_data_len = int(np.ceil( len(xdataset) * .95 ))
    ytraining_data_len = int(np.ceil( len(ydataset) * .95 ))
    # print("training_data_len: %s" %training_data_len )

    #scaling
    # Scale the data
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0,1))
    xscaled_data = scaler.fit_transform(xdataset)
    yscaled_data = scaler.fit_transform(ydataset)
    # scaled_data
    # Create the training data set
    # Create the scaled training data set
    xtrain_data = xscaled_data[0:int(xtraining_data_len), :]
    ytrain_data = yscaled_data[0:int(ytraining_data_len), :]
    # Split the data into x_train and y_train data sets
    x_train = []
    y_train = []



# sequence prep
look_back = 25
for i in range(look_back, len(xtrain_data)):
    x_train.append(xtrain_data[i - look_back:i, 0])
    y_train.append(ytrain_data[i, 0])
    if i <= look_back + 1:
        print(x_train)
        print(y_train)
        print()

# Convert the x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data
x_train = np.atleast_2d(x_train)  # experimenting to solve the tuple index out of range issue

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
# print("x_train.shape:")
# print(x_train.shape)


# LSTM
from keras.models import Sequential
from keras.layers import Dense, LSTM, Masking

# Build the LSTM model
model = Sequential()
model.add(Masking(mask_value=0., input_shape=(
x_train.shape[1], 1)))  # handle nans https://stackoverflow.com/questions/52570199/multivariate-lstm-with-missing-values
model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, batch_size=1, epochs=30)

# Test
# Create the testing data set
# Create a new array containing scaled values from index 1543 to 2002

# training and validating
xtest_data = xscaled_data[xtraining_data_len - look_back:, :]

# Create the data sets x_test and y_test
x_test = []
y_test = ydataset[ytraining_data_len:, :]
for i in range(look_back, len(xtest_data)):
    x_test.append(xtest_data[i - look_back:i, 0])

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
print("rmse %s" % rmse)

## Plot the data Again
# Plot the data
train = ydata[:ytraining_data_len]
valid = ydata[ytraining_data_len:]
valid['Predictions'] = predictions

# plot_predicted()

print(valid)

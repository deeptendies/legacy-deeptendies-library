from datetime import datetime
from pandas_datareader import DataReader
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# sudo apt-get install python3-tk

stock_sym='GME'
start='2020-12-01'
# the next line can be High, Low, Open, Close, Volume, Adj Close
metric_interested='High'
# Get the stock quote
df = DataReader(stock_sym, data_source='yahoo', start=start, end=datetime.now())
# Show teh data
print(df.head())

plt.figure(figsize=(16,8))
plt.title(stock_sym +" "+ metric_interested + ' Price History')
plt.plot(df[metric_interested])
plt.xlabel('Date', fontsize=18)
plt.ylabel(stock_sym + " " + metric_interested + ' Price USD ($)', fontsize=18)
plt.show()



# Create a new dataframe with only the 'Close column
data = df.filter([metric_interested])
# Convert the dataframe to a numpy array
dataset = data.values
# Get the number of rows to train the model on
training_data_len = int(np.ceil( len(dataset) * .95 ))

print("training_data_len: %s" %training_data_len )


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
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
print("x_train.shape:")
print(x_train.shape)


#LSTM
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Build the LSTM model
model = Sequential()
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
predictions = scaler.inverse_transform(predictions)

# Get the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
print("rmse %s" %rmse)




## Plot the data Again
# Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
# Visualize the data
plt.figure(figsize=(16,8))
plt.title(metric_interested+' Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel(metric_interested+' Price USD ($)', fontsize=18)
plt.plot(train[metric_interested])
plt.plot(valid[[metric_interested, 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()



print(valid)
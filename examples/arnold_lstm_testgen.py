import os
import pandas as pd
import numpy as np
# %matplotlib inline
import warnings

from deeptendies.utils import get_numerical_df
from sklearn.preprocessing import MinMaxScaler


warnings.filterwarnings('ignore')
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

# to get started developing your own arnold, clone this starter file, rename
# it as a new file, following mike's naming convention arnold_<model name>_<extra tag>.py
fdir, fname = 'temp', 'interm_data.csv'
file = os.path.join(fdir, fname)
df = pd.read_csv(file, header=0, index_col=0)
df = get_numerical_df(df)

split = int(0.8 * df.shape[1])
train=df[:split]
test=df[split:]

scaler=MinMaxScaler()
scaled_train=scaler.fit_transform(train)
scaled_test=scaler.transform(test)

n_input=12
n_features=1

train_generator=TimeseriesGenerator(scaled_train,
                                     scaled_train,
                                      n_input,
                                      batch_size=1)


model=Sequential()
model.add(LSTM(100,activation='relu',input_shape=(n_input,n_features),return_sequences=True))
model.add(LSTM(50,activation='relu',return_sequences=True))
model.add(LSTM(10,activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam',loss='mse')
model.summary()

model.fit(train_generator,epochs=30)



# predictions

test_predictions = []
#Select last n_input values from the train data
first_eval_batch = scaled_train[-n_input:]
#reshape the data into LSTM required (#batch,#timesteps,#features)
current_batch = first_eval_batch.reshape((1, n_input, n_features))
for i in range(len(test)):
    current_pred = model.predict(current_batch)[0]
    # get prediction, grab the exact number using the [0]
    pred = model.predict(current_batch)[0]
    # Add this prediction to the list
    test_predictions.append(pred)
    # The most critical part, update the (#batch,#timesteps,#features
    # using np.append(
    # current_batch[:        ,1:   ,:] ---------> read this as
    # current_batch[no_change,1:end,no_change]
    # (Do note the second part has the timesteps)
    # [[pred]] need the double brackets as current_batch is a 3D array
    # axis=1, remember we need to add to the second part i.e. 1st axis
    current_batch = np.append(current_batch[:,1:,:],
                          [[current_pred]],
                          axis=1)

test_predictions


actual_predictions = scaler.inverse_transform(test_predictions)
actual_predictions

test['Predictions'] = actual_predictions
test.plot(figsize=(12,8));



# https://sailajakarra.medium.com/lstm-for-time-series-predictions-cc68cc11ce4f
# https://www.kaggle.com/vaibhavsxn/lstm-rnn-on-timeseries-data-tf-2-0
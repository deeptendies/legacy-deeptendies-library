import os
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import warnings

from sklearn.preprocessing import MinMaxScaler

from deeptendies.utils import get_numerical_df

warnings.filterwarnings('ignore')
import tensorflow as tf
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



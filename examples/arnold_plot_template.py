import os
import pandas as pd
import numpy as np
from pandas import read_csv
from matplotlib import pyplot

fdir, fname = 'temp', 'interm_data.csv'
file = os.path.join(fdir, fname)

# load dataset
df = pd.read_csv(file, header=0, index_col=0)
values = df.values
# specify columns to plot

# to plot all groups
# groups = range(0, df.columns.size)

# to plot where there's interesting keystrings
groups = []
key_to_search = ['high', 'low']
for key in key_to_search:
    groups += [idx for idx, i in enumerate(df.columns) if key in i]
print(groups)

exit()

i = 1
# plot each column
pyplot.figure()
for group in groups:
    pyplot.subplot(len(groups), 1, i)
    pyplot.plot(values[:, group])
    pyplot.title(df.columns[group], y=0.5, loc='right')
    i += 1
pyplot.show()

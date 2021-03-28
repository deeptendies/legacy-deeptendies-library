import os
import pandas as pd
import numpy as np

fdir, fname = 'temp', 'interm_data.csv'
file = os.path.join(fdir, fname)
df = pd.read_csv(file, header=0, index_col=0)

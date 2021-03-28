import os
import pandas as pd
import numpy as np

# to get started developing your own arnold, clone this starter file, rename
# it as a new file, following mike's naming convention arnold_<model name>_<extra tag>.py
fdir, fname = 'temp', 'interm_data.csv'
file = os.path.join(fdir, fname)
df = pd.read_csv(file, header=0, index_col=0)

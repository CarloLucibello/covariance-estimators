#%%
from scipy.io import loadmat
import pandas as pd
from glob import glob
import os

#%%
dirname = "../dati_CamCAN/"
files = glob(dirname + "subj*.txt")
for file in files:
    basename = os.path.basename(file)
    df = pd.read_csv(file, sep=' ', header=None)
    df.drop([108,109], axis=1, inplace=True)
    df.to_csv(os.path.join(dirname, basename) + ".preprocessed", sep=' ', header=None, index=None)

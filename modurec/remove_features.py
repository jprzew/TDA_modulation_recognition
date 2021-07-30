import path
import pandas as pd
import modurec as mr
import pickle
import os
import features
from features import SignalFeatures

data_path = '../data'
data_files = ['stats_train.pkl', 'stats_test.pkl']
to_remove = ['cloud_3D', 'H0_3D', 'H1_3D','H0_life_time_3D',
             'H1_life_time_3D', 'no_H0_3D', 'no_H1_3D',
             'H0_mean_3D', 'H1_mean_3D', 'H0_var_3D', 'H1_var_3D']

for filename in data_files:
    df = pd.read_pickle(os.path.join(data_path, filename))
    df = df.drop(columns=to_remove)
    df.to_pickle(os.path.join(data_path, filename)) 

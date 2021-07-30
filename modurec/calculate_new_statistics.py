"""Features calculator

Use this script calculate features defined in SignalFeatures class. 
REMARK: the script calculates the features that are not yet calculated. 

    * Files to modify are defined in data_files variable
    * Files to modify are assumed to be located according to data_path variable
"""

import path
import pandas as pd
import modurec as mr
import pickle
import os
import features
from features import SignalFeatures

data_path = '../data'
data_files = ['stats_train.pkl', 'stats_test.pkl']

for filename in data_files:
    df = pd.read_pickle(os.path.join(data_path, filename))

    calculated_features = set(df.columns)
    all_features = set([f for f in dir(SignalFeatures)
                        if not f.startswith('_')])

    to_calculate = all_features.difference(calculated_features)

    for feature in to_calculate:
        df.feat[feature]

    df.to_pickle(os.path.join(data_path, filename)) 

"""Features calculator

Use this script calculate features defined in SignalFeatures class. 
REMARK: the script calculates the features that are not yet calculated. 

    * Files to modify are defined in data_files variable
    * Files to modify are assumed to be located according to data_path variable
WARNING: This script is deprecated since SignalFeatures class is deprecated
"""

# import path
import pandas as pd
import pickle
import os
from modurec import features


data_path = 'data'
data_files = ['stats_train.pkl', 'stats_test.pkl']

for filename in data_files:
    df = pd.read_pickle(os.path.join(data_path, filename))

    to_calculate = [df.ff.create_feature('diagram', dim=2, step='symbol_rate'),
                    df.ff.create_feature('diagram', dim=2, step='symbol_rate'),
                    df.ff.create_feature('diagram', dim=3, step='symbol_rate'),
                    df.ff.create_feature('diagram', dim=3, step='symbol_rate'),
                    df.ff.create_feature('diagram', dim=4, step='symbol_rate'),
                    df.ff.create_feature('diagram', dim=4, step='symbol_rate')]


    # all_features = set([f for f in dir(SignalFeatures)
    #                     if not f.startswith('_')])

    # to_calculate = all_features.difference(calculated_features)

    # for feature in to_calculate:
    #     df.feat[feature]

    df.to_pickle(os.path.join(data_path, filename)) 

import path
import pandas as pd
import modurec as mr
import pickle
import os

data_path = '../data'
stats_path = '../ml_statistics'
source_filename = 'train.pkl'
dest_filename = 'stats_train.pkl'

# df = pickle.load(open(os.path.join(data_path, source_filename), 'rb'))
df = pd.read_pickle(os.path.join(data_path, source_filename))

modulations = {'16PSK', '16QAM', '32PSK', '32QAM', '64QAM', '8PSK',
               'BPSK', 'FM', 'GMSK', 'OQPSK', 'QPSK'}

# df = df.loc[df['modulation_type'].isin(['BPSK', 'QPSK', '8PSK'])]
df = df.loc[df['modulation_type'].isin(modulations)]

# import pdb; pdb.set_trace()

df.mr.add_statistics(window=1, inplace=True)

# pickle.dump(df, open(os.path.join(stats_path, dest_filename), 'wb'))
df.to_pickle(os.path.join(stats_path, dest_filename))

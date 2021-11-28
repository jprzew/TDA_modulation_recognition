"""WARNING: This script is deprecated
It seems to read data from numpy and create from it train/test datasets """
import signal_reader
import pickle
import os
from sklearn.model_selection import train_test_split

seed = 42
data_path = './data'


df = signal_reader.get_signal_df_from_numpy()

# TODO: Make it ballanced
df_train, df_test = train_test_split(df, random_state=42, test_size=0.3)

# pickle.dump(df_train, open(os.path.join(data_path, 'train.pkl'), "wb" ))
# pickle.dump(df_train, open(os.path.join(data_path, 'test.pkl'), "wb" ))

df_train.to_pickle(os.path.join(data_path, 'train.pkl'))
df_test.to_pickle(os.path.join(data_path, 'test.pkl'))

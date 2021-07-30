# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# %load_ext pycodestyle_magic
# %matplotlib inline
# %flake8_on --ignore E703,E702
# %load_ext autoreload
# %autoreload 2

# %%
import path
import random
import modurec as mr
from modurec import signal_reader

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, plot_confusion_matrix


import matplotlib.pyplot as plt
import re
from ripser import Rips
import math
import numpy as np
import numpy.ma as ma
import pandas as pd
import importlib
import pickle

# %%
# df = pickle.load(open('../ml_statistics/stats_train.pkl', 'rb'))
df = pd.read_pickle('../ml_statistics/stats_train.pkl')

# %%
df.columns


# %% [markdown]
# **Some new elements**

# %%
def __remove_below_quantile(x, q=0.1):
    array = x.copy()
    quantile = np.quantile(array, q=q)
    return array[(array >= quantile) & (array < float('inf'))]

q = 0.95



df['new1'] = df.H1_life_time.apply(lambda x:
                                   np.mean(__remove_below_quantile(x, q=q)))
df['new2'] = df.H0_life_time.apply(lambda x:
                                   np.mean(__remove_below_quantile(x, q=q)))
df['new3'] = df.H1_life_time_4D.apply(lambda x:
                                   np.mean(__remove_below_quantile(x, q=q)))
df['new4'] = df.H0_life_time_4D.apply(lambda x:
                                   np.mean(__remove_below_quantile(x, q=q)))

df['new5'] = df.H1_life_time.apply(lambda x:
                                   np.var(__remove_below_quantile(x, q=q)))
df['new6'] = df.H0_life_time.apply(lambda x:
                                   np.var(__remove_below_quantile(x, q=q)))
df['new7'] = df.H1_life_time_4D.apply(lambda x:
                                   np.var(__remove_below_quantile(x, q=q)))
df['new8'] = df.H0_life_time_4D.apply(lambda x:
                                   np.var(__remove_below_quantile(x, q=q)))



# %%
def __remove_below_threshold(x, thr=0.1):
    array = x.copy()
    return array[(array >= thr) & (array < float('inf'))]

df['new9'] = df.H0_life_time.apply(lambda x:
                                   len(__remove_below_threshold(x, thr=0.15)))

df['new10'] = df.H1_life_time.apply(lambda x:
                                   len(__remove_below_threshold(x, thr=0.05)))


# %%
print(df.new10.to_string())

# %%
h_vector = ['no_H0', 'no_H1', 'H0_mean', 'H1_mean', 'H0_var',
       'H1_var', 'no_H0_3D', 'no_H1_3D', 'H0_mean_3D', 'H1_mean_3D',
       'H0_var_3D', 'H1_var_3D', 'no_H0_4D', 'no_H1_4D', 'H0_mean_4D', 'H1_mean_4D',
        'H0_var_4D', 'H1_var_4D']

# h_vector = ['no_H0', 'no_H1', 'H0_mean', 'H1_mean', 'H0_var',
#             'H1_var', 'no_H0_4D', 'no_H1_4D', 'H0_mean_4D', 'H1_mean_4D',
#             'H0_var_4D', 'H1_var_4D', 'new1', 'new3', 'new2', 'new4',
#            'new5', 'new6', 'new7', 'new8', 'new9', 'new10']

# h_vector = ['no_H0', 'no_H1', 'H0_mean', 'H1_mean', 'H0_var',
#        'H1_var', 'no_H0_3D', 'no_H1_3D', 'H0_mean_3D', 'H1_mean_3D',
#        'H0_var_3D', 'H1_var_3D', 'no_H0_4D', 'no_H1_4D', 'H0_mean_4D', 'H1_mean_4D',
#         'H0_var_4D', 'H1_var_4D', 'new1', 'new3', 'new2', 'new4',
#         'new5', 'new6', 'new7', 'new8', 'new9', 'new10']


# %%
X = df[h_vector]
y = df.modulation_type

models = mr.spot_check(X, y, k_fold=None, test_size=0.2, seed=42)

# df['signal_fft'] = df['signal_sample'].apply(np.fft.rfft)
# df['signal_fft'][0]

# %%
# df['r_fft'] = df['signal_fft'].map(lambda x: x.real)
# df['i_fft'] = df['signal_fft'].map(lambda x: x.imag)
# df['r_fft'][0]

# %%
model = models['rf']

# %%
importances = model.feature_importances_
plt.barh(h_vector, importances)

# %%
from sklearn import preprocessing

# %%
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# %%
scaler.fit(X)

# %%
# df_new = df
df_new = df.loc[df.SNR == 10]
X_new = df_new[h_vector]
y_new = df_new.modulation_type

predictions = model.predict(scaler.transform(X_new))
plot_confusion_matrix(model, scaler.transform(X_new), y_new)

# %%
accuracy_score(y_new, predictions)

# %%
predictions = model.predict(scaler.transform(X))
accuracy_score(y, predictions)

# %%
sorted(df.SNR.unique())

# %%
df.mr.plot_persistence_diagrams(data_col='diagram_4D', max_rows=300)

# %%
y

# %%
df.loc[df.SNR == 30, h_vector]

# %%
from sklearn import tree


# %%
tree.plot_tree(models['cart'])

# %%
import seaborn as sns
from sklearn import preprocessing

# %%
fig, ax1 = plt.subplots(1, 1)
df_temp = pd.DataFrame(preprocessing.scale(X), columns=X.columns)
df_temp['type'] = df.reset_index().modulation_type

sns.scatterplot(x='H1_mean', y='H0_mean', hue='type', data=df_temp, ax=ax1)

# %%
fig, ax2 = plt.subplots(1, 1)
df_temp = X.copy().reset_index()
df_temp['type'] = df.reset_index().modulation_type
sns.scatterplot(x='H1_mean', y='no_H1', hue='type', data=df_temp, ax=ax2)

# %%
fig, ax2 = plt.subplots(1, 1)
df_temp = X.copy().reset_index()
df_temp['type'] = df.reset_index().modulation_type
sns.scatterplot(x='H1_var', y='H0_var', hue='type', data=df_temp, ax=ax2)

# %%
df_temp['type'] = y

# %%
df_temp.type

# %%
import graphviz

# %%
dot_data = tree.export_graphviz(models['cart'], out_file=None, 
                                feature_names=X.columns,  
                                class_names=pd.unique(df.modulation_type),
                                filled=True, rounded=True,
                                special_characters=True)  
graph = graphviz.Source(dot_data)  
graph 

# %%
df_red = df[['signal_sample', 'signal_sampleQ']]
df_red = df_red.iloc[[1]]

# %%
df_red.mr.add_statistics(inplace=True)

# %%
df_red

# %%
x = df_red.cloud_3D.iloc[0]

# %%
np.column_stack([x, range(x.shape[0])])

# %%
x

# %%

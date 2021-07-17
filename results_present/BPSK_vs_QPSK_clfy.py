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
import path
import modurec as mr
from modurec import signal_reader

import matplotlib.pyplot as plt
import re
from ripser import Rips
import math
import numpy as np
import numpy.ma as ma
import pandas as pd
import importlib
import pickle
import graphviz
import seaborn as sns
from sklearn import preprocessing
from sklearn import tree

# %%
# df = pickle.load(open('../ml_statistics/stats_train.pkl', 'rb'))
df = pd.read_pickle('../ml_statistics/stats_BPSK_QPSK.pkl')

# %% [markdown]
# **Description of the features**
#
# *no_H0* - number of zeroth homology classes
#
# *no_H1* - number of first homology classes
#
# *H0_mean* - mean lifetime of zeroth homology classes
#
# *H1_mean* - mean lifetime of first homology classes
#
# *H0_var* - variance of lifetime of zeroth homology classes
#
# *H1_var* - variance of lifetime of first homology classes
#
# **The other features are defined analogusly and calculated from 3D-cloud. The cloud, however, is created with real part of the signal only**
#
# *no_H0_3D*
#
# *no_H1_3D*
#
# *H0_mean_3D*
#
# *H1_mean_3D*
#
# *H0_var_3D*
#
# *H1_var_3D*
#

# %%
h_vector = ['no_H0', 'no_H1', 'H0_mean', 'H1_mean', 'H0_var',
       'H1_var', 'no_H0_3D', 'no_H1_3D', 'H0_mean_3D', 'H1_mean_3D',
       'H0_var_3D', 'H1_var_3D']

# %%
X = df[h_vector]
y = df.modulation_id

models = mr.spot_check(X, y, k_fold=None, test_size=0.2)

# %%
dot_data = tree.export_graphviz(models['cart'], out_file=None, 
                                feature_names=X.columns,  
                                class_names=pd.unique(df.modulation_type),
                                filled=True, rounded=True,
                                special_characters=True)  
graph = graphviz.Source(dot_data)  
graph 

# %%

# %%
fig, ax1 = plt.subplots(1, 1)
df_temp = pd.DataFrame(preprocessing.scale(X), columns=X.columns)
df_temp['type'] = df.reset_index().modulation_type

sns.scatterplot(x='H1_var', y='H0_var', hue='type', data=df_temp, ax=ax1)

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
fig, ax2 = plt.subplots(1, 1)
df_temp = X.copy().reset_index()
df_temp['type'] = df.reset_index().modulation_type
sns.scatterplot(x='H1_mean_3D', y='H0_mean_3D', hue='type', data=df_temp, ax=ax2)

# %%
fig, ax2 = plt.subplots(1, 1)
df_temp = X.copy().reset_index()
df_temp['type'] = df.reset_index().modulation_type
sns.scatterplot(x='H1_var_3D', y='H0_var_3D', hue='type', data=df_temp, ax=ax2)

# %%
fig, ax2 = plt.subplots(1, 1)
df_temp = X.copy().reset_index()
df_temp['snr'] = df.reset_index().SNR
sns.scatterplot(x='H1_var', y='H0_var', hue='snr', data=df_temp, ax=ax2)

# %%
fig, ax2 = plt.subplots(1, 1)
df_temp = X.copy().reset_index()
df_temp['snr'] = df.reset_index().SNR
sns.scatterplot(x='H1_mean', y='H0_mean', hue='snr', data=df_temp, ax=ax2)

# %%
df.columns

# %%

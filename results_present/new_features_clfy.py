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
from src import modurec as mr
from src.modurec import signal_reader

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
df = pd.read_pickle('../ml_statistics/stats_train.pkl')

# %%
# df = df.loc[df.modulation_type.isin(['16QAM', 'FM', 'QPSK', 'GMSK', 'BPSK', 'OQPSK', '8PSK'])]

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
       'H0_var_3D', 'H1_var_3D', 'no_H0_4D', 'no_H1_4D', 'H0_mean_4D', 'H1_mean_4D',
        'H0_var_4D', 'H1_var_4D']

# %%
X = df[h_vector]
y = df.modulation_type

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
importances = models['rf'].feature_importances_
plt.barh(h_vector, importances)

# %% [markdown]
# **New dataset**

# %%
df = pd.read_pickle('../ml_statistics/stats_test.pkl')
model = models['cart']

# %%
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, plot_confusion_matrix
scaler = StandardScaler()

# %%
scaler.fit(X)


# %%
def conf_matrix(SNR=6):
    df_new = df.loc[df.SNR == SNR]
    X_new = df_new[h_vector]
    y_new = df_new.modulation_type

    predictions = model.predict(scaler.transform(X_new))
    plot_confusion_matrix(model, scaler.transform(X_new), y_new)


# %%
conf_matrix(6)
conf_matrix(10)
conf_matrix(16)
conf_matrix(26)

# %%
from sklearn.metrics import accuracy_score
def accuracies(SNR=range(6, 26, 2)):

    output = []
    for snr in SNR:
        df_new = df.loc[df.SNR == snr]
        X_new = df_new[h_vector]
        y_new = df_new.modulation_type

        predictions = model.predict(scaler.transform(X_new))
        output.append(accuracy_score(y_new, predictions))
    return output


# %%
accuracies()

# %%

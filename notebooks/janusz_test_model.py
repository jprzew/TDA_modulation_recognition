# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Using feature testing framework

# %%
# %load_ext pycodestyle_magic
# %matplotlib inline
# %flake8_on --ignore E703,E702
# %load_ext autoreload
# %autoreload 2

# %%
import path
import modurec as mr
from modurec import features
import pandas as pd
import numpy as np
import modurec.test_models as tm
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
import matplotlib.pyplot as plt

# %% [markdown]
# **Downloading the data**. The dataset with diagrams and some features can be downloaded from Xeon, by typing the following commands:
#
# *scp jprzew@153.19.6.218:~/TDA/TDA_modulation_recognition/data/stats_train.pkl .*
#
# *scp jprzew@153.19.6.218:~/TDA/TDA_modulation_recognition/data/stats_test.pkl .*
#

# %% [markdown]
# **Reading the data**

# %%
df = pd.read_pickle('../data/stats_train.pkl')

# %% [markdown]
# **What is in the data**

# %%
df.columns

# %% [markdown]
# As you can see the data contains point clouds (e.g. *cloud*, *cloud_3D*), persistence diagrams (e.g. *diagram*, *diagram_4D*). And many precalculated features. In the next section, we use these precalculated features to assess our models.

# %% [markdown]
# **Testing the models**

# %% [markdown]
# Say, for example, that we want to compare our models based on features *['H0_mean', 'H0_var']*. We do it in the following way:

# %%
tm.test_models(data=df, features=['H0_mean', 'H0_var'])

# %% [markdown]
# **Adding new features**

# %% [markdown]
# Suppose we want to add a new feature that is a normalised mean lifetime of H1-homology classes normalised with respect to number of these clases. Then we add the following method to *SignalFeatures* class. 
#
# *
#
# def H1_mean_norm(self):
#
#         return self.df.feat['H1_mean'] / self.df.feat['no_H1']*

# %% [markdown]
# We access these features in the following way

# %%
df.feat['H0_mean_norm']

# %% [markdown]
# After the above command, the feature is calculated, and written in our dataframe. Now we can test our models with the feature

# %%
tm.test_models(data=df, features=['H0_mean', 'H0_var', 'H1_mean_norm'])

# %% [markdown]
# We do not need to precalculate the new features. They can be calculated "on the fly", while testing the models. For example we can add normalised mean of zeroth homology:

# %%
tm.test_models(data=df, features=['H0_mean', 'H0_var',
                                  'H1_mean_norm', 'H0_mean_norm'])

# %% [markdown]
# **Taking many features**

# %%
features = ['no_H0', 'no_H1', 'H0_mean', 'H1_mean', 'H0_var',
       'H1_var', 'no_H0_4D', 'no_H1_4D', 'H0_mean_4D', 'H1_mean_4D',
       'H0_var_4D', 'H1_var_4D', 'H0_var_3D', 'H1_mean_3D', 'no_H0_3D', 'H1_var_3D',
       'no_H1_3D', 'H0_mean_3D']

models = tm.test_models(data=df, features=features, seed=42)

# %%
cart = models[4]

# %%
importances = cart['model'].feature_importances_
plt.barh(features, importances)

# %%
rf = models[3]
importances = rf['model'].feature_importances_
plt.barh(features, importances)

# %%
features = ['H0_mean', 'H1_var', 'no_H1_4D', 'H0_var_3D', 'H1_var_3D']

tm.test_models(data=df, features=features, seed=42)

# %%
df_reduced = df[['no_H1',
 'H0_mean',
 'H1_mean',
 'H0_var',
 'H1_var',
 'no_H1_4D',
 'H0_mean_4D',
 'H1_mean_4D',
 'H0_var_4D',
 'H1_var_4D',
 'H0_var_3D',
 'H1_mean_3D',
 'H1_var_3D',
 'no_H1_3D',
 'H0_mean_3D']]


corr = df_reduced.corr()
corr.style.background_gradient(cmap='coolwarm')
# 'RdBu_r', 'BrBG_r', & PuOr_r are other good diverging colormaps

# %% [markdown]
# Calculating features from **Khasawneh, Munch, Perea** paper.

# %%
# df.drop(columns=['kmp_features',
#                  'kmp_f1', 'kmp_f2', 'kmp_f3',
#                  'kmp_f4', 'kmp_f5'], inplace=True)

# %%
df.feat['kmp_f1'], df.feat['kmp_f2'], df.feat['kmp_f3'], df.feat['kmp_f4'], df.feat['kmp_f5']  

# %%
feat = ['no_H0', 'no_H1', 'H0_mean', 'H1_mean', 'H0_var',
       'H1_var', 'no_H0_4D', 'no_H1_4D', 'H0_mean_4D', 'H1_mean_4D',
       'H0_var_4D', 'H1_var_4D', 'H0_var_3D', 'H1_mean_3D', 'no_H0_3D', 'H1_var_3D',
       'no_H1_3D', 'H0_mean_3D', 'kmp_f1', 'kmp_f2', 'kmp_f3', 'kmp_f4', 'kmp_f5']

models = tm.test_models(data=df, features=feat, seed=42)

# %%
rf = models[3]
importances = rf['model'].feature_importances_
plt.barh(feat, importances)

# %%
cart = models[4]
importances = cart['model'].feature_importances_
plt.barh(feat, importances)

# %% [markdown]
# ## Checking new point clouds

# %%
features = ['no_H0', 'no_H1', 'H0_mean', 'H1_mean', 'H0_var',
       'H1_var', 'no_H0_4D', 'no_H1_4D', 'H0_mean_4D', 'H1_mean_4D',
       'H0_var_4D', 'H1_var_4D', 'H0_var_3D', 'H1_mean_3D', 'no_H0_3D', 'H1_var_3D',
       'no_H1_3D', 'H0_mean_3D', 'no_H1_sr',
       'H1_mean_sr', 'H0_var_sr', 'no_H0_sr', 'H1_var_sr', 'H0_mean_sr']

models = tm.test_models(data=df, features=features, seed=42)

# %%
cart = models[4]
importances = cart['model'].feature_importances_
plt.barh(feat, importances)

# %%
rf = models[3]
importances = rf['model'].feature_importances_
plt.barh(feat, importances)

# %%
df_reduced = df[['no_H1',
 'H0_mean',
 'H1_mean',
 'H0_var',
 'H1_var',
 'no_H1_4D',
 'H0_mean_4D',
 'H1_mean_4D',
 'H0_var_4D',
 'H1_var_4D',
 'H0_var_3D',
 'H1_mean_3D',
 'H1_var_3D',
 'no_H1_3D',
 'H0_mean_3D', 'no_H1_sr',
 'H1_mean_sr', 'H0_var_sr', 'no_H0_sr', 'H1_var_sr', 'H0_mean_sr']]


corr = df_reduced.corr()
corr.style.background_gradient(cmap='coolwarm')

# %%
df.feat['f1', 'H0']

# %%
df.feat['f1', 'H0', 'diag', 'cloud_4D']

# %%
df.feat['mean', 'life_time' 'H0', 'diag', 'cloud_4D']
df.feat['var', 'life_time' 'H0', 'diag', 'cloud_4D']


# %%

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

# %% [markdown]
# **Downloading the data**. The dataset with diagrams and some features can be downloaded from Xeon, by typing the following commands:
#
# *scp jprzew@153.19.6.218:~/TDA/learn_github/ml_statistics/stats_train.pkl .*
#
# *scp jprzew@153.19.6.218:~/TDA/learn_github/ml_statistics/stats_test.pkl .*
#

# %% [markdown]
# **Reading the data**

# %%
df = pd.read_pickle('../ml_statistics/stats_train.pkl')

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

# %%

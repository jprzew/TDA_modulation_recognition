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
# # %flake8_on --ignore E703,E702
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
# As you can see the dataset still contains columns with the old nomenclature (*H0_4D*, *H0_3D*, *cloud*, *cloud_3D*). Point clouds and diagrams are named due to the new nomenclature (e.g. *point_cloud_dim=3*). 

# %% [markdown]
# **List of all possible features**

# %%
mean_lifetimes = [df.ff.create_feature('mean', n=0, dim=2),
                  df.ff.create_feature('mean', n=1, dim=2),
                  df.ff.create_feature('mean', n=0, dim=3),
                  df.ff.create_feature('mean', n=1, dim=3),
                  df.ff.create_feature('mean', n=0, dim=4),
                  df.ff.create_feature('mean', n=1, dim=4)]

var_lifetimes = [df.ff.create_feature('var', n=0, dim=2),
                 df.ff.create_feature('var', n=1, dim=2),
                 df.ff.create_feature('var', n=0, dim=3),
                 df.ff.create_feature('var', n=1, dim=3),
                 df.ff.create_feature('var', n=0, dim=4),
                 df.ff.create_feature('var', n=1, dim=4)]

counting_features = [df.ff.create_feature('no', n=0, dim=2),
                     df.ff.create_feature('no', n=1, dim=2),
                     df.ff.create_feature('no', n=0, dim=3),
                     df.ff.create_feature('no', n=1, dim=3),
                     df.ff.create_feature('no', n=0, dim=4),
                     df.ff.create_feature('no', n=1, dim=4)]


feat = mean_lifetimes + var_lifetimes + counting_features


# %% [markdown]
# **Testing the models**

# %% [markdown]
# Comparing models based on the features

# %%
models = tm.test_models(data=df, features=feat, seed=42)

# %% [markdown]
# We can also access model's importances measure

# %%
[model['model'] for model in models]

# %%
cart = models[4]

importances = cart['model'].feature_importances_
plt.barh([str(f) for f in feat], importances)

# %%
rf = models[3]
importances = rf['model'].feature_importances_
plt.barh([str(f) for f in feat], importances)

# %% [markdown]
# **Models with few selected features**

# %%
mean_lifetimes = [df.ff.create_feature('mean', n=0, dim=2),
                  df.ff.create_feature('mean', n=1, dim=2)]

var_lifetimes = [df.ff.create_feature('var', n=0, dim=2),
                 df.ff.create_feature('var', n=1, dim=2)]

counting_features = [df.ff.create_feature('no', n=0, dim=2),
                     df.ff.create_feature('no', n=1, dim=2)]

feat = mean_lifetimes + var_lifetimes + counting_features

# %%
tm.test_models(data=df, features=feat)


# %% [markdown]
# **Adding new features**

# %% [markdown]
# Implementing new features requires implementing a new class. For example consider a case when we want to have a new feature being **mean lifetime of homology normalised with respect to number of homology classes**. We implement the following class 

# %%
class norm_mean(features.FeaturesFactory.Feature):

    def __init__(self, n, dim=2, step=1):
        self.dim = dim
        self.n = n
        self.step = step

    def compute(self):
        mean = self.creator.create_feature('mean',
                                           n=self.n,
                                           dim=self.dim,
                                           step=self.step)
        no = self.creator.create_feature('no',
                                         n=self.n,
                                         dim=self.dim,
                                         step=self.step)
        
        
        return mean.values() / no.values()

# %% [markdown]
# Notice that __init__ of the class needs to have arguments the same as homology (n - dimension of homology, dim - dimension of point cloud, step - step with respect to which the point cloud is created). Of course you can add your own new arguments. 

# %% [markdown]
# In the **compute** method we create features required for calculation of our feature. The method **.values** return the pandas.Series with values of our feature. 

# %% [markdown]
# Notice that our class inherits from **features.FeaturesFactory.Feature**. This is because we implement it here. Ideally it shoud be implemented directly in *features.py* file. This is also the reason why we need to add this class to our **FeaturesFactory**. We do it wit the following command 

# %%
features.FeaturesFactory.norm_mean = norm_mean

# %% [markdown]
# Now we can create and calculate tne new feature

# %%
new_feature = df.ff.create_feature('norm_mean', n=0, dim=2)

# %% [markdown]
# We access our results with

# %%
df[str(new_feature)]

# %% [markdown]
# Now we can test our models with the new feature added

# %%
tm.test_models(data=df, features=feat + [new_feature])

# %% [markdown]
# Please add here models' tests with features from **Khasawneh, Munch, Perea** paper.

# %%

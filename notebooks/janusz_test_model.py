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
from src.modurec import features
import pandas as pd
import src.modurec.test_models as tm
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
# df = pd.read_pickle('../data/stats_train_plain.pkl')
df = pd.read_pickle('/media/sf_VM/stats_train_plain.pkl')

# %% [markdown]
# **What is in the data**

# %%
df.columns

# %% [markdown]
# As you can see the dataset still contains columns with the old nomenclature (*H0_4D*, *H0_3D*, *cloud*, *cloud_3D*). Point clouds and diagrams are named due to the new nomenclature (e.g. *point_cloud_dim=3*). 

# %% [markdown]
# **List of all possible features**

# %%
epsilon = 0.05

mean_lifetimes = [df.ff.create_feature('mean', n=0, dim=2),
                  df.ff.create_feature('mean', n=1, dim=2),
                  df.ff.create_feature('mean', n=0, dim=3),
                  df.ff.create_feature('mean', n=1, dim=3),
                  df.ff.create_feature('mean', n=0, dim=4),
                  df.ff.create_feature('mean', n=1, dim=4),
                  df.ff.create_feature('mean', n=0, dim=10),
                  df.ff.create_feature('mean', n=1, dim=10),
                  df.ff.create_feature('mean', n=0, dim=2, kind='abs'),
                  df.ff.create_feature('mean', n=1, dim=2, kind='abs'),
                  df.ff.create_feature('mean', n=0, dim=10, kind='abs'),
                  df.ff.create_feature('mean', n=1, dim=10, kind='abs'),
                  df.ff.create_feature('mean', n=0, dim=2, kind='phi'),
                  df.ff.create_feature('mean', n=1, dim=2, kind='phi'),
                  df.ff.create_feature('mean', n=0, dim=10, kind='phi'),
                  df.ff.create_feature('mean', n=1, dim=10, kind='phi'),
                  df.ff.create_feature('mean', n=0, dim=2, kind='abs', fil='star'),
                  df.ff.create_feature('mean', n=0, dim=2, kind='phi', fil='star'),
                  df.ff.create_feature('mean', n=0, dim=2, step=30),
                  df.ff.create_feature('mean', n=1, dim=2, step=30),
                  df.ff.create_feature('mean', n=0, dim=4, step=30),
                  df.ff.create_feature('mean', n=1, dim=4, step=30)]
                  # df.ff.create_feature('mean', n=0, dim=4, step='symbol_rate'),
                  # df.ff.create_feature('mean', n=1, dim=4, step='symbol_rate')]

counting_features = [df.ff.create_feature('no', n=1, dim=2),
                     df.ff.create_feature('no', n=1, dim=3),
                     df.ff.create_feature('no', n=1, dim=4),
                     df.ff.create_feature('no', n=1, dim=10),
                     df.ff.create_feature('no', n=0, dim=2, eps=epsilon),
                     df.ff.create_feature('no', n=0, dim=3, eps=epsilon),
                     df.ff.create_feature('no', n=0, dim=4, eps=epsilon),
                     df.ff.create_feature('no', n=0, dim=10, eps=epsilon),
                     df.ff.create_feature('no', n=0, dim=2, kind='abs', fil='star'),
                     df.ff.create_feature('no', n=0, dim=2, kind='phi', fil='star'),]
                     # df.ff.create_feature('no', n=0, dim=4, step='symbol_rate'),
                     # df.ff.create_feature('no', n=1, dim=4, step='symbol_rate')]

variance_features = [df.ff.create_feature('var', n=0, dim=2),
                     df.ff.create_feature('var', n=1, dim=2),
                     df.ff.create_feature('var', n=0, dim=3),
                     df.ff.create_feature('var', n=1, dim=3),
                     df.ff.create_feature('var', n=0, dim=4),
                     df.ff.create_feature('var', n=1, dim=4),
                     df.ff.create_feature('var', n=0, dim=10),
                     df.ff.create_feature('var', n=1, dim=10),
                     df.ff.create_feature('var', n=0, dim=2, kind='abs'),
                     df.ff.create_feature('var', n=1, dim=2, kind='abs'),
                     df.ff.create_feature('var', n=0, dim=10, kind='abs'),
                     df.ff.create_feature('var', n=1, dim=10, kind='abs'),
                     df.ff.create_feature('var', n=0, dim=2, kind='phi'),
                     df.ff.create_feature('var', n=1, dim=2, kind='phi'),
                     df.ff.create_feature('var', n=0, dim=10, kind='phi'),
                     df.ff.create_feature('var', n=1, dim=10, kind='phi'),
                     df.ff.create_feature('var', n=0, dim=2, kind='abs', fil='star'),
                     df.ff.create_feature('var', n=0, dim=2, kind='phi', fil='star'),
                     df.ff.create_feature('var', n=0, dim=2, step=30),
                     df.ff.create_feature('var', n=1, dim=2, step=30),
                     df.ff.create_feature('var', n=0, dim=4, step=30),
                     df.ff.create_feature('var', n=1, dim=4, step=30)]
                         # df.ff.create_feature('var', n=0, dim=4, step='symbol_rate'),
                     # df.ff.create_feature('var', n=1, dim=4, step='symbol_rate')]
        
entropy_features = [df.ff.create_feature('entropy', n=0, dim=2),
                    df.ff.create_feature('entropy', n=1, dim=2),
                    df.ff.create_feature('entropy', n=0, dim=3),
                    df.ff.create_feature('entropy', n=1, dim=3),
                    df.ff.create_feature('entropy', n=0, dim=4),
                    df.ff.create_feature('entropy', n=1, dim=4),
                    df.ff.create_feature('entropy', n=0, dim=10),
                    df.ff.create_feature('entropy', n=1, dim=10),
                    df.ff.create_feature('entropy', n=0, dim=2, kind='abs'),
                    df.ff.create_feature('entropy', n=1, dim=2, kind='abs'),
                    df.ff.create_feature('entropy', n=0, dim=10, kind='abs'),
                    df.ff.create_feature('entropy', n=1, dim=10, kind='abs'),
                    df.ff.create_feature('entropy', n=0, dim=2, kind='phi'),
                    df.ff.create_feature('entropy', n=1, dim=2, kind='phi'),
                    df.ff.create_feature('entropy', n=0, dim=10, kind='phi'),
                    df.ff.create_feature('entropy', n=1, dim=10, kind='phi'),
                    df.ff.create_feature('entropy', n=0, dim=2, kind='abs', fil='star'),
                    df.ff.create_feature('entropy', n=0, dim=2, kind='phi', fil='star'),
                    df.ff.create_feature('entropy', n=0, dim=2, step=30),
                    df.ff.create_feature('entropy', n=1, dim=2, step=30),
                    df.ff.create_feature('entropy', n=0, dim=4, step=30),
                    df.ff.create_feature('entropy', n=1, dim=4, step=30, input='births'),
                    df.ff.create_feature('entropy', n=0, dim=2, input='deaths'),
                    df.ff.create_feature('entropy', n=1, dim=2, input='births'),
                    df.ff.create_feature('entropy', n=0, dim=3, input='deaths'),
                    df.ff.create_feature('entropy', n=1, dim=3, input='births'),
                    df.ff.create_feature('entropy', n=0, dim=4, input='deaths'),
                    df.ff.create_feature('entropy', n=1, dim=4, input='births'),
                    df.ff.create_feature('entropy', n=0, dim=10, input='deaths'),
                    df.ff.create_feature('entropy', n=1, dim=10, input='births'),
                    df.ff.create_feature('entropy', n=1, dim=2, kind='abs', input='births'),
                    df.ff.create_feature('entropy', n=1, dim=10, kind='abs', input='births'),
                    df.ff.create_feature('entropy', n=1, dim=2, kind='phi', input='births'),
                    df.ff.create_feature('entropy', n=1, dim=10, kind='phi', input='births'),
                    df.ff.create_feature('entropy', n=1, dim=2, kind='abs', input='deaths'),
                    df.ff.create_feature('entropy', n=1, dim=10, kind='abs', input='deaths'),
                    df.ff.create_feature('entropy', n=1, dim=2, kind='phi', input='deaths'),
                    df.ff.create_feature('entropy', n=1, dim=10, kind='phi', input='deaths')]
        
# df.ff.create_feature('entropy', n=0, dim=20),
# df.ff.create_feature('entropy', n=1, dim=20),
     
p = 2
wasser_features=[df.ff.create_feature('wasser_ampl', n=0, p=p, dim=2),
                 df.ff.create_feature('wasser_ampl', n=1, p=p, dim=2),
                 df.ff.create_feature('wasser_ampl', n=1, p=p, dim=3),
                 df.ff.create_feature('wasser_ampl', n=1, p=p, dim=3),
                 df.ff.create_feature('wasser_ampl', n=0, p=p, dim=4),
                 df.ff.create_feature('wasser_ampl', n=1, p=p, dim=4),
                 df.ff.create_feature('wasser_ampl', n=0, p=p, dim=10),
                 df.ff.create_feature('wasser_ampl', n=1, p=p, dim=10),
                 df.ff.create_feature('wasser_ampl', n=0, dim=2, p=p, kind='abs'),
                 df.ff.create_feature('wasser_ampl', n=1, dim=2, p=p, kind='abs'),
                 df.ff.create_feature('wasser_ampl', n=0, dim=10, p=p, kind='abs'),
                 df.ff.create_feature('wasser_ampl', n=1, dim=10, p=p, kind='abs'),
                 df.ff.create_feature('wasser_ampl', n=0, dim=2, p=p, kind='phi'),
                 df.ff.create_feature('wasser_ampl', n=1, dim=2, p=p, kind='phi'),
                 df.ff.create_feature('wasser_ampl', n=0, dim=10, p=p, kind='phi'),
                 df.ff.create_feature('wasser_ampl', n=1, dim=10, p=p, kind='phi'),
                 df.ff.create_feature('wasser_ampl', n=0, dim=2, kind='abs', fil='star', p=p),
                 df.ff.create_feature('wasser_ampl', n=0, dim=2, kind='phi', fil='star', p=p),
                 df.ff.create_feature('wasser_ampl', n=0, dim=2, step=30, p=p),
                 df.ff.create_feature('wasser_ampl', n=1, dim=2, step=30, p=p),
                 df.ff.create_feature('wasser_ampl', n=0, dim=4, step=30, p=p),
                 df.ff.create_feature('wasser_ampl', n=1, dim=4, step=30, p=p)]


feat = mean_lifetimes + counting_features + variance_features + entropy_features + wasser_features


# %% [markdown]
# **Testing the models**

# %% [markdown]
# Comparing models based on the features

# %%
models = tm.test_models(data=df, features=feat, seed=42)

# %% [markdown]
# We can also access model's importances measure

# %%
df.filter(regex='^diagram(.*)$')

# %%
[model['model'] for model in models]

# %%
cart = models[4]

importances = cart['model'].feature_importances_
plt.figure(figsize=(15, 15))
plt.barh([str(f) for f in feat], importances)

# %%
rf = models[3]
importances = rf['model'].feature_importances_
plt.figure(figsize=(15, 15))
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
df.columns

# %%
df.mr.plot_persistence_diagrams('diagram_kind=phi_fil=star', modulation_type='32PSK')

# %%
df.mr.plot_persistence_diagrams('diagram_kind=phi_fil=star', modulation_type='16PSK')

# %%
df.mr.plot_persistence_diagrams('diagram_kind=phi_fil=star', modulation_type='BPSK')

# %%
df.mr.plot_persistence_diagrams('diagram_kind=phi_fil=star', modulation_type='QPSK')

# %%
df.mr.plot_persistence_diagrams('diagram_kind=phi_fil=star', modulation_type='8PSK')

# %%
df.mr.plot_persistence_diagrams('diagram_kind=abs_fil=star', modulation_type='32PSK')

# %%
df.mr.plot_persistence_diagrams('diagram_kind=abs_fil=star', modulation_type='16PSK')

# %%
df.mr.plot_persistence_diagrams(data_col='diagram_dim=4_step=30', modulation_type='32PSK')

# %%
df.mr.plot_persistence_diagrams(data_col='diagram_dim=4_step=30', modulation_type='16PSK')

# %%

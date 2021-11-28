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
# # Dimension reduction analysis (t-SNE)

# %%
# %load_ext pycodestyle_magic
# %matplotlib inline
# # %flake8_on --ignore E703,E702
# %load_ext autoreload
# %autoreload 2

# %%
import path
import modurec
import pandas as pd
import numpy as np
import modurec.test_models as tm
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from bioinfokit.visuz import cluster
from IPython.display import Image
import plotly.express as px

from sklearn.manifold import TSNE

# %% [markdown]
# **Loading data**

# %%
df = pd.read_pickle('../data/stats_train.pkl')

# %% [markdown]
# **Preparing the data**

# %%
epsilon = 0.05

mean_lifetimes = [df.ff.create_feature('mean', n=0, dim=2),
                  df.ff.create_feature('mean', n=1, dim=2),
                  df.ff.create_feature('mean', n=0, dim=3),
                  df.ff.create_feature('mean', n=1, dim=3),
                  df.ff.create_feature('mean', n=0, dim=4),
                  df.ff.create_feature('mean', n=1, dim=4)]
                  # df.ff.create_feature('mean', n=0, dim=4, step='symbol_rate'),
                  # df.ff.create_feature('mean', n=1, dim=4, step='symbol_rate')]

counting_features = [df.ff.create_feature('no', n=1, dim=2),
                     df.ff.create_feature('no', n=1, dim=3),
                     df.ff.create_feature('no', n=1, dim=4),
                     df.ff.create_feature('no', n=0, dim=2, eps=epsilon),
                     df.ff.create_feature('no', n=0, dim=3, eps=epsilon),
                     df.ff.create_feature('no', n=0, dim=4, eps=epsilon)]
                     # df.ff.create_feature('no', n=0, dim=4, step='symbol_rate'),
                     # df.ff.create_feature('no', n=1, dim=4, step='symbol_rate')]

variance_features = [df.ff.create_feature('var', n=0, dim=2),
                     df.ff.create_feature('var', n=1, dim=2),
                     df.ff.create_feature('var', n=0, dim=3),
                     df.ff.create_feature('var', n=1, dim=3),
                     df.ff.create_feature('var', n=0, dim=4),
                     df.ff.create_feature('var', n=1, dim=4)]
                     # df.ff.create_feature('var', n=0, dim=4, step='symbol_rate'),
                     # df.ff.create_feature('var', n=1, dim=4, step='symbol_rate')]


feat = mean_lifetimes + counting_features + variance_features

# %%
numeric_cols = list(map(str, feat))
# X - explanatory data
# y - response variable
X = df.loc[:, numeric_cols].values 
X = StandardScaler().fit_transform(X)
y = df.loc[:, 'modulation_type']

# %%
RS=42
tsne = TSNE(random_state=RS).fit_transform(X)

# %%
tsne

# %%
tsne_df = pd.DataFrame(data = tsne,
                           columns = ['y1', 'y2'])

final_df = pd.concat([tsne_df, df[['modulation_type']].reset_index()], axis = 1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('y1', fontsize = 15)
ax.set_ylabel('y2', fontsize = 15)
ax.set_title('2 dimensional t-SNE', fontsize = 20)

targets = y.unique()
colors = ['r', 'g', 'b']


sns.scatterplot(data=final_df, x='y1', y='y2', hue='modulation_type')

# %%
df.columns

# %%
list(map(str, mean_lifetimes))

# %%

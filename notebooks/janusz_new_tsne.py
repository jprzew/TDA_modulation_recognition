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
# # t-SNE after adding new features

# %%
# %load_ext pycodestyle_magic
# %matplotlib inline
# # %flake8_on --ignore E703,E702
# %load_ext autoreload
# %autoreload 2

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# from bioinfokit.visuz import cluster
# from IPython.display import Image
# import plotly.express as px

from sklearn.manifold import TSNE

# %% [markdown]
# **Loading data**

# %%
# df = pd.read_pickle('../data/stats_train_plain.pkl')
# df = pd.read_pickle('../data/stats_large.pkl')
df = pd.read_pickle('/media/sf_VM/stats_train_plain.pkl')

# %% [markdown]
# **Preparing the data**

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
                  df.ff.create_feature('mean', n=1, dim=10, kind='phi')]
                  # df.ff.create_feature('mean', n=0, dim=4, step='symbol_rate'),
                  # df.ff.create_feature('mean', n=1, dim=4, step='symbol_rate')]

counting_features = [df.ff.create_feature('no', n=1, dim=2),
                     df.ff.create_feature('no', n=1, dim=3),
                     df.ff.create_feature('no', n=1, dim=4),
                     df.ff.create_feature('no', n=1, dim=10),
                     df.ff.create_feature('no', n=0, dim=2, eps=epsilon),
                     df.ff.create_feature('no', n=0, dim=3, eps=epsilon),
                     df.ff.create_feature('no', n=0, dim=4, eps=epsilon),
                     df.ff.create_feature('no', n=0, dim=10, eps=epsilon)]
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
                     df.ff.create_feature('var', n=1, dim=10, kind='phi')]
                         # df.ff.create_feature('var', n=0, dim=4, step='symbol_rate'),
                     # df.ff.create_feature('var', n=1, dim=4, step='symbol_rate')]
        
p = 2
new_features = [df.ff.create_feature('entropy', n=0, dim=2),
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
                # df.ff.create_feature('entropy', n=0, dim=20),
                # df.ff.create_feature('entropy', n=1, dim=20),
                df.ff.create_feature('wasser_ampl', n=0, p=p, dim=2),
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
                df.ff.create_feature('wasser_ampl', n=1, dim=10, p=p, kind='phi')]


feat = mean_lifetimes + counting_features + variance_features + new_features

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


plot = sns.scatterplot(data=final_df, x='y1', y='y2', hue='modulation_type')
fig = plot.get_figure()
fig.savefig('figure.png') 



# %%
import plotly.express as px
fig = px.scatter(final_df, x='y1', y='y2', color='modulation_type')

# %%
fig

# %%

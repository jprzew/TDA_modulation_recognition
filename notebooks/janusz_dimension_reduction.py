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
# # Dimension reduction analyses

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
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from bioinfokit.visuz import cluster
from IPython.display import Image
import plotly.express as px

# %% [markdown]
# **Loading data**

# %%
df = pd.read_pickle('../data/stats_train.pkl')

# %% [markdown]
# **Preparing the data**

# %%
numeric_cols = ['no_H1',
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
                'H0_mean_3D',
                'no_H1_sr',
                'H1_mean_sr',
                'H0_var_sr',
                'no_H0_sr',
                'H1_var_sr',
                'H0_mean_sr']

# %%
# x - explanatory data
# y - response variable
x = df.loc[:, numeric_cols].values 
x = StandardScaler().fit_transform(x)
y = df.loc[:, 'modulation_type']

# %% [markdown]
# **Creating PCA-model**

# %%
pca = PCA(n_components=4)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents,
                           columns = ['PC1', 'PC2', 'PC3',
                                      'PC4'])

# %%
principalDf

# %%
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 

ax.plot(pca.explained_variance_ratio_)

# %% [markdown]
# From the above plot we can see that three principal component are sufficient for our analysis. Therefore we refit the PCA-model. 

# %%
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents,
                           columns = ['PC1', 'PC2', 'PC3'])

# %%
finalDf = pd.concat([principalDf, df[['modulation_type']].reset_index()], axis = 1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)

targets = y.unique()
colors = ['r', 'g', 'b']


sns.scatterplot(data=finalDf, x='PC1', y='PC2', hue='modulation_type')

# %%
total_var = pca.explained_variance_ratio_.sum() * 100

fig = px.scatter_3d(
    principalComponents, x=0, y=1, z=2, color=finalDf['modulation_type'],
    title=f'Total Explained Variance: {total_var:.2f}%',
    labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}
)
fig.show()

# %%
total_var = pca.explained_variance_ratio_.sum() * 100

labels = {str(i): f"PC{i+1}" for i in range(3)}
labels['color'] = 'modulation_type'

fig = px.scatter_matrix(
    principalComponents,
    color=finalDf.modulation_type,
    dimensions=range(3),
    labels=labels,
    title=f'Total Explained Variance: {total_var:.2f}%',
)
fig.update_traces(diagonal_visible=False)
fig.show()

# %% [markdown]
# **Restricting to the more problematic modulations**

# %%
finalDf_restricted = finalDf.loc[~finalDf.modulation_type.isin(['BPSK', 'FM', 'QPSK', 'GMSK', 'OQPSK'])]

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)


sns.scatterplot(data=finalDf_restricted,
                x='PC1', y='PC2', hue='modulation_type')

# %%
total_var = pca.explained_variance_ratio_.sum() * 100

fig = px.scatter_3d(
    finalDf_restricted[['PC1', 'PC2', 'PC3']], x='PC1', y='PC2', z='PC3', color=finalDf_restricted['modulation_type'],
    title=f'Total Explained Variance: {total_var:.2f}%',
    labels={'PC1': 'PC1', 'PC2': 'PC2', 'PC3': 'PC3'}
)
fig.show()

# %% [markdown]
# **Analysis of the variables**

# %%
loadings = pca.components_ # coefficients of the components
num_pc = pca.n_features_
pc_list = ["PC"+str(i) for i in list(range(1, num_pc+1))]

loadings_df = pd.DataFrame.from_dict(dict(zip(pc_list, loadings)))
loadings_df['variable'] = numeric_cols
loadings_df = loadings_df.set_index('variable')
loadings_df

fig = plt.figure(figsize = (8,8))
ax = sns.heatmap(loadings_df, annot=True, cmap='Spectral')
plt.show()

# %%
# 2D
cluster.pcaplot(x=loadings[0], y=loadings[1], labels=numeric_cols, 
                var1=round(pca.explained_variance_ratio_[0]*100, 2),
                var2=round(pca.explained_variance_ratio_[1]*100, 2))


display(Image('pcaplot_2d.png'))

# %%
cluster.pcaplot(x=loadings[1], y=loadings[2], labels=numeric_cols, 
                var1=round(pca.explained_variance_ratio_[1]*100, 2),
                var2=round(pca.explained_variance_ratio_[2]*100, 2))

display(Image('pcaplot_2d.png'))

# %%
pca_scores = PCA().fit_transform(x)

cluster.biplot(cscore=pca_scores, loadings=loadings, labels=numeric_cols,
               var1=round(pca.explained_variance_ratio_[0]*100, 2),
               var2=round(pca.explained_variance_ratio_[1]*100, 2))
display(Image('biplot_2d.png'))

# %% [markdown]
# ### Conclusions
#
# Groups of variables wrt. PC1, PC2, PC3
#
# **GROUP1**
#
# *subgroup 1*
#
# H0_mean_3D
#
# H1_mean_3D  +
# H1_mean_sr  +
#
# *subgroup 2*
#
# H0_var		+
# H0_mean	    +
# H0_var_3D	+
# H0_mean_4D	-
# H0_var_sr	+
# H0_mean_sr	+
# H0_var_4D	-
# no_H1_sr	
#
#
# **GROUP2**
#
# no_H1
# no_H1_4D
#
#
# **GROUP3**
#
# no_H0_sr
#
#
# **GROUP4**
#
# no_H1_3D
#
#
# **GROUP5**
#
# *subgroup1*
#
# H1_mean_4D	=
#
# *subgroup2*
# H1_var_4D	=
# H1_var_sr
# H1_var_3D
# H1_mean		+
# H1_var		+

# %% [markdown]
# ## PCA after removing easy modulations

# %%
df_restricted = df.loc[~df.modulation_type.isin(['BPSK', 'FM', 'QPSK', 'GMSK', 'OQPSK'])]

x = df_restricted.loc[:, numeric_cols].values 
x = StandardScaler().fit_transform(x)
y = df_restricted.loc[:, 'modulation_type']

# %%
pca = PCA(n_components=4)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents,
                           columns = ['PC1', 'PC2', 'PC3', 'PC4'])

# %%
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 

ax.plot(pca.explained_variance_ratio_)

# %%
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents,
                           columns = ['PC1', 'PC2'])

# %%
finalDf = pd.concat([principalDf, df_restricted[['modulation_type']].reset_index()], axis = 1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)

targets = y.unique()
colors = ['r', 'g', 'b']


sns.scatterplot(data=finalDf, x='PC1', y='PC2', hue='modulation_type')

# %%
loadings = pca.components_ # coefficients of the components
num_pc = pca.n_features_
pc_list = ["PC"+str(i) for i in list(range(1, num_pc+1))]

loadings_df = pd.DataFrame.from_dict(dict(zip(pc_list, loadings)))
loadings_df['variable'] = numeric_cols
loadings_df = loadings_df.set_index('variable')
loadings_df

fig = plt.figure(figsize = (8,8))
ax = sns.heatmap(loadings_df, annot=True, cmap='Spectral')
plt.show()

# %%
# 2D
cluster.pcaplot(x=loadings[0], y=loadings[1], labels=numeric_cols, 
                var1=round(pca.explained_variance_ratio_[0]*100, 2),
                var2=round(pca.explained_variance_ratio_[1]*100, 2))


display(Image('pcaplot_2d.png'))

# %% [markdown]
#
# GROUP 1
#
# H0_var_sr
# H0_mean_3D
# H0_var_3D
# H0_mean_sr
# H0_mean_4D
#
# GROUP2
#
# H1_mean_sr
# H1_var_sr
#
#
# GROUP3
# no_H1_4D
# no_H1_sr
#
#
# GROUP4
#
# H1_mean_3D
# H1_mean
# H1_var_4D
# H1_var_3D
# H1_mean_3D
# H1_var
#
# GrOUP5
# no_H1
#
# Group6
# no_H0_sr
#
# Group7
# no_H1_3D
#
#
# GROUP8
# H1_mean_4D
#
# GROUP9
# H0_mean
# H0_var

# %%

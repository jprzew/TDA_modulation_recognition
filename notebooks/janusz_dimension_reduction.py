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
# # Dimension reduction analyses

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
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from bioinfokit.visuz import cluster

# %%
df = pd.read_pickle('../data/stats_train.pkl')

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
                'H0_mean_3D', 'no_H1_sr',
                'H1_mean_sr', 'H0_var_sr', 'no_H0_sr', 'H1_var_sr', 'H0_mean_sr']

x = df.loc[:, numeric_cols].values 
x = StandardScaler().fit_transform(x)
y = df.loc[:, 'modulation_type']

# %%
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents,
                           columns = ['principal component 1', 'principal component 2'])

# %%
principalDf

# %%
finalDf = pd.concat([principalDf, df[['modulation_type']].reset_index()], axis = 1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)

targets = y.unique()
colors = ['r', 'g', 'b']


sns.scatterplot(data=finalDf, x='principal component 1', y='principal component 2', hue='modulation_type')

# for target, color in zip(targets,colors):
#     indicesToKeep = finalDf['modulation_type'] == target
#     ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
#                , finalDf.loc[indicesToKeep, 'principal component 2']
#                , c = color
#                , s = 50)
# ax.legend(targets)
# ax.grid()

# %%
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)


sns.scatterplot(data=finalDf.loc[~finalDf.modulation_type.isin(['BPSK', 'FM', 'QPSK', 'GMSK', 'OQPSK'])],
                x='principal component 1', y='principal component 2', hue='modulation_type')

# %%
pca.components_

# %%
pca.components_

# %%
# get the component variance
# Proportion of Variance (from PC1 to PC2)
pca.explained_variance_ratio_

# Cumulative proportion of variance (from PC1 to PC2)   
np.cumsum(pca.explained_variance_ratio_)
       
# component loadings or weights (correlation coefficient between original variables and the component) 
# component loadings represents the elements of the eigenvector
# the squared loadings within the PCs always sums to 1
loadings = pca.components_
num_pc = pca.n_features_
pc_list = ["PC"+str(i) for i in list(range(1, num_pc+1))]

loadings_df = pd.DataFrame.from_dict(dict(zip(pc_list, loadings)))
loadings_df['variable'] = numeric_cols
loadings_df = loadings_df.set_index('variable')
loadings_df


# positive and negative values in component loadings reflects the positive and negative 
# correlation of the variables with the PCs. Except A and B, all other variables have 
# positive projection on first PC.

# get correlation matrix plot for loadings
# import seaborn as sns
# import matplotlib.pyplot as plt
ax = sns.heatmap(loadings_df, annot=True, cmap='Spectral')
plt.show()

# %%
# 2D
cluster.pcaplot(x=loadings[0], y=loadings[1], labels=numeric_cols, 
    var1=round(pca.explained_variance_ratio_[0]*100, 2),
    var2=round(pca.explained_variance_ratio_[1]*100, 2))

# %%
pca_scores = PCA().fit_transform(x)

cluster.biplot(cscore=pca_scores, loadings=loadings, labels=numeric_cols,
               var1=round(pca.explained_variance_ratio_[0]*100, 2),
               var2=round(pca.explained_variance_ratio_[1]*100, 2))

# %%

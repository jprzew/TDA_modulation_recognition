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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import path
import modurec as mr
import numpy as np

from modurec import signal_reader
from ripser import Rips

from mpl_toolkits import mplot3d


# %%
# magic functions that do not work in current ipython --version
# %load_ext pycodestyle_magic

# auto check each cell, E703 - "statement ends with a semicolon"
# %flake8_on --ignore E703
# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

# %%
df = pd.read_pickle('../ml_statistics/stats_train.pkl')

# %%
df.cloud_3D.iloc[0]

# %%
rips = Rips()

# %%
cloud = df.cloud_3D.iloc[0]
x, y, z = cloud[:, 0], cloud[:, 1], cloud[:, 2]


# %%
# Creating figure
fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
 
# Creating plot
ax.scatter3D(x, y, z, color = "green")
plt.title("simple 3D scatter plot")
 
# show plot
plt.show()

# %%
rips.fit_transform(cloud)

# %%

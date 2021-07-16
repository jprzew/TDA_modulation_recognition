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
# TODO: Tidy it up

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import path
import modurec as mr
import numpy as np

from modurec import signal_reader


# %%
# magic functions that do not work in current ipython --version
# %load_ext pycodestyle_magic

# auto check each cell, E703 - "statement ends with a semicolon"
# %flake8_on --ignore E703
# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

# %%
df = signal_reader.get_signal_df_from_numpy()

# %%
df

# %%
df.mr.add_statistics(inplace=True)

# %%

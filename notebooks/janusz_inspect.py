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
# %load_ext pycodestyle_magic
# %matplotlib inline
# auto check each cell, E703 - "statement ends with a semicolon"
# %flake8_on --ignore E703,E702
# %load_ext autoreload
# %autoreload 2

# %%
# !ipython --version

# %%
import path
import modurec as mr
from modurec import signal_reader
import pandas as pd
import re
import numpy as np

# %%
df_ext = signal_reader.get_signal_df_from_numpy(data_path='../../temp')

# %%
# modulations = {'16PSK', '16QAM', '32PSK', '32QAM', '64APSK', '8PSK',
#               'BPSK', 'FM', 'GMSK','OQPSK', 'QPSK'}
modulations = {'BPSK'}

df = df_ext.loc[df_ext['modulation_type'].isin(modulations)]
df = df.loc[df.SNR == 30]

# %%
df.mr.plot_IQ(max_rows=100)

# %%
print(df_ext.groupby(by=['modulation_type', 'SNR']).count().to_string())

# %%

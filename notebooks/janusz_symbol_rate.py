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
import path
import modurec as mr
from modurec import signal_reader
import numpy as np
import pandas as pd

# %%
df_ext = signal_reader.get_signal_df_from_numpy()

# %%
modulations = {'16PSK', '16QAM', '32PSK', '32QAM', '64APSK', '8PSK',
               'BPSK', 'FM', 'GMSK', 'OQPSK', 'QPSK'}

# %%
df = df_ext.loc[df_ext['modulation_type'].isin(modulations)]
# df = df.loc[df['SNR'] == 10.0]

# %%
df.mr.plot_samples(max_rows=float('inf'))

# %%
df.mr.add_power_spectr()

# %%
df['power_spectr_log'] = df['power_spectr'].apply(lambda x: np.log(x))
df.mr.plot_samples(data_col='power_spectr_log', max_rows=float('inf'))

# %%
df.mr.plot_samples(data_col='power_spectr', max_rows=float('inf'))

# %%
df.mr.estimate_symbol_rate()

# %%
df['symbol_rate']

# %%
print(df['symbol_rate'].to_string())

# %%
df['SNR']

# %%
int(np.around(float('inf')))

# %%

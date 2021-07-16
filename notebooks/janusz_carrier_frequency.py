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
# TODO: TIDY IT UP

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
import pandas as pd
import re
import numpy as np

from scipy.signal import hilbert
from scipy.stats import trim_mean


# %%
def series_to_numpy(series):
    series = series.map(lambda x: re.sub(r'[\[\]\n]', r'', x))
    series = series.map(lambda x: np.fromstring(x, sep=' '))
    return series  


# %%
def read_signals_df(csv_path):
    df = pd.read_csv(csv_path, index_col=[0])
    df['signal_sample'] = series_to_numpy(df['np_signal_sample'])
    return df

# %%
# def add_phase(df, data_col='signal_sample'):
#     column = df[data_col].apply(hilbert)
#     column = column.apply(lambda x: np.unwrap(np.angle(x))/(2*np.pi))
#     df['phase'] = column


# %%

# %%
def differentiate(df, data_col='signal_sample'):
    df[data_col+'_diff'] = df[data_col].apply(np.diff)


# %% [markdown]
# # Reading in the simulated data

# %%
df = read_signals_df('../data/signal_sample.csv')

# %%
df.mr.plot_samples()

# %% [markdown]
# ## Autocorrelation

# %%
df.mr.add_autocorr()
df.mr.plot_samples(data_col='signal_autocorr')

# %%
df.mr.add_fourier(data_col='signal_autocorr', fft_col='signal_spectrum')

# %%
# df['power_spectrum'] = df['signal_spectrum'].apply(lambda x: np.log(abs(x)))
df['power_spectrum'] = df['signal_spectrum'].apply(lambda x: abs(x))

# %%
df.mr.plot_samples(data_col='power_spectrum')

# %%
df.mr.add_fourier()
# df['fft_amplitude'] = df['signal_fft'].apply(lambda x: np.log(abs(x)))
df['fft_amplitude'] = df['signal_fft'].apply(lambda x: abs(x)**2)

df.mr.plot_samples(data_col='fft_amplitude')

# %% [markdown]
# ## Calculating instantenous phase

# %%
df.mr.add_phase()
df.mr.plot_samples(data_col='signal_phase')

# %% [markdown]
# ## Calculating instantenous frequency

# %%
differentiate(df, data_col='signal_phase')

# %%
df.mr.plot_samples(data_col='signal_phase_diff')

# %%
# df['freq_estim'] = df['signal_phase_diff'].apply(lambda x: 1/trim_mean(x, proportiontocut=0.3))

# %%
df.mr.estimate_carrier(estim_col='estim')

# %%
df.mr.add_carrier(estim_col='estim')

# %%

# %%
with pd.option_context('display.max_rows',
                       None,
                       'display.max_columns',
                       None):  
    print(df[['modulation_type', 'samples_per_period', 'estim']])

# %% [markdown]
# # Reading in the Radioml 2018 data

# %%
df_ext = signal_reader.get_signal_df_from_numpy()

# %% [markdown]
# **Subsetting modulations and selecting SNR**

# %%
modulations = {'16PSK', '16QAM', '32PSK', '32QAM', '64APSK', '8PSK',
               'BPSK', 'FM', 'GMSK','OQPSK', 'QPSK'}

# %%
df = df_ext.loc[df_ext['modulation_type'].isin(modulations)]
#df = df.loc[df['SNR'] == 10.0]

# %%
df.mr.plot_samples(max_rows=200)

# %% [markdown]
# # IQ - plots

# %%
df.mr.plot_IQ(max_rows=200)

# %% [markdown]
# ## Fourier transform

# %%
df.mr.add_fourier()
# df['fft_amplitude'] = df['signal_fft'].apply(lambda x: np.log(abs(x)))
df['fft_amplitude'] = df['signal_fft'].apply(lambda x: abs(x)**2)

df.mr.plot_samples(data_col='fft_amplitude')

# %% [markdown]
# ## Periodic autocorrelation

# %%
df.mr.add_autocorr()
df.mr.plot_samples(data_col='signal_autocorr')

# %% [markdown]
# ## Calculating instantenous phase

# %%
df.mr.add_phase()
df.mr.plot_samples(data_col='signal_phase')

# %% [markdown]
# ## Calculating instantenous frequency

# %%
differentiate(df, data_col='signal_phase')

# %%
df.mr.plot_samples(data_col='signal_phase_diff')

# %%
# df['freq_estim'] = df['phase_diff'].apply(lambda x:\
#                                           1/trim_mean(x,
#                                                       proportiontocut=0.3))

# %%
df.mr.estimate_carrier()

# %%
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(df[['modulation_type', 'carrier_estim']])

# %% [markdown]
# # Power spectrum

# %%
df.mr.add_power_spectr()

# %%
df['power_spectr_log'] = df['power_spectr'].apply(lambda x: np.log(x))
df.mr.plot_samples(data_col='power_spectr_log', max_rows=float('inf'))

# %%
df.mr.estimate_symbol_rate()

# %%
df['symbol_rate']

# %% [markdown]
# # Garbage

# %%
from sklearn.cluster import KMeans

x = np.array([1, 1, 1, 0, 0, 0, 0])
# x = np.reshape(x, (1, -1))
kmeans = KMeans(n_clusters=2).fit(x.reshape(-1, 1))

# %%
df.loc[[1870, 1871]].mr.plot_samples(data_col='power_spectr_log')

# %%
x = df.loc[1871]['power_spectr_log']
# x = np.reshape(x, (1, -1))
kmeans = KMeans(n_clusters=2).fit(x.reshape(-1, 1))
np.min(kmeans.cluster_centers_)

# %%
kmeans.cluster_centers_

# %%
a = np.array([1, 2, 3, 4])

# %%
np.fft.fft(np.correlate(a, a, mode='same'))

# %%
abs(np.fft.fft(a))**2


# %%
def periodic_corr(x, y):
    """Periodic correlation, implemented using np.correlate.

    x and y must be real sequences with the same length.
    """
    return np.correlate(x, np.hstack((y[1:], y)), mode='valid')


# %%
np.fft.fft(periodic_corr(a, a))

# %%
abs(np.fft.fft(a))**2

# %%
sort(np.array([3, 2, 10]))

# %%

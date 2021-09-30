import sys
import warnings
import re
import math
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import hilbert
from scipy.stats import trim_mean
from scipy.signal import welch
from scipy.signal.windows import hann
from ripser import Rips
from sklearn.cluster import KMeans

from .utility import rolling_window

# TODO: refactor the code:
#  remove unnecessary functions in SignalDataFrame...
#  and remove these imports
from scipy import signal

# #############################################################################
#                          Package configuration
# #############################################################################

inf = np.inf
# config = PkgConfig()

try:

    from ._config import personalize
    config = personalize(config)

except ImportError:

    msg = "personal_config.py does not exist.\n"
    msg += "Please create a file to personalize your settings."
    # warnings.warn(msg)


# #############################################################################
#
#                     DataFrame and Series extensions
#
# #############################################################################
with warnings.catch_warnings():

    warnings.simplefilter(action='ignore', category=UserWarning)
    #
    # @pd.api.extensions.register_extension_dtype
    # class MyExtensionDtype(pd.api.extensions.ExtensionDtype):
    #     name = "NumpySeries"

    # ##########################################################################
    #                        Series 'np' extensions
    # ##########################################################################
    @pd.api.extensions.register_series_accessor('np')
    class NumpySeries:

        np.set_printoptions(suppress=True)
        np.set_printoptions(threshold=np.inf)

        def __init__(self, series):
            self.series = series

        # @property
        # def shape(self):
        #      return 0

        def mean(self):
            return self.series.map(lambda x: np.mean(np.ma.masked_invalid(x)))

        def var(self):
            return self.series.map(lambda x: np.var(np.ma.masked_invalid(x)))

        def diff(self, axis=-1):
            return self.series.map(lambda x: np.diff(x, axis=axis).reshape(-1))

        def rolling_window(self, window=2, **kargs):
            def rw(a): return rolling_window(a, window=window, **kargs)
            return self.series.map(rw)

    # ##########################################################################
    #                        Series 'mr' extensions
    # ##########################################################################
    @pd.api.extensions.register_series_accessor('mr')
    class SignalSeries:

        def __init__(self, series):
            self.series = series

        def to_numpy_fromstring(self):
            ser = (self.series.map(lambda x: re.sub(r'[\[\]\n]', r'', x))
                              .map(lambda x: np.fromstring(x, sep=' ')))
            return ser

        def to_numpy(self):
            ser = (self.series.map(lambda x: re.sub(r'\]\n \[', r'], [', x))
                              .map(lambda x: re.sub(r'(\d+)\s+', r'\1, ', x))
                              .map(lambda x: re.sub(r'(\d+\.)\s+', r'\1, ', x))
                              .map(lambda x: np.array(eval(x))))
            return ser

    # ##########################################################################
    #                        DataFrame 'mr' extensions
    # ##########################################################################

    @pd.api.extensions.register_dataframe_accessor('mr')
    class SignalDataFrame:

        def __init__(self, df):
            self.df = df

        def add_point_cloud(self,
                            data_col='signal_sample',
                            data_colQ='signal_sampleQ',
                            point_cloud_col='point_cloud',
                            window=2):

            def __to_real(x):
                array = np.vstack((np.real(x), np.imag(x))).T
                return array.ravel()

            def __complex_cloud_to_real(cloud):
                return np.apply_along_axis(lambda x:
                                           __to_real(x), axis=1, arr=cloud)

            data = self.df.apply(lambda x: x[data_col] + 1j*x[data_colQ],
                                 axis=1)
            data = data.np.rolling_window(window=window)
            data = data.apply(__complex_cloud_to_real)
            self.df[point_cloud_col] = data

# TODO: Refactor this function (window size is not used!)
# TODO: Use new version of features.py in the function below
        def add_statistics(self,
                           data_col='signal_sample',
                           inplace=False,
                           window=2):
            rips = Rips()
            if inplace:
                df = self.df
            else:
                df = self.df.copy()

            # df['point_cloud'] = df[data_col].np.rolling_window()
            df.mr.add_point_cloud(window=1)
            df['diagram'] = df['point_cloud'].map(rips.fit_transform)

            df[['H0', 'H1']] =\
                pd.DataFrame(df['diagram'].tolist(), index=df.index)

            df['H0_life_time'] = df['H0'].np.diff(axis=1)
            df['H1_life_time'] = df['H1'].np.diff(axis=1)
            #
            #
            df['no_H0'] = df['H0'].map(lambda x: x.shape[0])
            df['no_H1'] = df['H1'].map(lambda x: x.shape[0])

            df['H0_mean'] = df['H0_life_time'].np.mean()
            df['H1_mean'] = df['H1_life_time'].np.mean()

            df['H0_var'] = df['H0_life_time'].np.var()
            df['H1_var'] = df['H1_life_time'].np.var()

            # 4D-cloud calculation
            df.mr.add_point_cloud(window=2, point_cloud_col='cloud_4D')
            df['diagram_4D'] = df['cloud_4D'].map(rips.fit_transform)

            df[['H0_4D', 'H1_4D']] =\
                pd.DataFrame(df['diagram_4D'].tolist(), index=df.index)

            df['H0_life_time_4D'] = df['H0_4D'].np.diff(axis=1)
            df['H1_life_time_4D'] = df['H1_4D'].np.diff(axis=1)

            df['no_H0_4D'] = df['H0_4D'].map(lambda x: x.shape[0])
            df['no_H1_4D'] = df['H1_4D'].map(lambda x: x.shape[0])

            df['H0_mean_4D'] = df['H0_life_time_4D'].np.mean()
            df['H1_mean_4D'] = df['H1_life_time_4D'].np.mean()

            df['H0_var_4D'] = df['H0_life_time_4D'].np.var()
            df['H1_var_4D'] = df['H1_life_time_4D'].np.var()

            # 3D-cloud calculation
            df.mr.add_point_cloud(window=1, point_cloud_col='cloud_3D')
            df['cloud_3D'] = \
                self.df.cloud_3D.apply(lambda x:
                                       np.column_stack([x, range(x.shape[0])]))

            # Deprecated code for 3D-cloud calculation
            # df['cloud_3D'] = df[data_col].np.rolling_window(window=3)

            diagram_3D = df['cloud_3D'].map(rips.fit_transform)
            df[['H0_3D', 'H1_3D']] =\
                pd.DataFrame(diagram_3D.tolist(), index=df.index)

            df['H0_life_time_3D'] = df['H0_3D'].np.diff(axis=1)
            df['H1_life_time_3D'] = df['H1_3D'].np.diff(axis=1)

            df['no_H0_3D'] = df['H0_3D'].map(lambda x: x.shape[0])
            df['no_H1_3D'] = df['H1_3D'].map(lambda x: x.shape[0])

            df['H0_mean_3D'] = df['H0_life_time_3D'].np.mean()
            df['H1_mean_3D'] = df['H1_life_time_3D'].np.mean()

            df['H0_var_3D'] = df['H0_life_time_3D'].np.var()
            df['H1_var_3D'] = df['H1_life_time_3D'].np.var()

            if inplace:
                self.df = df
            else:
                return df

        def estimate_symbol_rate(self,
                                 data_I='signal_sample',
                                 data_Q='signal_sampleQ',
                                 sr_col='symbol_rate'):

            signal = self.df[data_I] + 1j * self.df[data_Q]
            power_spectrum = signal.apply(lambda x:
                                          welch(x,
                                                window=hann(300),
                                                average='median'))

            power_spectrum_log = \
                power_spectrum.apply(lambda x:
                                     np.log((x[1]).reshape(-1, 1)))
            kmeans = power_spectrum_log.apply(lambda x:
                                              KMeans(n_clusters=2).fit(x))
            low_values = kmeans.apply(lambda x: np.min(x.cluster_centers_))

            df = pd.DataFrame({'power_spectrum': power_spectrum,
                               'threshold': low_values})

            # import pdb; pdb.set_trace()

            df[sr_col] = df.apply(lambda x:
                                  np.argmax(np.log(x['power_spectrum'][1]) <
                                            x['threshold']), axis=1)

            df[sr_col] = df.apply(lambda x:
                                  (x['power_spectrum'][0][x[sr_col]]) ** (-1),
                                  axis=1)

            df[sr_col] = df[sr_col].apply(lambda x:
                                          np.nan if x == float('inf') else
                                          np.around(x))

            self.df[sr_col] = df[sr_col]

        def add_power_spectr(self, data_I='signal_sample',
                             data_Q='signal_sampleQ',
                             ps_col='power_spectr'):

            signal = self.df[data_I] + 1j * self.df[data_Q]
            self.df[ps_col] = \
                signal.apply(lambda x:
                             welch(x,
                                   window=hann(300),
                                   average='median')[1])

        def add_fourier(self, data_col='signal_sample',
                        fft_col='signal_fft'):

            self.df[fft_col] = self.df[data_col].apply(np.fft.fft)

        def add_phase(self, data_col='signal_sample',
                      phase_col='signal_phase'):

            column = self.df[data_col].apply(hilbert)
            column = column.apply(lambda x: np.unwrap(np.angle(x)) / (2*np.pi))
            self.df[phase_col] = column

        def add_autocorr(self, data_col='signal_sample',
                         col_name='signal_autocorr'):

            def periodic_corr(x, y):
                """Periodic correlation, implemented using np.correlate.

                x and y must be real sequences with the same length.
                """
                return np.correlate(x, np.hstack((y[1:], y)), mode='valid')

            column = self.df[data_col].apply(lambda x: periodic_corr(x, x))
            self.df[col_name] = column

        def estimate_carrier(self, phase_col='signal_phase',
                             estim_col='carrier_estim'):

            # def estimate_slope(y):
            #     coeff = np.polyfit(np.arange(y.shape[0]), y, 1)
            #     return coeff[0]

            # def estimate_slope(y):
            #     model = Ridge(alpha=50.0)

            #     X = np.column_stack([np.arange(y.shape[0]), y.shape[0]*[1]])

            #     model.fit(X, y)
            #     coeff = model.coef_

            #     print(coeff)

            #     return coeff[0]
            def estimate_slope(y):

                y = np.diff(y)
                coeff = trim_mean(y, proportiontocut=0.3)
                return coeff

            self.df[estim_col] =\
                self.df[phase_col].apply(lambda x: 1/estimate_slope(x))

        def add_carrier(self,
                        sample_col='signal_sample',
                        estim_col='carrier_estim',
                        carrier_I_col='carrier_I',
                        carrier_Q_col='carrier_Q'):

            N = self.df[sample_col].apply(lambda x: x.shape[0])
            fc = (N / self.df[estim_col])

            time_samples = N.apply(np.arange)
            phase = 2 * np.pi * fc * time_samples / N

            self.df[carrier_I_col] = phase.apply(lambda x: np.cos(x))
            self.df[carrier_Q_col] = phase.apply(lambda x: np.sin(x))

        def demodulate(self,
                       data_col='signal_sample',
                       carrier_I_col='carrier_I',
                       carrier_Q_col='carrier_Q',
                       filtered_I_col='filtered_I',
                       filtered_Q_col='filtered_Q',
                       order=3, coef=0.15):

            # b and a are series of coefficients of the Butterworth filter
            b, a = signal.butter(order, coef)

            # b, a = signal.butter(3, 0.35)
            # b, a = signal.cheby1(4, 5, 0.35, 'low')
            # b, a = signal.ellip(10, 5, 40, 0.35, btype='low')

            product_I = self.df[data_col] * self.df[carrier_I_col]
            product_Q = self.df[data_col] * self.df[carrier_Q_col]

            # here we apply the digital filter calculated above
            self.df[filtered_I_col] = product_I.apply(lambda x:
                                                      signal.filtfilt(b, a, x))
            self.df[filtered_Q_col] = product_Q.apply(lambda x:
                                                      signal.filtfilt(b, a, x))

        @staticmethod
        def __get_axes(size, ncols=3, hspace=0.4, wspace=0.4,
                       projection=None, figsize=None):

            nrows = math.ceil(size/ncols)
            figsize = figsize or (20, 2 * size)
            fig, axes = plt.subplots(ncols=ncols,
                                     nrows=nrows,
                                     figsize=figsize,
                                     subplot_kw={'projection': projection})
            fig.subplots_adjust(hspace=hspace, wspace=wspace)
            axes = axes.flatten()
            for i in range(size, ncols * nrows):
                fig.delaxes(axes[i])
            return axes[:size]

        def __get_view_for_plots(self, modulation_id, max_rows=24):
            df = self.df
            if modulation_id is not None:
                df = df.loc[df.modulation_id == modulation_id]
            # df = df.reset_index()
            if df.shape[0] > max_rows:
                # printWARNING:
                df = df.head(max_rows)
            return df  # .reset_index()

        def plot_samples(self,
                         data_col='signal_sample',
                         title_col='modulation_type',
                         modulation_id=None,
                         max_rows=24,
                         min_sample=float('-inf'),
                         max_sample=float('inf')):

            df = self.__get_view_for_plots(modulation_id, max_rows=max_rows)
            size = df.shape[0]
            axes = self.__get_axes(size=size, ncols=2, figsize=(20, size))

            for ind, (_, row) in enumerate(df.iterrows()):
                samples = row[data_col]
                subset = np.arange(start=0, stop=samples.shape[0])
                subset = subset[(subset > min_sample) & (subset < max_sample)]

                axes[ind].plot(samples[subset])
                axes[ind].set_title(row[title_col])

        def plot_persistence_diagrams(self,
                                      data_col='diagram',
                                      title_col='modulation_type',
                                      modulation_id=None,
                                      max_rows=24):

            rips = Rips()
            df = self.__get_view_for_plots(modulation_id, max_rows=max_rows)
            axes = self.__get_axes(size=df.shape[0])

            for ind, (_, row) in enumerate(df.iterrows()):
                plt.sca(axes[ind])
                rips.plot(row[data_col], show=False)
                axes[ind].set_title(row[title_col])

        def plot_histograms(self,
                            data_col='signal_sample',
                            title_col='modulation_type',
                            modulation_id=None,
                            max_rows=24,
                            bins=15):

            df = self.__get_view_for_plots(modulation_id, max_rows=max_rows)
            axes = self.__get_axes(size=df.shape[0])

            for ind, (_, row) in enumerate(df.iterrows()):
                axes[ind].hist(row[data_col], bins=bins)
                axes[ind].set_title(row[title_col])

        def plot_IQ(self,
                    data_I='signal_sample',
                    data_Q='signal_sampleQ',
                    title_col='modulation_type',
                    max_rows=24,
                    modulation_id=None):

            df = self.__get_view_for_plots(modulation_id, max_rows=max_rows)
            size = df.shape[0]
            axes = self.__get_axes(size=size, ncols=2)

            for ind, (_, row) in enumerate(df.iterrows()):
                axes[ind].scatter(row[data_I], row[data_Q])
                axes[ind].set_title(row[title_col])

        def plot_clouds(self,
                        data_col='point_cloud',
                        title_col='modulation_type',
                        projection=None,
                        modulation_id=None,
                        max_rows=24):

            df = self.__get_view_for_plots(modulation_id, max_rows=max_rows)
            axes = self.__get_axes(size=df.shape[0], projection=projection)

            for ind, (_, row) in enumerate(df.iterrows()):
                if projection is None:
                    axes[ind].scatter(row[data_col][:, 0], row[data_col][:, 1])
                else:
                    axes[ind].scatter3D(row[data_col][:, 0],
                                        row[data_col][:, 1],
                                        row[data_col][:, 2])
                axes[ind].set_title(row[title_col])

        def to_csv(self, path_or_buf=None, **kwargs):

            np.set_printoptions(threshold=sys.maxsize)
            df = self.df.copy()
            for col in df.columns:
                if isinstance(df.iloc[0][col], np.ndarray):
                    df.rename(columns={col: 'np_' + col}, inplace=True)

            df.to_csv(path_or_buf=path_or_buf, **kwargs)


# #############################################################################
#                        read_csv override                                    #
# #############################################################################
def read_csv(filepath_or_buffer, **kwargs):
    try:
        df = pd.read_csv(filepath_or_buffer, **kwargs)
    except FileNotFoundError as e:
        msg = str(e)
        msg += "\nFile {} not found.".format(filepath_or_buffer)
        msg += "\nDefault file {} with test data will be open instead."\
               .format(config.default_data_file)
        warnings.warn(msg)
        dir_path = os.path.dirname(filepath_or_buffer)
        filepath_or_buffer = os.path.join(dir_path, config.default_data_file)
        df = pd.read_csv(filepath_or_buffer, **kwargs)
    for col_name in df.columns:
        if col_name[:3] == "np_":
            try:
                df[col_name] = df[col_name].mr.to_numpy()
                df.rename(columns={col_name: col_name[3:]}, inplace=True)
            except TypeError:
                print(col_name)
    # array = np.array
    # df['diagram'] = df['diagram'].apply(lambda x: eval(x))
    return df

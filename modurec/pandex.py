import warnings
import re
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.signal.windows import hann
from ripser import Rips
from sklearn.cluster import KMeans

from .utility import rolling_window


# #############################################################################
#                          Package configuration
# #############################################################################

inf = np.inf

# #############################################################################
#
#                     DataFrame and Series extensions
#
# #############################################################################
with warnings.catch_warnings():   # catch_warnings is useful when autoreload is applied

    warnings.simplefilter(action='ignore', category=UserWarning)

    # ##########################################################################
    #                        Series 'np' extensions
    # ##########################################################################
    @pd.api.extensions.register_series_accessor('np')
    class NumpySeries:

        np.set_printoptions(suppress=True)
        np.set_printoptions(threshold=np.inf)

        def __init__(self, series):
            self.series = series

        def mean(self):
            return self.series.map(lambda x: np.mean(np.ma.masked_invalid(x)))

        def var(self):
            return self.series.map(lambda x: np.var(np.ma.masked_invalid(x)))

        def diff(self, axis=-1):  # Needed to calculate homology life-time
            return self.series.map(lambda x: np.diff(x, axis=axis).reshape(-1))

        def rolling_window(self, window=2, **kargs):
            def rw(a): return rolling_window(a, window=window, **kargs)
            return self.series.map(rw)


    # ##########################################################################
    #                        DataFrame 'mr' extensions
    # ##########################################################################

    @pd.api.extensions.register_dataframe_accessor('mr')
    class SignalDataFrame:

        def __init__(self, df):
            self.df = df

        def add_point_cloud(self,
                            data_colI='signalI',
                            data_colQ='signalQ',
                            point_cloud_col='point_cloud',
                            window=2, step=1):

            def __to_real(x):  # converts complex-valued array to real-valued
                array = np.vstack((np.real(x), np.imag(x))).T
                return array.ravel()

            # needed for converting point-clouds in C^n...
            # ...into clouds in R^(2*n)
            def __complex_cloud_to_real(cloud):
                return np.apply_along_axis(lambda x:
                                           __to_real(x), axis=1, arr=cloud)

            df = self.df[[data_colI, data_colQ]].copy()

            if isinstance(step, int):
                df['step'] = step
            elif isinstance(step, str):
                df['step'] = self.df[step]
            else:
                raise ValueError('Incorrect value of step.')

            df['data'] = self.df.apply(lambda x: x[data_colI] + 1j * x[data_colQ],
                                       axis=1)

            result = df.apply(lambda x: rolling_window(x['data'],
                                                       window=window,
                                                       stride_step=x['step']), axis=1)
            result = result.apply(__complex_cloud_to_real)
            self.df[point_cloud_col] = result

        # symbol rate
        # WARNING: This implementation returns 1 if there is a division by zero problem!!!
        # TODO: Deal with the above problem
        def add_symbol_rate(self,
                            data_I='signalI',
                            data_Q='signalQ',
                            sr_col='symbol_rate'):

            signal = self.df[data_I] + 1j * self.df[data_Q]

            # The function 'welch' estimates power spectrum
            # it returns a tuple with vector of frequencies as a first element
            # and power spectrum as a second element
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

            df = pd.DataFrame({'power_spectrum': power_spectrum.apply(lambda x: x[1]),
                               'freq': power_spectrum.apply(lambda x: x[0]),
                               'threshold': low_values})

            df[sr_col] = df.apply(lambda x:
                                  np.argmax(np.log(x['power_spectrum']) < x['threshold']), axis=1)

            df[sr_col] = df.apply(lambda x:
                                  (x['freq'][x[sr_col]]) ** (-1), axis=1)

            df[sr_col] = df[sr_col].apply(lambda x:
                                          1.0 if x == float('inf') else
                                          np.around(x))

            self.df[sr_col] = df[sr_col].astype(int)

        # TODO: remove this unused function
        def add_fourier(self, data_col, fft_col):

            self.df[fft_col] = self.df[data_col].apply(np.fft.fft)

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

        def __get_view_for_plots(self, modulation_type, max_rows=24):
            df = self.df
            if modulation_type is not None:
                df = df.loc[df.modulation_type == modulation_type]
            # df = df.reset_index()
            if df.shape[0] > max_rows:
                # printWARNING:
                df = df.head(max_rows)
            return df  # .reset_index()

        def plot_samples(self,
                         data_col='signal_sample',
                         title_col='modulation_type',
                         modulation_type=None,
                         max_rows=24,
                         min_sample=float('-inf'),
                         max_sample=float('inf')):

            df = self.__get_view_for_plots(modulation_type, max_rows=max_rows)
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
                                      modulation_type=None,
                                      max_rows=24):

            rips = Rips()
            df = self.__get_view_for_plots(modulation_type, max_rows=max_rows)
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
                    data_I='signal_sampleI',
                    data_Q='signalQ',
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
                        modulation_type=None,
                        max_rows=24,
                        ncols=3):

            df = self.__get_view_for_plots(modulation_type, max_rows=max_rows)
            axes = self.__get_axes(size=df.shape[0], projection=projection,
                                   ncols=ncols)

            for ind, (_, row) in enumerate(df.iterrows()):
                if projection is None:
                    axes[ind].scatter(row[data_col][:, 0], row[data_col][:, 1])
                else:
                    axes[ind].scatter3D(row[data_col][:, 0],
                                        row[data_col][:, 1],
                                        row[data_col][:, 2])
                axes[ind].set_title(row[title_col])


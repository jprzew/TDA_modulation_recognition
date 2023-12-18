import warnings
import re
import math
from typing import Optional, List, Tuple

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

        @staticmethod
        def __get_axes(size, ncols=3, hspace=0.4, wspace=0.4,
                       projection=None, figsize=None):

            nrows = math.ceil(size/ncols)
            figsize = figsize or (20, 2 * size)
            fig, axes = plt.subplots(ncols=ncols,
                                     nrows=nrows,
                                     figsize=figsize,
                                     subplot_kw={'projection': None if not projection else ''})  # TODO: refactor projection
            fig.subplots_adjust(hspace=hspace, wspace=wspace)
            axes = axes.flatten()
            for i in range(size, ncols * nrows):
                fig.delaxes(axes[i])
            return axes[:size]

        def __get_view_for_plots(self, modulation_type, max_rows=24):
            df = self.df

            if modulation_type == 'all':
                df = df.groupby('modulation_type', as_index=False).sample(n=max_rows)
            elif modulation_type is not None:
                df = df.loc[df.modulation_type == modulation_type]
                df = df.sample(n=max_rows)
            elif df.shape[0] > max_rows:
                df = df.head(max_rows)
            return df

        def plot_diagrams(self,
                          data_col='diagram',
                          modulation_type: Optional[str] = 'all',
                          max_rows=24):
            """Plot persistence diagrams.

            Arguments:
            ----------
            data_col : str
                Name of the column with persistence diagrams.
            modulation_type : str or None (default: 'all')
                Indicates which modulations to plot.
                If None, then the first max_rows rows are plotted.
            max_rows : int (default: 24) indicates how many rows to plot.
                if modulation_type is 'all', then max_rows rows are sampled from each modulation.
            """

            rips = Rips()
            df = self.__get_view_for_plots(modulation_type, max_rows=max_rows)
            axes = self.__get_axes(size=df.shape[0])

            for ind, (original_index, row) in enumerate(df.iterrows()):
                plt.sca(axes[ind])
                rips.plot(row[data_col], show=False)
                axes[ind].set_title(f'Modulation: {row["modulation_type"]}. Index: {original_index}. SNR={row["SNR"]}')

        # TODO: This function requires refactoring; together with __get_view_for_plots
        def plot_clouds(self,
                        data_col='point_cloud',
                        three_dimensional: bool = False,
                        modulation_type: Optional[str] = 'all',
                        max_rows=24,
                        ncols=3,
                        xylim: Tuple[Tuple, Tuple] = None):
            """Plot point-clouds.

            Arguments:
            ----------
            data_col : str
                Name of the column with point-clouds.
            three_dimensional : bool (default: False) indicates whether to plot in 3D or 2D.
            modulation_type : str or None (default: 'all')
                Indicates which modulations to plot.
                If None, then the first max_rows rows are plotted.
            max_rows : int (default: 24) indicates how many rows to plot.
                if modulation_type is 'all', then max_rows rows are sampled from each modulation.
            ncols : int (default: 3) number of columns with plots.
            xylim : Tuple[Tuple, Tuple] or None (default: None) indicates the limits of the plot.
                If None, then the limits are calculated automatically.
                First tuple is for x-axis, second for y-axis.
            """

            df = self.__get_view_for_plots(modulation_type, max_rows=max_rows)
            axes = self.__get_axes(size=df.shape[0], projection=three_dimensional,
                                   ncols=ncols)

            for ind, (original_index, row) in enumerate(df.iterrows()):
                if not three_dimensional:
                    axes[ind].scatter(row[data_col][:, 0], row[data_col][:, 1])
                else:
                    axes[ind].scatter3D(row[data_col][:, 0],
                                        row[data_col][:, 1],
                                        row[data_col][:, 2])
                axes[ind].set_title(f'Modulation: {row["modulation_type"]}. Index: {original_index}. SNR={row["SNR"]}')
                if xylim is not None:
                    xlim = xylim[0]
                    ylim = xylim[1]
                    axes[ind].set_xlim(left=xlim[0], right=xlim[1])
                    axes[ind].set_ylim(bottom=ylim[0], top=ylim[1])

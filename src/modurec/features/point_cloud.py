from .feature import Feature
import numpy as np
import pandas as pd


def windowed_cloud(point_cloud, window, step):

    samples = point_cloud.shape[0]
    try:
        dim = point_cloud.shape[1]
    except IndexError:
        dim = 1

    no_windows = samples - step*(window-1)  # number of windows

    # we create indices to rearrange points so that
    # points in the same window are neighbours
    indices = np.array(range(samples))
    stride = indices.strides[0]

    # the first dimension describes different windows
    # the second dimension descibes points in windows
    indices = np.lib.stride_tricks.as_strided(indices,
                                              shape=(no_windows, window),
                                              strides=(stride, step * stride))
    indices = indices.flatten()

    return np.lib.stride_tricks.as_strided(point_cloud[indices, ...],
                                           shape=(no_windows, dim * window),
                                           strides=(stride * dim * window, stride))


class PointCloud(Feature):

    def __init__(self, dim=2, step=1, kind=None, preproc=None):
        self.dim = dim
        self.step = step
        self.kind = kind
        self.preproc = preproc

    @staticmethod
    def compute3D(df):

        def standardize(x):
            range0 = np.max(x[:, 0]) - np.min(x[:, 0])
            range1 = np.max(x[:, 1]) - np.min(x[:, 1])
            range2 = np.max(x[:, 2]) - np.min(x[:, 2])
            x[:, 2] = x[:, 2] * 0.5 * (range0 + range1) / range2
            return x

        df['point_cloud'] = df.point_cloud.apply(lambda x:
                                                 np.column_stack([x, range(x.shape[0])]))
        df['point_cloud'] = df.point_cloud.apply(standardize)

        return df['point_cloud']

    @staticmethod
    def fft_cloud(point_cloud):
        """Treats cloud as a complex signal and computes its fft, thus producing new cloud"""
        new_cloud = np.fft.fft(point_cloud[:, 0] + 1j * point_cloud[:, 1])
        return np.stack((np.real(new_cloud), np.imag(new_cloud)),
                        axis=-1)

    def compute(self):

        df = self.creator.df['point_cloud']

        if self.preproc == 'fft':
            df = df.apply(self.fft_cloud)

        if self.kind is None:
            window = self.dim / 2
        elif self.kind == 'abs':
            def amplitude(point_cloud):
                return np.apply_along_axis(lambda x: np.linalg.norm(x), 1, point_cloud)

            df = df.apply(amplitude)
            window = float(self.dim)
        elif self.kind == 'phi':
            def argument(point_cloud):
                return np.apply_along_axis(lambda x: phase(x[0] + x[1] * 1j), 1, point_cloud)

            df = df.apply(argument)
            window = float(self.dim)

        else:
            raise NotImplemented('Point cloud type not implemented.')

        if isinstance(self.step, str):
            step_col = self.step
            df = pd.concat([df, self.df[step_col]], axis=1)
        elif isinstance(self.step, int):
            step_col = 'step'
            df = pd.DataFrame(df)
            df[step_col] = self.step
        else:
            raise ValueError('Parameter step need to be str or int')

        if window == 1:
            return df['point_cloud']

        if window.is_integer():
            return df.apply(lambda x: windowed_cloud(x['point_cloud'],
                                                     window=int(window),
                                                     step=x[step_col]), axis=1)
        else:
            if self.dim == 3:
                return self.compute3D(df)
            else:
                raise NotImplemented('Dimension not implemented.')

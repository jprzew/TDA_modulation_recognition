from .feature import Feature, Computer
import numpy as np
import pandas as pd
from cmath import phase


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

    stride = min(point_cloud.strides)
    return np.lib.stride_tricks.as_strided(point_cloud[indices, ...],
                                           shape=(no_windows, dim * window),
                                           strides=(stride * dim * window, stride))


class PointCloud(Feature):
    """Computes multidimensional point clouds

    Attributes
    ----------
    dim : int - dimension of output cloud
    step : int or str - step between windows (if str it denotes a column name with step values)
    kind : str - type of point cloud (None, 'abs' - takes absolute values, 'phi' - takes phases)
    preproc : str - preprocessing of point cloud ('fft' - computes fft of cloud)
    """

    def __init__(self, dim=2, step=1, kind=None, preproc=None):
        self.dim = dim
        self.step = step
        self.kind = kind
        self.preproc = preproc

    @staticmethod
    def compute_3d(df):

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
        df = self.creator.df

        # Compute fft-clouds if necessary
        if self.preproc == 'fft':
            clouds = df['point_cloud'].apply(self.fft_cloud)
        else:
            clouds = df['point_cloud']

        # Add column with values of step
        if isinstance(self.step, str):  # user gave column name
            if self.step not in df.columns:
                raise ValueError(f'Column {self.step} does not exist')

            steps = df[self.step]
        elif isinstance(self.step, int):  # user gave step value
            steps = pd.Series(self.step, index=df.index)
        else:
            raise ValueError('Parameter step need to be str or int')

        # Create computer-responsibility chain
        computer = StandardCloudComputer()
        computer = ReducedCloudComputer(previous=computer)

        return computer.handle(clouds=clouds, dim=self.dim, steps=steps, kind=self.kind)


class StandardCloudComputer(Computer):
    """Computes standard type of point cloud (kind = None)"""

    def can_compute(self, **kwargs):
        # This case is computable if kind is None and dim is even or 3 (but then step be constant and equal to 1)
        conditions = [kwargs['kind'] is None,
                      kwargs['dim'] % 2 == 0 or (kwargs['dim'] == 3 and (kwargs['step'] == 1).all())]

        return np.array(conditions).all()

    def compute(self, **kwargs):
        clouds = kwargs['clouds']
        dim = kwargs['dim']
        steps = kwargs['steps']
        window = dim / 2

        if window == 1:
            return clouds
        elif window.is_integer():
            return (pd.concat([clouds, steps], axis=1)
                    .apply(lambda x: windowed_cloud(x.iloc[0], window=int(window), step=x.iloc[1]), axis=1))

        elif dim == 3:
            return self._compute_3d(clouds)

    @staticmethod
    def _compute_3d(clouds):

        def standardize(x):
            range0 = np.max(x[:, 0]) - np.min(x[:, 0])
            range1 = np.max(x[:, 1]) - np.min(x[:, 1])
            range2 = np.max(x[:, 2]) - np.min(x[:, 2])
            x[:, 2] = x[:, 2] * 0.5 * (range0 + range1) / range2
            return x

        # Third dimension of a point is set to be the index of that point
        result = clouds.apply(lambda x: np.column_stack([x, range(x.shape[0])]))
        return result.apply(standardize)


class ReducedCloudComputer(Computer):
    """Reduced cloud is derived from one-dimensional signal of amplitudes or phases"""

    def can_compute(self, **kwargs):
        # This case is computable if kind is 'abs' or 'phi'
        # dim must be an integer
        conditions = [kwargs['kind'] in {'abs', 'phi'},
                      isinstance(kwargs['dim'], int)]

        return np.array(conditions).all()

    def compute(self, **kwargs):
        clouds = kwargs['clouds']
        dim = kwargs['dim']
        steps = kwargs['steps']
        kind = kwargs['kind']

        if kind == 'abs':
            def amplitude(point_cloud):
                return np.apply_along_axis(lambda x: np.linalg.norm(x), 1, point_cloud)

            clouds = clouds.apply(amplitude)
        elif kind == 'phi':
            def argument(point_cloud):
                return np.apply_along_axis(lambda x: phase(x[0] + x[1] * 1j), 1, point_cloud)

            clouds = clouds.apply(argument)

        window = dim
        if window == 1:
            return clouds
        else:
            return (pd.concat([clouds, steps], axis=1)
                    .apply(lambda x: windowed_cloud(x.iloc[0], window=int(window), step=x.iloc[1]), axis=1))

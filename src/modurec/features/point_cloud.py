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
        df = self.creator.df.copy()  # TODO: This is risky memorywise; may require refactoring

        if self.preproc == 'fft':
            df['point_cloud'] = df['point_cloud'].apply(self.fft_cloud)

        computer = StandardCloudComputer()
        computer = ReducedCloudComputer(previous=computer)
        return computer.handle(df=df, dim=self.dim, step=self.step, kind=self.kind)

    #
    # def compute(self):
    #
    #     df = self.creator.df['point_cloud']
    #
    #     if self.preproc == 'fft':
    #         df = df.apply(self.fft_cloud)
    #
    #     if self.kind is None:
    #         window = self.dim / 2
    #     elif self.kind == 'abs':
    #         def amplitude(point_cloud):
    #             return np.apply_along_axis(lambda x: np.linalg.norm(x), 1, point_cloud)
    #
    #         df = df.apply(amplitude)
    #         window = float(self.dim)
    #     elif self.kind == 'phi':
    #         def argument(point_cloud):
    #             return np.apply_along_axis(lambda x: phase(x[0] + x[1] * 1j), 1, point_cloud)
    #
    #         df = df.apply(argument)
    #         window = float(self.dim)
    #
    #     else:
    #         raise NotImplemented('Point cloud type not implemented.')
    #
    #     if isinstance(self.step, str):
    #         step_col = self.step
    #         df = pd.concat([df, self.creator.df[step_col]], axis=1)
    #     elif isinstance(self.step, int):
    #         step_col = 'step'
    #         df = pd.DataFrame(df)
    #         df[step_col] = self.step
    #     else:
    #         raise ValueError('Parameter step need to be str or int')
    #
    #     if window == 1:
    #         return df['point_cloud']
    #
    #     if window.is_integer():
    #         return df.apply(lambda x: windowed_cloud(x['point_cloud'],
    #                                                  window=int(window),
    #                                                  step=x[step_col]), axis=1)
    #     else:
    #         if self.dim == 3:
    #             return self.compute_3d(df)
    #         else:
    #             raise NotImplemented('Dimension not implemented.')


class StandardCloudComputer(Computer):
    """Computes standard type of point cloud (kind = None)"""

    def can_compute(self, **kwargs):
        # This case is computable if kind is None & step is int or str
        # dim must be even or 3 (but then step must be 1)
        conditions = [kwargs['kind'] is None,
                      isinstance(kwargs['step'], (int, str)),
                      kwargs['dim'] % 2 == 0 or (kwargs['dim'] == 3 and kwargs['step'] == 1)]

        return np.array(conditions).all()

    def compute(self, **kwargs):
        df = kwargs['df']
        dim = kwargs['dim']
        step = kwargs['step']
        window = dim / 2

        # Add column with values of step
        if isinstance(step, str):  # user gave column name
            if step not in df.columns:
                raise ValueError(f'Column {step} does not exist')

            step_col = step
        elif isinstance(step, int):  # user gave step value
            step_col = 'step'
            df[step_col] = step
        else:
            raise ValueError('Parameter step need to be str or int')

        if window == 1:
            return df['point_cloud']
        elif window.is_integer():
            return df.apply(lambda x: windowed_cloud(x['point_cloud'],
                                                     window=int(window),  # window needs to be int
                                                     step=x[step_col]), axis=1)
        elif dim == 3:
            return self._compute_3d(df)

    @staticmethod
    def _compute_3d(df):

        def standardize(x):
            range0 = np.max(x[:, 0]) - np.min(x[:, 0])
            range1 = np.max(x[:, 1]) - np.min(x[:, 1])
            range2 = np.max(x[:, 2]) - np.min(x[:, 2])
            x[:, 2] = x[:, 2] * 0.5 * (range0 + range1) / range2
            return x

        # Third dimension of a point is set to be the index of that point
        result = df.point_cloud.apply(lambda x: np.column_stack([x, range(x.shape[0])]))
        return result.apply(standardize)


class ReducedCloudComputer(Computer):
    """Reduced cloud is derived from one-dimensional signal of amplitudes or phases"""

    def can_compute(self, **kwargs):
        # This case is computable if kind is 'abs' or 'phi' & step is int or str
        # dim must be an integer
        conditions = [kwargs['kind'] in {'abs', 'phi'},
                      isinstance(kwargs['step'], (int, str)),
                      isinstance(kwargs['dim'], int)]

        return np.array(conditions).all()

    def compute(self, **kwargs):
        df = kwargs['df']
        dim = kwargs['dim']
        step = kwargs['step']
        kind = kwargs['kind']

        if kind == 'abs':
            def amplitude(point_cloud):
                return np.apply_along_axis(lambda x: np.linalg.norm(x), 1, point_cloud)

            df['point_cloud'] = df.point_cloud.apply(amplitude)
        elif kind == 'phi':
            def argument(point_cloud):
                return np.apply_along_axis(lambda x: phase(x[0] + x[1] * 1j), 1, point_cloud)

            df['point_cloud'] = df.point_cloud.apply(argument)

        # Add column with values of step
        # TODO: This part of code may be worth moving to PointCloud class
        if isinstance(step, str):  # user gave column name
            if step not in df.columns:
                raise ValueError(f'Column {step} does not exist')

            step_col = step
        elif isinstance(step, int):  # user gave step value
            step_col = 'step'
            df[step_col] = step
        else:
            raise ValueError('Parameter step need to be str or int')

        window = dim
        if window == 1:
            return df['point_cloud']
        else:
            return df.apply(lambda x: windowed_cloud(x['point_cloud'],
                                                     window=int(window),
                                                     step=x[step_col]), axis=1)

        df['point_cloud'] = df.point_cloud.apply(lambda x:
                                                 np.column_stack([x, range(x.shape[0])]))
        df['point_cloud'] = df.point_cloud.apply(standardize)

        return df['point_cloud']



from inspect import signature
from abc import abstractmethod
from cmath import phase

import pandas as pd
import numpy as np
import h5py
from ripser import Rips, ripser
from math import sqrt
from scipy import sparse

from . import pandex  # Necessary for mr and np accessors
from .utility import rolling_window


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


def trim_diagrams(diagrams, eps):
    trimmed_diagrams = []
    for diagram in diagrams:
        lifetimes = diagram[:, 1] - diagram[:, 0]
        trimmed_diagrams.append(diagram[lifetimes > eps])

    return trimmed_diagrams

def wasserstein_amplitude(diagram, p):
    lifetimes = diagram[:, 1] - diagram[:, 0]
    lifetimes = lifetimes[lifetimes < float('inf')]

    return np.linalg.norm(lifetimes, p) / sqrt(2)

def persistent_entropy(diagram):
    lifetimes = diagram[:, 1] - diagram[:, 0]
    lifetimes = lifetimes[lifetimes < float('inf')]
    L = sum(lifetimes)

    return sum((lifetimes / L) * np.log(lifetimes / L))

@pd.api.extensions.register_dataframe_accessor('ff')
class FeaturesFactory:

    def __init__(self, df):
        self.df = df
        self.features = []
        self.rips = Rips()

    def create_feature(self, name, **kwargs):

        creator = getattr(FeaturesFactory, name)
        instance = creator(**kwargs)

        self.features.append(instance)
        instance.df = self.df
        instance.creator = self

        column_name = str(instance)
        if column_name not in self.df.columns:
            self.df[str(instance)] = instance.compute()

        return instance

    def get_values(self, feature):

        return self.df[str(feature)]

    class Feature:

        def __str__(self):
            params = signature(self.__init__).parameters

            keys = [key for key in params
                    if params[key].kind not in {params[key].VAR_KEYWORD,
                                                params[key].VAR_POSITIONAL} and
                    params[key].default != getattr(self, key)
                    ]

            # keys = [key for key in keys
            #         if params[key].default != getattr(self, key)]

            # Excluding parameters with default value
            # keys = [key for key in keys
            #         if params[key].default != getattr(self, key)]
            values = [self.__dict__[key] for key in keys]

            string = self.__class__.__name__

            if keys:
                string += '_' + '={}_'.join(keys) + '={}'
                string = string.format(*values)

            return string

        def values(self):
            return self.df[str(self)]

        @abstractmethod
        def compute(self):
            pass

    class point_cloud(Feature):

        def __init__(self, dim=2, step=1, kind=None):
            self.dim = dim
            self.step = step
            self.kind = kind

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


        def compute(self):
            
            df = self.df['point_cloud']

            if self.kind is None:
                window = self.dim / 2
            elif self.kind == 'abs':
                def amplitude(point_cloud):
                    return np.apply_along_axis(lambda x: np.linalg.norm(x), 1, point_cloud)

                df = df.apply(amplitude)
                window = float(self.dim)
            elif self.kind == 'phi':
                def argument(point_cloud):
                    return np.apply_along_axis(lambda x: phase(x[0] + x[1]*1j), 1, point_cloud)

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


    class diagram(Feature):

        @staticmethod
        def star_1D_diagram(time_series):

            # Add edges between adjacent points in the time series, with the "distance"
            # along the edge equal to the max value of the points it connects
            N = time_series.shape[0]
            I = np.arange(N-1)
            J = np.arange(1, N)
            W = np.maximum(time_series[0:-1], time_series[1::])

            # Add vertex birth times along the diagonal of the distance matrix
            I = np.concatenate((I, np.arange(N)))
            J = np.concatenate((J, np.arange(N)))
            W = np.concatenate((W, time_series))

            #Create the sparse distance matrix
            D = sparse.coo_matrix((W, (I, J)), shape=(N, N)).tocsr()
            return ripser(D, maxdim=0, distance_matrix=True)['dgms']



        def __init__(self, dim=2, step=1, eps=0, kind=None, fil=None):
            self.step = step
            self.dim = dim
            self.eps = eps
            self.kind = kind
            self.fil = fil


        def compute(self):
            if self.fil == 'star':
                if self.kind not in {'abs', 'phi'}:
                    raise NotImplemented('Options fil=star is implemented only with kind in (abs, phi)')

                point_cloud = self.creator.create_feature('point_cloud', dim=1,
                                                          step=self.step, kind=self.kind)

                return point_cloud.values().map(self.star_1D_diagram)



            if self.eps == 0:
                point_cloud = self.creator.create_feature('point_cloud', dim=self.dim,
                                                          step=self.step, kind=self.kind)
                return point_cloud.values().map(self.creator.rips.fit_transform)
            else:
                full_diagram = self.creator.create_feature('diagram', dim=self.dim,
                                                           step=self.step, eps=0, kind=self.kind)
                return full_diagram.values().map(lambda x:
                                                 trim_diagrams(x, self.eps))

    class H(Feature):

        def __init__(self, n, dim=2, step=1, eps=0, kind=None, fil=None):
            self.dim = dim
            self.n = n
            self.step = step
            self.eps = eps
            self.kind = kind
            self.fil = fil


        def compute(self):
            diagram = self.creator.create_feature('diagram', dim=self.dim, step=self.step,
                                                  eps=self.eps, kind=self.kind, fil=self.fil)

            return pd.DataFrame(diagram.values().tolist(),
                                    index=self.df.index)[self.n]

    class life_time(Feature):

        def __init__(self, n, dim=2, step=1, eps=0, kind=None, fil=None):
            self.dim = dim
            self.n = n
            self.step = step
            self.eps = eps
            self.kind = kind
            self.fil = fil


        def compute(self):
            homology = self.creator.create_feature('H',
                                                   n=self.n,
                                                   dim=self.dim,
                                                   step=self.step,
                                                   eps=self.eps,
                                                   kind=self.kind,
                                                   fil=self.fil)

            return homology.values().np.diff(axis=1)

    class no(Feature):

        def __init__(self, n, dim=2, step=1, eps=0, kind=None, fil=None):
            self.dim = dim
            self.n = n
            self.step = step
            self.eps = eps
            self.kind = kind
            self.fil = fil


        def compute(self):
            homology = self.creator.create_feature('H',
                                                   n=self.n,
                                                   dim=self.dim,
                                                   step=self.step,
                                                   eps=self.eps,
                                                   kind=self.kind,
                                                   fil=self.fil)

            return homology.values().map(lambda x: x.shape[0])

    class mean(Feature):

        def __init__(self, n, dim=2, step=1, eps=0, kind=None, fil=None):
            self.dim = dim
            self.n = n
            self.step = step
            self.eps = eps
            self.kind = kind
            self.fil = fil



        def compute(self):
            life_time = self.creator.create_feature('life_time',
                                                    n=self.n,
                                                    dim=self.dim,
                                                    step=self.step,
                                                    eps=self.eps,
                                                    kind=self.kind,
                                                    fil=self.fil)
            return life_time.values().np.mean()

    class var(Feature):

        def __init__(self, n, dim=2, step=1, eps=0, kind=None, fil=None):
            self.dim = dim
            self.n = n
            self.step = step
            self.eps = eps
            self.kind = kind
            self.fil = fil



        def compute(self):
            life_time = self.creator.create_feature('life_time',
                                                    n=self.n,
                                                    dim=self.dim,
                                                    step=self.step,
                                                    eps=self.eps,
                                                    kind=self.kind,
                                                    fil=self.fil)
            return life_time.values().np.var()

    class kmp_features(Feature):
        """Features from Chatter Classification in Turning using
           Machine Learning and Topological Data Analysis"""

        def __init__(self, k, n, dim=2, step=1, eps=0, kind=None, fil=None):
            self.dim = dim  # point cloud dimension
            self.n = n  # homology dimension
            self.k = k  # number of kmp-feature
            self.step = step
            self.eps = eps
            self.kind = kind
            self.fil = fil



        def compute(self):

            def __features(D):
                mean_y = np.array([y for _, y in D if y != float('inf')]).mean()

                return (sum([x * (y - x) for x, y in D if y != float('inf')]),
                        sum([(mean_y - y) * (y - x) for x, y in D
                             if y != float('inf')]),
                        sum([x ** 2 * (y - x) for x, y in D if y != float('inf')]),
                        sum([(mean_y - y) ** 2 * (y - x) ** 4 for x, y in D
                             if y != float('inf')]),
                        max([y - x for x, y in D if y != float('inf')]))

            homology = self.creator.create_feature('H',
                                                   n=self.n,
                                                   dim=self.dim,
                                                   step=self.step,
                                                   eps=self.eps,
                                                   kind=self.kind,
                                                   fil=self.fil)

            return homology.values().map(lambda x:
                                         __features(x)[self.k - 1])

    class wasser_ampl(Feature):
        def __init__(self, p, n, dim=2, step=1, eps=0, kind=None, fil=None):
            self.dim = dim  # point cloud dimension
            self.n = n  # homology dimension
            self.p = p  # p-parameter of the Lp-norm
            self.step = step
            self.eps = eps
            self.kind = kind
            self.fil = fil


        def compute(self):
            homology = self.creator.create_feature('H',
                                                   n=self.n,
                                                   dim=self.dim,
                                                   step=self.step,
                                                   eps=self.eps,
                                                   kind=self.kind,
                                                   fil=self.fil)

            return homology.values().map(lambda x: wasserstein_amplitude(x, self.p))

    class entropy(Feature):
        def __init__(self, n, dim=2, step=1, eps=0, kind=None, fil=None):
            self.dim = dim  # point cloud dimension
            self.n = n  # homology dimension
            self.step = step
            self.eps = eps
            self.kind = kind
            self.fil = fil


        def compute(self):
            homology = self.creator.create_feature('H',
                                                   n=self.n,
                                                   dim=self.dim,
                                                   step=self.step,
                                                   eps=self.eps,
                                                   kind=self.kind,
                                                   fil=self.fil)

            return homology.values().map(persistent_entropy)

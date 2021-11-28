from inspect import signature
from abc import abstractmethod

import pandas as pd
import numpy as np
import h5py
from ripser import Rips
from math import sqrt

from . import pandex  # Necessary for mr and np accessors

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


    class signalI(Feature):

        # def __init__(self):
        #     pass

        def compute(self):
            return self.df['signalI']

    class signalQ(Feature):

        # def __init__(self):
        #     pass

        def compute(self):
            return self.df['signalQ']

    class point_cloud(Feature):

        def __init__(self, dim=2, step=1):
            self.dim = dim
            self.step = step

        def compute(self):
            def __compute2D(samples):
                samples.mr.add_point_cloud(window=1, step=self.step)

                return samples['point_cloud']

            def __compute3D(samples):

                def __standardize(x):
                    range0 = np.max(x[:, 0]) - np.min(x[:, 0])
                    range1 = np.max(x[:, 1]) - np.min(x[:, 1])
                    range2 = np.max(x[:, 2]) - np.min(x[:, 2])
                    x[:, 2] = x[:, 2] * 0.5 * (range0 + range1) / range2
                    return x

                samples.mr.add_point_cloud(window=1, point_cloud_col='cloud_3D', step=self.step)
                samples['cloud_3D'] = \
                    samples.cloud_3D.apply(lambda x:
                                           np.column_stack([x, range(x.shape[0])]))
                samples['cloud_3D'] = samples.cloud_3D.apply(__standardize)

                return samples['cloud_3D']

            def __compute4D(samples):
                samples.mr.add_point_cloud(window=2, point_cloud_col='cloud_4D', step=self.step)
                return samples['cloud_4D']

            signalI = self.creator.create_feature('signalI')
            signalQ = self.creator.create_feature('signalQ')
            if isinstance(self.step, str):
                step_col = self.creator.df[self.step]
            else:
                step_col = None

            samples = pd.concat([signalI.values(),
                                 signalQ.values(),
                                 step_col],
                                axis=1)

            if self.dim == 2:
                return __compute2D(samples)
            elif self.dim == 3:
                return __compute3D(samples)
            elif self.dim == 4:
                return __compute4D(samples)

            raise(NotImplemented('Dimension not implemented.'))

    class diagram(Feature):

        def __init__(self, dim=2, step=1, eps=0):
            self.step = step
            self.dim = dim
            self.eps = eps

        def compute(self):
            if self.eps == 0:
                point_cloud = self.creator.create_feature('point_cloud', dim=self.dim,
                                                          step=self.step)
                return point_cloud.values().map(self.creator.rips.fit_transform)
            else:
                full_diagram = self.creator.create_feature('diagram', dim=self.dim,
                                                           step=self.step, eps=0)
                return full_diagram.values().map(lambda x:
                                                 trim_diagrams(x, self.eps))

    class H(Feature):

        def __init__(self, n, dim=2, step=1, eps=0):
            self.dim = dim
            self.n = n
            self.step = step
            self.eps = eps

        def compute(self):
            diagram = self.creator.create_feature('diagram', dim=self.dim, step=self.step,
                                                  eps=self.eps)

            return pd.DataFrame(diagram.values().tolist(),
                                    index=self.df.index)[self.n]

    class life_time(Feature):

        def __init__(self, n, dim=2, step=1, eps=0):
            self.dim = dim
            self.n = n
            self.step = step
            self.eps = eps

        def compute(self):
            homology = self.creator.create_feature('H',
                                                   n=self.n,
                                                   dim=self.dim,
                                                   step=self.step,
                                                   eps=self.eps)
            return homology.values().np.diff(axis=1)

    class no(Feature):

        def __init__(self, n, dim=2, step=1, eps=0):
            self.dim = dim
            self.n = n
            self.step = step
            self.eps = eps

        def compute(self):
            homology = self.creator.create_feature('H',
                                                   n=self.n,
                                                   dim=self.dim,
                                                   step=self.step,
                                                   eps=self.eps)

            return homology.values().map(lambda x: x.shape[0])

    class mean(Feature):

        def __init__(self, n, dim=2, step=1, eps=0):
            self.dim = dim
            self.n = n
            self.step = step
            self.eps = eps

        def compute(self):
            life_time = self.creator.create_feature('life_time',
                                                    n=self.n,
                                                    dim=self.dim,
                                                    step=self.step,
                                                    eps=self.eps)
            return life_time.values().np.mean()

    class var(Feature):

        def __init__(self, n, dim=2, step=1, eps=0):
            self.dim = dim
            self.n = n
            self.step = step
            self.eps = eps

        def compute(self):
            life_time = self.creator.create_feature('life_time',
                                                    n=self.n,
                                                    dim=self.dim,
                                                    step=self.step,
                                                    eps=self.eps)
            return life_time.values().np.var()

    class kmp_features(Feature):
        """Features from Chatter Classification in Turning using
           Machine Learning and Topological Data Analysis"""

        def __init__(self, k, n, dim=2, step=1, eps=0):
            self.dim = dim  # point cloud dimension
            self.n = n  # homology dimension
            self.k = k  # number of kmp-feature
            self.step = step
            self.eps = eps

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
                                                   eps=self.eps)

            return homology.values().map(lambda x:
                                         __features(x)[self.k - 1])

    class wasser_ampl(Feature):
        def __init__(self, p, n, dim=2, step=1, eps=0):
            self.dim = dim  # point cloud dimension
            self.n = n  # homology dimension
            self.p = p  # p-parameter of the Lp-norm
            self.step = step
            self.eps = eps

        def compute(self):
            homology = self.creator.create_feature('H',
                                                   n=self.n,
                                                   dim=self.dim,
                                                   step=self.step,
                                                   eps=self.eps)

            return homology.values().map(lambda x: wasserstein_amplitude(x, self.p))

    class entropy(Feature):
        def __init__(self, n, dim=2, step=1, eps=0):
            self.dim = dim  # point cloud dimension
            self.n = n  # homology dimension
            self.step = step
            self.eps = eps

        def compute(self):
            homology = self.creator.create_feature('H',
                                                   n=self.n,
                                                   dim=self.dim,
                                                   step=self.step,
                                                   eps=self.eps)

            return homology.values().map(persistent_entropy)

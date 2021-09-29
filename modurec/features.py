# TODO: correct imports in this package
# TODO: better implementation of __str__ (defalult arguments)
# TODO: Remove from features dim = None, etc.
# TODO: Abstract classes / methods
from functools import partial
from inspect import getfullargspec

import pandas as pd
import numpy as np
from ripser import Rips

from . import pandex  # Necessary for mr and np accessors

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
            df[str(instance)] = instance.compute()

        return instance

    def get_values(self, feature):

        return self.df[str(feature)]

    class Feature:

        def __str__(self):
            keys = getfullargspec(self.__init__).args[1:]
            values = [self.__dict__[key] for key in keys]

            # values = [value for key, value in self.__dict__.items()
            #           if key in keys]

            string = self.__class__.__name__

            if keys:
                string += '_' + '={}_'.join(keys) + '={}'
                string = string.format(*values)

            #
            # string = '={}_'.join(keys)
            # string = self.__class__.__name__ + '_' + string
            # string += '={}'

            return string

        def values(self):
            return self.df[str(self)]

    # TODO: change to signalI signalQ
    class signal_sample(Feature):

        # def __init__(self):
        #     pass

        def compute(self):
            return self.df['signal_sample']

    class signal_sampleQ(Feature):

        # def __init__(self):
        #     pass

        def compute(self):
            return self.df['signal_sampleQ']

    class point_cloud(Feature):

        dim = None

        def __init__(self, dim=2):
            self.dim = dim

        def compute(self):
            def __compute2D(samples):
                samples.mr.add_point_cloud(window=1)

                return samples['point_cloud']

            def __compute3D(samples):

                def __standardize(x):
                    range0 = np.max(x[:, 0]) - np.min(x[:, 0])
                    range1 = np.max(x[:, 1]) - np.min(x[:, 1])
                    range2 = np.max(x[:, 2]) - np.min(x[:, 2])
                    x[:, 2] = x[:, 2] * 0.5 * (range0 + range1) / range2
                    return x

                samples.mr.add_point_cloud(window=1, point_cloud_col='cloud_3D')
                samples['cloud_3D'] = \
                    samples.cloud_3D.apply(lambda x:
                                           np.column_stack([x, range(x.shape[0])]))
                samples['cloud_3D'] = samples.cloud_3D.apply(__standardize)

                return samples['cloud_3D']

            def __compute4D(sample):
                samples.mr.add_point_cloud(window=2, point_cloud_col='cloud_4D')
                return samples['cloud_4D']

            signal_sample = self.creator.create_feature('signal_sample')
            signal_sampleQ = self.creator.create_feature('signal_sampleQ')

            samples = pd.concat([signal_sample.values(),
                                 signal_sampleQ.values()],
                                axis=1)

            if self.dim == 2:
                return __compute2D(samples)
            elif self.dim == 3:
                return __compute3D(samples)
            elif self.dim == 4:
                return __compute4D(samples)

            raise(NotImplemented('Dimension not implemented.'))

    class diagram(Feature):

        dim: int

        def __init__(self, dim=2):
            self.dim = dim

        def compute(self):
            point_cloud = self.creator.create_feature('point_cloud', dim=self.dim)
            return point_cloud.values().map(self.creator.rips.fit_transform)

    class H(Feature):

        n: int
        dim: int

        def __init__(self, n, dim=2):
            self.dim = dim
            self.n = n

        def compute(self):
            diagram = self.creator.create_feature('diagram', dim=self.dim)

            return pd.DataFrame(diagram.values().tolist(),
                                    index=self.df.index)[self.n]

    class life_time(Feature):

        n = None
        dim = None

        def __init__(self, n, dim=2):
            self.dim = dim
            self.n = n

        def compute(self):
            homology = self.creator.create_feature('H',
                                                   n=self.n,
                                                   dim=self.dim)
            return homology.values().np.diff(axis=1)

    class no(Feature):

        n = None
        dim = None

        def __init__(self, n, dim=2):
            self.dim = dim
            self.n = n

        def compute(self):
            homology = self.creator.create_feature('H',
                                                   n=self.n,
                                                   dim=self.dim)
            return homology.values().map(lambda x: x.shape[0])

    class mean(Feature):

        n = None
        dim = None

        def __init__(self, n, dim=2):
            self.dim = dim
            self.n = n

        def compute(self):
            life_time = self.creator.create_feature('life_time',
                                                    n=self.n,
                                                    dim=self.dim)
            return life_time.values().np.mean()

    class var(Feature):

        n = None
        dim = None

        def __init__(self, n, dim=2):
            self.dim = dim
            self.n = n

        def compute(self):
            life_time = self.creator.create_feature('life_time',
                                                    n=self.n,
                                                    dim=self.dim)
            return life_time.values().np.var()



@pd.api.extensions.register_dataframe_accessor('feat')
class SignalFeatures:

    def __init__(self, df):
        self.df = df
        self.rips = Rips()

    def __getitem__(self, feature_name):

        if feature_name not in self.df.columns:
            f = getattr(SignalFeatures, feature_name, None)
            if f is None:
                return 'no such attribute'
            else:
                self.df[feature_name] = f(self)

        return self.df[feature_name]

    # point cloud
    def point_cloud(self):
        samples = self.df[['signal_sample', 'signal_sampleQ']].copy()
        samples.mr.add_point_cloud(window=1)
        return samples['point_cloud']

    def cloud_4D(self):
        samples = self.df[['signal_sample', 'signal_sampleQ']].copy()
        samples.mr.add_point_cloud(window=2, point_cloud_col='cloud_4D')
        return samples['cloud_4D']

    def cloud_3D(self):
        def __standardize(x):
            range0 = np.max(x[:, 0]) - np.min(x[:, 0])
            range1 = np.max(x[:, 1]) - np.min(x[:, 1])
            range2 = np.max(x[:, 2]) - np.min(x[:, 2])
            # import pdb; pdb.set_trace()
            x[:, 2] = x[:, 2] * 0.5 * (range0 + range1) / range2
            return x

        samples = self.df[['signal_sample', 'signal_sampleQ']].copy()
        samples.mr.add_point_cloud(window=1, point_cloud_col='cloud_3D')
        samples['cloud_3D'] = \
            samples.cloud_3D.apply(lambda x:
                                   np.column_stack([x, range(x.shape[0])]))
        samples['cloud_3D'] = samples.cloud_3D.apply(__standardize)

        return samples['cloud_3D']

    # persistence diagrams
    def diagram(self):
        return self.df.feat['point_cloud'].map(self.rips.fit_transform)

    def diagram_3D(self):
        return self.df.feat['cloud_3D'].map(self.rips.fit_transform)

    def diagram_4D(self):
        return self.df.feat['cloud_4D'].map(self.rips.fit_transform)

    # 2-dimensional features
    def H0(self):
        return pd.DataFrame(self.df['diagram'].tolist(),
                            index=self.df.index)[0]

    def H1(self):
        return pd.DataFrame(self.df['diagram'].tolist(),
                            index=self.df.index)[1]

    def H0_life_time(self):
        return self.df.feat['H0'].np.diff(axis=1)

    def H1_life_time(self):
        return self.df.feat['H1'].np.diff(axis=1)

    def no_H0(self):
        return self.df.feat['H0'].map(lambda x: x.shape[0])

    def no_H1(self):
        return self.df.feat['H1'].map(lambda x: x.shape[0])

    def H0_mean(self):
        return self.feat.df['H0_life_time'].np.mean()

    def H1_mean(self):
        return self.feat.df['H1_life_time'].np.mean()

    def H0_var(self):
        return self.feat.df['H0_life_time'].np.var()

    def H1_var(self):
        return self.feat.df['H1_life_time'].np.var()

    # 4-dimensional features
    def H0_4D(self):
        return pd.DataFrame(self.df['diagram_4D'].tolist(),
                            index=self.df.index)[0]

    def H1_4D(self):
        return pd.DataFrame(self.df['diagram_4D'].tolist(),
                            index=self.df.index)[1]

    def H0_life_time_4D(self):
        return self.df.feat['H0_4D'].np.diff(axis=1)

    def H1_life_time_4D(self):
        return self.df.feat['H1_4D'].np.diff(axis=1)

    def no_H0_4D(self):
        return self.df.feat['H0_4D'].map(lambda x: x.shape[0])

    def no_H1_4D(self):
        return self.df.feat['H1_4D'].map(lambda x: x.shape[0])

    def H0_mean_4D(self):
        return self.feat.df['H0_life_time_4D'].np.mean()

    def H1_mean_4D(self):
        return self.feat.df['H1_life_time_4D'].np.mean()

    def H0_var_4D(self):
        return self.feat.df['H0_life_time_4D'].np.var()

    def H1_var_4D(self):
        return self.feat.df['H1_life_time_4D'].np.var()

    # 3-dimensional features
    def H0_3D(self):
        return pd.DataFrame(self.df.feat['diagram_3D'].tolist(),
                            index=self.df.index)[0]

    def H1_3D(self):
        return pd.DataFrame(self.df.feat['diagram_3D'].tolist(),
                            index=self.df.index)[1]

    def H0_life_time_3D(self):
        return self.df.feat['H0_3D'].np.diff(axis=1)

    def H1_life_time_3D(self):
        return self.df.feat['H1_3D'].np.diff(axis=1)

    def no_H0_3D(self):
        return self.df.feat['H0_3D'].map(lambda x: x.shape[0])

    def no_H1_3D(self):
        return self.df.feat['H1_3D'].map(lambda x: x.shape[0])

    def H0_mean_3D(self):
        return self.df.feat['H0_life_time_3D'].np.mean()

    def H1_mean_3D(self):
        return self.df.feat['H1_life_time_3D'].np.mean()

    def H0_var_3D(self):
        return self.df.feat['H0_life_time_3D'].np.var()

    def H1_var_3D(self):
        return self.df.feat['H1_life_time_3D'].np.var()

    def H0_mean_norm(self):
        return self.df.feat['H0_mean'] / self.df.feat['no_H0']

    def H1_mean_norm(self):
        return self.df.feat['H1_mean'] / self.df.feat['no_H1']

    def kmp_features(self):

        def __features(D):
            mean_y = np.array([y for _, y in D if y != float('inf')]).mean()

            return (sum([x * (y - x) for x, y in D if y != float('inf')]),
                    sum([(mean_y - y) * (y - x) for x, y in D
                         if y != float('inf')]),
                    sum([x**2 * (y - x) for x, y in D if y != float('inf')]),
                    sum([(mean_y - y)**2 * (y - x)**4 for x, y in D
                         if y != float('inf')]),
                    max([y - x for x, y in D if y != float('inf')]))

        D = self.df.feat['H0']

        return D.map(__features)

    def kmp_f1(self):
        return self.df.feat['kmp_features'].map(lambda x: x[0])

    def kmp_f2(self):
        return self.df.feat['kmp_features'].map(lambda x: x[1])

    def kmp_f3(self):
        return self.df.feat['kmp_features'].map(lambda x: x[2])

    def kmp_f4(self):
        return self.df.feat['kmp_features'].map(lambda x: x[3])

    def kmp_f5(self):
        return self.df.feat['kmp_features'].map(lambda x: x[4])

    def kmp_features_H1(self):

        def __features(D):
            mean_y = np.array([y for _, y in D if y != float('inf')]).mean()

            return (sum([x * (y - x) for x, y in D if y != float('inf')]),
                    sum([(mean_y - y) * (y - x) for x, y in D
                         if y != float('inf')]),
                    sum([x**2 * (y - x) for x, y in D if y != float('inf')]),
                    sum([(mean_y - y)**2 * (y - x)**4 for x, y in D
                         if y != float('inf')]),
                    max([y - x for x, y in D if y != float('inf')]))

        D = self.df.feat['H1']

        return D.map(__features)

    def kmp_f1_H1(self):
        return self.df.feat['kmp_features_H1'].map(lambda x: x[0])

    def kmp_f2_H1(self):
        return self.df.feat['kmp_features_H1'].map(lambda x: x[1])

    def kmp_f3_H1(self):
        return self.df.feat['kmp_features_H1'].map(lambda x: x[2])

    def kmp_f4_H1(self):
        return self.df.feat['kmp_features_H1'].map(lambda x: x[3])

    def kmp_f5_H1(self):
        return self.df.feat['kmp_features_H1'].map(lambda x: x[4])

    def kmp_features_H1_4D(self):

        def __features(D):
            mean_y = np.array([y for _, y in D if y != float('inf')]).mean()

            return (sum([x * (y - x) for x, y in D if y != float('inf')]),
                    sum([(mean_y - y) * (y - x) for x, y in D
                         if y != float('inf')]),
                    sum([x**2 * (y - x) for x, y in D if y != float('inf')]),
                    sum([(mean_y - y)**2 * (y - x)**4 for x, y in D
                         if y != float('inf')]),
                    max([y - x for x, y in D if y != float('inf')]))

        D = self.df.feat['H1_4D']

        return D.map(__features)

    def kmp_f1_H1_4D(self):
        return self.df.feat['kmp_features_H1_4D'].map(lambda x: x[0])

    def kmp_f2_H1_4D(self):
        return self.df.feat['kmp_features_H1_4D'].map(lambda x: x[1])

    def kmp_f3_H1_4D(self):
        return self.df.feat['kmp_features_H1_4D'].map(lambda x: x[2])

    def kmp_f4_H1_4D(self):
        return self.df.feat['kmp_features_H1_4D'].map(lambda x: x[3])

    def kmp_f5_H1_4D(self):
        return self.df.feat['kmp_features_H1_4D'].map(lambda x: x[4])


if __name__ == '__main__':
    df = pd.read_pickle('../data/testpickle.pkl')
    homology0 = df.ff.create_feature('H', n=0, dim=2)
    homology1 = df.ff.create_feature('H', n=1, dim=2)
    life_time_0 = df.ff.create_feature('life_time', n=0, dim=2)
    life_time_1 = df.ff.create_feature('life_time', n=1, dim=2)
    no_0 = df.ff.create_feature('no', n=0, dim=2)
    no_1 = df.ff.create_feature('no', n=1, dim=2)
    mean_0 = df.ff.create_feature('mean', n=0, dim=2)
    mean_1 = df.ff.create_feature('mean', n=1, dim=2)
    var_0 = df.ff.create_feature('var', n=0, dim=2)
    var_1 = df.ff.create_feature('var', n=1, dim=2)

    assert(df['H_n=0_dim=2'].equals(df['H0']))
    assert(df['H_n=1_dim=2'].equals(df['H1']))
    assert(df['life_time_n=0_dim=2'].equals(df['H0_life_time']))
    assert(df['life_time_n=1_dim=2'].equals(df['H1_life_time']))
    assert(df['no_n=1_dim=2'].equals(df['no_H1']))
    assert(df['no_n=0_dim=2'].equals(df['no_H0']))
    assert(df['mean_n=1_dim=2'].equals(df['H1_mean']))
    assert(df['mean_n=0_dim=2'].equals(df['H0_mean']))
    assert(df['var_n=1_dim=2'].equals(df['H1_var']))
    assert(df['var_n=0_dim=2'].equals(df['H0_var']))









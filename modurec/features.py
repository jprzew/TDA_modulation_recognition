# TODO: Abstract classes / methods
# TODO: Implement new version of symbol-rate features
# from inspect import getfullargspec
import inspect

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
            self.df[str(instance)] = instance.compute()

        return instance

    def get_values(self, feature):

        return self.df[str(feature)]

    class Feature:

        def __str__(self):
            keys = inspect.getfullargspec(self.__init__).args[1:]
            params = inspect.signature(self.__init__).parameters

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

    # TODO: change to signalI signalQ
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

            signalI = self.creator.create_feature('signalI')
            signalQ = self.creator.create_feature('signalQ')

            samples = pd.concat([signalI.values(),
                                 signalQ.values()],
                                axis=1)

            if self.dim == 2:
                return __compute2D(samples)
            elif self.dim == 3:
                return __compute3D(samples)
            elif self.dim == 4:
                return __compute4D(samples)

            raise(NotImplemented('Dimension not implemented.'))

    class diagram(Feature):

        def __init__(self, dim=2):
            self.dim = dim

        def compute(self):
            point_cloud = self.creator.create_feature('point_cloud', dim=self.dim)
            return point_cloud.values().map(self.creator.rips.fit_transform)

    class H(Feature):

        def __init__(self, n, dim=2):
            self.dim = dim
            self.n = n

        def compute(self):
            diagram = self.creator.create_feature('diagram', dim=self.dim)

            return pd.DataFrame(diagram.values().tolist(),
                                    index=self.df.index)[self.n]

    class life_time(Feature):

        def __init__(self, n, dim=2):
            self.dim = dim
            self.n = n

        def compute(self):
            homology = self.creator.create_feature('H',
                                                   n=self.n,
                                                   dim=self.dim)
            return homology.values().np.diff(axis=1)

    class no(Feature):

        def __init__(self, n, dim=2):
            self.dim = dim
            self.n = n

        def compute(self):
            homology = self.creator.create_feature('H',
                                                   n=self.n,
                                                   dim=self.dim)
            return homology.values().map(lambda x: x.shape[0])

    class mean(Feature):

        def __init__(self, n, dim=2):
            self.dim = dim
            self.n = n

        def compute(self):
            life_time = self.creator.create_feature('life_time',
                                                    n=self.n,
                                                    dim=self.dim)
            return life_time.values().np.mean()

    class var(Feature):

        def __init__(self, n, dim=2):
            self.dim = dim
            self.n = n

        def compute(self):
            life_time = self.creator.create_feature('life_time',
                                                    n=self.n,
                                                    dim=self.dim)
            return life_time.values().np.var()


    class kmp_features(Feature):
        """Features from Chatter Classification in Turning using
           Machine Learning and Topological Data Analysis"""

        def __init__(self, k, n, dim=2):
            self.dim = dim  # point cloud dimension
            self.n = n  # homology dimension
            self.k = k  # number of kmp-feature

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
                                                   dim=self.dim)

            return homology.values().map(lambda x:
                                         __features(x)[self.k -1])


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

    # symbol rate
    # WARNING: This implementation returns 1 if there is a division by zero problem!!!
    # TODO: Deal with the above problem
    def symbol_rate(self,
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

        df['symbol_rate'] = df.apply(lambda x:
                                     np.argmax(np.log(x['power_spectrum']) < x['threshold']), axis=1)

        df['symbol_rate'] = df.apply(lambda x:
                                     (x['freq'][x['symbol_rate']]) ** (-1), axis=1)

        df['symbol_rate'] = df['symbol_rate'].apply(lambda x:
                                                    1.0 if x == float('inf') else
                                                    np.around(x))

        return df['symbol_rate'].astype(int)


    # point cloud
    def point_cloud_sr(self,
                       data_col='signalI',
                       data_colQ='signalQ',
                       sr_col='symbol_rate',
                       window=2):

        def __to_real(x):
            array = np.vstack((np.real(x), np.imag(x))).T
            return array.ravel()

        def __complex_cloud_to_real(cloud):
            return np.apply_along_axis(lambda x:
                                       __to_real(x), axis=1, arr=cloud)


        df = self.df[[data_col, data_colQ]].copy()
        df[sr_col] = self.df.feat[sr_col]

        df['data'] = self.df.apply(lambda x: x[data_col] + 1j*x[data_colQ],
                                          axis=1)

        result = df.apply(lambda x: rolling_window(x['data'],
                                                   window=window,
                                                   stride_step=x[sr_col]), axis=1)
        result = result.apply(__complex_cloud_to_real)
        return result

    def point_cloud(self):
        samples = self.df[['signalI', 'signalQ']].copy()
        samples.mr.add_point_cloud(window=1)
        return samples['point_cloud']

    def cloud_4D(self):
        samples = self.df[['signalI', 'signalQ']].copy()
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

        samples = self.df[['signalI', 'signalQ']].copy()
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

    def diagram_sr(self):
        return self.df.feat['point_cloud_sr'].map(self.rips.fit_transform)

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

    # symbol-rate features
    def H0_sr(self):
        return pd.DataFrame(self.df.feat['diagram_sr'].tolist(),
                            index=self.df.index)[0]

    def H1_sr(self):
        return pd.DataFrame(self.df.feat['diagram_sr'].tolist(),
                            index=self.df.index)[1]

    def H0_life_time_sr(self):
        return self.df.feat['H0_sr'].np.diff(axis=1)

    def H1_life_time_sr(self):
        return self.df.feat['H1_sr'].np.diff(axis=1)

    def no_H0_sr(self):
        return self.df.feat['H0_sr'].map(lambda x: x.shape[0])

    def no_H1_sr(self):
        return self.df.feat['H1_sr'].map(lambda x: x.shape[0])

    def H0_mean_sr(self):
        return self.df.feat['H0_life_time_sr'].np.mean()

    def H1_mean_sr(self):
        return self.df.feat['H1_life_time_sr'].np.mean()

    def H0_var_sr(self):
        return self.df.feat['H0_life_time_sr'].np.var()

    def H1_var_sr(self):
        return self.df.feat['H1_life_time_sr'].np.var()

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

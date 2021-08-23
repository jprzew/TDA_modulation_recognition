import pandas as pd
import numpy as np
from modurec import pandex    # Necessary for the mr dataframe accessor
from ripser import Rips


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

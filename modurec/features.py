import pandas as pd


@pd.api.extensions.register_dataframe_accessor('feat')
class SignalFeatures:

    def __init__(self, df):
        self.df = df

    def __getitem__(self, feature_name):

        if feature_name not in self.df.columns:
            f = getattr(SignalFeatures, feature_name, None)
            if f is None:
                return 'no such attribute'
            else:
                self.df[feature_name] = f(self)

        return self.df[feature_name]

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
        return pd.DataFrame(self.df['diagram_3D'].tolist(),
                            index=self.df.index)[0]

    def H1_3D(self):
        return pd.DataFrame(self.df['diagram_3D'].tolist(),
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

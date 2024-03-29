import pandas as pd
from .point_cloud import PointCloud
from .diagram import Diagram
from .feature import Feature
from .lifetime import Lifetime
from .homology import Homology
from .statistics import No, Mean, Var, KmpFeatures, WasserAmpl, Entropy

registered_features = {'point_cloud': PointCloud,
                       'diagram': Diagram,
                       'homology': Homology,
                       'lifetime': Lifetime,
                       'no': No,
                       'mean': Mean,
                       'var': Var,
                       'kmp_features': KmpFeatures,
                       'wasser_ampl': WasserAmpl,
                       'entropy': Entropy}


@pd.api.extensions.register_dataframe_accessor('ff')
class FeaturesFactory:

    def __init__(self, df):
        self.df = df
        self.features = []

    def create_feature(self, name: str, **kwargs):

        feature_cls = registered_features[name]
        instance = feature_cls(**kwargs)

        self.features.append(instance)
        instance.creator = self

        # If df does not contain column with the name of the feature, compute it
        column_name = str(instance)
        if column_name not in self.df.columns:
            self.df[column_name] = instance.compute()

        return instance

    def get_values(self, feature: 'Feature'):
        return self.df[str(feature)]

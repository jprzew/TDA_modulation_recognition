import pandas as pd
from utils import get_repo_path
import config as cfg
from feature_list import all_features
from modurec.features.feature import calculate_feature


# Reading the data
df = pd.read_pickle(get_repo_path() / cfg.SampleData.sampled_data_file)

# Reading the diagrams
diagrams = pd.read_pickle(get_repo_path() / cfg.Diagrams.diagrams_file)

# Calculate features
print('Calculating features...')
results = []
for case, feature_data in enumerate(all_features):
    print(f'Calculating {case+1}/{len(all_features)}. Feature data: {feature_data}')
    print('------------------------')
    results.append(calculate_feature(df=diagrams, feature_data=feature_data))

# Save to file
pd.concat(results, axis=1).to_pickle(get_repo_path() / cfg.Featurize.features_file)

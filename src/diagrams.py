import config as cfg
import pandas as pd
from utils import get_repo_path
from modurec.features.feature import calculate_feature
from joblib import Memory
from dvc.api import params_show
import numpy as np

# Setting random seed
params = params_show()['diagrams']
np.random.seed(params['random_seed'])

# Prepare caching
CACHEDIR = get_repo_path() / '.cache'
memory = Memory(CACHEDIR, verbose=0)


@memory.cache
def calculate_feature_cached(feature_data):
    """Cached version of calculate_feature"""

    return calculate_feature(df=df, feature_data=feature_data)


# Read data
df = pd.read_pickle(get_repo_path() / cfg.SampleData.sampled_data_file)

# Subsample modulations and cases
df = df[df.modulation_type.isin(cfg.Diagrams.modulation_subset)]

# Filter by SNR
if cfg.Diagrams.snr_threshold is not None:
    df = df[df.SNR >= cfg.Diagrams.snr_threshold]

# Sample cases
if cfg.Diagrams.sample_size is not None:
    df = (df.groupby(['modulation_type', 'SNR'], as_index=False)
            .apply(lambda x: x.sample(n=cfg.Diagrams.sample_size))
            .droplevel(0))

# Calculate diagrams
print('Calculating diagrams...')
results = []
for case, feature_data in enumerate(cfg.Diagrams.to_calculate):
    print(f'Calculating {case+1}/{len(cfg.Diagrams.to_calculate)}. Feature data: {feature_data}')
    print('------------------------')
    results.append(calculate_feature_cached(feature_data=feature_data))

# Save to file
pd.concat(results, axis=1).to_pickle(get_repo_path() / cfg.Diagrams.diagrams_file)

# Clean cache
memory.clear(warn=False)

import config as cfg
import pandas as pd
from utils import get_repo_path
import modurec.features  # Needed for feature factory to work

df = pd.read_pickle(get_repo_path() / cfg.SampledData.sampled_data_file)

# Subsample modulations and cases
df = df[df.modulation_type.isin(cfg.Diagrams.modulation_subset)]

if cfg.Diagrams.sample_size is not None:
    df = df.groupby('modulation_type').apply(lambda x: x.sample(n=cfg.Diagrams.sample_size))

# Calculate diagrams
print('Calculating diagrams...')
diagram_features = []
for case, feature_data in enumerate(cfg.Diagrams.to_calculate):
    print(f'Calculating {case+1}/{len(cfg.Diagrams.to_calculate)}')
    print('------------------------')
    diagram_features.append(df.ff.create_feature(feature_data.name, **feature_data.params))

# Save to file
to_take = list(map(str, diagram_features))
df[to_take].to_pickle(get_repo_path() / cfg.Diagrams.diagrams_file)

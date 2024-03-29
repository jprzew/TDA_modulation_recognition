"""Calculates features out of persistence diagrams"""

import pandas as pd
import warnings
from utils import get_repo_path
import config as cfg
from feature_list import all_features
from modurec.features.feature import calculate_feature


def main():

    # Reading the diagrams
    diagrams = pd.read_pickle(get_repo_path() / cfg.Diagrams.diagrams_file)

    # Calculate features
    print('Calculating features...')
    results = []
    for case, feature_data in enumerate(all_features):
        print(f'Calculating {case+1}/{len(all_features)}. Feature data: {feature_data}')
        print('------------------------')
        with warnings.catch_warnings(record=True) as w:
            results.append(calculate_feature(df=diagrams, feature_data=feature_data))
            if (len(w) > 0) and (w[0].category == pd.errors.PerformanceWarning):  # In case of dataframe fragmentation
                diagrams = diagrams.copy()

    # Save to file
    pd.concat(results, axis=1).to_pickle(get_repo_path() / cfg.Featurize.features_file)


if __name__ == '__main__':
    main()

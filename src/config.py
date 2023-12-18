from modurec.features.feature import FeatureData
from typing import Optional


class General:
    hdf_data_file = 'data/GOLD_XYZ_OSC.0001_1024.hdf5'


class TrainTestSplit:
    test_proportion = 0.1
    test_indices_file = 'data/test_indices.csv'
    train_indices_file = 'data/train_indices.csv'
    random_seed = 42


class SampleData:
    sampled_data_file = 'data/data.pkl'
    cases_per_class = 100
    random_seed = 42


class Diagrams:
    modulation_subset: list = ['BPSK', 'QPSK', '8PSK', '16PSK',
                               '32PSK', '16QAM', '32QAM', '64QAM',
                               'FM', 'GMSK', 'OQPSK']

    to_calculate = [FeatureData(name='diagram', params={'dim': 2}),
                    FeatureData(name='diagram', params={'dim': 3}),
                    FeatureData(name='diagram', params={'dim': 4}),
                    FeatureData(name='diagram', params={'dim': 10}),
                    FeatureData(name='diagram', params={'dim': 2, 'kind': 'abs'}),
                    FeatureData(name='diagram', params={'dim': 10, 'kind': 'abs'}),
                    FeatureData(name='diagram', params={'dim': 2, 'kind': 'phi'}),
                    FeatureData(name='diagram', params={'dim': 10, 'kind': 'phi'}),
                    FeatureData(name='diagram', params={'dim': 2, 'kind': 'abs', 'fil': 'star'}),
                    FeatureData(name='diagram', params={'dim': 2, 'kind': 'phi', 'fil': 'star'}),
                    FeatureData(name='diagram', params={'dim': 2, 'step': 30}),
                    FeatureData(name='diagram', params={'dim': 4, 'step': 30})]

    sample_size: Optional[int] = 2  # sample size per modulation
    snr_threshold: Optional[int] = 10

    diagrams_file = 'data/diagrams.pkl'


class Featurize:
    modulation_subset = ['BPSK', 'QPSK', '8PSK', '16PSK',
                         '32PSK', '16QAM', '32QAM', '64QAM',
                         'FM', 'GMSK', 'OQPSK']
    features_file = 'data/features.pkl'


class Spotcheck:
    input_file = 'data/features.pkl'




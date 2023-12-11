class General:
    hdf_data_file = 'data/GOLD_XYZ_OSC.0001_1024.hdf5'


class TrainTestSplit:
    test_proportion = 0.1
    test_indices_file = 'data/test_indices.csv'
    train_indices_file = 'data/train_indices.csv'


class SampledData:
    sampled_data_file = 'data/data.pkl'
    cases_per_class = 100


class Diagrams:
    pass


class Features:
    modulation_subset = ['BPSK', 'QPSK', '8PSK', '16PSK',
                         '32PSK', '16QAM', '32QAM', '64QAM',
                         'FM', 'GMSK', 'OQPSK']
    features_file = 'data/features.pkl'



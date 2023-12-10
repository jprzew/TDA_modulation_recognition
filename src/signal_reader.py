#!/usr/bin/env python3
import os
import sys
import numpy as np
import pandas as pd
import h5py
import pickle
import random as rnd
from modurec import features
from utils import get_repo_path

# Class labelings 2018
class_to_index = {'OOK': 0,
                  '4ASK': 1,
                  '8ASK': 2,
                  'BPSK': 3,
                  'QPSK': 4,
                  '8PSK': 5,
                  '16PSK': 6,
                  '32PSK': 7,
                  '16APSK': 8,
                  '32APSK': 9,
                  '64APSK': 10,
                  '128APSK': 11,
                  '16QAM': 12,
                  '32QAM': 13,
                  '64QAM': 14,
                  '128QAM': 15,
                  '256QAM': 16,
                  'AM-SSB-WC': 17,
                  'AM-SSB-SC': 18,
                  'AM-DSB-WC': 19,
                  'AM-DSB-SC': 20,
                  'FM': 21,
                  'GMSK': 22,
                  'OQPSK': 23}

index_to_class = {v: k for k, v in class_to_index.items()}


class_to_id_2016 = {'QPSK': 0,
                    'PAM4': 1,
                    'AM-DSB': 2,
                    'GFSK': 3,
                    'QAM64': 4,
                    'AM-SSB': 5,
                    '8PSK': 6,
                    'QAM16': 7,
                    'WBFM': 8,
                    'CPFSK': 9,
                    'BPSK': 10}


repo_path = get_repo_path()

# to_compute = [{'dim': 2, 'step': 1, 'eps': 0},
#               {'dim': 3, 'step': 1, 'eps': 0},
#               {'dim': 4, 'step': 1, 'eps': 0}]

# Temporary dictionary: only one diagram will be computed
to_compute = [{'dim': 2, 'step': 1, 'eps': 0}]

dictionary_pickle_file_path = r'../data/RML2016.10a_dict.pkl'

hdf5_input_file_name = 'GOLD_XYZ_OSC.0001_1024.hdf5'
hdf5_input_dict = 'hdf5_data'
hdf5_file = os.path.join(hdf5_input_dict, hdf5_input_file_name)

test_indices_file_name = 'test_indices.csv'
train_indices_file_name = 'train_indices.csv'
indices_dict = 'hdf5_data'
test_indices_file = os.path.join(indices_dict, test_indices_file_name)
train_indices_file = os.path.join(indices_dict, train_indices_file_name)

signal_samples_fileI_name = 'signalsI.csv'
signal_samples_fileQ_name = 'signalsQ.csv'
modulations_id_file_name = 'modulations.csv'
snrs_file_name = 'snrs.csv'


def split_and_save_indices_rml18(data_file,
                                 test_indices_file,
                                 train_indices_file,
                                 proportion=0.3,
                                 seed=42):
    """The function is needed in order to divide the datasets into the training and testing data."""

    data_file = repo_path / data_file
    test_indices_file = repo_path / test_indices_file
    train_indices_file = repo_path / train_indices_file

    rnd.seed(seed)
    with h5py.File(data_file, 'r') as f:
        num_of_samples = f['Z'][:].shape[0]

    test_size = int(num_of_samples * proportion)
    test_ind = rnd.sample(range(num_of_samples), test_size)
    train_ind = np.setdiff1d(range(num_of_samples), test_ind)

    np.savetxt(test_indices_file, test_ind, delimiter=',', fmt='%i')
    np.savetxt(train_indices_file, train_ind, delimiter=',', fmt='%i')



def sample_indices(indices_file, number, seed=42):
    """Returns a random sample of indices from indices_file"""

    indices = np.loadtxt(indices_file, delimiter=',', dtype=int)

    np.random.seed(seed)
    return np.random.choice(indices, size=number)


# TODO: Diagrams should not require signalI, etc. Correct this.
def compute_diagrams(point_cloud): 
    """Computes diagrams defined by to_compute dict
       Return value: dictionary of the form
       {'name': diagram}
    """

    df = pd.DataFrame({'point_cloud': [point_cloud]})
    df['signalI'] = df.point_cloud.map(lambda x: x[:, 0])
    df['signalQ'] = df.point_cloud.map(lambda x: x[:, 1])

    feat = [df.ff.create_feature('diagram', **params) for
            params in to_compute]

    diagrams = map(lambda x: x.values().iloc[0], feat)

    return {name: diags for name, diags in
            zip(map(str, feat), diagrams)}


def add_row_to_hdf5(hdf5_file, cloud, modulation, snr,
                    index, diagrams, row):
    """Function adds row to hdf5_file

    Args:
        hdf5_file: hdf5-file object (from h5py)
        cloud: pt. cloud will be written in 'point_cloud' array of the file
        modulation: modulation id in the one hot form
            will be written in modulation_one_hot array
        snr: value of SNR - will be written in SNR array
        index: index of this particular observation (from the original file)
            will be writen in index array
        diagrams: list of diagrams (wrt homology dimension)
            will be written in 'diagram' group consisting of two
            subgroups '0' and '1'. Each subgroup is a set of arrays
            indexed by str(row)
        row: row number
    """

    point_cloud = hdf5_file['point_cloud']
    modulation_one_hot = hdf5_file['modulation_one_hot']
    SNR = hdf5_file['SNR']
    ind = hdf5_file['index']
    diagram0 = hdf5_file['diagram']['0']
    diagram1 = hdf5_file['diagram']['1']

    point_cloud[row, :, :] = cloud
    modulation_one_hot[row, :] = modulation
    SNR[row] = snr
    ind[row] = index
    diagram0[str(row)] = diagrams['diagram'][0]
    diagram1[str(row)] = diagrams['diagram'][1]


def create_structure_hdf5(hdf5_file, length,
                          no_samples, no_components, no_modulations):
    """Creates structure of the hdf5-file with diagrams

    Args:
        hdf5_file: hdf5-file object (from h5py)
        length: number of observations
        no_samples: number of samples of each point cloud
        no_components: number of dimensions of the point cloud
        no_modulations: number of modulations

    The file will consist of: 'point_cloud', 'modulation_one_hot'
    'SNR', 'index' - arrays; and /diagram/0 /diagram/1 that are groups

        """

    hdf5_file.create_dataset('point_cloud',
                                 (length, no_samples, no_components))
    hdf5_file.create_dataset('modulation_one_hot',
                             (length, no_modulations))
    hdf5_file.create_dataset('SNR', (length,))
    hdf5_file.create_dataset('index', (length,))
    hdf5_file.create_group('/diagram', track_order=True)
    hdf5_file.create_group('/diagram/0', track_order=True)
    hdf5_file.create_group('/diagram/1', track_order=True)


def select_train_hdf5(number, output_file, parameters=to_compute,
                      hdf5_file=hdf5_file, seed=42):
    """Selects random sample from indices defined by train_indices_file
       and computes diagrams defined by parameters argument
       Result is written to output_file (str)
    """

    data = h5py.File(hdf5_file, 'r')
    modulation_one_hot = data['Y']
    SNR = data['Z']
    point_cloud = data['X']

    no_samples = point_cloud.shape[1]
    no_components = point_cloud.shape[2]
    no_modulations = modulation_one_hot.shape[1]

    indices = sample_indices(train_indices_file, number,
                             seed=seed)

    with h5py.File(output_file, 'w') as f:
        create_structure_hdf5(hdf5_file=f,
                              length=len(indices),
                              no_samples=no_samples,
                              no_components=no_components,
                              no_modulations=no_modulations)

        for row, index in enumerate(indices):
            cloud = point_cloud[index, :, :]
            diagrams = compute_diagrams(cloud)
            add_row_to_hdf5(hdf5_file=f,
                            cloud=cloud,
                            modulation=modulation_one_hot[index, :],
                            snr=SNR[index].flatten(),
                            index=index,
                            diagrams=diagrams,
                            row=row)


def create_index_df(hdf5_file=hdf5_file,
                    indices_file=train_indices_file):
    """Creates dataframe of indices. Helper function to be used
    when selecting random sample of indices"""

    indices = np.loadtxt(indices_file, delimiter=',', dtype=int)

    data = h5py.File(hdf5_file, 'r')
    modulation_one_hot = data['Y']
    modulations = np.apply_along_axis(lambda x: x.argmax(),
                                      1,
                                      modulation_one_hot[:, :])
    SNR = data['Z'][:].flatten()

    return pd.DataFrame({'index': indices,
                         'modulation_id': modulations[indices],
                         'SNR': SNR[indices]})


def filter_df(df,
              condition_snr=lambda x: x >= 6,
              condition_mod_id=lambda x: x in range(24)):
    """Filters dataframe wrt. to conditions on SNR and modulation id"""

    return df.loc[df.SNR.apply(condition_snr) & 
                  df.modulation_id.apply(condition_mod_id)]


def random_subsample(df, size, seed=42):
    """Creates a random subsample from 'df' with 'size' observations
    in each subgroup"""


    np.random.seed(seed)
    grouped = df.groupby(['modulation_id', 'SNR'], as_index=False)

    df = grouped.apply(lambda x: x.loc[np.random.choice(x.index,
                                                        size,
                                                        False)])
    return df.sort_index()


def create_pickle(indices, output_file, hdf5_file=hdf5_file):
    """Creates pickle from selected cases from hdf5-file

    Args:
        indices: indices of selected cases
        output_file: output file name
        hdf5_file: hdf5-file name
    """

    data = h5py.File(hdf5_file, 'r')
    modulation_one_hot = data['Y']
    SNR = data['Z']
    point_cloud = data['X']

    # no_samples = point_cloud.shape[1]
    # no_components = point_cloud.shape[2]
    # no_modulations = modulation_one_hot.shape[1]

    point_cloud = point_cloud[indices, :, :]
    modulations = np.apply_along_axis(lambda x: x.argmax(),
                                      1,
                                      modulation_one_hot[indices, :])
    SNR = SNR[indices].flatten()

    df = pd.DataFrame({'point_cloud': list(point_cloud),
                       'modulation_id': modulations,
                       'SNR': SNR}, index=indices)
    df['modulation_type'] = df.modulation_id.apply(lambda x: index_to_class[x])

    # TODO: Erase signalI and signalQ properties!
    df['signalI'] = df['point_cloud'].apply(lambda x: x[:, 0])
    df['signalQ'] = df['point_cloud'].apply(lambda x: x[:, 1])
    
    df.to_pickle(output_file)


def select_train_pkl(size,
                     output_file,
                     condition_snr=lambda x: x >= 6,
                     condition_mod_id=lambda x: x in range(24),
                     hdf5_file=hdf5_file,
                     indices_file=train_indices_file,
                     seed=42):
    """Selects a subsample from hdf5-file and saves is as a pickle

    Args:
        size: number of observations in each group defined by SNR and modulation
        output_file: name of the output pickle
        condition_snr: condition for SNR
        condition_mod_id: condition for modulation id
        hdf5_file: name of the hdf5-file
        indices_file: file with indices of the training data
        seed: random seed
    """

    df = create_index_df(hdf5_file=hdf5_file, indices_file=indices_file)
    df = filter_df(df, condition_snr=condition_snr,
                   condition_mod_id=condition_mod_id)
    df = random_subsample(df, size=size, seed=seed)
    
    create_pickle(indices=df['index'],
                  output_file=output_file,
                  hdf5_file=hdf5_file)

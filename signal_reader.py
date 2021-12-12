#!/usr/bin/env python3
import os
import sys
import numpy as np
import pandas as pd
import h5py
import pickle
import random as rnd
from modurec import features
# import vaex

# Class labelings 2018
class_to_index = {  'OOK': 0,
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


def split_and_save_indices_rml18(proportion=0.3,
                                 data_file=hdf5_file,
                                 test_indices_file=test_indices_file,
                                 train_indices_file=train_indices_file,
                                 seed=42):
    """The function is needed in order to divide the datasets into the training and testing data."""

    rnd.seed(seed)
    with h5py.File(data_file, 'r') as f:
        num_of_samples = f['Z'][:].shape[0]

    test_size = int(num_of_samples * proportion)
    test_ind = rnd.sample(range(num_of_samples), test_size)
    train_ind = np.setdiff1d(range(num_of_samples), test_ind)

    np.savetxt(test_indices_file, test_ind, delimiter=',', fmt='%i')
    np.savetxt(train_indices_file, train_ind, delimiter=',', fmt='%i')

# TODO: This function should probably be removed
# def select_sample_from_train(number_of_signals,  # per class / pairs (snr, modulation)
#                              data_file=hdf5_file,
#                              total_random=True,
#                              output_signals_fileI=signal_samples_fileI_name,
#                              output_signals_fileQ=signal_samples_fileQ_name,
#                              output_modulations_file=modulations_id_file_name,
#                              output_snr_file=snrs_file_name,
#                              output_path='../numpy_data',
#                              random=False,
#                              seed=None,
#                              snr_min=float('-inf'),
#                              snr_max=float('inf'),
#                              train_indices_file=train_indices_file,
#                              test_indices_file=test_indices_file):

#     rnd.seed(seed)
#     train_ind = np.loadtxt(train_indices_file, delimiter=',', dtype=int)
#     test_ind = np.loadtxt(test_indices_file, delimiter=',', dtype=int)
#     number_of_all_signals = train_ind.shape[0] + test_ind.shape[0]

#     if total_random:
#         if number_of_signals > train_ind.shape[0]:
#             number_of_signals = train_ind.shape[0]
#             print("All train data chosen")
#         indices_taken = rnd.sample(range(train_ind.shape[0]), number_of_signals)
#         indices_taken.sort()
#         indices_np = np.zeros(number_of_all_signals)
#         indices_np[indices_taken] = 1
#         df = vaex.open(data_file)
#         assert len(df) == number_of_all_signals
#         df['index'] = indices_np
#         df = df[df.index == 1]
#         df.rename('Y', 'modulation_one_hot')
#         df.rename('Z', 'SNR')
#         df.rename('X', 'point_cloud')
#         return df


#     else:
#         dataX, modulations, snrs = read_data_from_h5py(data_file)
#         modulations = modulations[train_ind]
#         snrs = snrs[train_ind]
#         indices_taken = __create_index_list(modulations, snrs, snr_max, snr_min, random, number_of_signals)


#     # reindexing
#     allowed_indices_taken = train_ind[indices_taken]

#     signalsI = dataX[allowed_indices_taken][:, :, 0]
#     signalsQ = dataX[allowed_indices_taken][:, :, 1]

#     np.savetxt(os.path.join(output_path, output_signals_fileI), signalsI,
#                delimiter=',')
#     np.savetxt(os.path.join(output_path, output_signals_fileQ), signalsQ,
#                delimiter=',')

#     np.savetxt(os.path.join(output_path, output_modulations_file),
#                modulations[indices_taken],
#                delimiter=',',
#                fmt='%d')
#     np.savetxt(os.path.join(output_path, output_snr_file),
#                snrs[indices_taken],
#                delimiter=',')


# TODO: This function is probablly deprecated too, and should be removed
# def get_np_arrays_from_files(signal_samples_fileI=signal_samples_fileI_name,
#                              signal_samples_fileQ=signal_samples_fileQ_name,
#                              id_file=modulations_id_file_name,
#                              snr_file=snrs_file_name,
#                              data_path='../numpy_data'):


#     samples_pathI = os.path.join(data_path, signal_samples_fileI)
#     samples_pathQ = os.path.join(data_path, signal_samples_fileQ)
#     snr_path = os.path.join(data_path, snr_file)
#     id_path = os.path.join(data_path, id_file)

#     modulation_id = np.genfromtxt(id_path, delimiter=',')
#     signal_sampleI = np.genfromtxt(samples_pathI, delimiter=',')
#     signal_sampleQ = np.genfromtxt(samples_pathQ, delimiter=',')
#     snrs = np.genfromtxt(snr_path, delimiter=',')
#     return modulation_id, signal_sampleI, signal_sampleQ, snrs


# TODO: This function should probably be removed
# def get_signal_df_from_numpy(max_sample_len=None):

#     modulation_id, signal_sampleI, signal_sampleQ, snr = get_np_arrays_from_files()
#     np.set_printoptions(threshold=sys.maxsize)
#     df = pd.DataFrame({
#             'modulation_id': modulation_id,
#             'modulation_type': [index_to_class[idx] for idx in modulation_id],
#             'SNR': snr,
#             'signalI': [s[:max_sample_len] for s in signal_sampleI],
#             'signalQ': [s[:max_sample_len] for s in signal_sampleQ]})
#     return df

# TODO: This function probably should be removed too
# def read_data_from_h5py(input_file=hdf5_file):
#     # , indices=None):

#     data = h5py.File(input_file, 'r')

#     # data['Y'] contains indicators of modulations as 0-1 vectors (one-hot)
#     # the line below converts it to a numerical vector

#     modulation_one_hot = np.apply_along_axis(lambda x: x.argmax(),
#                                                  1,
#                                                  data['Y'][:, :])

#     # dataZ[:] has its second dimension of length one
#     # thus it is in fact one-dimenional; therefore .flatten is needed
#     SNR = data['Z'][:].flatten()

#     point_cloud = data['X']
#     # if indices is not None:
#     #     # dataX = dataX[indices]
#     #     snrs = snrs[indices]
#     #     modulations = modulations[indices]
#     return point_cloud, modulation_one_hot, SNR

# TODO: Another function to be removed
# def __create_index_list(modulations, snrs, snr_max, snr_min, random, number_of_signals):

#     cases = [(mod, snr) for mod in np.unique(modulations)
#              for snr in np.unique(snrs)
#              if (snr <= snr_max and snr >= snr_min)]

#     indices_taken = []
#     for mod, snr in cases:
#         indices = np.logical_and(modulations == mod,
#                                  snrs == snr)
#         indices = indices.nonzero()[0]
#         indices_taken.extend(__select_indices(indices, random, number_of_signals))

#     return sorted(indices_taken)

# TODO: This function should probably be removed
# def __select_indices(indices, random, number_of_signals):

#     if random:
#         return rnd.sample(list(indices), number_of_signals)
#     else:
#         take_these = list(range(number_of_signals))
#         return list(indices[take_these])


def sample_indices(indices_file, number, seed=42):

     indices = np.loadtxt(indices_file, delimiter=',', dtype=int)

     np.random.seed(seed)
     return np.random.choice(indices, size=number)


# TODO: Diagrams should not require signalI, etc. Correct this.
def compute_diagrams(point_cloud): 

    df = pd.DataFrame({'point_cloud': [point_cloud]})
    df['signalI'] = df.point_cloud.map(lambda x: x[:, 0])
    df['signalQ'] = df.point_cloud.map(lambda x: x[:, 1])

    feat = [df.ff.create_feature('diagram', **params) for
            params in to_compute]

    diagrams = map(lambda x: x.values().iloc[0], feat)

    return {name: diags for name, diags in
            zip(map(str, feat), diagrams)}

    # here we should write into HDF5


def add_row_to_hdf5(hdf5_file, cloud, modulation, snr,
                    index, diagrams, row):

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
                            snr = SNR[index].flatten(),
                            index=index,
                            diagrams=diagrams,
                            row=row)


def create_index_df(hdf5_file=hdf5_file,
                    indices_file=train_indices_file):
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

    return df.loc[df.SNR.apply(condition_snr) & 
                  df.modulation_id.apply(condition_mod_id)]


def random_subsample(df, size, seed=42):

     np.random.seed(seed)
     grouped = df.groupby(['modulation_id', 'SNR'], as_index=False)

     df = grouped.apply(lambda x: x.loc[np.random.choice(x.index,
                                                         size,
                                                         False)])
     return df.sort_index()


def create_pickle(indices, output_file, hdf5_file=hdf5_file):
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

    #TODO: Erase signalI and signalQ properties!
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

    df = create_index_df(hdf5_file=hdf5_file, indices_file=train_indices_file)
    df = filter_df(df, condition_snr=condition_snr,
                   condition_mod_id=condition_mod_id)
    df = random_subsample(df, size=size, seed=seed)
    
    create_pickle(indices=df['index'],
                  output_file=output_file,
                  hdf5_file=hdf5_file)

    
# for RML2016 data
def read_df_from_dictionary(file_path=dictionary_pickle_file_path):

    with open(file_path, 'rb') as f:
        X_dictionary = pickle.load(f, encoding='latin1')
    rows = []
    for key, val in X_dictionary.items():
        mod_type, snr = key
        mod_id = class_to_id_2016[mod_type]
        new_rows = [(
            mod_id,
            mod_type,
            snr,
            np.array(v[0]),
            np.array(v[1]))
            for v in val
            ]
        rows.extend(new_rows)
    df = pd.DataFrame(rows, columns=['modulation_id',
                                     'modulation_type',
                                     'SNR',
                                     'signal_sample',
                                     'imagine_part'])
    return df

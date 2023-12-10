import pandas as pd
import numpy as np
from collections import namedtuple

Feature = namedtuple('Feature', 'name params')

# to_add = [Feature('diagram', {'dim': 10, 'step': 1, 'eps': 0}),
#           Feature('diagram', {'dim': 20, 'step': 1, 'eps': 0}),
#           Feature('diagram', {'dim': 2, 'step': 1, 'eps': 0}),
#           Feature('diagram', {'dim': 2, 'step': 1, 'eps': 0, 'kind': 'abs'}),
#           Feature('diagram', {'dim': 10, 'step': 1, 'eps': 0, 'kind': 'abs'}),
#           Feature('diagram', {'dim': 2, 'step': 1, 'eps': 0, 'kind': 'phi'}),
#           Feature('diagram', {'dim': 10, 'step': 1, 'eps': 0, 'kind': 'phi'}),
#           Feature('diagram', {'dim': 2, 'step': 1, 'eps': 0, 'kind': 'abs', 'fil': 'star'}),
#           Feature('diagram', {'dim': 2, 'step': 1, 'eps': 0, 'kind': 'phi', 'fil': 'star'}),
#           Feature('diagram', {'dim': 2, 'step': 30, 'eps': 0}),
#           Feature('diagram', {'dim': 4, 'step': 30, 'eps': 0})]


# to_remove = ['H0', 'H1', 'H0_life_time', 'H1_life_time',
#              'no_H0', 'no_H1', 'H0_mean', 'H1_mean', 'H0_var', 'H1_var',
#              'point_cloud_dim=4', 'H0_4D', 'H1_4D',
#              'H0_life_time_4D', 'H1_life_time_4D', 'no_H0_4D', 'no_H1_4D',
#              'H0_mean_4D', 'H1_mean_4D', 'H0_var_4D', 'H1_var_4D',
#              'point_cloud_dim=3', 'H1_3D', 'H1_life_time_3D',
#              'H1_mean_norm', 'H0_mean_norm', 'H0_3D', 'H0_life_time_3D', 'H0_var_3D',
#              'H1_mean_3D', 'no_H0_3D', 'H1_var_3D', 'no_H1_3D', 'H0_mean_3D',
#              'point_cloud_sr', 'diagram_sr', 'H0_sr',
#              'H0_life_time_sr', 'H1_sr', 'no_H1_sr', 'H1_life_time_sr', 'H1_mean_sr',
#              'H0_var_sr', 'no_H0_sr', 'H1_var_sr', 'H0_mean_sr',
#              'point_cloud_step=symbol_rate', 'point_cloud_dim=3_step=symbol_rate', 
#              'point_cloud_dim=4_step=symbol_rate', 'point_cloud_dim=10',
#              'point_cloud_dim=20', 'diagram', 'diagram_dim=4', 'diagram_dim=3',
#              'diagram_dim=2', 'diagram_step=symbol_rate', 'diagram_dim=3_step=symbol_rate',
#              'diagram_dim=4_step=symbol_rate']
# to_add = [Feature('diagram', {'dim': 2, 'preproc': 'fft'}),
#           Feature('diagram', {'dim': 4, 'preproc': 'fft'}),
#           Feature('diagram', {'dim': 10, 'preproc': 'fft'})]


to_add = [Feature('diagram', {'dim': 2})]
to_remove = ['signalI', 'signalQ']

to_filter = ['modulation_id', 'modulation_type', 'SNR', 'point_cloud', 'symbol_rate']

input_file = 'data/stats_train_plain.pkl'
output_file = 'data/stats_train_plain_fft.pkl'

def add_features(df, to_add):
    for f in to_add:
        df.ff.create_feature(f.name, **f.params)


def clear(input_file, output_file, to_keep=['point_cloud', 'symbol_rate',
                                            'SNR', 'modulation_id', 'modulation_type',
                                            'signalI', 'signalQ']):

    df = pd.read_pickle(input_file)
    df.drop(set(df.columns) - set(to_keep), axis=1, inplace=True)
    df['signalI'] = df['point_cloud'].apply(lambda x: x[:, 0])
    df['signalQ'] = df['point_cloud'].apply(lambda x: x[:, 1])
    df.to_pickle(output_file)


def add_and_drop(input_file=input_file,
                 output_file=output_file,
                 to_remove=to_remove,
                 to_add=to_add):

    df = pd.read_pickle(input_file)

    df.drop(set(to_remove) & set(df.columns), axis=1, inplace=True)
    add_features(df, to_add)

    df.to_pickle(output_file)


def create_test_file(input_file=input_file,
                     output_file='data/stats_train_test.pkl'):

    df = pd.read_pickle(input_file)
    df = df.iloc[0:3]

    df.to_pickle(output_file)

def drop_samples(n, input_file=input_file,
                 output_file=output_file):


    df = pd.read_pickle(input_file)
    df['point_cloud'] = df['point_cloud'].apply(lambda x: x[1:n, :])
    df.to_pickle(output_file)

def normalise_point_clouds(input_file=input_file,
                           output_file=output_file,
                           normaliser=np.max):

    def normalise(point_cloud):
        amplitude = np.apply_along_axis(lambda x:
                                        np.linalg.norm(x), 1, point_cloud)
        return point_cloud / normaliser(amplitude)

    df = pd.read_pickle(input_file)
    df['point_cloud'] = df['point_cloud'].apply(normalise)
    df.to_pickle(output_file)
    

def filter_features(to_filter=to_filter,
                    input_file=input_file,
                    output_file=output_file):

    df = pd.read_pickle(input_file)
    df = df[to_filter]
    df.to_pickle(output_file)


if __name__ == "__main__":

    # normalise_point_clouds()

    # filter_features(input_file='data/stats_train_plain_max.pkl',
    #                 output_file='data/stats_train_plain_max.pkl')

    # signal_reader.select_train_pkl(80,
    #                                output_file='data/stats_size_1040.pkl',
    #                                seed=42)

    add_and_drop(input_file='data/stats_size_10010.pkl',
                 output_file='data/stats_size_10010.pkl',
                 to_remove=to_remove,
                 to_add=to_add)
    


    # filenames = ['data/stats_test_origin.pkl']
    # for name in filenames:
    #     df = pd.read_pickle(name)
    #     print('For file: ', name)
    #     print(df.columns)

    # add_and_drop()



    # signal_reader.select_train_pkl(770,
    #                                output_file='data/stats_size_10010.pkl',
    #                                seed=42)


    # names = ['data/stats_train080.pkl',
    #          'data/stats_train100.pkl',
    #          'data/stats_train200.pkl',
    #          'data/stats_train500.pkl',
    #          'data/stats_train800.pkl']


    # # for name in names:
    # #     print('Computation for: ', name)
    # #     clear(input_file=name, output_file=name)



    # for name in names:
    #     print('Computation for: ', name)

    #     add_and_drop(input_file=name,
    #                  output_file=name)





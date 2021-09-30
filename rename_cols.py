import pandas as pd

files = ['data/temp.pkl', 'data/stats_test_old.pkl',
         'data/stats_test.pkl', 'data/stats_train_old.pkl',
         'data/stats_train.pkl', 'data/temp.pkl', 'data/testpickle.pkl']

new_names = {'cloud_3D': 'point_cloud_dim=3',
             'cloud_4D': 'point_cloud_dim=4',
             'diagram_4D': 'diagram_dim=4',
             'diagram_3D': 'diagram_dim=3',
             'signal_sample': 'signalI',
             'signal_sampleQ': 'signalQ'}

for file in files:
    df = pd.read_pickle(file)
    df.rename(new_names, axis='columns', inplace=True)
    df.to_pickle(file)

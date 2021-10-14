"""This script reads signals from SigDataset and writes them into .csv files
WARNING: function select_sample_from_train might be deprecated"""
from modurec.signal_reader import select_sample_from_train


seed = 42

n = 40

select_sample_from_train(number_of_signals=n,
                         output_signals_fileI='signalsI.csv',
                         output_signals_fileQ='signalsQ.csv',
                         input_path='/home/inf/jprzew/TDA/SigDatasets/2018.01',
                         snr_min=5, random=True, seed=seed)

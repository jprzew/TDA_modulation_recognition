from venv.signal_reader import split_and_save_indices_rml18
from src.config import TrainTestSplit, General

split_and_save_indices_rml18(data_file=General.hdf_data_file,
                             test_indices_file=TrainTestSplit.test_indices_file,
                             train_indices_file=TrainTestSplit.train_indices_file)

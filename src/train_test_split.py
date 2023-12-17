from datasets import radioml_dataset
import config as cfg
import numpy as np

# Setting random seed
np.random.seed(cfg.TrainTestSplit.random_seed)

print('Splitting dataset into train/test subsets...')
splitter = radioml_dataset.get_splitter(input_file=cfg.General.hdf_data_file)
splitter.split(test_proportion=cfg.TrainTestSplit.test_proportion)
splitter.save_to_file(train_output_file=cfg.TrainTestSplit.train_indices_file,
                      test_output_file=cfg.TrainTestSplit.test_indices_file)


"""Splits the main dataset into test/train subsets"""

from datasets import radioml
import config as cfg
import numpy as np

# Setting random seed
np.random.seed(cfg.TrainTestSplit.random_seed)


def main():
    print('Splitting dataset into train/test subsets...')
    splitter = radioml.Splitter(input_file=cfg.General.hdf_data_file)
    splitter.split(test_proportion=cfg.TrainTestSplit.test_proportion)
    splitter.save_to_file(train_output_file=cfg.TrainTestSplit.train_indices_file,
                          test_output_file=cfg.TrainTestSplit.test_indices_file)


if __name__ == '__main__':
    main()

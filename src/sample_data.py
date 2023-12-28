from datasets import radioml_dataset
import config as cfg
import numpy as np

# Setting random seed
np.random.seed(cfg.SampleData.random_seed)


def main():

    print('Sampling dataset...')
    sampler = radioml_dataset.get_sampler(input_file=cfg.General.hdf_data_file,
                                          sampled_indices_file=cfg.TrainTestSplit.train_indices_file)
    sampler.sample(cases_per_class=cfg.SampleData.cases_per_class)
    sampler.format_data()
    sampler.save_to_file(output_file=cfg.SampleData.sampled_data_file)


if __name__ == '__main__':
    main()
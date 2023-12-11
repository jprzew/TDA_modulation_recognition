from datasets import radioml_dataset
import config as cfg

print('Sampling dataset...')
sampler = radioml_dataset.get_sampler(input_file=cfg.General.hdf_data_file,
                                      sampled_indices_file=cfg.TrainTestSplit.train_indices_file)
sampler.sample(cases_per_class=cfg.SampledData.cases_per_class)
sampler.format_data()
sampler.save_to_file(output_file=cfg.SampledData.sampled_data_file)

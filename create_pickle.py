from signal_reader import select_train_pkl
from params import CreatePickle
import config as cfg

select_train_pkl(CreatePickle.cases_per_class, output_file=cfg.CreatePickle.output, seed=42)

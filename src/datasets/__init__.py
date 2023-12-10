from abc import ABC, abstractmethod
from pathlib import Path
from .radioml import RadoiMLDatasetFactory

class DatasetFactory(ABC):

    @abstractmethod
    def get_splitter(self, **kwargs) -> 'DatasetSplitter':
        """Returns dataset splitter"""

    @abstractmethod
    def get_sampler(self, **kwargs) -> 'DatasetSampler':
        """Returns dataset sampler"""


class DatasetSplitter(ABC):
    """Splits large dataset into train/test subsets and saves them to file"""
    @abstractmethod
    def split(self, test_proportion: float) -> None:
        """Splits input dataset"""

    @abstractmethod
    def save_to_file(self, train_output_file: Path, test_output_file: Path) -> None:
        """Saves dataset split to file"""


class DatasetSampler(ABC):
    """Samples large dataset, formats it and saves to file"""

    @abstractmethod
    def sample(self, cases_per_class: int) -> None:
        """Samples input dataset"""

    @abstractmethod
    def format_data(self) -> None:
        """Formats sampled data"""

    @abstractmethod
    def save_to_file(self, output_file: Path) -> None:
        """Saves dataset sample to file"""


radioml_dataset = RadoiMLDatasetFactory()

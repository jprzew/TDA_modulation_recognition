# Standard library imports
from pathlib import Path
import os
import itertools

# Third-party imports

# Local imports


def get_repo_path():
    current_path = Path(os.path.abspath('..'))
    this_file_path = current_path / Path(__file__)
    return this_file_path.parent.parent.parent

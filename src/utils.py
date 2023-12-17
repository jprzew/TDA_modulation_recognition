from pathlib import Path
import os


def get_repo_path():
    current_path = Path(os.path.abspath('../..'))
    this_file_path = current_path / Path(__file__)
    return this_file_path.parent.parent

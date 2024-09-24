import os.path
import pathlib
from pathlib import Path

def get_models_path() -> Path:
    path = pathlib.Path(__file__).parent.parent.resolve()
    path = os.path.join(path, 'models')

    return path
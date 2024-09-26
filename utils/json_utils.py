import json
from pathlib import Path

def load_json_file(json_file_path: Path) -> dict:
    with open(json_file_path) as json_file:
        json_data = json.load(json_file)
        return json_data

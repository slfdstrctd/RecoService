import os
from typing import Any
import pickle


def load_model_from_pickle(file_path) -> Any:
    if os.path.isfile(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)
    else:
        print(f'Warning: {file_path} not found.')
        return None

userknn = load_model_from_pickle("saved_models/userknn.pkl")

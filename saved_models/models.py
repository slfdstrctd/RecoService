from typing import Any
import pickle


def load_model_from_pickle(file_path) -> Any:
    with open(file_path, "rb") as f:
        model = pickle.load(f)
    return model


userknn = load_model_from_pickle("saved_models/userknn.pkl")

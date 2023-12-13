import os
import pickle
from types import MethodType
from typing import Any
import pandas as pd


def load_model_from_pickle(file_path) -> Any:
    if os.path.isfile(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)
    else:
        print(f'Warning: {file_path} not found.')
        return None


def load_json(file_path) -> Any:
    if os.path.isfile(file_path):
        return pd.read_json(file_path, orient='index')
    else:
        print(f'Warning: {file_path} not found.')
        return None


popular = load_json('saved_models/popular.json')

userknn = load_model_from_pickle("saved_models/userknn.pkl")
als_ann = load_model_from_pickle("saved_models/als_ann.pkl")
lfm_ann = load_model_from_pickle("saved_models/lfm_ann.pkl")

dssm_offline = load_model_from_pickle("saved_models/dssm_offline.pkl")
ae_offline = load_model_from_pickle("saved_models/ae_offline.pkl")
recvae_offline = load_model_from_pickle("saved_models/recVAE_offline.pkl")


def recommend_ann(self, user_id, N_recs=10, popular=popular):
    if user_id in self.user_id_map.external_ids:
        return self.get_item_list_for_user(user_id, N_recs).tolist()
    else:
        return popular[0].values.tolist()[:10]


def recommend_offline(model_name, user_id, popular=popular):
    file = ""
    if model_name == 'dssm':
        file = dssm_offline
    elif model_name == 'ae':
        file = ae_offline
    elif model_name == 'recvae':
        file = recvae_offline

    if user_id in file:
        return file.get(user_id)
    else:
        return popular[0].values.tolist()[:10]


if als_ann and len(popular):
    als_ann.recommend = MethodType(recommend_ann, als_ann)

if lfm_ann and len(popular):
    lfm_ann.recommend = MethodType(recommend_ann, lfm_ann)

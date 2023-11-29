from typing import Dict, List, Any
from collections import Counter
import rectools.models
import pandas as pd
import numpy as np
import scipy as sp
from implicit.nearest_neighbours import ItemItemRecommender
from rectools.dataset import Dataset
from rectools.models import PopularModel


class UserKnn:
    """Class for fit-predict UserKNN model
       based on ItemKNN model from implicit.nearest_neighbours
    """

    def __init__(self, model: ItemItemRecommender, N_users: int = 50):
        self.N_users = N_users
        self.model = model
        self.is_fitted = False
        self.pop_model = PopularModel()

    def get_mappings(self, train):
        self.users_inv_mapping = dict(enumerate(train['user_id'].unique()))
        self.users_mapping = {v: k for k, v in self.users_inv_mapping.items()}

        self.items_inv_mapping = dict(enumerate(train['item_id'].unique()))
        self.items_mapping = {v: k for k, v in self.items_inv_mapping.items()}

    def get_matrix(self, df: pd.DataFrame,
                   user_col: str = 'user_id',
                   item_col: str = 'item_id',
                   weight_col: str = None,
                   users_mapping: Dict[int, int] = None,
                   items_mapping: Dict[int, int] = None):

        if weight_col:
            weights = df[weight_col].astype(np.float32)
        else:
            weights = np.ones(len(df), dtype=np.float32)

        self.interaction_matrix = sp.sparse.coo_matrix((
            weights,
            (
                df[item_col].map(self.items_mapping.get),
                df[user_col].map(self.users_mapping.get)
            )
        ))

        self.watched = df \
            .groupby(user_col, as_index=False) \
            .agg({item_col: list}) \
            .rename(columns={user_col: 'sim_user_id'})

        return self.interaction_matrix

    def idf(self, n: int, x: float):
        return np.log((1 + n) / (1 + x) + 1)

    def _count_item_idf(self, df: pd.DataFrame):
        item_cnt = Counter(df['item_id'].values)
        item_idf = pd.DataFrame.from_dict(item_cnt, orient='index',
                                          columns=['doc_freq']).reset_index()
        item_idf['idf'] = item_idf['doc_freq'].apply(
            lambda x: self.idf(self.n, x))
        self.item_idf = item_idf

    def fit(self, train: pd.DataFrame):
        self.user_knn = self.model
        self.get_mappings(train)
        self.weights_matrix = self.get_matrix(train,
                                              users_mapping=self.users_mapping,
                                              items_mapping=self.items_mapping)

        self.n = train.shape[0]
        self._count_item_idf(train)

        self.user_knn.fit(self.weights_matrix)

        self.pop_model = rectools.models.PopularModel()
        self.pop_model.fit(Dataset.construct(train))

        self.is_fitted = True

    def _generate_recs_mapper(self, model: ItemItemRecommender,
                              user_mapping: Dict[int, int],
                              user_inv_mapping: Dict[int, int], N: int):
        def _recs_mapper(user):
            user_id = self.users_mapping[user]
            users, sim = model.similar_items(user_id, N=N)
            return [self.users_inv_mapping[user] for user in users], sim

        return _recs_mapper

    def predict_knn(self, test: pd.DataFrame):

        if not self.is_fitted:
            raise ValueError("Please call fit before predict")

        mapper = self._generate_recs_mapper(
            model=self.user_knn,
            user_mapping=self.users_mapping,
            user_inv_mapping=self.users_inv_mapping,
            N=self.N_users
        )

        recs = pd.DataFrame({'user_id': test['user_id'].unique()})
        recs['sim_user_id'], recs['sim'] = zip(*recs['user_id'].map(mapper))
        recs = recs.set_index('user_id').apply(pd.Series.explode).reset_index()

        recs = recs[~(recs['user_id'] == recs['sim_user_id'])] \
            .merge(self.watched, on=['sim_user_id'], how='left') \
            .explode('item_id') \
            .sort_values(['user_id', 'sim'], ascending=False) \
            .drop_duplicates(['user_id', 'item_id'], keep='first') \
            .merge(self.item_idf, left_on='item_id', right_on='index',
                   how='left')

        return recs

    # Mixed recommendation method
    def predict(self, test: pd.DataFrame, N_recs: int = 10) -> pd.DataFrame:
        if not self.pop_model:
            raise Exception(
                "Popular model is not fitted yet. Please fit the popular model first.")

        if not self.is_fitted:
            raise ValueError("Please call fit before predict")

        recs = self.predict_knn(test)

        # Fill knn recommendations with popular if they are less than N_recs
        if len(recs) < N_recs:
            popular_recs = [self.items_inv_mapping[p] for p in
                            self.pop_model.popularity_list[0][:N_recs]]
            popular_recs_df = pd.DataFrame(
                {'user_id': np.repeat(test.user_id.values, len(popular_recs)),
                 'item_id': np.tile(popular_recs, len(test.user_id.values)),
                 'sim': 0.01,
                 'similar_user_id': -1,
                 'score': 0,
                 'idf': 1})
            recs = pd.concat([recs, popular_recs_df], ignore_index=True)
            recs = recs.drop_duplicates(['user_id', 'item_id'], keep='first')
            recs = recs.head(N_recs)

        recs['score'] = recs['sim'] * recs['idf']
        recs = recs.sort_values(['user_id', 'score'], ascending=False)
        recs['rank'] = recs.groupby('user_id').cumcount() + 1

        return recs[recs['rank'] <= N_recs][
            ['user_id', 'item_id', 'score', 'rank']]

    def recommend(self, user_id: int, N_recs: int = 10) -> list[Any]:
        user = pd.DataFrame([{'user_id': user_id}])
        recommendations = self.predict(user, N_recs)
        return list(recommendations.item_id)

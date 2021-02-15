from typing import List

import numpy as np
import pandas as pd
from category_encoders.backward_difference import BackwardDifferenceEncoder
from category_encoders.cat_boost import CatBoostEncoder
from category_encoders.helmert import HelmertEncoder
from category_encoders.james_stein import JamesSteinEncoder
from category_encoders.leave_one_out import LeaveOneOutEncoder
from category_encoders.m_estimate import MEstimateEncoder
from category_encoders.one_hot import OneHotEncoder
from category_encoders.ordinal import OrdinalEncoder
from category_encoders.sum_coding import SumEncoder
from category_encoders.target_encoder import TargetEncoder
from category_encoders.woe import WOEEncoder
from sklearn.model_selection import RepeatedStratifiedKFold


def get_single_encoder(encoder_name: str, cat_cols: list):
    """
    Get encoder by its name
    :param encoder_name: Name of desired encoder
    :param cat_cols: Cat columns for encoding
    :return: Categorical encoder
    """
    if encoder_name == "FrequencyEncoder":
        encoder = FrequencyEncoder(cols=cat_cols)

    if encoder_name == "WOEEncoder":
        encoder = WOEEncoder(cols=cat_cols)

    if encoder_name == "TargetEncoder":
        encoder = TargetEncoder(cols=cat_cols)

    if encoder_name == "SumEncoder":
        encoder = SumEncoder(cols=cat_cols)

    if encoder_name == "MEstimateEncoder":
        encoder = MEstimateEncoder(cols=cat_cols)

    if encoder_name == "LeaveOneOutEncoder":
        encoder = LeaveOneOutEncoder(cols=cat_cols)

    if encoder_name == "HelmertEncoder":
        encoder = HelmertEncoder(cols=cat_cols)

    if encoder_name == "BackwardDifferenceEncoder":
        encoder = BackwardDifferenceEncoder(cols=cat_cols)

    if encoder_name == "JamesSteinEncoder":
        encoder = JamesSteinEncoder(cols=cat_cols)

    if encoder_name == "OrdinalEncoder":
        encoder = OrdinalEncoder(cols=cat_cols)

    if encoder_name == "CatBoostEncoder":
        encoder = CatBoostEncoder(cols=cat_cols)

    if encoder_name == "MEstimateEncoder":
        encoder = MEstimateEncoder(cols=cat_cols)
    if encoder_name == "OneHotEncoder":
        encoder = OneHotEncoder(cols=cat_cols)
    if encoder is None:
        raise NotImplementedError("To be implemented")
    return encoder


class DoubleValidationEncoderNumerical:
    """
    Encoder with validation within
    """

    def __init__(self, cols, encoders_names_tuple=()):
        """
        :param cols: Categorical columns
        :param encoders_names_tuple: Tuple of str with encoders
        """
        self.cols, self.num_cols = cols, None
        self.encoders_names_tuple = encoders_names_tuple

        self.n_folds, self.n_repeats = 5, 3
        self.model_validation = RepeatedStratifiedKFold(
            n_splits=self.n_folds, n_repeats=self.n_repeats, random_state=0
        )
        self.encoders_dict = {}

        self.storage = None

    def fit_transform(self, X: pd.DataFrame, y: np.array) -> pd.DataFrame:
        self.num_cols = [col for col in X.columns if col not in self.cols]
        self.storage = []

        for encoder_name in self.encoders_names_tuple:
            for n_fold, (train_idx, val_idx) in enumerate(
                self.model_validation.split(X, y)
            ):
                encoder = get_single_encoder(encoder_name, self.cols)

                X_train, X_val = (
                    X.loc[train_idx].reset_index(drop=True),
                    X.loc[val_idx].reset_index(drop=True),
                )
                y_train, y_val = y[train_idx], y[val_idx]
                _ = encoder.fit_transform(X_train, y_train)

                # transform validation part and get all necessary cols
                val_t = encoder.transform(X_val)
                val_t = val_t[
                    [col for col in val_t.columns if col not in self.num_cols]
                ].values

                if encoder_name not in self.encoders_dict.keys():
                    cols_representation = np.zeros((X.shape[0], val_t.shape[1]))
                    self.encoders_dict[encoder_name] = [encoder]
                else:
                    self.encoders_dict[encoder_name].append(encoder)

                cols_representation[val_idx, :] += val_t / self.n_repeats

            cols_representation = pd.DataFrame(cols_representation)
            cols_representation.columns = [
                f"encoded_{encoder_name}_{i}"
                for i in range(cols_representation.shape[1])
            ]
            self.storage.append(cols_representation)

        for df in self.storage:
            X = pd.concat([X, df], axis=1)

        X.drop(self.cols, axis=1, inplace=True)
        return X

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.storage = []
        for encoder_name in self.encoders_names_tuple:
            cols_representation = None

            for encoder in self.encoders_dict[encoder_name]:
                test_tr = encoder.transform(X)
                test_tr = test_tr[
                    [col for col in test_tr.columns if col not in self.num_cols]
                ].values

                if cols_representation is None:
                    cols_representation = np.zeros(test_tr.shape)

                cols_representation = (
                    cols_representation + test_tr / self.n_folds / self.n_repeats
                )

            cols_representation = pd.DataFrame(cols_representation)
            cols_representation.columns = [
                f"encoded_{encoder_name}_{i}"
                for i in range(cols_representation.shape[1])
            ]
            self.storage.append(cols_representation)

        for df in self.storage:
            X = pd.concat([X, df], axis=1)

        X.drop(self.cols, axis=1, inplace=True)
        return X


class MultipleEncoder:
    """
    Multiple encoder for categorical columns
    """

    def __init__(self, cols: List[str], encoders_names_tuple=()):
        """
        :param cols: List of categorical columns
        :param encoders_names_tuple: Tuple of categorical encoders names. Possible values in tuple are:
        "FrequencyEncoder", "WOEEncoder", "TargetEncoder", "SumEncoder", "MEstimateEncoder", "LeaveOneOutEncoder",
        "HelmertEncoder", "BackwardDifferenceEncoder", "JamesSteinEncoder", "OrdinalEncoder""CatBoostEncoder"
        """

        self.cols = cols
        self.num_cols = None
        self.encoders_names_tuple = encoders_names_tuple
        self.encoders_dict = {}

        # list for storing results of transformation from each encoder
        self.storage = None

    def fit_transform(self, X: pd.DataFrame, y: np.array) -> pd.DataFrame:
        self.num_cols = [col for col in X.columns if col not in self.cols]
        self.storage = []
        for encoder_name in self.encoders_names_tuple:
            encoder = get_single_encoder(encoder_name=encoder_name, cat_cols=self.cols)

            cols_representation = encoder.fit_transform(X, y)
            self.encoders_dict[encoder_name] = encoder
            cols_representation = cols_representation[
                [col for col in cols_representation.columns if col not in self.num_cols]
            ].values
            cols_representation = pd.DataFrame(cols_representation)
            cols_representation.columns = [
                f"encoded_{encoder_name}_{i}"
                for i in range(cols_representation.shape[1])
            ]
            self.storage.append(cols_representation)

        # concat cat cols representations with initial dataframe
        for df in self.storage:
            X = pd.concat([X, df], axis=1)

        # remove all columns as far as we have their representations
        X.drop(self.cols, axis=1, inplace=True)
        return X

    def transform(self, X) -> pd.DataFrame:
        self.storage = []
        for encoder_name in self.encoders_names_tuple:
            # get representation of cat columns and form a pd.DataFrame for it
            cols_representation = self.encoders_dict[encoder_name].transform(X)
            cols_representation = cols_representation[
                [col for col in cols_representation.columns if col not in self.num_cols]
            ].values
            cols_representation = pd.DataFrame(cols_representation)
            cols_representation.columns = [
                f"encoded_{encoder_name}_{i}"
                for i in range(cols_representation.shape[1])
            ]
            self.storage.append(cols_representation)

        # concat cat cols representations with initial dataframe
        for df in self.storage:
            X = pd.concat([X, df], axis=1)

        # remove all columns as far as we have their representations
        X.drop(self.cols, axis=1, inplace=True)
        return X


class FrequencyEncoder:
    def __init__(self, cols):
        self.cols = cols
        self.counts_dict = None

    def fit(self, X: pd.DataFrame):
        counts_dict = {}
        for col in self.cols:
            values, counts = np.unique(X[col], return_counts=True)
            counts_dict[col] = dict(zip(values, counts))
        self.counts_dict = counts_dict

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        counts_dict_test = {}
        res = []
        for col in self.cols:
            values, counts = np.unique(X[col], return_counts=True)
            counts_dict_test[col] = dict(zip(values, counts))

            # if value is in "train" keys - replace "test" counts with "train" counts
            for k in [
                key
                for key in counts_dict_test[col].keys()
                if key in self.counts_dict[col].keys()
            ]:
                counts_dict_test[col][k] = self.counts_dict[col][k]

            res.append(X[col].map(counts_dict_test[col]).values.reshape(-1, 1))
        res = np.hstack(res)

        X[self.cols] = res
        return X

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        self.fit(X, y)
        X = self.transform(X)
        return X


if __name__ == "__main__":
    df = pd.DataFrame({})
    df["cat_col"] = [1, 2, 3, 1, 2, 3, 1, 1, 1]
    df["target"] = [0, 1, 0, 1, 0, 1, 0, 1, 0]

    #
    temp = df.copy()
    enc = CatBoostEncoder(cols=["cat_col"])
    print(enc.fit_transform(temp, temp["target"]))

    #
    temp = df.copy()
    enc = MultipleEncoder(cols=["cat_col"], encoders_names_tuple=("CatBoostEncoder",))
    print(enc.fit_transform(temp, temp["target"]))

    #
    temp = df.copy()
    enc = DoubleValidationEncoderNumerical(
        cols=["cat_col"], encoders_names_tuple=("CatBoostEncoder",)
    )
    print(enc.fit_transform(temp, temp["target"]))

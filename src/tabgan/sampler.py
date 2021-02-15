# -*- coding: utf-8 -*-
"""
todo write description
"""

import logging
import warnings
from typing import Tuple

import numpy as np
import pandas as pd

from _ctgan.synthesizer import _CTGANSynthesizer
from tabgan.abc_sampler import Sampler, SampleData
from tabgan.adversarial_model import AdversarialModel
from tabgan.utils import setup_logging

warnings.filterwarnings("ignore", category=FutureWarning)

__author__ = "Insaf Ashrapov"
__copyright__ = "Insaf Ashrapov"
__license__ = "Apache 2.0"

__all__ = [
    "SamplerOriginal",
    "SamplerGAN"
]


class OriginalGenerator(SampleData):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def get_object_generator(self) -> Sampler:
        return SamplerOriginal(*self.args, **self.kwargs)


class GANGenerator(SampleData):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def get_object_generator(self) -> Sampler:
        return SamplerGAN(*self.args, **self.kwargs)


class SamplerOriginal(Sampler):
    def __init__(self, gen_x_times, cat_cols=None, bot_filter_quantile=0.001,
                 top_filter_quantile=0.999,
                 is_post_process=True, adversaial_model_params={
                "metrics": "AUC",
                "max_depth": 2,
                "max_bin": 100,
                "n_estimators": 500,
                "learning_rate": 0.02,
                "random_state": 42,
            }, pregeneration_frac=2):
        """
        :param gen_x_times: Factor for which initial dataframe should be increased
        """
        self.gen_x_times = gen_x_times
        self.cat_cols = cat_cols
        self.is_post_process = is_post_process
        self.bot_filter_quantile = bot_filter_quantile
        self.top_filter_quantile = top_filter_quantile
        self.adversarial_model_params = adversaial_model_params
        self.pregeneration_frac = pregeneration_frac

    def preprocess_data(self, train_df, target, test_df, ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if "_temp_target" in train_df.columns:
            raise ValueError("Input train dataframe already have _temp_target, condiser removing it")
        if "test_similarity" in train_df.columns:
            raise ValueError("Input train dataframe already have test_similarity, condiser removing it")
        # todo check data types np -> pd

        return train_df, target, test_df

    def generate_data(self, train_df, target, test_df) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self._validate_data(train_df, target, test_df)
        train_df["_temp_target"] = target
        generated_df = train_df.sample(frac=(1 + self.pregeneration_frac * self.get_generated_shape(train_df)),
                                       replace=True, random_state=42)
        generated_df = generated_df.reset_index(drop=True)
        return generated_df.drop("_temp_target", axis=1), generated_df["_temp_target"]

    def postprocess_data(self, train_df, target, test_df, ):
        if not self.is_post_process:
            return train_df, target
        self._validate_data(train_df, target, test_df)
        train_df["_temp_target"] = target
        for num_col in train_df.columns:
            if (self.cat_cols is None or num_col not in self.cat_cols) \
                    and num_col != "_temp_target":
                min_val = test_df[num_col].quantile(self.bot_filter_quantile)
                max_val = test_df[num_col].quantile(self.top_filter_quantile)
                # todo add check if filtering is too heavy
                train_df = train_df.loc[
                    (train_df[num_col] >= min_val) & (train_df[num_col] <= max_val)
                    ]
        if self.cat_cols is not None:
            for cat_col in self.cat_cols:
                train_df = train_df[train_df[cat_col].isin(test_df[cat_col].unique())]
        return train_df.drop("_temp_target", axis=1), train_df["_temp_target"]

    def adversarial_filtering(self, train_df, target, test_df, ):
        # todo add more init params to AdversarialModel, think about kwargs
        # todo param to give user choose use or not use adversarial_filtering

        ad_model = AdversarialModel(cat_cols=self.cat_cols,
                                    model_params=self.adversarial_model_params)
        self._validate_data(train_df, target, test_df)
        train_df["_temp_target"] = target
        ad_model.adversarial_test(test_df, train_df.drop("_temp_target", axis=1))

        train_df["test_similarity"] = ad_model.trained_model.predict(train_df.drop("_temp_target", axis=1),
                                                                     return_shape=False)
        train_df.sort_values("test_similarity", ascending=False, inplace=True)

        train_df = train_df.head(self.get_generated_shape(train_df) * train_df.shape[0])
        return train_df.drop(["test_similarity", "_temp_target"], axis=1), train_df["_temp_target"]

    def _validate_data(self, train_df, target, test_df):
        if train_df.shape[0] < 10 or test_df.shape[0] < 10:
            raise ValueError("Shape of train is {} and test is {} should at least 10! "
                             "Consider disabling adversarial training".
                             format(train_df.shape[0], test_df.shape[0]))
        if train_df.shape[0] != target.shape[0]:
            raise ValueError("Something gone wrong: shape of train_df = {} is not equal to target = {} shape"
                             .format(train_df.shape[0], target.shape[0]))


class SamplerGAN(SamplerOriginal):
    def __init__(self, gen_x_times, cat_cols=None, bot_filter_quantile=0.001,
                 top_filter_quantile=0.999,
                 is_post_process=True, adversaial_model_params={
                "metrics": "AUC",
                "max_depth": 2,
                "max_bin": 100,
                "n_estimators": 500,
                "learning_rate": 0.02,
                "random_state": 42,
            }, epochs=500, pregeneration_frac=2):
        """
        :param gen_x_times: Factor for which initial dataframe should be increased
        """
        self.gen_x_times = gen_x_times
        self.cat_cols = cat_cols
        self.is_post_process = is_post_process
        self.bot_filter_quantile = bot_filter_quantile
        self.top_filter_quantile = top_filter_quantile
        self.adversarial_model_params = adversaial_model_params
        self.epochs = epochs
        self.pregeneration_frac = pregeneration_frac

    def generate_data(self, train_df, target, test_df) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self._validate_data(train_df, target, test_df)
        train_df["_temp_target"] = target
        ctgan = _CTGANSynthesizer()
        if self.cat_cols is None:
            ctgan.fit(train_df, [], epochs=self.epochs)
        else:
            ctgan.fit(train_df, self.cat_cols, epochs=self.epochs)
        generated_df = ctgan.sample(self.pregeneration_frac * self.get_generated_shape(train_df))
        data_dtype = train_df.dtypes.values

        for i in range(len(generated_df.columns)):
            generated_df[generated_df.columns[i]] = generated_df[generated_df.columns[i]
            ].astype(data_dtype[i])

        train_df = pd.concat([train_df, generated_df, ]).reset_index(drop=True)
        return train_df.drop("_temp_target", axis=1), train_df["_temp_target"]


def _sampler(creator: SampleData, in_train, in_target, in_test) -> None:
    # todo think about how user will be using library
    _logger = logging.getLogger(__name__)
    _logger.info("Starting generating data:")
    _logger.info(creator.generate_data_pipe(in_train, in_target, in_test))
    _logger.info("Finished generatation\n")


if __name__ == "__main__":
    setup_logging(logging.DEBUG)
    train = pd.DataFrame(np.random.randint(-10, 150, size=(50, 4)), columns=list('ABCD'))
    target = pd.DataFrame(np.random.randint(0, 2, size=(50, 1)), columns=list('Y'))
    test = pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), columns=list('ABCD'))

    _sampler(OriginalGenerator(gen_x_times=15), train, target, test, )
    _sampler(GANGenerator(gen_x_times=10), train, target, test, )

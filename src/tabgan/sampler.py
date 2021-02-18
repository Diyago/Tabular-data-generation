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
    "OriginalGenerator",
    "GANGenerator"
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
    def __init__(self, gen_x_times: float = 1.1, cat_cols: list = None, bot_filter_quantile: float = 0.001,
                 top_filter_quantile: float = 0.999,
                 is_post_process: bool = True, adversaial_model_params: dict = {
                "metrics": "AUC",
                "max_depth": 2,
                "max_bin": 100,
                "n_estimators": 500,
                "learning_rate": 0.02,
                "random_state": 42,
            }, pregeneration_frac: float = 2, epochs: int = 500):
        """

        @param gen_x_times: float = 1.1 - how much data to generate, output might be less because of postprocessing and
        adversarial filtering
        @param cat_cols: list = None - categorical columns
        @param bot_filter_quantile: float = 0.001 - bottom quantile for postprocess filtering
        @param top_filter_quantile: float = 0.999 - bottom quantile for postprocess filtering
        @param is_post_process: bool = True - perform or not postfiltering, if false bot_filter_quantile
         and top_filter_quantile ignored
        @param adversaial_model_params: dict params for adversarial filtering model, default values for binary task
        @param pregeneration_frac: float = 2 - for generataion step gen_x_times * pregeneration_frac amount of data
        will generated. However in postprocessing (1 + gen_x_times) % of original data will be returned
        @param epochs: int = 500 - for how many epochs train GAN samplers, ignored for OriginalGenerator
        """
        self.gen_x_times = gen_x_times
        self.cat_cols = cat_cols
        self.is_post_process = is_post_process
        self.bot_filter_quantile = bot_filter_quantile
        self.top_filter_quantile = top_filter_quantile
        self.adversarial_model_params = adversaial_model_params
        self.pregeneration_frac = pregeneration_frac
        self.epochs = epochs

    def preprocess_data(self, train_df, target, test_df, ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if type(train_df) != pd.DataFrame or type(target) != pd.DataFrame or type(test_df) != pd.DataFrame:
            raise ValueError("Input dataframes aren't pandas dataframes: train is {}; target is ; test is {}".
                             format(type(train_df, type(target), type(type(test_df)))))
        self.TEMP_TARGET = target.columns[0]
        if self.TEMP_TARGET in train_df.columns:
            raise ValueError("Input train dataframe already have {}, condiser removing it".format(self.TEMP_TARGET))
        if "test_similarity" in train_df.columns:
            raise ValueError("Input train dataframe already have test_similarity, condiser removing it")

        return train_df, target, test_df

    def generate_data(self, train_df, target, test_df) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self._validate_data(train_df, target, test_df)
        train_df[self.TEMP_TARGET] = target
        generated_df = train_df.sample(frac=(1 + self.pregeneration_frac * self.get_generated_shape(train_df)),
                                       replace=True, random_state=42)
        generated_df = generated_df.reset_index(drop=True)
        return generated_df.drop(self.TEMP_TARGET, axis=1), generated_df[self.TEMP_TARGET]

    def postprocess_data(self, train_df, target, test_df, ):
        if not self.is_post_process:
            return train_df, target
        self._validate_data(train_df, target, test_df)
        train_df[self.TEMP_TARGET] = target
        for num_col in train_df.columns:
            if (self.cat_cols is None or num_col not in self.cat_cols) \
                    and num_col != self.TEMP_TARGET:
                min_val = test_df[num_col].quantile(self.bot_filter_quantile)
                max_val = test_df[num_col].quantile(self.top_filter_quantile)
                # to avoid to heavy filtering
                filtered_df = train_df.loc[
                    (train_df[num_col] >= min_val) & (train_df[num_col] <= max_val)]
                if train_df.shape[0] * 0.8 < filtered_df.shape[0]:
                    train_df = filtered_df

        if self.cat_cols is not None:
            for cat_col in self.cat_cols:
                # to avoid to heavy filtering
                filtered_df = train_df[train_df[cat_col].isin(test_df[cat_col].unique())]
                if train_df.shape[0] * 0.8 < filtered_df.shape[0]:
                    train_df = filtered_df
        return train_df.drop(self.TEMP_TARGET, axis=1).reset_index(drop=True), train_df[self.TEMP_TARGET].reset_index(
            drop=True)

    def adversarial_filtering(self, train_df, target, test_df, ):
        ad_model = AdversarialModel(cat_cols=self.cat_cols,
                                    model_params=self.adversarial_model_params)
        self._validate_data(train_df, target, test_df)
        train_df[self.TEMP_TARGET] = target
        ad_model.adversarial_test(test_df, train_df.drop(self.TEMP_TARGET, axis=1))

        train_df["test_similarity"] = ad_model.trained_model.predict(train_df.drop(self.TEMP_TARGET, axis=1))
        train_df.sort_values("test_similarity", ascending=False, inplace=True)
        train_df = train_df.head(self.get_generated_shape(train_df) * train_df.shape[0])
        return train_df.drop(["test_similarity", self.TEMP_TARGET], axis=1).reset_index(drop=True), \
               train_df[self.TEMP_TARGET].reset_index(drop=True)

    def _validate_data(self, train_df, target, test_df):
        if train_df.shape[0] < 10 or test_df.shape[0] < 10:
            raise ValueError("Shape of train is {} and test is {} should at least 10! "
                             "Consider disabling adversarial training".
                             format(train_df.shape[0], test_df.shape[0]))
        if train_df.shape[0] != target.shape[0]:
            raise ValueError("Something gone wrong: shape of train_df = {} is not equal to target = {} shape"
                             .format(train_df.shape[0], target.shape[0]))


class SamplerGAN(SamplerOriginal):
    def generate_data(self, train_df, target, test_df) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self._validate_data(train_df, target, test_df)
        train_df[self.TEMP_TARGET] = target
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
        return train_df.drop(self.TEMP_TARGET, axis=1), train_df[self.TEMP_TARGET]


def _sampler(creator: SampleData, in_train, in_target, in_test) -> None:
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

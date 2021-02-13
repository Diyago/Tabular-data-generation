# -*- coding: utf-8 -*-
"""
todo write description

"""

from __future__ import annotations

import gc
from typing import Tuple
import logging

from abc import ABC, abstractmethod
import pandas as pd
from utils import setup_logging
from adversarial_model import AdversarialModel

__author__ = "Insaf Ashrapov"
__copyright__ = "Insaf Ashrapov"
__license__ = "Apache 2.0"

__all__ = [
    "SamplerOriginal",
    "SamplerGAN"
]


class SampleData(ABC):
    """
        Factory method for different sampler strategies. The goal is to generate more train data
        which should be more close to test, in other word we trying to fix uneven distribution.
    """

    @abstractmethod
    def get_object_generator(self):
        """
        Getter for object sampler aka generator, which is not a generator
        """
        raise NotImplementedError

    def generate_data(self, train_df, target, test_df) -> pd.DataFrame:
        """
        Defines logic for sampling
        """
        generator = self.get_object_generator()

        train_df, test_df = generator.preprocess_data(train_df, test_df)
        new_train = generator(train_df, test_df)
        new_train = generator.postprocess_data(new_train)
        new_train, new_target = generator. \
            adversarial_filtering(new_train, target, test_df)
        return new_train, new_target


class SamplerOriginalGenerator(SampleData):
    def get_object_generator(self) -> Sampler:
        return SamplerOriginal()


class SamplerGANGenerator(SampleData):
    def get_object_generator(self) -> Sampler:
        return SamplerGAN()


class Sampler(ABC):
    """
        Interface for each sampling strategy
    """

    def get_generated_shape(self, input_df):
        """
        Calcs final output shape
        """
        if self.gen_x_times <= 0:
            raise ValueError("Passed gen_x_times = {} should be bigger than 0".format(self.gen_x_times))
        return int(self.gen_x_times * input_df.shape[0] / input_df.shape[0])

    @abstractmethod
    def preprocess_data(self, train_df, test_df, ):
        """Before we can start data generation we might need some preprosing, numpy to pandas
        and etc"""
        raise NotImplementedError

    @abstractmethod
    def generate_data(self, train_df, target, test_df):
        raise NotImplementedError

    @abstractmethod
    def postprocess_data(self, train_df, test_df, ):
        """Filtering data which far beyond from test_df data distribution"""
        raise NotImplementedError

    def adversarial_filtering(self, train_df, test_df, ):
        raise NotImplementedError


class SamplerOriginal(Sampler):
    def __init__(self, gen_x_times, cat_cols=None, bot_filter_quantile=0.001, top_filter_quantile=0.999,
                 is_post_process=True, adversaial_model_params={
                "metrics": "AUC",
                "max_depth": 2,
                "max_bin": 100,
                "n_estimators": 500,
                "learning_rate": 0.02,
                "random_state": 42,
            }):
        """
        :param gen_x_times: Factor for which initial dataframe should be increased
        """
        self.gen_x_times = gen_x_times
        self.cat_cols = cat_cols
        self.is_post_process = is_post_process
        self.bot_filter_quantile = bot_filter_quantile
        self.top_filter_quantile = top_filter_quantile
        self.adversaial_model_params = adversaial_model_params

    def preprocess_data(self, train_df, test_df, ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return train_df, test_df

    def generate_data(self, train_df, target, test_df) -> Tuple[pd.DataFrame, pd.DataFrame]:
        generated_df = train_df.sample(frac=self.get_generated_shape(train_df),
                                       replace=True, random_state=42)
        generated_df = generated_df.reset_index(drop=True)
        train_df = pd.concat([train_df, generated_df], axis=0).reset_index(drop=True)
        del generated_df
        gc.collect()
        return train_df

    def postprocess_data(self, train_df, test_df, ):
        if not self.is_post_process:
            return train_df

        for num_col in train_df.columns:
            if self.cat_cols is None or num_col not in self.cat_cols:
                min_val = test_df[num_col].quantile(self.bot_filter_quantile)
                max_val = test_df[num_col].quantile(self.top_filter_quantile)
                # todo add check if filtering is too heavy
                generated_df = generated_df.loc[
                    (generated_df[num_col] >= min_val) & (generated_df[num_col] <= max_val)
                    ]
        if self.cat_cols is not None:
            for cat_col in self.cat_cols:
                train_df = train_df[train_df[cat_col].isin(test_df[cat_col].unique())]

        return train_df.reset_index(drop=True)

    def adversarial_filtering(self, train_df, target, test_df, ):
        # todo add more init params to AdversarialModel, think about kwargs
        ad_model = AdversarialModel(cat_cols=self.cat_cols,
                                    model_params=self.model_params)
        ad_model = ad_model.adversarial_test(test_df, train_df)

        train_df["test_similarity"] = ad_model.trained_model.predict(train_df, return_shape=False)
        train_df.sort_values("test_similarity", ascending=False, inplace=True)

        return train_df, target[train_df.index]


class SamplerGAN(Sampler):
    def preprocess_data(self, train_df, test_df, ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return train_df, test_df

    def generate_data(self, train_df, target, test_df):
        return train_df

    def adversarial_filtering(self, train_df, target, test_df, ):
        return train_df, target

    # generated_df = train_df.head(int(gen_x_times * x_train.shape[0]))


def client_code(creator: SampleData, in_train, in_target, in_test) -> None:
    # todo think about how user will be using soft
    _logger = logging.getLogger(__name__)
    _logger.info(f"Generated data.")
    _logger.info(creator.generate_data(in_train, in_target, in_test))
    _logger.info("\n")


if __name__ == "__main__":
    setup_logging(logging.DEBUG)
    train, test = pd.DataFrame([[1, 2], [3, 4]]), pd.DataFrame([[1, 2], [7, 10]])
    target = pd.DataFrame([234234, 23])
    client_code(SamplerOriginal(gen_x_times=15), train, target, test, )

    # _logger.debug("App: Launched SamplerGAN")
    # client_code(SamplerGAN(gen_x_times=1.5), train, test)
    #

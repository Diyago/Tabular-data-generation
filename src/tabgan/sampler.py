# -*- coding: utf-8 -*-
"""
todo write description

Based on factory method from https://refactoring.guru/ru/design-patterns/factory-method/python/example
"""

from __future__ import annotations

import logging
import sys
import gc

__author__ = "Insaf Ashrapov"
__copyright__ = "Insaf Ashrapov"
__license__ = "Apache 2.0"

from typing import Tuple

_logger = logging.getLogger(__name__)

from abc import ABC, abstractmethod
import pandas as pd


class SampleData(ABC):
    """
        todo write desc
    """

    @abstractmethod
    def get_object_generator(self):
        raise NotImplementedError

    def generate_data(self, train_df, test_df) -> pd.DataFrame:
        generator = self.get_object_generator()

        train_df, test_df = generator.preprocess_data(train_df, test_df)
        new_train = generator(train_df, test_df)

        return new_train


class SamplerOriginalGenerator(SampleData):
    def get_object_generator(self) -> Sampler:
        return SamplerOriginal()


class SamplerGANGenerator(SampleData):
    def get_object_generator(self) -> Sampler:
        return SamplerGAN()


class SamplerAdversarialGenerator(SampleData):
    def get_object_generator(self) -> Sampler:
        return SamplerAdversarial()


class Sampler(ABC):
    """
        Interface
    """

    def get_generated_shape(self, input_df):
        """
        Calcs finall output shape
        """
        if self.gen_x_times <= 0:
            raise ValueError("Passed gen_x_times = {} should be bigger than 0".format(self.gen_x_times))
        return int(self.gen_x_times * input_df.shape[0] / input_df.shape[0])

    @abstractmethod
    def preprocess_data(self, train_df, test_df, ):
        raise NotImplementedError

    @abstractmethod
    def generate_data(self, train_df, test_df):
        raise NotImplementedError


class SamplerOriginal(Sampler):
    def __init__(self, gen_x_times, **kwargs):
        """
        :param gen_x_times: Factor for which initial dataframe should be increased
        """
        self.args = kwargs
        self.gen_x_times = gen_x_times

    def preprocess_data(self, train_df, test_df, ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return train_df, test_df

    def generate_data(self, train_df, test_df) -> Tuple[pd.DataFrame, pd.DataFrame]:
        x_test_bigger = self.get_generated_shape(train_df)
        generated_df = train_df.sample(frac=x_test_bigger, replace=True, random_state=42)
        generated_df = generated_df.reset_index(drop=True)
        train_df = pd.concat([train_df, generated_df], axis=0).reset_index(drop=True)
        del generated_df
        gc.collect()
        return train_df


class SamplerGAN(Sampler):
    def preprocess_data(self, train_df, test_df, ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return train_df, test_df

    def generate_data(self, train_df, test_df):
        return train_df


class SamplerAdversarial(Sampler):
    def __init__(self, gen_x_times, **kwargs):
        """
        :param gen_x_times: Factor for which initial dataframe should be increased
        """
        self.args = kwargs
        self.gen_x_times = gen_x_times

    def preprocess_data(self, train_df, test_df, ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return train_df, test_df

    def generate_data(self, train_df, test_df) -> Tuple[pd.DataFrame, pd.DataFrame]:
        x_test_bigger = self.get_generated_shape(train_df)
        generated_df = train_df.sample(frac=x_test_bigger, replace=True, random_state=42)
        generated_df = generated_df.reset_index(drop=True)
        train_df = pd.concat([train_df, generated_df], axis=0).reset_index(drop=True)

        #todo adversarial training
        del generated_df
        gc.collect()
        return train_df


def client_code(creator: SampleData, in_train, in_test) -> None:
    _logger.info(f"Generated data.")
    _logger.info(creator.generate_data(in_train, in_test))
    _logger.info("\n")


def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(
        level=loglevel, stream=sys.stdout, format=logformat, datefmt="%Y-%m-%d %H:%M:%S"
    )


if __name__ == "__main__":
    setup_logging(logging.DEBUG)
    train, test = pd.DataFrame([[1, 2], [3, 4]]), pd.DataFrame([[1, 2], [7, 10]])
    _logger.debug("App: Launched SamplerOriginal")
    client_code(SamplerOriginal(gen_x_times=15), train, test)

    # _logger.debug("App: Launched SamplerGAN")
    # client_code(SamplerGAN(gen_x_times=1.5), train, test)
    #
    _logger.debug("App: Launched SamplerGAN")
    client_code(SamplerAdversarial(gen_x_times=0.6), train, test)
    _logger.info("Script ends here")

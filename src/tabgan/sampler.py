# -*- coding: utf-8 -*-
"""
todo write description

Based on factory method from https://refactoring.guru/ru/design-patterns/factory-method/python/example
"""

from __future__ import annotations

import logging
import sys

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

    def __init__(self, **kwargs):
        args = kwargs

    @abstractmethod
    def get_object_generator(self):
        pass

    def generate_data(self, train_df, test_df) -> pd.DataFrame:
        generator = self.get_object_generator()

        train_df, test_df = generator.preprocess_data(train_df, test_df)
        new_train = generator(train_df, test_df)

        return new_train


class SamplerOriginalGenerator(SampleData):
    def get_object_generator(self) -> Sampler:
        return SamplerOriginal()


class Sampler1GANGenerator(SampleData):
    def get_object_generator(self) -> Sampler:
        return SamplerGAN()


class SamplerAdversarialGenerator(SampleData):
    def get_object_generator(self) -> Sampler:
        return SamplerAdversarial()


class Sampler(ABC):
    """
        Interface
    """

    @abstractmethod
    def preprocess_data(self, train_df, test_df, ):
        pass

    @abstractmethod
    def generate_data(self, train_df, test_df):
        pass


class SamplerOriginal(Sampler):
    def preprocess_data(self, train_df, test_df, ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return train_df, test_df

    def generate_data(self, train_df, test_df) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return train_df


class SamplerGAN(Sampler):
    def preprocess_data(self, train_df, test_df, ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return train_df, test_df

    def generate_data(self, train_df, test_df):
        return train_df


class SamplerAdversarial(Sampler):
    def preprocess_data(self, train_df, test_df, ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return train_df, test_df

    def generate_data(self, train_df, test_df):
        return train_df


def client_code(creator: SampleData, in_train, in_test) -> None:
    print(f"Generated data.\n"
          f"{creator.generate_data(in_train, in_test)}", end="")
    print("\n")


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
    client_code(SamplerOriginal(), train, test)

    _logger.debug("App: Launched SamplerGAN")
    client_code(SamplerGAN(), train, test)

    _logger.debug("App: Launched SamplerGAN")
    client_code(SamplerAdversarial(), train, test)
    _logger.info("Script ends here")

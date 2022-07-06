import gc
import logging
from abc import ABC, abstractmethod
from typing import Tuple

import pandas as pd

__author__ = "Insaf Ashrapov"
__copyright__ = "Insaf Ashrapov"
__license__ = "Apache 2.0"


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

    def generate_data_pipe(
        self,
        train_df: pd.DataFrame,
        target: pd.DataFrame,
        test_df: pd.DataFrame,
        deep_copy: bool = True,
        only_adversarial: bool = False,
        use_adversarial: bool = True,
        only_generated_data: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Defines logic for sampling
        @param train_df: Train dataframe which has separate target
        @param target: Input target for the train dataset
        @param test_df: Test dataframe - newly generated train dataframe should be close to it
        @param deep_copy: make copy of input files or not. If not input dataframes will be overridden
        @param only_adversarial: only adversarial fitering to train dataframe will be performed
        @param use_adversarial: perform or not adversarial filtering
        @param only_generated_data: After generation get only newly generated, without concating input train dataframe.
        Only works for SamplerGAN.
        @return: Newly generated train dataframe and test data
        """
        generator = self.get_object_generator()
        if deep_copy:
            logging.info("Preprocessing input data with deep copying input data.")
            if target is None or test_df is None:
                new_train = generator.preprocess_data_df(train_df.copy())
                new_target = None
            else:
                new_train, new_target, test_df = generator.preprocess_data(
                    train_df.copy(), target.copy(), test_df
                )
        else:
            logging.info("Preprocessing input data with deep copying input data.")
            new_train, new_target, test_df = generator.preprocess_data(
                train_df, target, test_df
            )
        if only_adversarial and use_adversarial:
            logging.info("Applying adversarial filtering")
            return generator.adversarial_filtering(new_train, new_target, test_df)
        else:
            logging.info("Starting generation step.")
            new_train, new_target = generator.generate_data(
                new_train, new_target, test_df, only_generated_data
            )
            logging.info("Starting postprocessing step.")
            new_train, new_target = generator.postprocess_data(
                new_train, new_target, test_df
            )
            if use_adversarial:
                logging.info("Applying adversarial filtering")
                new_train, new_target = generator.adversarial_filtering(
                    new_train, new_target, test_df
                )
            gc.collect()

            logging.info("Total finishing, returning data")
            return new_train, new_target


class Sampler(ABC):
    """
        Interface for each sampling strategy
    """

    def get_generated_shape(self, input_df):
        """
        Calculates final output shape
        """
        if self.gen_x_times <= 0:
            raise ValueError(
                "Passed gen_x_times = {} should be bigger than 0".format(
                    self.gen_x_times
                )
            )
        return int(self.gen_x_times * input_df.shape[0])

    @abstractmethod
    def preprocess_data(self, train, target, test_df):
        """Before we can start data generation we might need some preprocessing, numpy to pandas
        and etc"""
        raise NotImplementedError

    @abstractmethod
    def generate_data(self, train_df, target, test_df):
        raise NotImplementedError

    @abstractmethod
    def postprocess_data(self, train_df, target, test_df):
        """Filtering data which far beyond from test_df data distribution"""
        raise NotImplementedError

    @abstractmethod
    def adversarial_filtering(self, train_df, target, test_df):
        raise NotImplementedError

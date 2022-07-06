# -*- coding: utf-8 -*-
"""
todo write description
"""

import gc
import logging
import warnings
from typing import Tuple

import numpy as np
import pandas as pd

from _ctgan.synthesizer import _CTGANSynthesizer as CTGAN
from tabgan.abc_sampler import Sampler, SampleData
from tabgan.adversarial_model import AdversarialModel
from tabgan.utils import setup_logging, get_year_mnth_dt_from_date, collect_dates

warnings.filterwarnings("ignore", category=FutureWarning)

__author__ = "Insaf Ashrapov"
__copyright__ = "Insaf Ashrapov"
__license__ = "Apache 2.0"

__all__ = ["OriginalGenerator", "GANGenerator"]


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
    def __init__(
            self,
            gen_x_times: float = 1.1,
            cat_cols: list = None,
            bot_filter_quantile: float = 0.001,
            top_filter_quantile: float = 0.999,
            is_post_process: bool = True,
            adversarial_model_params: dict = {
                "metrics": "AUC",
                "max_depth": 2,
                "max_bin": 100,
                "n_estimators": 500,
                "learning_rate": 0.02,
                "random_state": 42,
            },
            pregeneration_frac: float = 2,
            only_generated_data: bool = False,
            gan_params: dict = {'batch_size': 500, 'patience': 25, "epochs": 500, }
    ):
        """

        @param gen_x_times: float = 1.1 - how much data to generate, output might be less because of postprocessing and
        adversarial filtering
        @param cat_cols: list = None - categorical columns
        @param bot_filter_quantile: float = 0.001 - bottom quantile for postprocess filtering
        @param top_filter_quantile: float = 0.999 - bottom quantile for postprocess filtering
        @param is_post_process: bool = True - perform or not postfiltering, if false bot_filter_quantile
         and top_filter_quantile ignored
        @param adversarial_model_params: dict params for adversarial filtering model, default values for binary task
        @param pregeneration_frac: float = 2 - for generation step gen_x_times * pregeneration_frac amount of data
        will generated. However in postprocessing (1 + gen_x_times) % of original data will be returned
        @param only_generated_data: bool = False If True after generation get only newly generated, without
        concating input train dataframe.
        @param gan_params: dict params for GAN training
        Only works for SamplerGAN.
        """
        self.gen_x_times = gen_x_times
        self.cat_cols = cat_cols
        self.is_post_process = is_post_process
        self.bot_filter_quantile = bot_filter_quantile
        self.top_filter_quantile = top_filter_quantile
        self.adversarial_model_params = adversarial_model_params
        self.pregeneration_frac = pregeneration_frac
        self.only_generated_data = only_generated_data
        self.gan_params = gan_params
        self.TEMP_TARGET = "TEMP_TARGET"

    @staticmethod
    def preprocess_data_df(df) -> pd.DataFrame:
        logging.info("Input shape: {}".format(df.shape))
        if isinstance(df, pd.DataFrame) is False:
            raise ValueError(
                "Input dataframe aren't pandas dataframes: df is {}".format(type(df))
            )
        return df

    def preprocess_data(
            self, train, target, test_df
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train = self.preprocess_data_df(train)
        target = self.preprocess_data_df(target)
        test_df = self.preprocess_data_df(test_df)
        self.TEMP_TARGET = target.columns[0]
        if self.TEMP_TARGET in train.columns:
            raise ValueError(
                "Input train dataframe already have {} column, consider removing it".format(
                    self.TEMP_TARGET
                )
            )
        if "test_similarity" in train.columns:
            raise ValueError(
                "Input train dataframe already have test_similarity, consider removing it"
            )

        return train, target, test_df

    def generate_data(
            self, train_df, target, test_df, only_generated_data
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if only_generated_data:
            Warning(
                "For SamplerOriginal setting only_generated_data doesn't change anything, "
                "because generated data sampled from the train!"
            )
        self._validate_data(train_df, target, test_df)
        train_df[self.TEMP_TARGET] = target
        generated_df = train_df.sample(
            frac=(1 + self.pregeneration_frac), replace=True, random_state=42
        )
        generated_df = generated_df.reset_index(drop=True)
        gc.collect()
        logging.info(
            "Generated shape: {} and {}".format(
                generated_df.drop(self.TEMP_TARGET, axis=1).shape,
                generated_df[self.TEMP_TARGET].shape,
            )
        )
        return (
            generated_df.drop(self.TEMP_TARGET, axis=1),
            generated_df[self.TEMP_TARGET],
        )

    def postprocess_data(self, train_df, target, test_df):
        if not self.is_post_process or test_df is None:
            logging.info("Skipping postprocessing")
            return train_df, target

        self._validate_data(train_df, target, test_df)
        train_df[self.TEMP_TARGET] = target

        for num_col in test_df.columns:
            if self.cat_cols is None or num_col not in self.cat_cols:
                min_val = test_df[num_col].quantile(self.bot_filter_quantile)
                max_val = test_df[num_col].quantile(self.top_filter_quantile)
                filtered_df = train_df.loc[
                    (train_df[num_col] >= min_val) & (train_df[num_col] <= max_val)
                    ]
                if filtered_df.shape[0] < 10:
                    raise ValueError(
                        "After post-processing generated data's shape less than 10. For columns {} test "
                        "might be highly skewed. Filter conditions are min_val = {} and max_val = {}.".format(
                            num_col, min_val, max_val
                        )
                    )
                train_df = filtered_df

        if self.cat_cols is not None:
            for cat_col in self.cat_cols:
                filtered_df = train_df[
                    train_df[cat_col].isin(test_df[cat_col].unique())
                ]
                if filtered_df.shape[0] < 10:
                    raise ValueError(
                        "After post-processing generated data's shape less than 10. For columns {} test "
                        "might be highly skewed.".format(num_col)
                    )
                train_df = filtered_df
        gc.collect()
        logging.info(
            "Generated shapes after postprocessing: {} plus target".format(
                train_df.drop(self.TEMP_TARGET, axis=1).shape
            )
        )
        return (
            train_df.drop(self.TEMP_TARGET, axis=1).reset_index(drop=True),
            train_df[self.TEMP_TARGET].reset_index(drop=True),
        )

    def adversarial_filtering(self, train_df, target, test_df):
        if test_df is None:
            logging.info("Skipping adversarial filtering, because test_df is None.")
            return train_df, target
        ad_model = AdversarialModel(
            cat_cols=self.cat_cols, model_params=self.adversarial_model_params
        )
        self._validate_data(train_df, target, test_df)
        train_df[self.TEMP_TARGET] = target
        ad_model.adversarial_test(test_df, train_df.drop(self.TEMP_TARGET, axis=1))

        train_df["test_similarity"] = ad_model.trained_model.predict(
            train_df.drop(self.TEMP_TARGET, axis=1)
        )
        train_df.sort_values("test_similarity", ascending=False, inplace=True)
        train_df = train_df.head(self.get_generated_shape(train_df) * train_df.shape[0])
        del ad_model
        gc.collect()
        return (
            train_df.drop(["test_similarity", self.TEMP_TARGET], axis=1).reset_index(
                drop=True
            ),
            train_df[self.TEMP_TARGET].reset_index(drop=True),
        )

    @staticmethod
    def _validate_data(train_df, target, test_df):
        if test_df is not None:
            if train_df.shape[0] < 10 or test_df.shape[0] < 10:
                raise ValueError(
                    "Shape of train is {} and test is {}. Both should at least 10! "
                    "Consider disabling adversarial filtering".format(
                        train_df.shape[0], test_df.shape[0]
                    )
                )
        if target is not None:
            if train_df.shape[0] != target.shape[0]:
                raise ValueError(
                    "Something gone wrong: shape of train_df = {} is not equal to target = {} shape".format(
                        train_df.shape[0], target.shape[0]
                    )
                )


class SamplerGAN(SamplerOriginal):
    def generate_data(
            self, train_df, target, test_df, only_generated_data: bool
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self._validate_data(train_df, target, test_df)
        if target is not None:
            train_df[self.TEMP_TARGET] = target
        ctgan = CTGAN(batch_size=self.gan_params["batch_size"], patience=self.gan_params["patience"])
        logging.info("training GAN")
        if self.cat_cols is None:
            ctgan.fit(train_df, [], epochs=self.gan_params["epochs"])
        else:
            ctgan.fit(train_df, self.cat_cols, epochs=self.gan_params["epochs"])
        logging.info("Finished training GAN")
        generated_df = ctgan.sample(
            self.pregeneration_frac * self.get_generated_shape(train_df)
        )
        data_dtype = train_df.dtypes.values

        for i in range(len(generated_df.columns)):
            generated_df[generated_df.columns[i]] = generated_df[
                generated_df.columns[i]
            ].astype(data_dtype[i])
        gc.collect()
        if not only_generated_data:
            train_df = pd.concat([train_df, generated_df]).reset_index(drop=True)
            logging.info(
                "Generated shapes: {} plus target".format(
                    _drop_col_if_exist(train_df, self.TEMP_TARGET).shape
                )
            )
            return (
                _drop_col_if_exist(train_df, self.TEMP_TARGET),
                get_columns_if_exists(train_df, self.TEMP_TARGET),
            )
        else:
            logging.info(
                "Generated shapes: {} plus target".format(
                    _drop_col_if_exist(train_df, self.TEMP_TARGET).shape
                )
            )
            return (
                _drop_col_if_exist(generated_df, self.TEMP_TARGET),
                get_columns_if_exists(generated_df, self.TEMP_TARGET),
            )
        gc.collect()

        return (
            _drop_col_if_exist(train_df, self.TEMP_TARGET),
            get_columns_if_exists(train_df, self.TEMP_TARGET),
        )


def _sampler(creator: SampleData, in_train, in_target, in_test) -> None:
    _logger = logging.getLogger(__name__)
    _logger.info("Starting generating data")
    train, test = creator.generate_data_pipe(in_train, in_target, in_test)
    _logger.info(train, test)
    _logger.info("Finished generation\n")
    return train, test


def _drop_col_if_exist(df, col_to_drop) -> pd.DataFrame:
    """
    Drops col_to_drop from input dataframe df if such column exists
    """
    if col_to_drop in df.columns:
        return df.drop(col_to_drop, axis=1)
    else:
        return df


def get_columns_if_exists(df, col) -> pd.DataFrame:
    if col in df.columns:
        return df[col]
    else:
        return None


if __name__ == "__main__":
    setup_logging(logging.DEBUG)
    train_size = 100
    train = pd.DataFrame(
        np.random.randint(-10, 150, size=(train_size, 4)), columns=list("ABCD")
    )
    logging.info(train)
    target = pd.DataFrame(np.random.randint(0, 2, size=(train_size, 1)), columns=list("Y"))
    test = pd.DataFrame(np.random.randint(0, 100, size=(train_size, 4)), columns=list("ABCD"))
    _sampler(OriginalGenerator(gen_x_times=15), train, target, test)
    _sampler(
        GANGenerator(gen_x_times=10, only_generated_data=False,
                     gan_params={"batch_size": 500, "patience": 25, "epochs" : 500,}), train, target, test
    )

    _sampler(OriginalGenerator(gen_x_times=15), train, None, train)
    _sampler(
        GANGenerator(cat_cols=["A"], gen_x_times=20, only_generated_data=True),
        train,
        None,
        train,
    )
    min_date = pd.to_datetime('2019-01-01')
    max_date = pd.to_datetime('2021-12-31')

    d = (max_date - min_date).days + 1

    train['Date'] = min_date + pd.to_timedelta(pd.np.random.randint(d, size=train_size), unit='d')
    train = get_year_mnth_dt_from_date(train, 'Date')

    new_train, new_target = GANGenerator(gen_x_times=1.1, cat_cols=['year'], bot_filter_quantile=0.001,
                                         top_filter_quantile=0.999,
                                         is_post_process=True, pregeneration_frac=2, only_generated_data=False).\
                                         generate_data_pipe(train.drop('Date', axis=1), None,
                                                            train.drop('Date', axis=1)
                                                                        )
    new_train = collect_dates(new_train)
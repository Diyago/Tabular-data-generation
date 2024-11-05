# -*- coding: utf-8 -*-

import logging
import warnings
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from be_great import GReaT

from _ForestDiffusion import ForestDiffusionModel
from _ctgan.synthesizer import _CTGANSynthesizer as CTGAN
from tabgan.abc_sampler import Sampler, SampleData
from tabgan.adversarial_model import AdversarialModel
from tabgan.utils import setup_logging, _drop_col_if_exist, \
    get_columns_if_exists, _sampler, get_year_mnth_dt_from_date, collect_dates

warnings.filterwarnings("ignore")

__author__ = "Insaf Ashrapov"
__copyright__ = "Insaf Ashrapov"
__license__ = "Apache 2.0"

__all__ = ["OriginalGenerator", "GANGenerator", "ForestDiffusionGenerator", "LLMGenerator"]


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


class ForestDiffusionGenerator(SampleData):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def get_object_generator(self) -> Sampler:
        return SamplerDiffusion(*self.args, **self.kwargs)


class LLMGenerator(SampleData):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def get_object_generator(self) -> Sampler:
        return SamplerLLM(*self.args, **self.kwargs)


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
                "n_estimators": 150,
                "learning_rate": 0.02,
                "random_state": 42,
            },
            pregeneration_frac: float = 2,
            only_generated_data: bool = False,
            gen_params: dict = {"batch_size": 45, 'patience': 25, "epochs": 50, "llm": "distilgpt2"},
    ):
        """

        @param gen_x_times: float = 1.1 - how much data to generate, output might be less because of postprocessing and
        adversarial filtering
        @param cat_cols: list = None - categorical columns
        @param bot_filter_quantile: float = 0.001 - bottom quantile for postprocess filtering
        @param top_filter_quantile: float = 0.999 - top quantile for postprocess filtering
        @param is_post_process: bool = True - perform or not postfiltering, if false bot_filter_quantile
         and top_filter_quantile ignored
        @param adversarial_model_params: dict params for adversarial filtering model, default values for binary task
        @param pregeneration_frac: float = 2 - for generation step gen_x_times * pregeneration_frac amount of data
        will be generated. However, in postprocessing (1 + gen_x_times) % of original data will be returned
        @param only_generated_data: bool = False If True after generation get only newly generated, without
        concatenating input train dataframe.
        @param gen_params: dict params for GAN training. Only works for SamplerGAN, ForestDiffusionGenerator,
        LLMGenerator.
        """
        self.gen_x_times = gen_x_times
        self.cat_cols = cat_cols
        self.is_post_process = is_post_process
        self.bot_filter_quantile = bot_filter_quantile
        self.top_filter_quantile = top_filter_quantile
        self.adversarial_model_params = adversarial_model_params
        self.pregeneration_frac = pregeneration_frac
        self.only_generated_data = only_generated_data
        self.gen_params = gen_params
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

    def handle_generated_data(self, train_df, generated_df, only_generated_data):
        generated_df = pd.DataFrame(generated_df)
        generated_df.columns = train_df.columns
        for i in range(len(generated_df.columns)):
            generated_df[generated_df.columns[i]] = generated_df[
                generated_df.columns[i]
            ].astype(train_df.dtypes.values[i])
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
                    _drop_col_if_exist(generated_df, self.TEMP_TARGET).shape
                )
            )
            return (
                _drop_col_if_exist(generated_df, self.TEMP_TARGET),
                get_columns_if_exists(generated_df, self.TEMP_TARGET),
            )


class SamplerGAN(SamplerOriginal):
    def check_params(self):
        if self.gen_params["batch_size"] % 10 != 0:
            logging.warning(
                "Batch size should be divisible to 10, but provided {}. Fixing it".format(
                    self.gen_params["batch_size"]))
            self.gen_params["batch_size"] += 10 - (self.gen_params["batch_size"] % 10)

        if "patience" not in self.gen_params:
            logging.warning("patience param is not set for GAN params, so setting it to default ""25""")
            self.gen_params["patience"] = 25

        if "epochs" not in self.gen_params:
            logging.warning("patience param is not set for GAN params, so setting it to default ""50""")
            self.gen_params["epochs"] = 50

    def generate_data(
            self, train_df, target, test_df, only_generated_data: bool
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self.check_params()
        self._validate_data(train_df, target, test_df)
        if target is not None:
            train_df[self.TEMP_TARGET] = target
        ctgan = CTGAN(batch_size=self.gen_params["batch_size"], patience=self.gen_params["patience"])
        logging.info("training GAN")
        if self.cat_cols is None:
            ctgan.fit(train_df, [], epochs=self.gen_params["epochs"])
        else:
            ctgan.fit(train_df, self.cat_cols, epochs=self.gen_params["epochs"])
        logging.info("Finished training GAN")
        generated_df = ctgan.sample(
            self.pregeneration_frac * self.get_generated_shape(train_df)
        )
        return self.handle_generated_data(train_df, generated_df, only_generated_data)


class SamplerDiffusion(SamplerOriginal):
    def generate_data(
            self, train_df, target, test_df, only_generated_data: bool
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self._validate_data(train_df, target, test_df)
        if target is not None:
            train_df[self.TEMP_TARGET] = target
        logging.info("Fitting ForestDiffusion model")
        if self.cat_cols is None:
            forest_model = ForestDiffusionModel(train_df.to_numpy(), label_y=None, n_t=50,
                                                duplicate_K=100,
                                                diffusion_type='flow', n_jobs=-1)
        else:
            forest_model = ForestDiffusionModel(train_df.to_numpy(), label_y=None, n_t=50,
                                                duplicate_K=100,
                                                # todo fix bug with cat cols
                                                # cat_indexes=self.get_column_indexes(train_df, self.cat_cols),
                                                diffusion_type='flow', n_jobs=-1)
        logging.info("Finished training ForestDiffusionModel")
        generated_df = forest_model.generate(batch_size=int(self.gen_x_times * train_df.to_numpy().shape[0]))

        return self.handle_generated_data(train_df, generated_df, only_generated_data)

    @staticmethod
    def get_column_indexes(df, column_names):
        return [df.columns.get_loc(col) for col in column_names]


class SamplerLLM(SamplerOriginal):
    def check_params(self):
        if "llm" not in self.gen_params:
            logging.warning("llm param is not set for LLM params, so setting it to default ""distilgpt2""")
            self.gen_params["llm"] = "distilgpt2"
        if "max_length" not in self.gen_params:
            logging.warning("max_length param is not set for LLM params, so setting it to default ""500""")
            self.gen_params["max_length"] = "500"

        if self.gen_params["epochs"] < 3:
            logging.warning(
                "Current set epoch = {} for llm training is too low, setting to 3!""".format(
                    self.gen_params["epochs"]))
            self.gen_params["epochs"] = 3

    def generate_data(
            self, train_df, target, test_df, only_generated_data: bool
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self._validate_data(train_df, target, test_df)
        self.check_params()
        if target is not None:
            train_df[self.TEMP_TARGET] = target
        logging.info("Fitting LLM model")
        is_fp16 = torch.cuda.is_available()
        model = GReaT(llm=self.gen_params["llm"], batch_size=self.gen_params["batch_size"],
                      epochs=self.gen_params["epochs"], fp16=is_fp16)
        model.fit(train_df)

        logging.info("Finished training ForestDiffusionModel")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        generated_df = model.sample(int(self.gen_x_times * train_df.shape[0]), device=device,
                                    max_length=self.gen_params["max_length"])
        return self.handle_generated_data(train_df, generated_df, only_generated_data)


if __name__ == "__main__":
    setup_logging(logging.DEBUG)
    train_size = 75
    train = pd.DataFrame(np.random.randint(-10, 150, size=(train_size, 4)), columns=list("ABCD"))
    target = pd.DataFrame(np.random.randint(0, 2, size=(train_size, 1)), columns=list("Y"))
    test = pd.DataFrame(np.random.randint(0, 100, size=(train_size, 4)), columns=list("ABCD"))
    logging.info(train)

    generators = [
        OriginalGenerator(gen_x_times=15),
        GANGenerator(gen_x_times=10, only_generated_data=False,
                     gen_params={"batch_size": 500, "patience": 25, "epochs": 500}),
        LLMGenerator(gen_params={"batch_size": 32, "epochs": 4, "llm": "distilgpt2", "max_length": 500}),
        OriginalGenerator(gen_x_times=15),
        GANGenerator(cat_cols=["A"], gen_x_times=20, only_generated_data=True),
        ForestDiffusionGenerator(cat_cols=["A"], gen_x_times=1, only_generated_data=True),
        ForestDiffusionGenerator(gen_x_times=10, only_generated_data=False,
                                 gen_params={"batch_size": 500, "patience": 25, "epochs": 500})
    ]

    for gen in generators:
        _sampler(gen, train, target if 'LLMGenerator' not in str(type(gen)) else None, test)

    min_date, max_date = pd.to_datetime('2019-01-01'), pd.to_datetime('2021-12-31')
    train['Date'] = min_date + pd.to_timedelta(np.random.randint((max_date - min_date).days + 1, size=train_size),
                                               unit='d')
    train = get_year_mnth_dt_from_date(train, 'Date')

    new_train, new_target = GANGenerator(
        gen_x_times=1.1, cat_cols=['year'], bot_filter_quantile=0.001, top_filter_quantile=0.999,
        is_post_process=True, pregeneration_frac=2, only_generated_data=False
    ).generate_data_pipe(train.drop('Date', axis=1), None, train.drop('Date', axis=1))
    new_train = collect_dates(new_train)

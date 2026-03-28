# -*- coding: utf-8 -*-

import logging
import numpy as np
import warnings
from typing import Tuple

import pandas as pd
import torch
from be_great import GReaT

from _ForestDiffusion import ForestDiffusionModel
from _ctgan.synthesizer import _CTGANSynthesizer as CTGAN
from tabgan.abc_sampler import Sampler, SampleData
from tabgan.adversarial_model import AdversarialModel
from tabgan.utils import setup_logging, _drop_col_if_exist, \
    get_columns_if_exists, _sampler, get_year_mnth_dt_from_date, collect_dates
from tabgan.llm_config import LLMAPIConfig
from tabgan.llm_api_client import LLMAPIClient

warnings.filterwarnings("ignore")

__author__ = "Insaf Ashrapov"
__copyright__ = "Insaf Ashrapov"
__license__ = "Apache 2.0"

__all__ = ["OriginalGenerator", "GANGenerator", "ForestDiffusionGenerator", "LLMGenerator"]


class _BaseGenerator(SampleData):
    """Base factory that stores constructor arguments for the concrete sampler."""
    _sampler_class = None

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def get_object_generator(self) -> Sampler:
        return self._sampler_class(*self.args, **self.kwargs)


class OriginalGenerator(_BaseGenerator):
    _sampler_class = None  # set after SamplerOriginal is defined


class GANGenerator(_BaseGenerator):
    _sampler_class = None


class ForestDiffusionGenerator(_BaseGenerator):
    _sampler_class = None


class LLMGenerator(_BaseGenerator):
    _sampler_class = None


class SamplerOriginal(Sampler):
    def __init__(
            self,
            gen_x_times: float = 1.1,
            cat_cols: list = None,
            bot_filter_quantile: float = 0.001,
            top_filter_quantile: float = 0.999,
            is_post_process: bool = True,
            adversarial_model_params: dict = None,
            pregeneration_frac: float = 2,
            only_generated_data: bool = False,
            gen_params: dict = None,
            text_generating_columns: list = None,
            conditional_columns: list = None,
            llm_api_config: LLMAPIConfig = None,
    ):
        """
        Initialize an original sampler configuration.

        Args:
            gen_x_times (float): Factor controlling how many synthetic samples
                to generate relative to the training size. The final amount
                can be smaller after post-processing and adversarial filtering.
            cat_cols (list | None): Names of categorical columns in the
                training data.
            bot_filter_quantile (float): Lower quantile used for numeric
                post-processing filters.
            top_filter_quantile (float): Upper quantile used for numeric
                post-processing filters.
            is_post_process (bool): Whether to apply post-processing filters
                based on the distribution of `test_df`. If False, the
                quantile-based filters are skipped.
            adversarial_model_params (dict): Parameters for the adversarial
                filtering model used to keep generated samples close to the
                test distribution.
            pregeneration_frac (float): Oversampling factor applied before
                post-processing. The final number of rows is derived from
                `gen_x_times`.
            only_generated_data (bool): If True, return only synthetic rows.
                If False, append generated rows to the original training data.
            gen_params (dict): Model-specific generation parameters shared by
                subclasses (GAN, ForestDiffusion, LLM).
            text_generating_columns (list | None): Column names for which new
                text values should be generated (used by `SamplerLLM`).
            conditional_columns (list | None): Column names that condition
                text generation for `text_generating_columns`.
            llm_api_config (LLMAPIConfig | None): Configuration for external LLM
                API-based text generation. When provided, text generation will use
                the API instead of the local model. Useful for LM Studio, Ollama,
                OpenAI, etc.
        """
        if adversarial_model_params is None:
            adversarial_model_params = {
                "metrics": "AUC",
                "max_depth": 2,
                "max_bin": 100,
                "n_estimators": 150,
                "learning_rate": 0.02,
                "random_state": 42,
            }
        if gen_params is None:
            gen_params = {"batch_size": 45, "patience": 25, "epochs": 50, "llm": "distilgpt2"}
        super().__init__(
            gen_x_times=gen_x_times,
            cat_cols=cat_cols,
            bot_filter_quantile=bot_filter_quantile,
            top_filter_quantile=top_filter_quantile,
            is_post_process=is_post_process,
            adversarial_model_params=adversarial_model_params,
            pregeneration_frac=pregeneration_frac,
            only_generated_data=only_generated_data,
            gen_params=gen_params,
        )
        self.text_generating_columns = text_generating_columns
        self.conditional_columns = conditional_columns
        self.llm_api_config = llm_api_config
        if not hasattr(self, "TEMP_TARGET"):
            self.TEMP_TARGET = "TEMP_TARGET"

    @staticmethod
    def preprocess_data_df(df) -> pd.DataFrame:
        logging.info(f"Input shape: {df.shape}")
        if not isinstance(df, pd.DataFrame):
            raise ValueError(
                f"Input dataframe is not a pandas DataFrame: got {type(df)}"
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
                f"Input train dataframe already has '{self.TEMP_TARGET}' column, consider removing it"
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
            warnings.warn(
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
            f"Generated shape: {generated_df.drop(self.TEMP_TARGET, axis=1).shape} "
            f"and {generated_df[self.TEMP_TARGET].shape}"
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

        # Filter numerical columns
        for col in test_df.columns:
            if self.cat_cols is None or col not in self.cat_cols:
                min_val = test_df[col].quantile(self.bot_filter_quantile)
                max_val = test_df[col].quantile(self.top_filter_quantile)
                train_df = train_df[(train_df[col].isna()) | ((train_df[col] >= min_val) & (train_df[col] <= max_val))]

                if train_df.shape[0] < 10:
                    raise ValueError(f"Too few samples (<10) after filtering column {col}. "
                                     f"Test data may be skewed. Filter range: [{min_val}, {max_val}]")

        # Filter categorical columns
        if self.cat_cols:
            for col in self.cat_cols:
                train_df = train_df[train_df[col].isin(test_df[col].unique())]
                if train_df.shape[0] < 10:
                    raise ValueError(f"Too few samples (<10) after filtering categorical column {col}")

        logging.info(
            f"Generated shapes after postprocessing: {train_df.drop(self.TEMP_TARGET, axis=1).shape} plus target")

        result_df = train_df.reset_index(drop=True)
        return (
            result_df.drop(self.TEMP_TARGET, axis=1),
            result_df[self.TEMP_TARGET]
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
                    f"Shape of train is {train_df.shape[0]} and test is {test_df.shape[0]}. "
                    f"Both should be at least 10! Consider disabling adversarial filtering"
                )
        if target is not None:
            if train_df.shape[0] != target.shape[0]:
                raise ValueError(
                    f"Shape mismatch: train_df has {train_df.shape[0]} rows "
                    f"but target has {target.shape[0]} rows"
                )

    def handle_generated_data(self, train_df, generated_df, only_generated_data):
        """
        Align and optionally merge generated rows with the original training data.

        The generated data is cast to the dtypes and column order of `train_df`
        so that downstream models receive data with a consistent schema.

        Args:
            train_df (pd.DataFrame): Original training data used to infer the
                schema and target column.
            generated_df (pd.DataFrame or array-like): Newly generated
                samples to be aligned with `train_df`.
            only_generated_data (bool): If True, return only synthetic rows;
                otherwise, append them to `train_df` before returning.

        Returns:
            Tuple[pd.DataFrame, pd.Series | pd.DataFrame]: Features and
            corresponding target values.
        """
        generated_df = pd.DataFrame(generated_df)
        generated_df.columns = train_df.columns

        for column_index in range(len(generated_df.columns)):
            target_column = generated_df.columns[column_index]
            generated_df[target_column] = generated_df[target_column].astype(
                train_df.dtypes.values[column_index]
            )

        if not only_generated_data:
            train_df = pd.concat([train_df, generated_df]).reset_index(drop=True)
            logging.info(
                f"Generated shapes: {_drop_col_if_exist(train_df, self.TEMP_TARGET).shape} plus target"
            )
            return (
                _drop_col_if_exist(train_df, self.TEMP_TARGET),
                get_columns_if_exists(train_df, self.TEMP_TARGET),
            )
        else:
            logging.info(
                f"Generated shapes: {_drop_col_if_exist(generated_df, self.TEMP_TARGET).shape} plus target"
            )
            return (
                _drop_col_if_exist(generated_df, self.TEMP_TARGET),
                get_columns_if_exists(generated_df, self.TEMP_TARGET),
            )


class SamplerGAN(SamplerOriginal):
    def check_params(self):
        if self.gen_params["batch_size"] % 10 != 0:
            logging.warning(
                f"Batch size should be divisible by 10, but got {self.gen_params['batch_size']}. Fixing it")
            self.gen_params["batch_size"] += 10 - (self.gen_params["batch_size"] % 10)

        if "patience" not in self.gen_params:
            logging.warning("patience param is not set for GAN params, setting default to 25")
            self.gen_params["patience"] = 25

        if "epochs" not in self.gen_params:
            logging.warning("epochs param is not set for GAN params, setting default to 50")
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
            logging.warning("llm param is not set for LLM params, setting default to 'distilgpt2'")
            self.gen_params["llm"] = "distilgpt2"
        if "max_length" not in self.gen_params:
            logging.warning("max_length param is not set for LLM params, setting default to 500")
            self.gen_params["max_length"] = 500

        if self.gen_params["epochs"] < 3:
            logging.warning(
                f"Current epoch={self.gen_params['epochs']} for LLM training is too low, setting to 3")
            self.gen_params["epochs"] = 3

    def _build_training_frame(self, train_df: pd.DataFrame, target: pd.DataFrame | None) -> pd.DataFrame:
        """
        Return a copy of the training frame with TEMP_TARGET attached when a target is provided.
        """
        current_train_df = train_df.copy()
        if target is not None:
            current_train_df[self.TEMP_TARGET] = target
        return current_train_df

    def _fit_great_model(self, current_train_df: pd.DataFrame):
        """
        Fit a GReaT model on the provided training frame and return the instance and inference device.
        """
        logging.info("Fitting LLM model")
        is_fp16 = torch.cuda.is_available()
        try:
            from be_great import GReaT
        except ImportError:
            raise ImportError("be_great library is not installed. Please install it to use LLMGenerator.")

        great_model_instance = GReaT(
            llm=self.gen_params["llm"],
            batch_size=self.gen_params["batch_size"],
            epochs=self.gen_params["epochs"],
            fp16=is_fp16,
        )
        great_model_instance.fit(current_train_df)
        logging.info("Finished training LLM model")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        return great_model_instance, device

    def _conditional_text_generation(
        self,
        great_model_instance,
        current_train_df: pd.DataFrame,
        train_df: pd.DataFrame,
        target: pd.DataFrame | None,
        device: str,
    ) -> pd.DataFrame:
        """
        Generate rows when text and conditional columns are specified.
        """
        logging.info("Starting conditional generation of text columns.")
        num_samples_to_generate = int(self.gen_x_times * train_df.shape[0])

        original_unique_text_values: dict[str, set] = {}
        for col in self.text_generating_columns:
            if col not in current_train_df.columns:
                raise ValueError(f"Text generating column '{col}' not found in training data.")
            original_unique_text_values[col] = set(current_train_df[col].unique())

        attribute_distributions: dict[str, pd.Series] = {}
        for col in self.conditional_columns:
            if col not in current_train_df.columns:
                raise ValueError(f"Conditional column '{col}' not found in training data.")
            attribute_distributions[col] = current_train_df[col].value_counts(normalize=True)

        generated_rows: list[dict] = []
        all_train_columns = current_train_df.columns.tolist()

        for _ in range(num_samples_to_generate):
            current_row_data: dict = {}

            for attr_col in self.conditional_columns:
                dist = attribute_distributions[attr_col]
                current_row_data[attr_col] = np.random.choice(dist.index, p=dist.values)

            row_template_for_impute = pd.DataFrame(columns=all_train_columns, index=[0])
            for col in all_train_columns:
                if col in current_row_data:
                    row_template_for_impute.loc[0, col] = current_row_data[col]
                elif col not in self.text_generating_columns:
                    row_template_for_impute.loc[0, col] = np.nan

            imputed_full_row_df = great_model_instance.impute(
                row_template_for_impute.copy(),
                max_length=self.gen_params.get("max_length", 500),
            )

            for col in all_train_columns:
                if col not in self.text_generating_columns and col not in current_row_data:
                    current_row_data[col] = imputed_full_row_df.loc[0, col]

            for text_col in self.text_generating_columns:
                prompt_parts: list[str] = []
                for cond_col in self.conditional_columns:
                    prompt_parts.append(f"{cond_col}: {current_row_data[cond_col]}")
                for other_col in all_train_columns:
                    if (
                        other_col not in self.text_generating_columns
                        and other_col not in self.conditional_columns
                        and other_col in current_row_data
                    ):
                        val_str = str(current_row_data[other_col])
                        if len(val_str) > 30:
                            val_str = val_str[:27] + "..."
                        prompt_parts.append(f"{other_col}: {val_str}")

                prompt = ", ".join(prompt_parts) + f", Generate {text_col}: "

                generated_text_candidate = None
                max_retries = 10
                for _retry_attempt in range(max_retries):
                    generated_text_candidate = self._generate_via_prompt(
                        prompt,
                        great_model_instance,
                        device=device,
                    )
                    if generated_text_candidate not in original_unique_text_values[text_col]:
                        break
                else:
                    logging.warning(
                        f"Max retries reached for generating novel text for {text_col}. Using last candidate."
                    )
                current_row_data[text_col] = generated_text_candidate

            ordered_row = {col: current_row_data.get(col) for col in train_df.columns}
            if target is not None and self.TEMP_TARGET in current_row_data:
                ordered_row[self.TEMP_TARGET] = current_row_data[self.TEMP_TARGET]

            generated_rows.append(ordered_row)

        generated_df = pd.DataFrame(generated_rows)
        return generated_df.reindex(columns=current_train_df.columns)

    def _standard_llm_sampling(
        self,
        great_model_instance,
        current_train_df: pd.DataFrame,
        device: str,
    ) -> pd.DataFrame:
        """
        Fallback sampling when no explicit text/conditional columns are provided.
        """
        logging.info("Starting standard LLM sampling.")
        return great_model_instance.sample(
            int(self.gen_x_times * current_train_df.shape[0]),
            device=device,
            max_length=self.gen_params["max_length"],
        )

    def generate_data(
            self, train_df, target, test_df, only_generated_data: bool
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self._validate_data(train_df, target, test_df)
        self.check_params()

        current_train_df = self._build_training_frame(train_df, target)
        great_model_instance, device = self._fit_great_model(current_train_df)

        if self.text_generating_columns and self.conditional_columns:
            generated_df = self._conditional_text_generation(
                great_model_instance,
                current_train_df=current_train_df,
                train_df=train_df,
                target=target,
                device=device,
            )
        else:
            generated_df = self._standard_llm_sampling(
                great_model_instance,
                current_train_df=current_train_df,
                device=device,
            )

        # When a target is provided, ``current_train_df`` already includes the
        # TEMP_TARGET column and represents the true training frame used for
        # generation. Passing it to ``handle_generated_data`` keeps feature and
        # target alignment consistent for both conditional and standard LLM
        # sampling paths.
        base_train_for_handling = current_train_df if target is not None else train_df
        return self.handle_generated_data(base_train_for_handling, generated_df, only_generated_data)

    def _generate_via_prompt(self, prompt: str, great_model_instance, device: str, max_tokens_to_generate=50) -> str:
        """
        Generate a short text completion from the underlying GReaT LLM.

        Args:
            prompt (str): Serialized row description used as generation context.
            great_model_instance: Fitted GReaT instance providing `model` and
                `tokenizer` attributes.
            device (str): Target device for inference (for example, ``"cpu"``
                or ``"cuda"``).
            max_tokens_to_generate (int): Maximum number of new tokens to
                sample from the model.

        Returns:
            str: Post-processed generated text. Returns an empty string if
            generation fails.
        """
        llm_model = great_model_instance.model
        tokenizer = great_model_instance.tokenizer

        if llm_model is None or tokenizer is None:
            logging.error("LLM model or tokenizer not available in GReaT instance.")
            return ""  # Or raise an error

        llm_model.to(device)

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=tokenizer.model_max_length - max_tokens_to_generate)
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)

        try:
            outputs = llm_model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens_to_generate,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,  # Enable sampling for more diverse outputs
                temperature=0.7,  # Default temperature, can be tuned
                top_k=50,  # Default top_k, can be tuned
                top_p=0.95  # Default top_p, can be tuned
            )
            generated_text = tokenizer.decode(outputs[0, input_ids.shape[1]:], skip_special_tokens=True)

            generated_text = generated_text.split('\n')[0].split('|')[0].strip()

            return generated_text

        except Exception as e:
            logging.error(f"Error during text generation via prompt: {e}")
            return ""  # Fallback or re-raise


# Wire up factory classes to their concrete sampler implementations
OriginalGenerator._sampler_class = SamplerOriginal
GANGenerator._sampler_class = SamplerGAN
ForestDiffusionGenerator._sampler_class = SamplerDiffusion
LLMGenerator._sampler_class = SamplerLLM


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
        ForestDiffusionGenerator(cat_cols=["A"], gen_x_times=10, only_generated_data=True),
        ForestDiffusionGenerator(gen_x_times=15, only_generated_data=False,
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

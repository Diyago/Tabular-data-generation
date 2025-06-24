# -*- coding: utf-8 -*-

import logging
import numpy as np
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
            text_generating_columns: list = None,
            conditional_columns: list = None,
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
        @param text_generating_columns: list = None - List of column names for which new text values should be generated.
        @param conditional_columns: list = None - List of column names to condition the generation of text_generating_columns.
        """
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
        # Ensure TEMP_TARGET is initialized here if not by super()
        if not hasattr(self, 'TEMP_TARGET'):
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
        """
        Integrates synthetic data with the original dataset by preserving data types
        and structural alignment.

        This method transforms generated data to match the original dataset's structure
        and types. It can either combine synthetic with original data or return only
        the synthetic data.

        Args:
            train_df: The original dataset that defines the expected structure
            generated_df: The synthetic data to be processed
            only_generated_data: Boolean flag to return only synthetic data

        Returns:
            A tuple containing:
            - Feature matrix (with or without original data)
            - Corresponding target vector
        """
        generated_df = pd.DataFrame(generated_df)
        generated_df.columns = train_df.columns

        # Preserve original data types
        for column_index in range(len(generated_df.columns)):
            target_column = generated_df.columns[column_index]
            generated_df[target_column] = generated_df[target_column].astype(
                train_df.dtypes.values[column_index]
            )

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

        # Use a copy to avoid modifying the original train_df if target is added
        current_train_df = train_df.copy()
        if target is not None:
            current_train_df[self.TEMP_TARGET] = target

        logging.info("Fitting LLM model")
        is_fp16 = torch.cuda.is_available()
        # Ensure GReaT is imported
        try:
            from be_great import GReaT
        except ImportError:
            raise ImportError("be_great library is not installed. Please install it to use LLMGenerator.")

        great_model_instance = GReaT(llm=self.gen_params["llm"], batch_size=self.gen_params["batch_size"],
                                     epochs=self.gen_params["epochs"], fp16=is_fp16)
        great_model_instance.fit(current_train_df)
        logging.info("Finished training LLM model")

        device = "cuda" if torch.cuda.is_available() else "cpu" # Needed for _generate_via_prompt

        if self.text_generating_columns and self.conditional_columns:
            logging.info("Starting conditional generation of text columns.")
            num_samples_to_generate = int(self.gen_x_times * train_df.shape[0])

            original_unique_text_values = {}
            for col in self.text_generating_columns:
                if col not in current_train_df.columns:
                    raise ValueError(f"Text generating column '{col}' not found in training data.")
                original_unique_text_values[col] = set(current_train_df[col].unique())

            attribute_distributions = {}
            for col in self.conditional_columns:
                if col not in current_train_df.columns:
                    raise ValueError(f"Conditional column '{col}' not found in training data.")
                attribute_distributions[col] = current_train_df[col].value_counts(normalize=True)

            generated_rows = []
            all_train_columns = current_train_df.columns.tolist()

            for _ in range(num_samples_to_generate):
                current_row_data = {}

                # 1. Sample conditional attributes
                for attr_col in self.conditional_columns:
                    dist = attribute_distributions[attr_col]
                    current_row_data[attr_col] = np.random.choice(dist.index, p=dist.values)

                # 2. Generate other non-text columns using be_great.impute
                # Create a template row with NaNs for be_great to fill
                row_template_for_impute = pd.DataFrame(columns=all_train_columns, index=[0])
                for col in all_train_columns:
                    if col in current_row_data: # Conditional column already sampled
                        row_template_for_impute.loc[0, col] = current_row_data[col]
                    elif col not in self.text_generating_columns: # Column to be imputed
                        row_template_for_impute.loc[0, col] = np.nan
                    # else: text_generating_columns will be filled later by _generate_via_prompt

                # Impute NaNs for non-text, non-conditional columns
                # Note: be_great.impute might expect specific formatting or might modify inplace.
                # This part might need adjustment based on be_great's exact API for single row imputation.
                # For now, assuming it returns a DataFrame with NaNs filled.
                # Also, max_length might need to be dynamic or a class parameter.
                imputed_full_row_df = great_model_instance.impute(row_template_for_impute.copy(),
                                                                  max_length=self.gen_params.get("max_length", 500))

                for col in all_train_columns:
                    if col not in self.text_generating_columns and col not in current_row_data:
                        current_row_data[col] = imputed_full_row_df.loc[0, col]

                # 3. Generate novel text values
                for text_col in self.text_generating_columns:
                    # Construct prompt
                    prompt_parts = []
                    for cond_col in self.conditional_columns:
                        prompt_parts.append(f"{cond_col}: {current_row_data[cond_col]}")
                    # Include other relevant, already generated/imputed columns in the prompt
                    for other_col in all_train_columns:
                        if other_col not in self.text_generating_columns and other_col not in self.conditional_columns and other_col in current_row_data:
                             # Limit length of values in prompt
                            val_str = str(current_row_data[other_col])
                            if len(val_str) > 30: val_str = val_str[:27] + "..."
                            prompt_parts.append(f"{other_col}: {val_str}")

                    prompt = ", ".join(prompt_parts) + f", Generate {text_col}: "

                    generated_text_candidate = None
                    max_retries = 10
                    for _retry_attempt in range(max_retries):
                        # Placeholder for now, will be implemented in next step
                        # Pass great_model_instance.model and great_model_instance.tokenizer
                        generated_text_candidate = self._generate_via_prompt(prompt, great_model_instance, device=device)
                        if generated_text_candidate not in original_unique_text_values[text_col]:
                            break
                    else: # Max retries reached
                        logging.warning(f"Max retries reached for generating novel text for {text_col}. Using last candidate.")
                    current_row_data[text_col] = generated_text_candidate

                # Ensure all columns are present in the order of original train_df (excluding TEMP_TARGET for generated_df)
                ordered_row = {col: current_row_data.get(col) for col in train_df.columns} # Uses original train_df columns
                if target is not None and self.TEMP_TARGET in current_row_data: # Handle target if it was part of generation
                    ordered_row[self.TEMP_TARGET] = current_row_data[self.TEMP_TARGET]

                generated_rows.append(ordered_row)

            generated_df = pd.DataFrame(generated_rows)
            # Align columns with current_train_df (which includes TEMP_TARGET if target was not None)
            # This ensures generated_df has the TEMP_TARGET before handle_generated_data if it was part of training
            generated_df = generated_df.reindex(columns=current_train_df.columns)

        else:
            logging.info("Starting standard LLM sampling.")
            generated_df = great_model_instance.sample(int(self.gen_x_times * current_train_df.shape[0]),
                                                       device=device,
                                                       max_length=self.gen_params["max_length"])

        # current_train_df already includes TEMP_TARGET if target was not None
        # generated_df should also have TEMP_TARGET column if it was part of generation process (e.g. via impute or if it was a regular col)
        # handle_generated_data expects train_df without target, and generated_df potentially with target

        # If target was added to current_train_df, we pass the original train_df (without target)
        # to handle_generated_data, and generated_df (which might contain the target column from generation)
        return self.handle_generated_data(train_df, generated_df, only_generated_data)

    def _generate_via_prompt(self, prompt: str, great_model_instance, device: str, max_tokens_to_generate=50) -> str:
        """
        Generates text using the underlying LLM from the GReaT model instance based on a given prompt.
        """
        llm_model = great_model_instance.model
        tokenizer = great_model_instance.tokenizer

        if llm_model is None or tokenizer is None:
            logging.error("LLM model or tokenizer not available in GReaT instance.")
            return "" # Or raise an error

        # Ensure model is on the correct device (might already be, but good to ensure)
        llm_model.to(device)

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length - max_tokens_to_generate)
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)

        # Generate output tokens
        # Common parameters for generate:
        # - no_repeat_ngram_size: to prevent repetitive phrases
        # - early_stopping: if applicable
        # - temperature: for randomness
        # - top_k, top_p: for nucleus sampling
        # These could be exposed via self.gen_params if more control is needed
        try:
            outputs = llm_model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens_to_generate,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True, # Enable sampling for more diverse outputs
                temperature=0.7, # Default temperature, can be tuned
                top_k=50,        # Default top_k, can be tuned
                top_p=0.95       # Default top_p, can be tuned
            )
            # Decode the generated tokens, excluding the prompt tokens
            generated_text = tokenizer.decode(outputs[0, input_ids.shape[1]:], skip_special_tokens=True)

            # Basic post-processing:
            # GReaT often serializes as "Feature: Value | Feature: Value".
            # The prompt ends with "Generate ColumnName: ". We want the value part.
            # This might need to be smarter if the LLM generates more than just the value.
            # For now, a simple strip should work for many cases.
            # If the LLM generates "GeneratedColumn: Value", we might want to strip "GeneratedColumn: ".
            # This depends on how GReaT serializes and what the LLM learns.
            # For a generic approach, we might look for the first newline or pipe if the model generates more structured output.

            # Example: if prompt is "..., Generate Name: " and output is "John Doe | Age: ...", we want "John Doe"
            # A simple heuristic: take text until a potential separator or end.
            # This is a tricky part and might need refinement based on observed LLM outputs.
            generated_text = generated_text.split('\n')[0].split('|')[0].strip()

            return generated_text

        except Exception as e:
            logging.error(f"Error during text generation via prompt: {e}")
            return "" # Fallback or re-raise


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

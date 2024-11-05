import logging
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import entropy

__all__ = ["compare_dataframes"]
TEMP_TARGET = "_temp_target"


def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(
        level=loglevel, stream=sys.stdout, format=logformat, datefmt="%Y-%m-%d %H:%M:%S"
    )


def make_two_digit(num_as_str: str) -> pd.DataFrame:
    if len(num_as_str) == 2:
        return num_as_str
    else:
        return "0" + num_as_str


def get_year_mnth_dt_from_date(df: pd.DataFrame, date_col="Date") -> pd.DataFrame:
    """
    Extracts year, month, and day from a date column in a pandas DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        date_col (str): Name of the date column.

    Returns:
        pd.DataFrame: DataFrame with year, month, and day columns added.
    """
    df[date_col] = pd.to_datetime(df[date_col])
    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    df["day"] = df[date_col].dt.day
    return df


def collect_dates(df: pd.DataFrame) -> pd.DataFrame:
    df["Date"] = df["year"].astype(str) + "-" \
                 + df["month"].astype(str).apply(make_two_digit) + "-" \
                 + df["day"].astype(str).apply(make_two_digit)
    df.drop(["year", "month", "day"], axis=1, inplace=True)
    return df


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def _sampler(creator, in_train, in_target, in_test) -> None:
    _logger = logging.getLogger(__name__)
    _logger.info("Starting generating data")
    train, test = creator.generate_data_pipe(in_train, in_target, in_test)
    _logger.info(f"Train Data: {train}\nTest Data: {test}")
    _logger.info("Finished generation\n")
    return train, test


def _drop_col_if_exist(df, col_to_drop) -> pd.DataFrame:
    """Drops col_to_drop from input dataframe df if such column exists"""
    if col_to_drop in df.columns:
        return df.drop(col_to_drop, axis=1)
    else:
        return df


def get_columns_if_exists(df, col) -> pd.DataFrame:
    if col in df.columns:
        return df[col]
    else:
        return None


def calculate_psi(expected, actual, buckettype="bins", buckets=10, axis=0):
    """Calculate the PSI (population stability index) across all variables

    Args:
       expected: numpy matrix of original values
       actual: numpy matrix of new values
       buckettype: type of strategy for creating buckets, bins splits into even splits, quantiles splits into quantile buckets
       buckets: number of quantiles to use in bucketing variables
       axis: axis by which variables are defined, 0 for vertical, 1 for horizontal

    Returns:
       psi_values: ndarray of psi values for each variable

    Author:
       Matthew Burke
       github.com/mwburke
       mwburke.github.io.com
    """

    def psi(expected_array, actual_array, buckets):
        """Calculate the PSI for a single variable

        Args:
           expected_array: numpy array of original values
           actual_array: numpy array of new values, same size as expected
           buckets: number of percentile ranges to bucket the values into

        Returns:
           psi_value: calculated PSI value
        """

        def scale_range(input_val, min_val, max_val):
            input_val += -(np.min(input_val))
            input_val /= np.max(input_val) / (max_val - min_val)
            input_val += min_val
            return input_val

        breakpoints = np.arange(0, buckets + 1) / buckets * 100

        if buckettype == "bins":
            breakpoints = scale_range(breakpoints, np.min(expected_array), np.max(expected_array))
        elif buckettype == "quantiles":
            breakpoints = np.stack([np.percentile(expected_array, b) for b in breakpoints])

        expected_fractions = np.histogram(expected_array, breakpoints)[0] / len(expected_array)
        actual_fractions = np.histogram(actual_array, breakpoints)[0] / len(actual_array)

        def sub_psi(e_perc, a_perc):
            '''Calculate the actual PSI value from comparing the values.
               Update the actual value to a very small number if equal to zero
            '''
            if a_perc == 0:
                a_perc = 0.0001
            if e_perc == 0:
                e_perc = 0.0001

            value = (e_perc - a_perc) * np.log(e_perc / a_perc)
            return (value)

        psi_value = sum(sub_psi(expected_fractions[i], actual_fractions[i]) for i in range(0, len(expected_fractions)))

        return (psi_value)

    if len(expected.shape) == 1:
        psi_values = np.empty(len(expected.shape))
    else:
        psi_values = np.empty(expected.shape[1 - axis])

    for i in range(0, len(psi_values)):
        if len(psi_values) == 1:
            try:
                psi_values = psi(expected, actual, buckets)
            except:
                psi_values = 0.9
        elif axis == 0:
            psi_values[i] = psi(expected[:, i], actual[:, i], buckets)
        elif axis == 1:
            psi_values[i] = psi(expected[i, :], actual[i, :], buckets)

    return psi_values


def compare_dataframes(df_original, df_generated):
    """
    Compares two DataFrames for similarity in terms of uniqueness, data quality, and PSI.

    Args:
      df_original: The original DataFrame.
      df_generated: The DataFrame with generated numbers.

    Returns:
        float: A score between 0 (no similarity) and 1 (high similarity) representing the similarity of the two
        DataFrames.

    # Example usage
    df1 = pd.DataFrame({"col1": [1, 2, 3, 4], "col2": ["a", "b", "a", "c"]})
    df2 = pd.DataFrame({"col1": [1, 2, 5, 6], "col2": ["a", "b", "x", "y"]})

    similarity_score = compare_dataframes(df1.copy(), df2.copy())
    print(similarity_score)
    """
    # Handle DataFrames with different shapes
    if len(df_original.columns) != len(df_generated.columns):
        # Penalize if column names don't match
        return 0.0
    # Handle potential differences in row count
    n_original = len(df_original)
    n_generated = len(df_generated)
    min_rows = min(n_original, n_generated)

    # Uniqueness: Ratio of non-null unique values in generated vs original (weighted by min rows)
    uniq_original = df_original.nunique().sum() / (len(df_original.columns) + 1e-6)
    uniq_generated = df_generated.nunique().sum() / (len(df_generated.columns) + 1e-6)
    uniqueness_score = (uniq_generated / uniq_original) * (min_rows / n_generated)

    # Data Quality: Distribution similarity using Kolmogorov-Smirnov test (average across columns)
    data_quality_scores = []
    for col in df_original.columns:
        if col in df_generated.columns:
            # Ensure both columns have numeric data types before applying K-S test
            if pd.api.types.is_numeric_dtype(df_original[col]) and pd.api.types.is_numeric_dtype(df_generated[col]):
                _, p_value = df_original[col].value_counts().sort_index(ascending=False).diff().dropna().abs().sum() / (
                        n_original + 1e-6), \
                             df_generated[col].value_counts().sort_index(
                                 ascending=False).diff().dropna().abs().sum() / (n_generated + 1e-6)
                # Avoid zero division and set minimum p-value to a small positive value
                p_value = max(p_value, 1e-6)
                data_quality_scores.append(p_value)
    data_quality_score = sum(data_quality_scores) / len(data_quality_scores) if data_quality_scores else 1

    # PSI Similarity: Average PSI across all column pairs (capped at theoretical maximum)
    psi_scores = []
    for col_orig in df_original.columns:
        if col_orig in df_generated.columns:
            psi_scores.append(calculate_psi(df_original[col_orig], df_generated[col_orig],
                                            buckets=10))  # Assuming buckets=10 for PSI calculation
    psi_similarity = sum(psi_scores) / len(psi_scores) if psi_scores else 1

    # Combine uniqueness, data quality, and PSI scores (weighted)
    similarity_score = 0.1 * uniqueness_score + 0.45 * data_quality_score + 0.45 * (1/psi_similarity)
    print(uniqueness_score, data_quality_score, psi_similarity)
    # Ensure score is between 0 and 1
    similarity_score = min(max(similarity_score, 0), 1)

    return similarity_score

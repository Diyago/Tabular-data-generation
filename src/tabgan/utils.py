import logging
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import entropy

__all__ = ["compare_dataframes"]


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
        return '0' + num_as_str


def get_year_mnth_dt_from_date(df: pd.DataFrame, date_col='Date') -> pd.DataFrame:
    """
    Extracts year, month, and day from a date column in a pandas DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        date_col (str): Name of the date column.

    Returns:
        pd.DataFrame: DataFrame with year, month, and day columns added.
    """
    df[date_col] = pd.to_datetime(df[date_col])
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['day'] = df[date_col].dt.day
    return df


def collect_dates(df: pd.DataFrame) -> pd.DataFrame:
    df["Date"] = df['year'].astype(str) + '-' \
                 + df['month'].astype(str).apply(make_two_digit) + '-' \
                 + df['day'].astype(str).apply(make_two_digit)
    df.drop(['year', 'month', 'day'], axis=1, inplace=True)
    return df


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def _sampler(creator, in_train, in_target, in_test) -> None:
    _logger = logging.getLogger(__name__)
    _logger.info("Starting generating data")
    train, test = creator.generate_data_pipe(in_train, in_target, in_test)
    _logger.info(train, test)
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
            p_orig = df_original[col_orig].value_counts(normalize=True)
            p_gen = df_generated[col_orig].value_counts(normalize=True)
            # Handle potential division by zero with entropy function (uses log2 internally)
            h_orig = entropy(p_orig, base=2)
            h_gen = entropy(p_gen, base=2)
            h_joint = entropy(pd.concat([p_orig, p_gen], ignore_index=True), base=2)
            psi = max(0, min(h_orig + h_gen - h_joint, 1))  # Ensure non-negative and cap at 1
            psi_scores.append(psi)
    psi_similarity = sum(psi_scores) / len(psi_scores) if psi_scores else 1

    # Combine uniqueness, data quality, and PSI scores (weighted)
    similarity_score = 0.5 * uniqueness_score + 0.3 * data_quality_score + 0.2 * psi_similarity

    # Ensure score is between 0 and 1
    similarity_score = min(max(similarity_score, 0), 1)

    return similarity_score


TEMP_TARGET = "_temp_target"

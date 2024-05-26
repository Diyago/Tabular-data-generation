import logging
import os
import random
import sys

import numpy as np
import pandas as pd
import torch

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


def compare_dataframes(df1, df2):
    """
    Compares two DataFrames for similarity

    Args:
        df1 (pd.DataFrame): The first DataFrame (original)
        df2 (pd.DataFrame): The second DataFrame (generated)

    Returns:
        float: A score between 0 and 1 representing the similarity of the two DataFrames

    # Example usage
    df1 = pd.DataFrame({"col1": [1, 2, 3, 4], "col2": ["a", "b", "a", "c"]})
    df2 = pd.DataFrame({"col1": [1, 2, 5, 6], "col2": ["a", "b", "x", "y"]})

    similarity_score = compare_dataframes(df1.copy(), df2.copy())
    print(similarity_score)
    """

    if df1.shape != df2.shape:
        # Penalize if DataFrames have different shapes
        return 0.0

    # Calculate the intersection of unique elements between DataFrames
    intersection = len(set(df1.values.ravel()) & set(df2.values.ravel()))

    # Calculate the union of unique elements between DataFrames
    union = len(set(df1.values.ravel()) | set(df2.values.ravel()))

    # Avoid division by zero
    if union == 0:
        return 0.0

    # Jaccard similarity score - measure of set similarity
    similarity = intersection / union

    # Penalize if there are many missing values in generated DataFrame
    missing_values_penalty = 1 - df2.isnull().sum().sum() / df2.size

    # Penalize if the distribution of values in each column is very different
    chi_squared_penalty = 0
    for col in df1.columns:
        chi_squared_penalty += sum(
            (observed - expected) ** 2 / (expected + 1)
            for observed, expected in pd.crosstab(df1[col], df2[col]).fillna(0).to_numpy().ravel()
        )
    chi_squared_penalty = np.exp(-chi_squared_penalty)  # Normalize penalty

    # Combine Jaccard similarity, missing value penalty, and distribution penalty
    return similarity * missing_values_penalty * chi_squared_penalty


TEMP_TARGET = "_temp_target"

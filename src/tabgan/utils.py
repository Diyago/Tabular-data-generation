import logging
import sys
import os
import random

import pandas as pd
import numpy as np
import torch


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


TEMP_TARGET = "_temp_target"

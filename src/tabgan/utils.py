import logging
import sys

import pandas as pd


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
    df[date_col] = pd.to_datetime(df[date_col])
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['day'] = df[date_col].dt.day
    return df


def collect_dates(df: pd.DataFrame)-> pd.DataFrame:
    df["Date"] = df['year'].astype(str) + '-' \
                        + df['month'].astype(str).apply(make_two_digit) + '-' \
                        + df['day'].astype(str).apply(make_two_digit)
    df.drop(['year','month','day'], axis=1,inplace=True)
    return df


TEMP_TARGET = "_temp_target"

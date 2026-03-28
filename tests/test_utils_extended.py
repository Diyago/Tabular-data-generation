# -*- coding: utf-8 -*-
"""
Extended tests for tabgan.utils — covers compare_dataframes, calculate_psi,
collect_dates, and the time-series round-trip workflow.
"""

import unittest

import numpy as np
import pandas as pd

from tabgan.utils import (
    calculate_psi,
    collect_dates,
    compare_dataframes,
    get_year_mnth_dt_from_date,
)


# ---------------------------------------------------------------------------
# calculate_psi
# ---------------------------------------------------------------------------
class TestCalculatePSI(unittest.TestCase):
    def test_identical_distributions_returns_near_zero(self):
        arr = np.random.RandomState(42).normal(0, 1, size=500)
        psi = calculate_psi(arr, arr.copy(), buckets=10)
        # PSI of identical arrays should be ~0
        self.assertAlmostEqual(float(psi), 0.0, places=3)

    def test_shifted_distribution_returns_positive(self):
        rng = np.random.RandomState(42)
        expected = rng.normal(0, 1, size=1000)
        actual = rng.normal(2, 1, size=1000)  # shifted mean
        psi = calculate_psi(expected, actual, buckets=10)
        self.assertGreater(float(psi), 0.1)

    def test_2d_array_axis0(self):
        rng = np.random.RandomState(42)
        expected = rng.normal(0, 1, size=(200, 3))
        actual = rng.normal(0, 1, size=(200, 3))
        psi_values = calculate_psi(expected, actual, buckets=10, axis=0)
        self.assertEqual(len(psi_values), 3)
        for v in psi_values:
            self.assertGreaterEqual(v, 0.0)

    def test_quantile_bucket_type(self):
        rng = np.random.RandomState(42)
        expected = rng.normal(0, 1, size=500)
        actual = rng.normal(0, 1, size=500)
        psi = calculate_psi(expected, actual, buckettype="quantiles", buckets=10)
        self.assertGreaterEqual(float(psi), 0.0)


# ---------------------------------------------------------------------------
# compare_dataframes
# ---------------------------------------------------------------------------
class TestCompareDataframes(unittest.TestCase):
    def test_identical_dataframes_score_high(self):
        df = pd.DataFrame({"a": range(100), "b": range(100, 200)})
        score = compare_dataframes(df.copy(), df.copy())
        self.assertGreater(score, 0.5)
        self.assertLessEqual(score, 1.0)

    def test_completely_different_columns_returns_zero(self):
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        score = compare_dataframes(df1, df2)
        self.assertEqual(score, 0.0)

    def test_similar_dataframes_score_moderate(self):
        rng = np.random.RandomState(42)
        df1 = pd.DataFrame({"x": rng.normal(0, 1, 200), "y": rng.normal(5, 2, 200)})
        df2 = pd.DataFrame({"x": rng.normal(0, 1, 200), "y": rng.normal(5, 2, 200)})
        score = compare_dataframes(df1, df2)
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_score_is_between_0_and_1(self):
        rng = np.random.RandomState(0)
        df1 = pd.DataFrame({"a": rng.randint(0, 100, 50)})
        df2 = pd.DataFrame({"a": rng.randint(0, 100, 50)})
        score = compare_dataframes(df1, df2)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_with_only_numeric_columns(self):
        rng = np.random.RandomState(10)
        df1 = pd.DataFrame({"a": rng.randint(0, 50, 80), "b": rng.randint(0, 50, 80)})
        df2 = pd.DataFrame({"a": rng.randint(0, 50, 80), "b": rng.randint(0, 50, 80)})
        score = compare_dataframes(df1, df2)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_with_mixed_types_raises_on_string_psi(self):
        """compare_dataframes passes string columns to calculate_psi which
        cannot handle them. This documents the current behavior."""
        df1 = pd.DataFrame({"num": [1, 2, 3, 4], "cat": ["a", "b", "a", "c"]})
        df2 = pd.DataFrame({"num": [1, 2, 5, 6], "cat": ["a", "b", "x", "y"]})
        with self.assertRaises(TypeError):
            compare_dataframes(df1.copy(), df2.copy())


# ---------------------------------------------------------------------------
# collect_dates (round-trip with get_year_mnth_dt_from_date)
# ---------------------------------------------------------------------------
class TestCollectDates(unittest.TestCase):
    def test_round_trip(self):
        """Decompose dates then reassemble — result should match originals."""
        dates = pd.to_datetime(["2022-01-15", "2023-06-01", "2021-12-31"])
        df = pd.DataFrame({"Date": dates, "value": [10, 20, 30]})
        decomposed = get_year_mnth_dt_from_date(df.copy(), "Date")

        # Drop the original Date column to simulate the generation pipeline
        decomposed = decomposed.drop("Date", axis=1)
        reassembled = collect_dates(decomposed)

        self.assertIn("Date", reassembled.columns)
        self.assertNotIn("year", reassembled.columns)
        self.assertNotIn("month", reassembled.columns)
        self.assertNotIn("day", reassembled.columns)

        expected_dates = ["2022-01-15", "2023-06-01", "2021-12-31"]
        self.assertEqual(list(reassembled["Date"]), expected_dates)

    def test_single_digit_month_day_padded(self):
        """Months and days < 10 should be zero-padded."""
        df = pd.DataFrame({"year": [2020], "month": [3], "day": [5], "x": [1]})
        result = collect_dates(df)
        self.assertEqual(result["Date"].iloc[0], "2020-03-05")


# ---------------------------------------------------------------------------
# Time-series generation workflow (integration-like)
# ---------------------------------------------------------------------------
class TestTimeSeriesWorkflow(unittest.TestCase):
    def test_date_decompose_generate_collect(self):
        """Full round-trip: decompose dates → OriginalGenerator → collect dates."""
        from tabgan.sampler import OriginalGenerator

        rng = np.random.RandomState(42)
        train = pd.DataFrame(rng.randint(0, 100, size=(60, 3)), columns=list("ABC"))
        min_date = pd.to_datetime("2020-01-01")
        max_date = pd.to_datetime("2021-12-31")
        d = (max_date - min_date).days + 1
        train["Date"] = min_date + pd.to_timedelta(rng.randint(d, size=60), unit="D")
        train = get_year_mnth_dt_from_date(train, "Date")

        train_no_date = train.drop("Date", axis=1)

        new_train, _ = OriginalGenerator(
            gen_x_times=1.1,
            cat_cols=["year"],
            is_post_process=True,
            pregeneration_frac=2,
        ).generate_data_pipe(train_no_date, None, train_no_date)

        self.assertIn("year", new_train.columns)
        self.assertIn("month", new_train.columns)
        self.assertIn("day", new_train.columns)

        new_train = collect_dates(new_train)
        self.assertIn("Date", new_train.columns)
        self.assertNotIn("year", new_train.columns)

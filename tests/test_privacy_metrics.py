# -*- coding: utf-8 -*-
"""Tests for privacy metrics."""

import unittest

import numpy as np
import pandas as pd

from src.tabgan.privacy_metrics import PrivacyMetrics


def _make_numeric_data(seed=42):
    rng = np.random.RandomState(seed)
    original = pd.DataFrame(rng.normal(0, 1, size=(100, 4)), columns=list("ABCD"))
    synthetic = pd.DataFrame(rng.normal(0, 1, size=(80, 4)), columns=list("ABCD"))
    return original, synthetic


class TestDCR(unittest.TestCase):
    def test_dcr_returns_expected_keys(self):
        orig, synth = _make_numeric_data()
        pm = PrivacyMetrics(orig, synth)
        result = pm.dcr()
        self.assertIn("mean", result)
        self.assertIn("median", result)
        self.assertIn("5th_percentile", result)
        self.assertIn("distances", result)

    def test_identical_data_dcr_near_zero(self):
        orig, _ = _make_numeric_data()
        pm = PrivacyMetrics(orig, orig.copy())
        result = pm.dcr()
        self.assertAlmostEqual(result["mean"], 0.0, places=5)

    def test_distant_data_dcr_positive(self):
        rng = np.random.RandomState(42)
        orig = pd.DataFrame(rng.normal(0, 1, size=(100, 3)), columns=list("ABC"))
        synth = pd.DataFrame(rng.normal(10, 1, size=(80, 3)), columns=list("ABC"))
        pm = PrivacyMetrics(orig, synth)
        result = pm.dcr()
        self.assertGreater(result["mean"], 1.0)

    def test_dcr_with_sample_size(self):
        orig, synth = _make_numeric_data()
        pm = PrivacyMetrics(orig, synth)
        result = pm.dcr(sample_size=20)
        self.assertEqual(len(result["distances"]), 20)


class TestNNDR(unittest.TestCase):
    def test_nndr_returns_expected_keys(self):
        orig, synth = _make_numeric_data()
        pm = PrivacyMetrics(orig, synth)
        result = pm.nndr()
        self.assertIn("mean", result)
        self.assertIn("median", result)
        self.assertIn("ratios", result)

    def test_nndr_values_between_0_and_1(self):
        orig, synth = _make_numeric_data()
        pm = PrivacyMetrics(orig, synth)
        result = pm.nndr()
        self.assertGreater(result["mean"], 0.0)
        self.assertLessEqual(result["mean"], 1.0)


class TestMembershipInference(unittest.TestCase):
    def test_mi_returns_expected_keys(self):
        orig, synth = _make_numeric_data()
        pm = PrivacyMetrics(orig, synth)
        result = pm.membership_inference_risk()
        self.assertIn("auc", result)
        self.assertIn("accuracy", result)

    def test_mi_auc_in_range(self):
        orig, synth = _make_numeric_data()
        pm = PrivacyMetrics(orig, synth)
        result = pm.membership_inference_risk()
        self.assertGreaterEqual(result["auc"], 0.0)
        self.assertLessEqual(result["auc"], 1.0)


class TestSummary(unittest.TestCase):
    def test_summary_returns_overall_score(self):
        orig, synth = _make_numeric_data()
        pm = PrivacyMetrics(orig, synth)
        s = pm.summary()
        self.assertIn("overall_privacy_score", s)
        self.assertGreaterEqual(s["overall_privacy_score"], 0.0)
        self.assertLessEqual(s["overall_privacy_score"], 1.0)

    def test_summary_contains_all_sections(self):
        orig, synth = _make_numeric_data()
        s = PrivacyMetrics(orig, synth).summary()
        self.assertIn("dcr", s)
        self.assertIn("nndr", s)
        self.assertIn("membership_inference", s)

    def test_with_cat_cols(self):
        rng = np.random.RandomState(42)
        orig = pd.DataFrame({
            "num": rng.normal(0, 1, 80),
            "cat": rng.choice(["A", "B", "C"], 80),
        })
        synth = pd.DataFrame({
            "num": rng.normal(0, 1, 60),
            "cat": rng.choice(["A", "B", "C"], 60),
        })
        s = PrivacyMetrics(orig, synth, cat_cols=["cat"]).summary()
        self.assertIn("overall_privacy_score", s)


if __name__ == "__main__":
    unittest.main()

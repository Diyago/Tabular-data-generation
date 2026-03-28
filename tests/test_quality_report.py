# -*- coding: utf-8 -*-
"""Tests for the quality report."""

import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.tabgan.quality_report import QualityReport


def _make_data(seed=42):
    rng = np.random.RandomState(seed)
    original = pd.DataFrame({
        "A": rng.normal(0, 1, 100),
        "B": rng.normal(5, 2, 100),
        "target": rng.randint(0, 2, 100),
    })
    synthetic = pd.DataFrame({
        "A": rng.normal(0, 1, 80),
        "B": rng.normal(5, 2, 80),
        "target": rng.randint(0, 2, 80),
    })
    return original, synthetic


class TestQualityReportCompute(unittest.TestCase):
    def test_compute_returns_self(self):
        orig, synth = _make_data()
        report = QualityReport(orig, synth).compute()
        self.assertIsInstance(report, QualityReport)

    def test_summary_has_required_keys(self):
        orig, synth = _make_data()
        s = QualityReport(orig, synth).compute().summary()
        self.assertIn("column_stats", s)
        self.assertIn("psi", s)
        self.assertIn("ml_utility", s)
        self.assertIn("overall_score", s)

    def test_overall_score_range(self):
        orig, synth = _make_data()
        s = QualityReport(orig, synth).compute().summary()
        self.assertGreaterEqual(s["overall_score"], 0.0)
        self.assertLessEqual(s["overall_score"], 1.0)

    def test_column_stats_numeric(self):
        orig, synth = _make_data()
        s = QualityReport(orig, synth).compute().summary()
        self.assertIn("A", s["column_stats"])
        self.assertEqual(s["column_stats"]["A"]["dtype"], "numeric")
        self.assertIn("orig_mean", s["column_stats"]["A"])

    def test_column_stats_categorical(self):
        orig = pd.DataFrame({"cat": ["a", "b", "a", "c"], "num": [1, 2, 3, 4]})
        synth = pd.DataFrame({"cat": ["a", "b", "b", "c"], "num": [1, 2, 3, 4]})
        s = QualityReport(orig, synth, cat_cols=["cat"]).compute().summary()
        self.assertEqual(s["column_stats"]["cat"]["dtype"], "categorical")

    def test_psi_per_column(self):
        orig, synth = _make_data()
        s = QualityReport(orig, synth).compute().summary()
        self.assertIn("A", s["psi"])
        self.assertIn("mean", s["psi"])

    def test_ml_utility_with_target(self):
        orig, synth = _make_data()
        s = QualityReport(orig, synth, target_col="target").compute().summary()
        ml = s["ml_utility"]
        self.assertIn("tstr_auc", ml)
        self.assertIn("trtr_auc", ml)
        self.assertIn("utility_ratio", ml)

    def test_ml_utility_without_target(self):
        orig, synth = _make_data()
        s = QualityReport(orig, synth).compute().summary()
        self.assertEqual(s["ml_utility"]["utility_ratio"], 0.0)


class TestQualityReportHTML(unittest.TestCase):
    def test_to_html_creates_file(self):
        orig, synth = _make_data()
        report = QualityReport(orig, synth, target_col="target").compute()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "report.html")
            report.to_html(path)
            self.assertTrue(os.path.exists(path))
            with open(path) as f:
                content = f.read()
            self.assertIn("TabGAN Quality Report", content)
            self.assertIn("Overall Score", content)

    def test_html_contains_charts(self):
        orig, synth = _make_data()
        report = QualityReport(orig, synth).compute()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "report.html")
            report.to_html(path)
            with open(path) as f:
                content = f.read()
            # Should contain base64 images
            self.assertIn("data:image/png;base64", content)

    def test_auto_compute_on_to_html(self):
        """to_html should auto-compute if compute() wasn't called."""
        orig, synth = _make_data()
        report = QualityReport(orig, synth)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "report.html")
            report.to_html(path)
            self.assertTrue(os.path.exists(path))


if __name__ == "__main__":
    unittest.main()

# -*- coding: utf-8 -*-
"""Tests for AutoSynth."""

import pandas as pd
import pytest
from sklearn.datasets import load_iris

from tabgan.auto_synth import AutoSynth, AutoSynthResult


@pytest.fixture
def iris_df():
    data = load_iris(as_frame=True)
    df = data.frame
    df.columns = [c.replace(" (cm)", "").replace(" ", "_") for c in df.columns]
    return df


def test_autosynth_basic(iris_df):
    result = AutoSynth(iris_df, target_col="target").run(verbose=False)
    assert isinstance(result, AutoSynthResult)
    assert result.best_name in ("GAN (CTGAN)", "Forest Diffusion", "Random Baseline")
    assert result.best_score > 0
    assert len(result.best_data) > 0
    assert isinstance(result.report, pd.DataFrame)
    assert "Generator" in result.report.columns
    assert "Score" in result.report.columns


def test_autosynth_without_target(iris_df):
    df = iris_df.drop(columns=["target"])
    result = AutoSynth(df).run(verbose=False)
    assert isinstance(result, AutoSynthResult)
    assert len(result.best_data) > 0


def test_autosynth_custom_weights(iris_df):
    result = AutoSynth(
        iris_df,
        target_col="target",
        quality_weight=1.0,
        privacy_weight=0.0,
    ).run(verbose=False)
    assert isinstance(result, AutoSynthResult)


def test_autosynth_report_columns(iris_df):
    result = AutoSynth(iris_df, target_col="target").run(verbose=False)
    expected_cols = {"Generator", "Status", "Score", "Quality", "Privacy", "Rows", "Time (s)"}
    assert expected_cols == set(result.report.columns)

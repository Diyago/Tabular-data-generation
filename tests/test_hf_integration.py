# -*- coding: utf-8 -*-
"""Tests for HuggingFace integration."""

import pandas as pd
import pytest

from tabgan.hf_integration import synthesize_hf_dataset, SynthesizeResult
from tabgan.sampler import OriginalGenerator


def test_synthesize_hf_dataset_iris():
    """Test with a small well-known HF dataset."""
    result = synthesize_hf_dataset(
        "scikit-learn/iris",
        target_col="target",
        generator_class=OriginalGenerator,
        gen_x_times=1.0,
        evaluate=True,
    )
    assert isinstance(result, SynthesizeResult)
    assert len(result.synthetic_df) > 0
    assert len(result.original_df) > 0
    assert result.quality_summary is not None


def test_synthesize_hf_dataset_no_eval():
    result = synthesize_hf_dataset(
        "scikit-learn/iris",
        generator_class=OriginalGenerator,
        gen_x_times=1.0,
        evaluate=False,
    )
    assert isinstance(result, SynthesizeResult)
    assert result.quality_summary is None
    assert result.privacy_summary is None


def test_synthesize_hf_dataset_max_rows():
    result = synthesize_hf_dataset(
        "scikit-learn/iris",
        generator_class=OriginalGenerator,
        gen_x_times=1.0,
        max_rows=50,
        evaluate=False,
    )
    assert len(result.original_df) == 50

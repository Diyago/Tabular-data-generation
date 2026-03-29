# -*- coding: utf-8 -*-
"""HuggingFace Hub integration — synthesize any HF dataset in one call.

Usage::

    from tabgan import synthesize_hf_dataset

    # Generate synthetic version of any tabular HF dataset
    result = synthesize_hf_dataset("scikit-learn/iris", target_col="target")
    print(result.synthetic_df.head())

    # Push synthetic dataset back to Hub
    result = synthesize_hf_dataset(
        "scikit-learn/iris",
        target_col="target",
        push_to_hub=True,
        hub_repo_id="myuser/iris-synthetic",
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Type

import pandas as pd

from .sampler import (
    GANGenerator,
    ForestDiffusionGenerator,
    OriginalGenerator,
    BayesianGenerator,
    _BaseGenerator,
)

logger = logging.getLogger(__name__)


@dataclass
class SynthesizeResult:
    """Container for synthesize_hf_dataset results."""

    original_df: pd.DataFrame
    synthetic_df: pd.DataFrame
    quality_summary: Optional[Dict] = None
    privacy_summary: Optional[Dict] = None
    hub_url: Optional[str] = None


def synthesize_hf_dataset(
    dataset_name: str,
    *,
    subset: Optional[str] = None,
    split: str = "train",
    target_col: Optional[str] = None,
    cat_cols: Optional[List[str]] = None,
    generator_class: Type[_BaseGenerator] = GANGenerator,
    gen_x_times: float = 1.1,
    gen_params: Optional[Dict] = None,
    max_rows: int = 5000,
    evaluate: bool = True,
    push_to_hub: bool = False,
    hub_repo_id: Optional[str] = None,
    hub_token: Optional[str] = None,
    hub_private: bool = False,
) -> SynthesizeResult:
    """Synthesize any tabular dataset from HuggingFace Hub.

    Parameters
    ----------
    dataset_name : str
        HuggingFace dataset identifier (e.g. ``"scikit-learn/iris"``).
    subset : str, optional
        Dataset subset/config name.
    split : str
        Which split to use (default ``"train"``).
    target_col : str, optional
        Target column name for ML utility evaluation.
    cat_cols : list[str], optional
        Categorical column names.  Auto-detected if *None*.
    generator_class : type
        Generator class to use (default :class:`GANGenerator`).
    gen_x_times : float
        Synthetic-to-real ratio (default 1.1).
    gen_params : dict, optional
        Extra parameters for the generator.
    max_rows : int
        Maximum rows to sample from the original dataset (default 5000).
    evaluate : bool
        Whether to compute quality & privacy metrics (default True).
    push_to_hub : bool
        Push synthetic dataset to HuggingFace Hub (default False).
    hub_repo_id : str, optional
        Repository ID for the pushed dataset.  Defaults to
        ``"{dataset_name}-synthetic"``.
    hub_token : str, optional
        HuggingFace API token.  Uses cached token if *None*.
    hub_private : bool
        Whether the pushed dataset should be private (default False).

    Returns
    -------
    SynthesizeResult
        Contains original and synthetic DataFrames, metrics, and hub URL.
    """
    # --- Load dataset from HuggingFace ---
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The `datasets` package is required for HuggingFace integration. "
            "Install it with: pip install datasets"
        )

    print(f"  ▶ Loading dataset '{dataset_name}' from HuggingFace Hub ...")
    load_kwargs = {"path": dataset_name, "split": split}
    if subset:
        load_kwargs["name"] = subset

    hf_dataset = load_dataset(**load_kwargs)
    df = hf_dataset.to_pandas()

    print(f"    Loaded {len(df)} rows × {len(df.columns)} columns")

    # Subsample if too large
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)
        print(f"    Subsampled to {max_rows} rows")

    # Drop non-tabular columns (images, audio, etc.)
    tabular_cols = [
        c
        for c in df.columns
        if df[c].dtype.kind in ("i", "f", "u", "b", "O", "U", "S")
    ]
    if len(tabular_cols) < len(df.columns):
        dropped = set(df.columns) - set(tabular_cols)
        print(f"    Dropped non-tabular columns: {dropped}")
        df = df[tabular_cols]

    if len(df.columns) < 2:
        raise ValueError(
            f"Dataset has only {len(df.columns)} tabular columns — need at least 2."
        )

    # Auto-detect categorical columns
    if cat_cols is None:
        cat_cols = [
            c
            for c in df.columns
            if df[c].dtype == "object" or df[c].nunique() < 15
        ]

    # --- Generate synthetic data ---
    cat_cols_clean = [c for c in cat_cols if c != target_col]

    if target_col and target_col in df.columns:
        target_series = df[[target_col]]
        train_df = df.drop(columns=[target_col])
    else:
        target_series = None
        train_df = df.copy()

    gen_kwargs = {}
    if gen_params:
        gen_kwargs["gen_params"] = gen_params

    print(f"  ▶ Generating synthetic data with {generator_class.__name__} ...")
    generator = generator_class(
        gen_x_times=gen_x_times,
        cat_cols=cat_cols_clean or None,
        **gen_kwargs,
    )

    new_train, new_target = generator.generate_data_pipe(
        train_df, target_series, train_df, only_generated_data=True
    )

    if new_target is not None and isinstance(new_target, pd.Series):
        new_target = new_target.to_frame()

    if target_col and new_target is not None and len(new_target.columns) > 0:
        synthetic_df = pd.concat([new_train, new_target], axis=1)
    else:
        synthetic_df = new_train

    print(f"    Generated {len(synthetic_df)} synthetic rows")

    # --- Evaluate ---
    quality_summary = None
    privacy_summary = None

    if evaluate:
        print("  ▶ Evaluating quality & privacy ...")
        try:
            from .quality_report import QualityReport

            shared_cols = [c for c in df.columns if c in synthetic_df.columns]
            qr = QualityReport(
                df[shared_cols],
                synthetic_df[shared_cols],
                cat_cols=[c for c in cat_cols if c in shared_cols],
                target_col=(
                    target_col
                    if target_col and target_col in shared_cols
                    else None
                ),
            )
            qr.compute()
            quality_summary = qr.summary()
            q_score = quality_summary.get("overall_score", "N/A")
            print(f"    Quality score: {q_score}")
        except Exception as e:
            logger.warning("Quality evaluation failed: %s", e)

        try:
            import numpy as np
            from .privacy_metrics import PrivacyMetrics

            shared_cols = [c for c in df.columns if c in synthetic_df.columns]
            num_cols = (
                df[shared_cols]
                .select_dtypes(include=[np.number])
                .columns.tolist()
            )
            if len(num_cols) >= 2:
                pm = PrivacyMetrics(
                    df[shared_cols],
                    synthetic_df[shared_cols],
                    cat_cols=[c for c in cat_cols if c in shared_cols],
                )
                privacy_summary = pm.summary()
                p_score = privacy_summary.get("overall_privacy_score", "N/A")
                print(f"    Privacy score: {p_score}")
        except Exception as e:
            logger.warning("Privacy evaluation failed: %s", e)

    # --- Push to Hub ---
    hub_url = None
    if push_to_hub:
        try:
            from datasets import Dataset
            from huggingface_hub import HfApi
        except ImportError:
            raise ImportError(
                "The `datasets` and `huggingface_hub` packages are required "
                "to push to Hub.  Install with: pip install datasets huggingface_hub"
            )

        repo_id = hub_repo_id or f"{dataset_name}-synthetic"
        print(f"  ▶ Pushing to HuggingFace Hub as '{repo_id}' ...")

        synth_dataset = Dataset.from_pandas(synthetic_df)
        synth_dataset.push_to_hub(
            repo_id,
            token=hub_token,
            private=hub_private,
            commit_message=f"Synthetic data generated by tabgan from {dataset_name}",
        )
        hub_url = f"https://huggingface.co/datasets/{repo_id}"
        print(f"    Published: {hub_url}")

    result = SynthesizeResult(
        original_df=df,
        synthetic_df=synthetic_df,
        quality_summary=quality_summary,
        privacy_summary=privacy_summary,
        hub_url=hub_url,
    )

    print("  ✓ Done!")
    return result

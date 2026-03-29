# -*- coding: utf-8 -*-
"""AutoSynth — automatically select the best synthetic data generator.

Usage::

    from tabgan import AutoSynth

    result = AutoSynth(df, target_col="label").run()
    print(result.best_name)          # e.g. "GAN (CTGAN)"
    print(result.best_score)         # overall quality score
    synthetic_df = result.best_data  # best synthetic DataFrame
    result.report                    # comparison table of all generators
"""

from __future__ import annotations

import logging
import time
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .sampler import (
    GANGenerator,
    ForestDiffusionGenerator,
    OriginalGenerator,
    BayesianGenerator,
)

logger = logging.getLogger(__name__)


@dataclass
class _GeneratorResult:
    name: str
    synthetic_df: Optional[pd.DataFrame] = None
    synthetic_target: Optional[pd.DataFrame] = None
    quality: Optional[Dict] = None
    privacy: Optional[Dict] = None
    score: float = 0.0
    elapsed: float = 0.0
    error: Optional[str] = None


@dataclass
class AutoSynthResult:
    """Container for AutoSynth results."""

    best_name: str
    best_score: float
    best_data: pd.DataFrame
    best_target: Optional[pd.DataFrame]
    best_quality: Dict
    best_privacy: Dict
    report: pd.DataFrame
    all_results: Dict[str, _GeneratorResult] = field(repr=False)


# Default generator configurations
_DEFAULT_GENERATORS = {
    "GAN (CTGAN)": {
        "cls": GANGenerator,
        "params": {"gen_params": {"epochs": 30, "patience": 15, "batch_size": 50}},
    },
    "Forest Diffusion": {
        "cls": ForestDiffusionGenerator,
        "params": {},
    },
    "Bayesian (Copula)": {
        "cls": BayesianGenerator,
        "params": {},
    },
    "Random Baseline": {
        "cls": OriginalGenerator,
        "params": {},
    },
}


class AutoSynth:
    """Run multiple generators, evaluate quality & privacy, pick the best.

    Parameters
    ----------
    df : pd.DataFrame
        Original dataset.
    target_col : str, optional
        Name of the target column (for ML utility evaluation).
    cat_cols : list[str], optional
        Categorical column names.  Auto-detected if *None*.
    gen_x_times : float
        Synthetic-to-real ratio (default 1.1).
    generators : dict, optional
        Custom generator configs.  Keys are display names, values are dicts
        with ``"cls"`` (generator class) and ``"params"`` (extra kwargs).
        Falls back to CTGAN + ForestDiffusion + Random baseline.
    quality_weight : float
        Weight for quality score in ranking (0-1, default 0.7).
    privacy_weight : float
        Weight for privacy score in ranking (0-1, default 0.3).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        cat_cols: Optional[List[str]] = None,
        gen_x_times: float = 1.1,
        generators: Optional[Dict] = None,
        quality_weight: float = 0.7,
        privacy_weight: float = 0.3,
    ):
        self.df = df
        self.target_col = target_col
        self.gen_x_times = gen_x_times
        self.quality_weight = quality_weight
        self.privacy_weight = privacy_weight
        self.generators = generators or _DEFAULT_GENERATORS

        # Auto-detect categorical columns
        if cat_cols is None:
            self.cat_cols = [
                c
                for c in df.columns
                if df[c].dtype == "object" or df[c].nunique() < 15
            ]
        else:
            self.cat_cols = cat_cols

    def run(self, verbose: bool = True) -> AutoSynthResult:
        """Execute all generators and return the best result."""
        results: Dict[str, _GeneratorResult] = {}

        for name, cfg in self.generators.items():
            if verbose:
                logger.info("Running %s ...", name)
                print(f"  ▶ Running {name} ...")

            result = self._run_single(name, cfg)
            results[name] = result

            if verbose:
                if result.error:
                    print(f"    ✗ {name} failed: {result.error[:80]}")
                else:
                    print(
                        f"    ✓ {name} — score: {result.score:.3f} "
                        f"({result.elapsed:.1f}s)"
                    )

        # Build comparison report
        report = self._build_report(results)

        # Pick best
        valid = {k: v for k, v in results.items() if v.error is None}
        if not valid:
            raise RuntimeError(
                "All generators failed. Errors:\n"
                + "\n".join(f"  {k}: {v.error}" for k, v in results.items())
            )

        best_name = max(valid, key=lambda k: valid[k].score)
        best = valid[best_name]

        if verbose:
            print(f"\n  ★ Best generator: {best_name} (score {best.score:.3f})")

        return AutoSynthResult(
            best_name=best_name,
            best_score=best.score,
            best_data=best.synthetic_df,
            best_target=best.synthetic_target,
            best_quality=best.quality or {},
            best_privacy=best.privacy or {},
            report=report,
            all_results=results,
        )

    def _run_single(self, name: str, cfg: Dict) -> _GeneratorResult:
        """Run a single generator and evaluate."""
        t0 = time.time()
        result = _GeneratorResult(name=name)

        try:
            cls = cfg["cls"]
            extra = cfg.get("params", {})

            cat_cols_clean = [
                c for c in self.cat_cols if c != self.target_col
            ]

            generator = cls(
                gen_x_times=self.gen_x_times,
                cat_cols=cat_cols_clean or None,
                **extra,
            )

            # Prepare data
            if self.target_col and self.target_col in self.df.columns:
                target_series = self.df[[self.target_col]]
                train_df = self.df.drop(columns=[self.target_col])
            else:
                target_series = None
                train_df = self.df.copy()

            new_train, new_target = generator.generate_data_pipe(
                train_df, target_series, train_df, only_generated_data=True
            )

            # Ensure new_target is a DataFrame
            if new_target is not None and isinstance(new_target, pd.Series):
                new_target = new_target.to_frame()

            # Reconstruct full synthetic df
            if (
                self.target_col
                and new_target is not None
                and len(new_target.columns) > 0
            ):
                synthetic_df = pd.concat([new_train, new_target], axis=1)
            else:
                synthetic_df = new_train

            result.synthetic_df = synthetic_df
            result.synthetic_target = new_target

        except Exception:
            result.error = traceback.format_exc()
            result.elapsed = time.time() - t0
            return result

        # Evaluate quality
        try:
            from .quality_report import QualityReport

            shared_cols = [
                c for c in self.df.columns if c in synthetic_df.columns
            ]
            qr = QualityReport(
                self.df[shared_cols],
                synthetic_df[shared_cols],
                cat_cols=[c for c in self.cat_cols if c in shared_cols],
                target_col=(
                    self.target_col
                    if self.target_col and self.target_col in shared_cols
                    else None
                ),
            )
            qr.compute()
            result.quality = qr.summary()
        except Exception:
            result.quality = {"overall_score": 0.0, "error": traceback.format_exc()}

        # Evaluate privacy
        try:
            import numpy as np
            from .privacy_metrics import PrivacyMetrics

            shared_cols = [
                c for c in self.df.columns if c in synthetic_df.columns
            ]
            num_cols = (
                self.df[shared_cols]
                .select_dtypes(include=[np.number])
                .columns.tolist()
            )
            if len(num_cols) >= 2:
                pm = PrivacyMetrics(
                    self.df[shared_cols],
                    synthetic_df[shared_cols],
                    cat_cols=[c for c in self.cat_cols if c in shared_cols],
                )
                result.privacy = pm.summary()
            else:
                result.privacy = {"overall_privacy_score": 0.5}
        except Exception:
            result.privacy = {
                "overall_privacy_score": 0.5,
                "error": traceback.format_exc(),
            }

        # Composite score
        q_score = self._extract_quality_score(result.quality)
        p_score = self._extract_privacy_score(result.privacy)
        result.score = (
            self.quality_weight * q_score + self.privacy_weight * p_score
        )

        result.elapsed = time.time() - t0
        return result

    @staticmethod
    def _extract_quality_score(quality: Dict) -> float:
        score = quality.get("overall_score", 0.0)
        if isinstance(score, (int, float)):
            return float(score)
        return 0.0

    @staticmethod
    def _extract_privacy_score(privacy: Dict) -> float:
        score = privacy.get("overall_privacy_score", 0.5)
        if isinstance(score, (int, float)):
            return float(score)
        return 0.5

    @staticmethod
    def _build_report(results: Dict[str, _GeneratorResult]) -> pd.DataFrame:
        rows = []
        for name, r in results.items():
            row = {"Generator": name}
            if r.error:
                row["Status"] = "FAILED"
                row["Score"] = None
                row["Quality"] = None
                row["Privacy"] = None
                row["Rows"] = None
                row["Time (s)"] = round(r.elapsed, 1)
            else:
                row["Status"] = "OK"
                row["Score"] = round(r.score, 3)
                row["Quality"] = round(
                    AutoSynth._extract_quality_score(r.quality or {}), 3
                )
                row["Privacy"] = round(
                    AutoSynth._extract_privacy_score(r.privacy or {}), 3
                )
                row["Rows"] = (
                    len(r.synthetic_df) if r.synthetic_df is not None else 0
                )
                row["Time (s)"] = round(r.elapsed, 1)
            rows.append(row)

        df = pd.DataFrame(rows)
        df = df.sort_values("Score", ascending=False, na_position="last")
        return df.reset_index(drop=True)

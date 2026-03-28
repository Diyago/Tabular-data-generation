# -*- coding: utf-8 -*-
"""
Privacy metrics for assessing re-identification risk in synthetic data.

Provides Distance to Closest Record (DCR), Nearest Neighbor Distance Ratio
(NNDR), and a membership inference risk score.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

__all__ = ["PrivacyMetrics"]


def _encode_for_distance(
    original: pd.DataFrame,
    synthetic: pd.DataFrame,
    cat_cols: Optional[List[str]] = None,
) -> tuple:
    """Encode and scale DataFrames for distance computation."""
    original = original.copy()
    synthetic = synthetic.copy()

    if cat_cols:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        original[cat_cols] = encoder.fit_transform(original[cat_cols].astype(str))
        synthetic[cat_cols] = encoder.transform(synthetic[cat_cols].astype(str))

    # Fill NaN with column medians
    for col in original.columns:
        if original[col].isna().any():
            med = original[col].median()
            original[col] = original[col].fillna(med)
            synthetic[col] = synthetic[col].fillna(med)

    scaler = StandardScaler()
    orig_scaled = scaler.fit_transform(original.select_dtypes(include=[np.number]))
    synth_scaled = scaler.transform(synthetic.select_dtypes(include=[np.number]))

    return orig_scaled, synth_scaled


class PrivacyMetrics:
    """Evaluate privacy risk of synthetic data relative to original data.

    Args:
        original_df: The real / training DataFrame.
        synthetic_df: The generated / synthetic DataFrame.
        cat_cols: Names of categorical columns (encoded before distance computation).

    Example::

        from tabgan.privacy_metrics import PrivacyMetrics
        pm = PrivacyMetrics(original_df, synthetic_df, cat_cols=["gender"])
        print(pm.summary())
    """

    def __init__(
        self,
        original_df: pd.DataFrame,
        synthetic_df: pd.DataFrame,
        cat_cols: Optional[List[str]] = None,
    ):
        shared_cols = [c for c in original_df.columns if c in synthetic_df.columns]
        self.original_df = original_df[shared_cols].copy()
        self.synthetic_df = synthetic_df[shared_cols].copy()
        self.cat_cols = [c for c in (cat_cols or []) if c in shared_cols]
        self._orig_scaled, self._synth_scaled = _encode_for_distance(
            self.original_df, self.synthetic_df, self.cat_cols
        )

    # ------------------------------------------------------------------
    # DCR — Distance to Closest Record
    # ------------------------------------------------------------------
    def dcr(self, sample_size: Optional[int] = None) -> Dict:
        """Compute the distance from each synthetic row to the nearest original row.

        Higher distances indicate better privacy (synthetic rows are not
        trivially close to any real record).

        Returns:
            dict with ``mean``, ``median``, ``5th_percentile``, and ``distances``.
        """
        synth = self._synth_scaled
        if sample_size and sample_size < len(synth):
            idx = np.random.choice(len(synth), sample_size, replace=False)
            synth = synth[idx]

        nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
        nn.fit(self._orig_scaled)
        distances, _ = nn.kneighbors(synth)
        distances = distances.ravel()

        return {
            "mean": float(np.mean(distances)),
            "median": float(np.median(distances)),
            "5th_percentile": float(np.percentile(distances, 5)),
            "distances": distances,
        }

    # ------------------------------------------------------------------
    # NNDR — Nearest Neighbor Distance Ratio
    # ------------------------------------------------------------------
    def nndr(self, sample_size: Optional[int] = None) -> Dict:
        """Nearest-neighbor distance ratio for each synthetic row.

        Ratio = dist(nearest_original) / dist(2nd_nearest_original).
        A ratio close to 1 means the synthetic row is equidistant to
        multiple originals (lower risk); a ratio near 0 means it is
        suspiciously close to exactly one real record.

        Returns:
            dict with ``mean``, ``median``, and ``ratios``.
        """
        synth = self._synth_scaled
        if sample_size and sample_size < len(synth):
            idx = np.random.choice(len(synth), sample_size, replace=False)
            synth = synth[idx]

        k = min(2, len(self._orig_scaled))
        nn = NearestNeighbors(n_neighbors=k, algorithm="auto")
        nn.fit(self._orig_scaled)
        distances, _ = nn.kneighbors(synth)

        if k < 2:
            ratios = np.ones(len(synth))
        else:
            d1 = distances[:, 0]
            d2 = np.where(distances[:, 1] == 0, 1e-10, distances[:, 1])
            ratios = d1 / d2

        return {
            "mean": float(np.mean(ratios)),
            "median": float(np.median(ratios)),
            "ratios": ratios,
        }

    # ------------------------------------------------------------------
    # Membership Inference Risk
    # ------------------------------------------------------------------
    def membership_inference_risk(self, holdout_frac: float = 0.3) -> Dict:
        """Estimate membership inference risk.

        Splits the original data into a *member* set (simulating the training
        data the generator saw) and a *holdout* set. If the generator
        memorised the members, synthetic rows will be closer to members than
        to holdout rows. The risk is quantified as the AUC of a simple
        classifier trained on this distance signal.

        Returns:
            dict with ``auc`` (0.5 = good privacy, 1.0 = full memorisation)
            and ``accuracy``.
        """
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import cross_val_predict
        from sklearn.linear_model import LogisticRegression

        n = len(self._orig_scaled)
        n_holdout = max(int(n * holdout_frac), 1)
        perm = np.random.permutation(n)
        member_idx = perm[n_holdout:]
        holdout_idx = perm[:n_holdout]

        members = self._orig_scaled[member_idx]
        holdout = self._orig_scaled[holdout_idx]

        # For each original row, compute distance to nearest synthetic
        nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
        nn.fit(self._synth_scaled)

        d_members, _ = nn.kneighbors(members)
        d_holdout, _ = nn.kneighbors(holdout)

        X = np.concatenate([d_members.ravel(), d_holdout.ravel()]).reshape(-1, 1)
        y = np.concatenate([np.ones(len(d_members)), np.zeros(len(d_holdout))])

        if len(np.unique(y)) < 2:
            return {"auc": 0.5, "accuracy": 0.5}

        clf = LogisticRegression(solver="lbfgs", max_iter=200)
        try:
            proba = cross_val_predict(clf, X, y, cv=min(3, len(y)), method="predict_proba")[:, 1]
            auc = float(roc_auc_score(y, proba))
        except Exception:
            auc = 0.5

        accuracy = float(np.mean((proba > 0.5) == y)) if 'proba' in dir() else 0.5

        return {"auc": auc, "accuracy": accuracy}

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    def summary(self) -> Dict:
        """Aggregate all privacy metrics into a single report.

        The ``overall_privacy_score`` ranges from 0 (high risk) to 1 (private).
        """
        dcr_res = self.dcr()
        nndr_res = self.nndr()
        mi_res = self.membership_inference_risk()

        # Score components (each normalised to 0-1, higher = more private)
        # DCR: 5th percentile > 0 is good; cap contribution at 1
        dcr_score = min(dcr_res["5th_percentile"], 1.0)

        # NNDR: mean closer to 1 is better
        nndr_score = min(nndr_res["mean"], 1.0)

        # MI: AUC closer to 0.5 is better → score = 1 - 2*|AUC - 0.5|
        mi_score = max(1.0 - 2.0 * abs(mi_res["auc"] - 0.5), 0.0)

        overall = 0.4 * dcr_score + 0.3 * nndr_score + 0.3 * mi_score

        return {
            "dcr": {k: v for k, v in dcr_res.items() if k != "distances"},
            "nndr": {k: v for k, v in nndr_res.items() if k != "ratios"},
            "membership_inference": mi_res,
            "overall_privacy_score": round(overall, 4),
        }

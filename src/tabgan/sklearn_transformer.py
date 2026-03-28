# -*- coding: utf-8 -*-
"""
sklearn-compatible transformer for TabGAN data augmentation.

Allows inserting synthetic data generation into a ``sklearn.pipeline.Pipeline``.
"""

import logging
from typing import List, Optional, Type

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from tabgan.sampler import GANGenerator

__all__ = ["TabGANTransformer"]


class TabGANTransformer(BaseEstimator, TransformerMixin):
    """Augment training data with TabGAN synthetic rows inside an sklearn Pipeline.

    During ``fit`` the generator is trained and synthetic data produced.
    ``transform`` returns the augmented DataFrame (original + synthetic).

    Because sklearn's ``transform`` only returns X, the augmented target
    is available via :meth:`get_augmented_target` after ``fit_transform``.

    Args:
        generator_class: A TabGAN generator class (e.g. ``GANGenerator``).
        gen_x_times: Multiplier for synthetic sample count.
        cat_cols: Categorical column names.
        gen_params: Generator-specific hyperparameters.
        only_generated_data: If True, return only synthetic rows.
        constraints: Optional list of ``Constraint`` instances.
        use_adversarial: Whether to use adversarial filtering.
        **generator_kwargs: Extra keyword arguments forwarded to the generator.

    Example::

        from sklearn.pipeline import Pipeline
        from sklearn.ensemble import RandomForestClassifier
        from tabgan.sklearn_transformer import TabGANTransformer

        pipe = Pipeline([
            ("augment", TabGANTransformer(gen_x_times=1.5)),
            ("model", RandomForestClassifier()),
        ])
        pipe.fit(X_train, y_train)
    """

    def __init__(
        self,
        generator_class: Type = None,
        gen_x_times: float = 1.1,
        cat_cols: Optional[List[str]] = None,
        gen_params: Optional[dict] = None,
        only_generated_data: bool = False,
        constraints: Optional[list] = None,
        use_adversarial: bool = True,
        **generator_kwargs,
    ):
        self.generator_class = generator_class
        self.gen_x_times = gen_x_times
        self.cat_cols = cat_cols
        self.gen_params = gen_params
        self.only_generated_data = only_generated_data
        self.constraints = constraints
        self.use_adversarial = use_adversarial
        self.generator_kwargs = generator_kwargs

        # Internal state (set after fit)
        self._augmented_X: Optional[pd.DataFrame] = None
        self._augmented_y: Optional[pd.Series] = None

    def fit(self, X, y=None):
        """Train the generator and produce synthetic data.

        Args:
            X: Training features (DataFrame or ndarray).
            y: Target variable (Series, DataFrame, ndarray, or None).
        """
        gen_cls = self.generator_class or GANGenerator

        X_df = pd.DataFrame(X).copy() if not isinstance(X, pd.DataFrame) else X.copy()

        target_df = None
        if y is not None:
            if isinstance(y, pd.DataFrame):
                target_df = y.copy()
            elif isinstance(y, pd.Series):
                target_df = y.to_frame().copy()
            else:
                target_df = pd.DataFrame(y, columns=["target"])

        gen_kwargs = dict(
            gen_x_times=self.gen_x_times,
            cat_cols=self.cat_cols,
            only_generated_data=self.only_generated_data,
        )
        if self.gen_params is not None:
            gen_kwargs["gen_params"] = self.gen_params
        gen_kwargs.update(self.generator_kwargs)

        generator = gen_cls(**gen_kwargs)

        new_train, new_target = generator.generate_data_pipe(
            X_df,
            target_df,
            X_df,  # use train as test for distribution alignment
            use_adversarial=self.use_adversarial,
            constraints=self.constraints,
        )

        self._augmented_X = new_train
        if new_target is not None and not new_target.isna().all():
            self._augmented_y = (
                new_target.iloc[:, 0] if isinstance(new_target, pd.DataFrame) else new_target
            )
        else:
            self._augmented_y = None

        return self

    def transform(self, X, y=None):
        """Return the augmented training data.

        During training (when ``_augmented_X`` is available), returns the
        augmented data. At inference time, returns X unchanged.
        """
        if self._augmented_X is not None:
            result = self._augmented_X
            # Clear after first transform to avoid leaking into predict
            self._augmented_X = None
            return result
        return X

    def fit_transform(self, X, y=None, **fit_params):
        """Fit and return augmented data in one step."""
        self.fit(X, y)
        return self.transform(X, y)

    def get_augmented_target(self) -> Optional[pd.Series]:
        """Return the augmented target produced during ``fit``.

        Call this after ``fit`` or ``fit_transform`` to get the target
        values corresponding to the augmented training data.
        """
        return self._augmented_y

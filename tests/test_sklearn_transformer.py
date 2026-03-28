# -*- coding: utf-8 -*-
"""Tests for the sklearn-compatible TabGANTransformer."""

import unittest

import numpy as np
import pandas as pd

from src.tabgan.sampler import OriginalGenerator, GANGenerator
from src.tabgan.sklearn_transformer import TabGANTransformer
from src.tabgan.constraints import RangeConstraint


class TestTabGANTransformerBasic(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(42)
        self.X = pd.DataFrame(rng.randint(0, 100, size=(60, 3)), columns=list("ABC"))
        self.y = pd.Series(rng.randint(0, 2, 60), name="target")

    def test_fit_transform_returns_dataframe(self):
        t = TabGANTransformer(generator_class=OriginalGenerator, gen_x_times=1.1)
        result = t.fit_transform(self.X, self.y)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result), 0)

    def test_augmented_target_available(self):
        t = TabGANTransformer(generator_class=OriginalGenerator, gen_x_times=1.1)
        t.fit(self.X, self.y)
        aug_y = t.get_augmented_target()
        self.assertIsNotNone(aug_y)

    def test_augmented_shapes_aligned(self):
        t = TabGANTransformer(generator_class=OriginalGenerator, gen_x_times=1.1)
        X_aug = t.fit_transform(self.X, self.y)
        y_aug = t.get_augmented_target()
        self.assertEqual(len(X_aug), len(y_aug))

    def test_without_target(self):
        t = TabGANTransformer(generator_class=OriginalGenerator, gen_x_times=1.1)
        result = t.fit_transform(self.X)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIsNone(t.get_augmented_target())

    def test_transform_at_inference_passthrough(self):
        """After first transform, subsequent transforms pass data through."""
        t = TabGANTransformer(generator_class=OriginalGenerator, gen_x_times=1.1)
        t.fit(self.X, self.y)
        _ = t.transform(self.X)  # First transform consumes augmented data
        result2 = t.transform(self.X)  # Second should pass through
        self.assertEqual(len(result2), len(self.X))


class TestTabGANTransformerWithGAN(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(42)
        self.X = pd.DataFrame(rng.randint(0, 100, size=(60, 3)), columns=list("ABC"))
        self.y = pd.Series(rng.randint(0, 2, 60), name="target")

    def test_gan_generator(self):
        t = TabGANTransformer(
            generator_class=GANGenerator,
            gen_x_times=1.0,
            gen_params={"batch_size": 50, "patience": 5, "epochs": 2},
        )
        result = t.fit_transform(self.X, self.y)
        self.assertGreater(len(result), 0)
        self.assertEqual(result.shape[1], self.X.shape[1])


class TestTabGANTransformerWithConstraints(unittest.TestCase):
    def test_constraints_applied(self):
        rng = np.random.RandomState(42)
        X = pd.DataFrame(rng.randint(-50, 200, size=(60, 3)), columns=list("ABC"))
        y = pd.Series(rng.randint(0, 2, 60))

        t = TabGANTransformer(
            generator_class=OriginalGenerator,
            gen_x_times=1.5,
            constraints=[RangeConstraint("A", min_val=0, max_val=100)],
        )
        result = t.fit_transform(X, y)
        self.assertGreaterEqual(result["A"].min(), 0)
        self.assertLessEqual(result["A"].max(), 100)


class TestTabGANTransformerSklearnCompat(unittest.TestCase):
    def test_get_params(self):
        t = TabGANTransformer(gen_x_times=2.0, cat_cols=["A"])
        params = t.get_params()
        self.assertEqual(params["gen_x_times"], 2.0)
        self.assertEqual(params["cat_cols"], ["A"])

    def test_set_params(self):
        t = TabGANTransformer(gen_x_times=1.0)
        t.set_params(gen_x_times=3.0)
        self.assertEqual(t.gen_x_times, 3.0)

    def test_in_sklearn_pipeline(self):
        """Verify it can be placed in a Pipeline (doesn't run the full pipeline)."""
        from sklearn.pipeline import Pipeline
        from sklearn.ensemble import RandomForestClassifier

        pipe = Pipeline([
            ("augment", TabGANTransformer(
                generator_class=OriginalGenerator,
                gen_x_times=1.1,
            )),
            ("model", RandomForestClassifier(n_estimators=5, random_state=42)),
        ])
        # Pipeline should be constructable and have proper steps
        self.assertEqual(len(pipe.steps), 2)
        self.assertEqual(pipe.steps[0][0], "augment")


if __name__ == "__main__":
    unittest.main()

# -*- coding: utf-8 -*-
"""
Tests for generate_data_pipe parameter combinations and cat_cols handling
in postprocess / adversarial filtering.
"""

import unittest

import numpy as np
import pandas as pd

from src.tabgan.sampler import (
    OriginalGenerator,
    GANGenerator,
    ForestDiffusionGenerator,
    SamplerOriginal,
)


def _make_data(n_train=80, n_test=80, seed=42):
    """Create reproducible train / target / test DataFrames."""
    rng = np.random.RandomState(seed)
    train = pd.DataFrame(rng.randint(0, 100, size=(n_train, 4)), columns=list("ABCD"))
    target = pd.DataFrame(rng.randint(0, 2, size=(n_train, 1)), columns=["Y"])
    test = pd.DataFrame(rng.randint(0, 100, size=(n_test, 4)), columns=list("ABCD"))
    return train, target, test


def _make_data_with_cat(n_train=80, n_test=80, seed=42):
    """Create data with an explicit categorical column."""
    rng = np.random.RandomState(seed)
    train = pd.DataFrame({
        "num1": rng.randint(0, 100, n_train),
        "num2": rng.randint(0, 100, n_train),
        "cat": rng.choice(["X", "Y", "Z"], n_train),
    })
    target = pd.DataFrame({"Y": rng.randint(0, 2, n_train)})
    test = pd.DataFrame({
        "num1": rng.randint(0, 100, n_test),
        "num2": rng.randint(0, 100, n_test),
        "cat": rng.choice(["X", "Y", "Z"], n_test),
    })
    return train, target, test


# ---------------------------------------------------------------------------
# generate_data_pipe parameter combinations
# ---------------------------------------------------------------------------
class TestGenerateDataPipeParams(unittest.TestCase):
    """Test various parameter combinations of generate_data_pipe."""

    def test_only_adversarial_true(self):
        """only_adversarial=True should skip generation and only filter."""
        train, target, test = _make_data()
        new_train, new_target = OriginalGenerator(gen_x_times=1.1).generate_data_pipe(
            train, target, test,
            only_adversarial=True,
            use_adversarial=True,
        )
        self.assertEqual(new_train.shape[0], new_target.shape[0])
        # With only adversarial filtering on original data, output rows <= input rows
        self.assertLessEqual(new_train.shape[0], train.shape[0])

    def test_use_adversarial_false(self):
        """use_adversarial=False should skip adversarial filtering entirely."""
        train, target, test = _make_data()
        new_train, new_target = OriginalGenerator(gen_x_times=1.5).generate_data_pipe(
            train, target, test,
            use_adversarial=False,
        )
        self.assertEqual(new_train.shape[0], new_target.shape[0])
        # Without adversarial filtering, we should have more rows than original
        self.assertGreater(new_train.shape[0], train.shape[0])

    def test_deep_copy_false(self):
        """deep_copy=False should still produce valid output."""
        train, target, test = _make_data()
        new_train, new_target = OriginalGenerator(gen_x_times=1.1).generate_data_pipe(
            train.copy(), target.copy(), test.copy(),
            deep_copy=False,
        )
        self.assertEqual(new_train.shape[0], new_target.shape[0])
        self.assertGreater(new_train.shape[0], 0)

    def test_only_generated_data_true_original(self):
        """only_generated_data=True with OriginalGenerator — data is sampled from train."""
        train, target, test = _make_data()
        new_train, new_target = OriginalGenerator(
            gen_x_times=1.1,
            only_generated_data=True,
        ).generate_data_pipe(
            train, target, test,
            only_generated_data=True,
        )
        self.assertEqual(new_train.shape[0], new_target.shape[0])

    def test_only_generated_data_true_gan(self):
        """only_generated_data=True with GANGenerator returns purely synthetic rows."""
        train, target, test = _make_data()
        new_train, new_target = GANGenerator(
            gen_x_times=1.0,
            only_generated_data=True,
            gen_params={"batch_size": 50, "patience": 5, "epochs": 2},
        ).generate_data_pipe(
            train, target, test,
            only_generated_data=True,
        )
        self.assertEqual(new_train.shape[0], new_target.shape[0])
        self.assertGreater(new_train.shape[0], 0)
        self.assertEqual(new_train.shape[1], train.shape[1])

    def test_target_none(self):
        """Passing target=None should work for all generators."""
        train, _, test = _make_data()
        new_train, new_target = OriginalGenerator(gen_x_times=1.1).generate_data_pipe(
            train, None, test,
        )
        self.assertIsNotNone(new_train)
        self.assertGreater(new_train.shape[0], 0)

    def test_test_df_none(self):
        """Passing test_df=None should skip postprocess and adversarial."""
        train, _, _ = _make_data()
        new_train, new_target = OriginalGenerator(
            gen_x_times=1.1,
            is_post_process=False,
        ).generate_data_pipe(
            train, None, None,
        )
        self.assertIsNotNone(new_train)
        self.assertGreater(new_train.shape[0], 0)


# ---------------------------------------------------------------------------
# cat_cols in postprocess and adversarial filtering
# ---------------------------------------------------------------------------
class TestCatColsPostprocessAndAdversarial(unittest.TestCase):
    """Test that cat_cols are handled correctly in postprocess and adversarial."""

    def test_postprocess_with_cat_cols(self):
        """Postprocessing with cat_cols should filter by category membership."""
        train, target, test = _make_data_with_cat()

        sampler = OriginalGenerator(
            gen_x_times=2.0,
            cat_cols=["cat"],
        ).get_object_generator()

        new_train, new_target, test_df = sampler.preprocess_data(
            train.copy(), target.copy(), test.copy()
        )
        gen_train, gen_target = sampler.generate_data(
            new_train, new_target, test_df, only_generated_data=False
        )
        post_train, post_target = sampler.postprocess_data(gen_train, gen_target, test_df)

        self.assertEqual(post_train.shape[0], post_target.shape[0])
        # All categorical values in result should be present in test
        result_cats = set(post_train["cat"].unique())
        test_cats = set(test_df["cat"].unique())
        self.assertTrue(result_cats.issubset(test_cats))

    def test_adversarial_with_cat_cols(self):
        """Adversarial filtering with cat_cols should produce valid output."""
        train, target, test = _make_data_with_cat()

        sampler = OriginalGenerator(
            gen_x_times=2.0,
            cat_cols=["cat"],
        ).get_object_generator()

        new_train, new_target, test_df = sampler.preprocess_data(
            train.copy(), target.copy(), test.copy()
        )
        gen_train, gen_target = sampler.generate_data(
            new_train, new_target, test_df, only_generated_data=False
        )
        post_train, post_target = sampler.postprocess_data(gen_train, gen_target, test_df)
        adv_train, adv_target = sampler.adversarial_filtering(post_train, post_target, test_df)

        self.assertEqual(adv_train.shape[0], adv_target.shape[0])
        self.assertGreater(adv_train.shape[0], 0)

    def test_full_pipeline_with_cat_cols(self):
        """End-to-end generate_data_pipe with cat_cols."""
        train, target, test = _make_data_with_cat()
        new_train, new_target = OriginalGenerator(
            gen_x_times=1.5,
            cat_cols=["cat"],
        ).generate_data_pipe(train, target, test)

        self.assertEqual(new_train.shape[0], new_target.shape[0])
        self.assertGreater(new_train.shape[0], 0)
        self.assertIn("cat", new_train.columns)

    def test_gan_with_cat_cols(self):
        """GANGenerator with cat_cols should train and generate correctly."""
        train, target, test = _make_data_with_cat()
        new_train, new_target = GANGenerator(
            gen_x_times=1.1,
            cat_cols=["cat"],
            gen_params={"batch_size": 50, "patience": 5, "epochs": 2},
        ).generate_data_pipe(train, target, test)

        self.assertEqual(new_train.shape[0], new_target.shape[0])
        self.assertGreater(new_train.shape[0], 0)


if __name__ == "__main__":
    unittest.main()

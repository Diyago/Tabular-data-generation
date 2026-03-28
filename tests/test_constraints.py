# -*- coding: utf-8 -*-
"""Tests for the constraint system."""

import unittest

import numpy as np
import pandas as pd

from src.tabgan.constraints import (
    RangeConstraint,
    UniqueConstraint,
    FormulaConstraint,
    RegexConstraint,
    ConstraintEngine,
)
from src.tabgan.sampler import OriginalGenerator


class TestRangeConstraint(unittest.TestCase):
    def test_is_satisfied(self):
        df = pd.DataFrame({"age": [5, 25, 150, -1]})
        c = RangeConstraint("age", min_val=0, max_val=120)
        mask = c.is_satisfied(df)
        self.assertEqual(list(mask), [True, True, False, False])

    def test_fix_clips_values(self):
        df = pd.DataFrame({"age": [5, 25, 150, -1]})
        c = RangeConstraint("age", min_val=0, max_val=120)
        fixed = c.fix(df)
        self.assertEqual(list(fixed["age"]), [5, 25, 120, 0])

    def test_min_only(self):
        df = pd.DataFrame({"x": [-5, 0, 10]})
        c = RangeConstraint("x", min_val=0)
        self.assertEqual(list(c.is_satisfied(df)), [False, True, True])

    def test_max_only(self):
        df = pd.DataFrame({"x": [-5, 0, 10]})
        c = RangeConstraint("x", max_val=5)
        self.assertEqual(list(c.is_satisfied(df)), [True, True, False])

    def test_requires_at_least_one_bound(self):
        with self.assertRaises(ValueError):
            RangeConstraint("x")


class TestUniqueConstraint(unittest.TestCase):
    def test_is_satisfied(self):
        df = pd.DataFrame({"id": [1, 2, 3, 2, 1]})
        c = UniqueConstraint("id")
        mask = c.is_satisfied(df)
        # First occurrences are True, duplicates are False
        self.assertEqual(list(mask), [True, True, True, False, False])

    def test_fix_drops_duplicates(self):
        df = pd.DataFrame({"id": [1, 2, 3, 2, 1], "val": [10, 20, 30, 40, 50]})
        c = UniqueConstraint("id")
        fixed = c.fix(df)
        self.assertEqual(len(fixed), 3)
        self.assertEqual(list(fixed["id"]), [1, 2, 3])


class TestFormulaConstraint(unittest.TestCase):
    def test_is_satisfied(self):
        df = pd.DataFrame({"start": [1, 5, 10], "end": [10, 3, 15]})
        c = FormulaConstraint("end > start")
        mask = c.is_satisfied(df)
        self.assertEqual(list(mask), [True, False, True])

    def test_fix_filters_violations(self):
        df = pd.DataFrame({"start": [1, 5, 10], "end": [10, 3, 15]})
        c = FormulaConstraint("end > start")
        fixed = c.fix(df)
        self.assertEqual(len(fixed), 2)
        self.assertEqual(list(fixed["start"]), [1, 10])


class TestRegexConstraint(unittest.TestCase):
    def test_is_satisfied(self):
        df = pd.DataFrame({"email": ["a@b.com", "invalid", "x@y.org"]})
        c = RegexConstraint("email", r".+@.+\..+")
        mask = c.is_satisfied(df)
        self.assertEqual(list(mask), [True, False, True])

    def test_fix_filters(self):
        df = pd.DataFrame({"code": ["AB12", "XY34", "bad!"]})
        c = RegexConstraint("code", r"[A-Z]{2}\d{2}")
        fixed = c.fix(df)
        self.assertEqual(len(fixed), 2)
        self.assertEqual(list(fixed["code"]), ["AB12", "XY34"])


class TestConstraintEngine(unittest.TestCase):
    def test_filter_strategy(self):
        df = pd.DataFrame({"age": [5, 150, 30], "id": [1, 2, 3]})
        engine = ConstraintEngine(
            [RangeConstraint("age", min_val=0, max_val=120)],
            strategy="filter",
        )
        result = engine.apply(df)
        self.assertEqual(len(result), 2)

    def test_fix_strategy(self):
        df = pd.DataFrame({"age": [5, 150, 30], "id": [1, 2, 3]})
        engine = ConstraintEngine(
            [RangeConstraint("age", min_val=0, max_val=120)],
            strategy="fix",
        )
        result = engine.apply(df)
        self.assertEqual(len(result), 3)  # All rows kept after clipping
        self.assertEqual(result["age"].max(), 120)

    def test_multiple_constraints(self):
        df = pd.DataFrame({
            "age": [5, 150, 30, 25, 25],
            "id": [1, 2, 3, 4, 4],
        })
        engine = ConstraintEngine([
            RangeConstraint("age", min_val=0, max_val=120),
            UniqueConstraint("id"),
        ], strategy="fix")
        result = engine.apply(df)
        # After fix: age clipped, then unique id kept
        self.assertTrue(result["age"].max() <= 120)
        self.assertEqual(result["id"].nunique(), len(result))

    def test_invalid_strategy_raises(self):
        with self.assertRaises(ValueError):
            ConstraintEngine([], strategy="invalid")


class TestConstraintsInPipeline(unittest.TestCase):
    def test_generate_data_pipe_with_constraints(self):
        rng = np.random.RandomState(42)
        train = pd.DataFrame(rng.randint(-10, 200, size=(60, 3)), columns=list("ABC"))
        target = pd.DataFrame(rng.randint(0, 2, size=(60, 1)), columns=["Y"])
        test = pd.DataFrame(rng.randint(0, 100, size=(60, 3)), columns=list("ABC"))

        constraints = [RangeConstraint("A", min_val=0, max_val=100)]

        new_train, new_target = OriginalGenerator(gen_x_times=1.5).generate_data_pipe(
            train, target, test, constraints=constraints,
        )

        self.assertEqual(new_train.shape[0], new_target.shape[0])
        self.assertGreaterEqual(new_train["A"].min(), 0)
        self.assertLessEqual(new_train["A"].max(), 100)


if __name__ == "__main__":
    unittest.main()

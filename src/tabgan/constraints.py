# -*- coding: utf-8 -*-
"""
Constraint system for enforcing business rules on generated data.

Constraints are applied as a post-generation step — after the main
generation pipeline produces synthetic rows, the ConstraintEngine filters
or repairs rows that violate the declared rules.
"""

import logging
import re
from abc import ABC, abstractmethod
from typing import List, Optional

import pandas as pd

__all__ = [
    "Constraint",
    "RangeConstraint",
    "UniqueConstraint",
    "FormulaConstraint",
    "RegexConstraint",
    "ConstraintEngine",
]


class Constraint(ABC):
    """Base class for data constraints."""

    @abstractmethod
    def is_satisfied(self, df: pd.DataFrame) -> pd.Series:
        """Return a boolean Series — True for rows that satisfy the constraint."""
        raise NotImplementedError

    @abstractmethod
    def fix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attempt to repair violating rows in-place and return the DataFrame."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class RangeConstraint(Constraint):
    """Enforce numeric column values within [min_val, max_val]."""

    def __init__(self, column: str, min_val: float = None, max_val: float = None):
        if min_val is None and max_val is None:
            raise ValueError("At least one of min_val or max_val must be specified")
        self.column = column
        self.min_val = min_val
        self.max_val = max_val

    def is_satisfied(self, df: pd.DataFrame) -> pd.Series:
        col = df[self.column]
        mask = pd.Series(True, index=df.index)
        if self.min_val is not None:
            mask &= col >= self.min_val
        if self.max_val is not None:
            mask &= col <= self.max_val
        return mask

    def fix(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[self.column] = df[self.column].clip(lower=self.min_val, upper=self.max_val)
        return df

    def __repr__(self) -> str:
        return f"RangeConstraint(column={self.column!r}, min={self.min_val}, max={self.max_val})"


class UniqueConstraint(Constraint):
    """Enforce uniqueness of values in a column (drop duplicate rows)."""

    def __init__(self, column: str):
        self.column = column

    def is_satisfied(self, df: pd.DataFrame) -> pd.Series:
        return ~df[self.column].duplicated(keep="first")

    def fix(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop_duplicates(subset=[self.column], keep="first").reset_index(drop=True)

    def __repr__(self) -> str:
        return f"UniqueConstraint(column={self.column!r})"


class FormulaConstraint(Constraint):
    """Enforce a boolean expression evaluated via ``pd.DataFrame.eval``.

    Example expressions:
        - ``"end_date > start_date"``
        - ``"price * quantity == total"``
        - ``"age >= 0"``
    """

    def __init__(self, expression: str):
        self.expression = expression

    def is_satisfied(self, df: pd.DataFrame) -> pd.Series:
        return df.eval(self.expression)

    def fix(self, df: pd.DataFrame) -> pd.DataFrame:
        mask = self.is_satisfied(df)
        return df[mask].reset_index(drop=True)

    def __repr__(self) -> str:
        return f"FormulaConstraint({self.expression!r})"


class RegexConstraint(Constraint):
    """Enforce that string values in a column match a regular expression."""

    def __init__(self, column: str, pattern: str):
        self.column = column
        self.pattern = pattern
        self._compiled = re.compile(pattern)

    def is_satisfied(self, df: pd.DataFrame) -> pd.Series:
        return df[self.column].astype(str).str.fullmatch(self.pattern).fillna(False)

    def fix(self, df: pd.DataFrame) -> pd.DataFrame:
        mask = self.is_satisfied(df)
        return df[mask].reset_index(drop=True)

    def __repr__(self) -> str:
        return f"RegexConstraint(column={self.column!r}, pattern={self.pattern!r})"


class ConstraintEngine:
    """Apply a list of constraints to a DataFrame.

    Args:
        constraints: List of ``Constraint`` instances to enforce.
        strategy: ``"filter"`` drops violating rows; ``"fix"`` attempts
            repair first, then filters remaining violations.
    """

    def __init__(self, constraints: List[Constraint], strategy: str = "filter"):
        if strategy not in ("filter", "fix"):
            raise ValueError(f"strategy must be 'filter' or 'fix', got {strategy!r}")
        self.constraints = constraints
        self.strategy = strategy

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        initial_len = len(df)
        for constraint in self.constraints:
            if self.strategy == "fix":
                df = constraint.fix(df)
            # After fix (or directly if filter), drop remaining violations
            mask = constraint.is_satisfied(df)
            df = df[mask].reset_index(drop=True)

        dropped = initial_len - len(df)
        if dropped > 0:
            logging.info(f"ConstraintEngine: dropped {dropped} rows ({dropped / initial_len:.1%})")
        return df

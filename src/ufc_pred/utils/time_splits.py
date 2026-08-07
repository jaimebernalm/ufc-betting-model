"""Frozen time-based train/val/test splits.

DO NOT CHANGE THESE DATES once models start being compared. The evaluation
protocol must be stable across versions or comparisons are meaningless.
See PLAN.md §2.3 and §2.5.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRAIN_END = pd.Timestamp("2022-12-31")
VAL_END = pd.Timestamp("2023-12-31")
TEST_END = pd.Timestamp("2026-03-28")

RECENCY_HALF_LIFE_YEARS = 4.0


@dataclass
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def summary(self) -> pd.DataFrame:
        rows = []
        for name, df in [("train", self.train), ("val", self.val), ("test", self.test)]:
            rows.append(
                {
                    "split": name,
                    "n_fights": len(df),
                    "first_date": df["date"].min().date(),
                    "last_date": df["date"].max().date(),
                }
            )
        return pd.DataFrame(rows)


def split(df: pd.DataFrame, date_col: str = "date") -> Splits:
    """Split a fights dataframe by date. Input must be sorted/sortable by date_col."""
    df = df.sort_values(date_col).reset_index(drop=True)
    d = df[date_col]
    train = df[d <= TRAIN_END].copy()
    val = df[(d > TRAIN_END) & (d <= VAL_END)].copy()
    test = df[(d > VAL_END) & (d <= TEST_END)].copy()
    return Splits(train=train, val=val, test=test)


def recency_weights(
    dates: pd.Series,
    reference_date: pd.Timestamp = TRAIN_END,
    half_life_years: float = RECENCY_HALF_LIFE_YEARS,
) -> np.ndarray:
    """Exponential-decay weights for older fights.

    weight = 0.5 ** (years_before_reference / half_life_years)

    Fights at reference_date get weight 1.0; fights `half_life_years` earlier
    get 0.5; fights `2 * half_life_years` earlier get 0.25. Etc.
    """
    age_years = (reference_date - pd.to_datetime(dates)).dt.days / 365.25
    age_years = age_years.clip(lower=0)
    return np.power(0.5, age_years / half_life_years).to_numpy()

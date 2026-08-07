"""Feature joins shared across the model ladder.

Every model version differs mainly in *which extra columns get attached to the
fight table* and *which of those columns must flip sign under the symmetry
augmentation*. Both concerns live here so the individual `baseline_v*.py`
modules stay declarative.

Symmetry note: `features.static_v1.prepare(augment_symmetry=True)` concatenates
[original, red/blue-swapped] and flips the label. Any column expressing a
*signed difference* (Red minus Blue) must be negated on the swapped half;
magnitude-only columns (standard deviations) must not.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from ufc_pred.features.derived_v2 import compute as _compute_derived
from ufc_pred.features.skill_v3_1_pipeline import OUTPUT as SKILL_V3_1_PARQUET
from ufc_pred.features.skill_v3_pipeline import OUTPUT as SKILL_V3_PARQUET

_KEYS = ["date", "R_fighter", "B_fighter"]
_SKILL_COLS = ["skill_diff_mean", "skill_diff_std"]


def _merge_skill(fights: pd.DataFrame, parquet_path, rename: dict[str, str] | None = None) -> pd.DataFrame:
    """Left-join a skill-features parquet onto the fight table on (date, R, B)."""
    df = fights.copy()
    df["date"] = pd.to_datetime(df["date"])

    skill = pd.read_parquet(parquet_path)
    if rename:
        skill = skill.rename(columns=rename)
    skill["date"] = pd.to_datetime(skill["date"])

    value_cols = [rename[c] for c in _SKILL_COLS] if rename else list(_SKILL_COLS)
    return df.merge(
        skill[_KEYS + value_cols],
        on=_KEYS,
        how="left",
        validate="many_to_one",
    )


def join_skill_v3(fights: pd.DataFrame) -> pd.DataFrame:
    """Scalar Bradley-Terry skill posterior (v3)."""
    return _merge_skill(fights, SKILL_V3_PARQUET)


def join_skill_v3_1(fights: pd.DataFrame) -> pd.DataFrame:
    """Time-varying random-walk skill posterior (v3.1). Same column names as v3."""
    return _merge_skill(fights, SKILL_V3_1_PARQUET)


def join_skill_stacked(fights: pd.DataFrame) -> pd.DataFrame:
    """Both skill versions side by side, suffixed to avoid collision (v3.3)."""
    df = _merge_skill(
        fights,
        SKILL_V3_PARQUET,
        rename={"skill_diff_mean": "skill_diff_mean_v3", "skill_diff_std": "skill_diff_std_v3"},
    )
    return _merge_skill(
        df,
        SKILL_V3_1_PARQUET,
        rename={
            "skill_diff_mean": "skill_diff_mean_v3_1",
            "skill_diff_std": "skill_diff_std_v3_1",
        },
    )


def join_derived(fights: pd.DataFrame) -> pd.DataFrame:
    """Derived no-scrape features: layoff, activity, career ratios (v2)."""
    return _compute_derived(fights)


def join_skill_v3_and_derived(fights: pd.DataFrame) -> pd.DataFrame:
    """v3.2 = scalar skill + derived. Skill is joined first, matching the
    original v3.2 ordering (derived features read columns skill does not touch,
    so the order is cosmetic — but it is preserved to keep runs reproducible)."""
    return join_derived(join_skill_v3(fights))


def flip_signed_columns(X: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Negate `columns` on the swapped half of a symmetry-augmented matrix.

    `prepare()` emits [original (n rows), swapped (n rows)]. If `X` is not an
    even split it was not augmented (the val/test path) and is returned as-is.
    """
    n = len(X) // 2
    if len(X) != 2 * n or not columns:
        return X
    X = X.copy()
    second = X.index[n:]
    for c in columns:
        if c in X.columns:
            X.loc[second, c] = -X.loc[second, c]
    return X

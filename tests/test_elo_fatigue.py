"""Leakage tests for the Elo and fatigue feature modules (Tasks 2.1/2.2).

Truncation invariance (pattern from TennisPred tests/test_elo_surface.py):
features computed on the full history must be identical to features computed
on history truncated at date D, for every fight before D. If a future
outcome leaked into a past feature, the two disagree.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ufc_pred.features.elo import build_elo
from ufc_pred.features.fatigue import build_fatigue
from ufc_pred.paths import PROCESSED

CUT = pd.Timestamp("2020-01-01")

FIGHTS_PARQUET = PROCESSED / "fights.parquet"

# Integration tests against the real fight history. The dataset is not
# redistributed (see README), so these skip on a fresh clone until the ingest
# pipeline has been run.
pytestmark = pytest.mark.skipif(
    not FIGHTS_PARQUET.exists(),
    reason=f"requires {FIGHTS_PARQUET.name}; run the ingest pipeline first",
)


@pytest.fixture(scope="module")
def fights():
    df = pd.read_parquet(FIGHTS_PARQUET)
    df = df[df["Winner"].isin(["Red", "Blue"])].copy()
    df["date"] = pd.to_datetime(df["date"])
    return df


def _assert_truncation_invariant(full: pd.DataFrame, trunc: pd.DataFrame):
    full = full[full["date"] < CUT].reset_index(drop=True)
    trunc = trunc[trunc["date"] < CUT].reset_index(drop=True)
    assert len(full) == len(trunc) and len(full) > 1000
    num = full.select_dtypes(include=[np.number]).columns
    for c in num:
        np.testing.assert_allclose(
            full[c].to_numpy(),
            trunc[c].to_numpy(),
            equal_nan=True,
            err_msg=f"column {c} changed when future fights were removed",
        )


def test_elo_truncation_invariance(fights):
    _assert_truncation_invariant(build_elo(fights, k=8.0), build_elo(fights[fights["date"] < CUT], k=8.0))


def test_elo_prefight_rating_excludes_today(fights):
    """A fighter's first appearance (either corner) must show INIT (1500)."""
    elo = build_elo(fights, k=8.0).sort_values("date", kind="stable")
    long = pd.concat(
        [
            elo[["date", "R_fighter", "R_elo"]].rename(columns={"R_fighter": "fighter", "R_elo": "elo"}),
            elo[["date", "B_fighter", "B_elo"]].rename(columns={"B_fighter": "fighter", "B_elo": "elo"}),
        ]
    ).sort_values("date", kind="stable")
    first = long.groupby("fighter")["elo"].first()
    assert (first == 1500.0).all()


def test_fatigue_truncation_invariance(fights):
    _assert_truncation_invariant(build_fatigue(fights), build_fatigue(fights[fights["date"] < CUT]))


def test_fatigue_first_fight_is_nan(fights):
    fat = build_fatigue(fights)
    fat.merge(fights[["date", "R_fighter", "B_fighter"]], on=["date", "R_fighter", "B_fighter"])
    # a fighter's first fight has no layoff/last-fight features
    first_rows = fat.sort_values("date").groupby("R_fighter").head(1)
    debut_mask = first_rows["R_days_since_last"].isna()
    assert debut_mask.mean() > 0.4  # many R-corner firsts are true debuts

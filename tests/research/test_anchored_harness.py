"""Invariants of the Oct-2026 research harness and market-anchored models."""

import numpy as np
import pandas as pd
import pytest

from ufc_pred.research import anchored as A
from ufc_pred.research.harness import bets_at_book_odds, load_base, logit, roi_summary

pytestmark = pytest.mark.skipif(
    not A.__file__
    or not __import__("pathlib")
    .Path("artifacts/shadow_candidate_2026_09_14_v2/training_features.parquet")
    .exists(),
    reason="needs the local dataset",
)


@pytest.fixture(scope="module")
def base():
    return load_base()


def _swap_rows(t: pd.DataFrame) -> pd.DataFrame:
    sw = A._swap(t, [])
    sw["Winner"] = sw["Winner"].map({"Red": "Blue", "Blue": "Red"})
    sw["y"] = 1 - t["y"].to_numpy()
    sw["debut"] = t["debut"].to_numpy()
    return sw


@pytest.mark.parametrize(
    "make",
    [
        lambda: A.make_resid_fn(
            logit_q_feature=True, flb_baseline=True, params=dict(A.RESID_V0, iterations=30)
        ),
        lambda: A.make_glm_resid_fn(),
        lambda: A.make_flb_fn(),
    ],
)
def test_corner_symmetry(base, make):
    train = base[(base.year < 2019) & base.has_odds].reset_index(drop=True)
    test = base[(base.year == 2019) & base.has_odds].head(60).reset_index(drop=True)
    fn = make()
    p = fn(train, test, 0)
    p_sw = fn(train, _swap_rows(test).reset_index(drop=True), 0)
    assert np.allclose(p, 1 - p_sw, atol=1e-9)


def test_flb_slope_uses_only_training_rows(base):
    train = base[base.year < 2019].reset_index(drop=True)
    a1 = A.fit_flb_slope(train)
    poisoned = pd.concat([train, base[base.year >= 2019]], ignore_index=True)
    a2 = A.fit_flb_slope(poisoned[poisoned.date < "2019-01-01"])
    assert a1 == pytest.approx(a2)
    assert 0.8 < a1 < 1.5


def test_nan_prediction_is_no_bet_and_keeps_opportunity(base):
    d = base[base.has_odds & (base.year == 2020)].head(20).copy()
    d["p_x"] = np.where(np.arange(len(d)) % 2 == 0, np.nan, d.q + 0.2)
    b = bets_at_book_odds(d, "p_x")
    assert len(b) == len(d)
    assert not b.bet.to_numpy()[::2].any()


def test_zero_bets_roi_undefined(base):
    d = base[base.has_odds & (base.year == 2020)].head(20).copy()
    d["p_x"] = d.q  # never beats the vig
    r = roi_summary(d, "p_x", 0.0)
    assert r["bets"] == 0 and np.isnan(r["roi"]) and r["pnl_per_opp"] == 0.0  # only 0 bets is undefined


def test_logit_q_roundtrip():
    q = np.array([0.1, 0.5, 0.9])
    assert np.allclose(1 / (1 + np.exp(-logit(q))), q)


def test_harness_metadata_never_reaches_features(base):
    from ufc_pred.research.harness import load_base as lb

    df = lb(fill_bfo_from="2024-01-01")
    X, _, _, _ = A._matrix(df.head(50), False, [])
    for c in ["q", "has_odds", "y", "card", "year", "debut", "odds_source", "Winner", "R_odds", "B_odds"]:
        assert c not in X.columns

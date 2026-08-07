"""Unit tests for bet_eval. Hand-built fights with known answers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ufc_pred.backtest.bet_eval import (
    american_to_decimal,
    evaluate_bets,
    evaluate_bets_kelly,
    evaluate_bets_sweep,
    probability_edge,
)


def test_american_to_decimal_basic():
    out = american_to_decimal(pd.Series([-150, +200, -110, +100]))
    np.testing.assert_allclose(out, [1.6667, 3.0, 1.9091, 2.0], atol=1e-3)


def test_american_to_decimal_handles_nan():
    out = american_to_decimal(pd.Series([np.nan, -150, None]))
    assert np.isnan(out[0])
    assert np.isnan(out[2])
    np.testing.assert_allclose(out[1], 1.6667, atol=1e-3)


def test_bets_one_fight_winning_bet():
    # Red @ +200 (decimal 3.0). Model says 50% Red. EV = 0.5*3 - 1 = +0.5.
    # Threshold 0.03 → bet. Red wins → realized = 3-1 = +2.
    r = evaluate_bets(
        model_prob_red=np.array([0.5]),
        y_red=np.array([1]),
        R_odds=pd.Series([200]),
        B_odds=pd.Series([-300]),
        edge_threshold=0.03,
    )
    assert r.n_bets == 1
    assert r.n_bets_red == 1
    assert r.hit_rate == 1.0
    np.testing.assert_allclose(r.roi_pct, 200.0, atol=1e-3)
    np.testing.assert_allclose(r.mean_ev_pct, 50.0, atol=1e-3)


def test_bets_one_fight_losing_bet():
    # Same setup but Blue wins.
    r = evaluate_bets(
        model_prob_red=np.array([0.5]),
        y_red=np.array([0]),
        R_odds=pd.Series([200]),
        B_odds=pd.Series([-300]),
        edge_threshold=0.03,
    )
    assert r.n_bets == 1
    assert r.hit_rate == 0.0
    np.testing.assert_allclose(r.roi_pct, -100.0, atol=1e-3)


def test_no_bet_below_threshold():
    # Decimal 2.0 for both sides (vig-free). Model says 50/50 → EV = 0 on both.
    # Threshold 0.03 → no bet.
    r = evaluate_bets(
        model_prob_red=np.array([0.5]),
        y_red=np.array([1]),
        R_odds=pd.Series([100]),  # decimal 2.0
        B_odds=pd.Series([100]),  # decimal 2.0
        edge_threshold=0.03,
    )
    assert r.n_bets == 0
    assert r.total_pnl == 0.0


def test_symmetry_swap_red_blue():
    # Same fight, encoded with R and B swapped. Model probs and labels mirror.
    # Should produce identical aggregate stats.
    p_red = np.array([0.7])
    y = np.array([1])
    R_odds = pd.Series([-150])
    B_odds = pd.Series([+130])

    r1 = evaluate_bets(p_red, y, R_odds, B_odds, edge_threshold=0.0)
    r2 = evaluate_bets(1 - p_red, 1 - y, B_odds, R_odds, edge_threshold=0.0)

    assert r1.n_bets == r2.n_bets
    np.testing.assert_allclose(r1.roi_pct, r2.roi_pct, atol=1e-6)
    np.testing.assert_allclose(r1.mean_ev_pct, r2.mean_ev_pct, atol=1e-6)
    np.testing.assert_allclose(r1.total_pnl, r2.total_pnl, atol=1e-6)


def test_fee_reduces_winnings():
    # +200 = decimal 3.0. With 10% fee on winnings, effective decimal = 1 + 0.9*2 = 2.8.
    # Model 50% Red → EV = 0.5*2.8 - 1 = +0.4 (vs +0.5 with no fee).
    r = evaluate_bets(
        model_prob_red=np.array([0.5]),
        y_red=np.array([1]),
        R_odds=pd.Series([200]),
        B_odds=pd.Series([-300]),
        edge_threshold=0.03,
        fee_rate=0.10,
    )
    np.testing.assert_allclose(r.mean_ev_pct, 40.0, atol=1e-3)
    # Won: realized = 2.8 - 1 = 1.8.
    np.testing.assert_allclose(r.roi_pct, 180.0, atol=1e-3)


def test_skips_fights_with_missing_odds():
    r = evaluate_bets(
        model_prob_red=np.array([0.6, 0.7]),
        y_red=np.array([1, 1]),
        R_odds=pd.Series([np.nan, -150]),
        B_odds=pd.Series([+130, +130]),
        edge_threshold=0.0,
    )
    assert r.n_eligible == 1
    assert r.n_bets <= 1


def test_probability_edge_no_vig():
    # Vig-free book: R @ +100 and B @ +100. Implied 50/50, no vig. Edge = model - 0.5.
    edge = probability_edge(
        model_prob_red=np.array([0.65, 0.5, 0.3]),
        R_odds=pd.Series([100, 100, 100]),
        B_odds=pd.Series([100, 100, 100]),
    )
    np.testing.assert_allclose(edge, [0.15, 0.0, -0.20], atol=1e-6)


def test_sweep_monotone_n_bets():
    # n_bets should be non-increasing as threshold rises.
    rng = np.random.default_rng(0)
    n = 100
    p = rng.uniform(0.2, 0.8, size=n)
    y = (rng.uniform(size=n) < p).astype(int)
    R_odds = rng.choice([-150, -110, +100, +130, +200], size=n)
    B_odds = rng.choice([-150, -110, +100, +130, +200], size=n)

    df = evaluate_bets_sweep(p, y, pd.Series(R_odds), pd.Series(B_odds))
    nb = df["n_bets"].to_numpy()
    assert all(nb[i] >= nb[i + 1] for i in range(len(nb) - 1)), (
        f"n_bets should be non-increasing with threshold: {nb}"
    )


def test_no_vig_uses_fair_decimal():
    # Vig-loaded book: R -120, B -120. Implied 0.545 each, sum 1.091.
    # No-vig: 0.5 each → fair decimal = 2.0 each side.
    # Model says 60% Red → EV (no-vig) = 0.6*2.0 - 1 = +0.2 (vs ~0.1 with vig).
    r_vig = evaluate_bets(
        model_prob_red=np.array([0.6]),
        y_red=np.array([1]),
        R_odds=pd.Series([-120]),
        B_odds=pd.Series([-120]),
        edge_threshold=0.0,
        use_no_vig=False,
    )
    r_nv = evaluate_bets(
        model_prob_red=np.array([0.6]),
        y_red=np.array([1]),
        R_odds=pd.Series([-120]),
        B_odds=pd.Series([-120]),
        edge_threshold=0.0,
        use_no_vig=True,
    )
    # No-vig EV is higher (fairer payout)
    assert r_nv.mean_ev_pct > r_vig.mean_ev_pct
    # No-vig payout: decimal 2.0, won → realized = 1.0 → 100% ROI on this single bet
    np.testing.assert_allclose(r_nv.roi_pct, 100.0, atol=1e-3)
    np.testing.assert_allclose(r_nv.mean_ev_pct, 20.0, atol=1e-3)


def test_kalshi_like_no_vig_plus_fee():
    # Kalshi-like: no vig, 7% fee on winnings.
    # R -120 / B -120 → no-vig decimal 2.0. 7% fee → eff dec = 1 + 0.93*1 = 1.93.
    # Model 60% Red → EV = 0.6 * 1.93 - 1 = +0.158.
    r = evaluate_bets(
        model_prob_red=np.array([0.6]),
        y_red=np.array([1]),
        R_odds=pd.Series([-120]),
        B_odds=pd.Series([-120]),
        edge_threshold=0.0,
        use_no_vig=True,
        fee_rate=0.07,
    )
    np.testing.assert_allclose(r.mean_ev_pct, 15.8, atol=1e-3)
    np.testing.assert_allclose(r.roi_pct, 93.0, atol=1e-3)  # 1.93 - 1 = 0.93


def test_kelly_no_bet_when_no_edge():
    # Model = market exactly → no positive EV, no bets, bankroll unchanged.
    r = evaluate_bets_kelly(
        model_prob_red=np.array([0.5]),
        y_red=np.array([1]),
        R_odds=pd.Series([100]),
        B_odds=pd.Series([100]),
        edge_threshold=0.0,
        kelly_fraction=0.25,
        starting_bankroll=100.0,
    )
    assert r["n_bets"] == 0
    np.testing.assert_allclose(r["final_bankroll"], 100.0)


def test_kelly_caps_at_max_bet_fraction():
    # Huge edge: model says 95% on a +200 bet. Full Kelly would stake heavily.
    # max_bet_fraction = 0.02 should cap us at 2% of bankroll.
    r = evaluate_bets_kelly(
        model_prob_red=np.array([0.95]),
        y_red=np.array([1]),  # we win
        R_odds=pd.Series([200]),  # decimal 3.0
        B_odds=pd.Series([-1000]),
        edge_threshold=0.03,
        kelly_fraction=1.0,
        max_bet_fraction=0.02,
        starting_bankroll=100.0,
    )
    # Stake = 100 * 0.02 = 2. Win 2 * (3.0 - 1) = 4. Final bankroll = 104.
    np.testing.assert_allclose(r["final_bankroll"], 104.0, atol=1e-6)

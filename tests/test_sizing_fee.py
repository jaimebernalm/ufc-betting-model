import pytest

from ufc_pred.inference.sizing import (
    AccountConfig,
    _fee_per_contract,
    _full_kelly,
    size_bets_combined,
)


def test_quadratic_fee_coefficient_is_venue_specific():
    assert _fee_per_contract(0.5, 0.07) == pytest.approx(0.0175)
    assert _fee_per_contract(0.5, 0.03) == pytest.approx(0.0075)
    assert _full_kelly(0.70, 0.50, 0.03) > _full_kelly(0.70, 0.50, 0.07)


def test_combined_sizing_uses_requested_fee_coefficient():
    account = AccountConfig("A", bankroll=300.0, kelly_fraction=0.10, bankroll_cap=None, model="real")
    asks_a = [(0.50, 10_000.0)]
    asks_b = [(0.52, 10_000.0)]

    kalshi = size_bets_combined([account], {"A": 0.70}, asks_a, asks_b, fee_coeff=0.07)[0]
    polymarket = size_bets_combined([account], {"A": 0.70}, asks_a, asks_b, fee_coeff=0.03)[0]

    assert kalshi.decision == polymarket.decision == "BET"
    assert polymarket.stake_usd > kalshi.stake_usd
    assert polymarket.fee_usd / polymarket.shares == pytest.approx(0.0075, abs=0.001)

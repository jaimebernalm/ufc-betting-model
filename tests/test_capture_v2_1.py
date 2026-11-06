"""Capture v2.1: UFC.com name fallback and shadow lead-time trigger (no network)."""

import pandas as pd

from ufc_pred.cli import bet_runner as r
from ufc_pred.inference.kalshi_card import CardFight
from ufc_pred.ingest.ufc_schedule import FightSchedule

START = pd.Timestamp("2026-10-24 20:00", tz="UTC")
HISTORY = pd.DataFrame(
    {
        "R_fighter": ["Alexander Volkanovski", "Movsar Evloev"],
        "B_fighter": ["Diego Lopes", "Arnold Allen"],
        "weight_class": ["Featherweight", "Featherweight"],
        "date": pd.to_datetime(["2026-01-31", "2025-03-22"]),
    }
)


def _kalshi(a: str, b: str) -> CardFight:
    return CardFight(
        event_title="UFC 333",
        fight_date=START,
        fighter_a=a,
        fighter_b=b,
        token_a="T-A",
        token_b="T-B",
        market_id="T",
        asks_a=[],
        asks_b=[],
        bids_a=[],
        bids_b=[],
        volume=None,
        liquidity=None,
    )


def _schedule(a: str, b: str) -> FightSchedule:
    return FightSchedule(a, b, START, "main_card", "Featherweight")


def test_unresolvable_kalshi_name_falls_back_to_ufc_spelling():
    fight = _schedule("Alexander Volkanovski", "Movsar Evloev")
    out = r.with_resolvable_names(_kalshi("Alex Volkanovski", "Movsar Evloev"), fight, False, HISTORY)
    assert (out.fighter_a, out.fighter_b) == ("Alexander Volkanovski", "Movsar Evloev")
    # Kalshi listing the corners the other way round: the fallback follows the swap.
    out = r.with_resolvable_names(_kalshi("Movsar Evloev", "Alex Volkanovski"), fight, True, HISTORY)
    assert (out.fighter_a, out.fighter_b) == ("Movsar Evloev", "Alexander Volkanovski")


def test_resolvable_kalshi_name_and_debutant_are_left_alone():
    kalshi = _kalshi("Alexander Volkanovski", "Movsar Evloev")
    fight = _schedule("Alex Volkanovski", "Movsar Evloev")
    assert r.with_resolvable_names(kalshi, fight, False, HISTORY) is kalshi
    debut = _kalshi("Joe Kropschot", "Movsar Evloev")
    out = r.with_resolvable_names(debut, _schedule("Joseph Kropschot", "Movsar Evloev"), False, HISTORY)
    assert out.fighter_a == "Joe Kropschot"


def _trigger(now, prev_live=True):
    opener = FightSchedule("A One", "B One", START - pd.Timedelta(minutes=30), "prelim", "Featherweight")
    ordered = [opener, _schedule("A Two", "B Two")]
    return r.fight_trigger(
        1,
        ordered[1],
        ordered,
        now,
        prev_resolution=lambda f: None,
        prev_market_live=lambda f: prev_live,
        this_resolution=lambda f: None,
    )


def test_scheduled_lead_only_in_shadow_mode(monkeypatch):
    at_lead = START - pd.Timedelta(minutes=r.FIRST_FIGHT_CAPTURE_MIN)
    monkeypatch.setattr(r, "CAPTURE_ALL_AT_LEAD", False)
    assert _trigger(at_lead) is None
    monkeypatch.setattr(r, "CAPTURE_ALL_AT_LEAD", True)
    assert _trigger(at_lead - pd.Timedelta(minutes=1)) is None
    assert _trigger(at_lead) == "scheduled_lead"

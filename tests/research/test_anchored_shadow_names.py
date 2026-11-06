"""Name handling in the paper logger when sources spell a fighter differently (no network)."""

import pandas as pd

import ufc_pred.research.anchored_shadow as S


def _cap(fighter_a, fighter_b, red, blue, swap=False):
    return {
        "fight": {"fighter_a_ufc": fighter_a, "fighter_b_ufc": fighter_b, "kalshi_swap_vs_ufc": swap},
        "kalshi_snapshot": {"a_ask": 0.80, "b_ask": 0.22, "a_bid": 0.79, "b_bid": 0.21},
        "features": {"forward": [{"R_fighter": red, "B_fighter": blue}]},
    }


def test_asks_orient_through_blue_when_red_is_spelled_differently():
    cap = _cap("Zachary Reese", "Mickey Gall", "Zach Reese", "Mickey Gall")
    assert S.kalshi_asks_for(cap, "Zach Reese")["R_ask"] == 0.80
    cap = _cap("Mickey Gall", "Zachary Reese", "Zach Reese", "Mickey Gall")
    assert S.kalshi_asks_for(cap, "Zach Reese")["R_ask"] == 0.22
    # Same surname on both sides still orients by the exact names.
    cap = _cap("Yana Santos", "Luana Santos", "Luana Santos", "Yana Santos")
    assert S.kalshi_asks_for(cap, "Luana Santos")["R_ask"] == 0.22


def test_bfo_matchup_found_by_surnames_and_relabelled(monkeypatch):
    rows = [
        {
            "matchup_id": 7,
            "fighter_name": n,
            "fighter_pos": pos,
            "book_name": "DraftKings",
            "american_odds": o,
        }
        for n, pos, o in [("Alex Volkanovski", 1, -120), ("Movsar Evloev", 2, 100)]
    ]
    monkeypatch.setattr(
        S.bfo, "list_event_urls", lambda refresh: [{"slug": "ufc-333", "url": "u", "lastmod": "2026-10-24"}]
    )
    monkeypatch.setattr(S.bfo, "fetch", lambda url, refresh: "page")
    monkeypatch.setattr(S.bfo, "parse_event", lambda page: {"rows": rows})
    slug, _, g, mu = S.find_bfo_event(pd.Timestamp("2026-10-24"), ("Alexander Volkanovski", "Movsar Evloev"))
    assert (slug, mu) == ("ufc-333", 7)
    assert set(g.fighter_name) == {"Alexander Volkanovski", "Movsar Evloev"}
    assert S.book_pair(g, "Alexander Volkanovski") == ("DraftKings", -120.0, 100.0)

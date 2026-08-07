"""Attach UFC rankings to scraped fight rows.

Uses martj42's rankings_history.csv (one snapshot per ranking update).
For each (fighter, weight class) we take the rank from the most recent
snapshot strictly prior to the fight date.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from ufc_pred.paths import RAW

RANKINGS_URL = "https://raw.githubusercontent.com/martj42/ufc_rankings_history/master/rankings_history.csv"
RANKINGS_CACHE = RAW / "martj42_rankings" / "rankings_history.csv"

WEIGHT_CLASSES = [
    "Women's Flyweight",
    "Women's Featherweight",
    "Women's Strawweight",
    "Women's Bantamweight",
    "Heavyweight",
    "Light Heavyweight",
    "Middleweight",
    "Welterweight",
    "Lightweight",
    "Featherweight",
    "Bantamweight",
    "Flyweight",
    "Pound-for-Pound",
]


def download_rankings(force: bool = False) -> Path:
    RANKINGS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if force or not RANKINGS_CACHE.exists():
        r = requests.get(RANKINGS_URL, timeout=30)
        r.raise_for_status()
        RANKINGS_CACHE.write_bytes(r.content)
    return RANKINGS_CACHE


def load_rankings(path: Path | None = None) -> pd.DataFrame:
    p = path or download_rankings()
    df = pd.read_csv(p, parse_dates=["date"])
    return df


def _rank_at(rankings: pd.DataFrame, fighter: str, weight_class: str, before: pd.Timestamp):
    """Most recent rank for fighter in weight class strictly before `before`."""
    mask = (
        (rankings["weightclass"] == weight_class)
        & (rankings["fighter"] == fighter)
        & (rankings["date"] < before)
    )
    sub = rankings[mask]
    if sub.empty:
        return None
    # Get the latest snapshot date the fighter appeared, then read the rank from it
    latest_date = sub["date"].max()
    row = sub[sub["date"] == latest_date].iloc[0]
    try:
        return int(row["rank"])
    except (ValueError, TypeError):
        return None


def attach_ranks(fights: pd.DataFrame, rankings: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return a copy of ``fights`` with rank columns populated."""
    if rankings is None:
        rankings = load_rankings()

    fights = fights.copy()
    fights["date"] = pd.to_datetime(fights["date"])

    for wc in WEIGHT_CLASSES:
        r_col = f"R_{wc}_rank"
        b_col = f"B_{wc}_rank"
        fights[r_col] = [
            _rank_at(rankings, f, wc, d) for f, d in zip(fights["R_fighter"], fights["date"], strict=False)
        ]
        fights[b_col] = [
            _rank_at(rankings, f, wc, d) for f, d in zip(fights["B_fighter"], fights["date"], strict=False)
        ]

    fights["R_match_weightclass_rank"] = [
        _rank_at(rankings, f, wc, d)
        for f, wc, d in zip(fights["R_fighter"], fights["weight_class"], fights["date"], strict=False)
    ]
    fights["B_match_weightclass_rank"] = [
        _rank_at(rankings, f, wc, d)
        for f, wc, d in zip(fights["B_fighter"], fights["weight_class"], fights["date"], strict=False)
    ]

    def _better(r, b):
        if pd.isna(r) and pd.isna(b):
            return "neither"
        if pd.isna(r):
            return "Blue"
        if pd.isna(b):
            return "Red"
        return "Red" if int(r) < int(b) else "Blue"

    fights["better_rank"] = [
        _better(r, b)
        for r, b in zip(fights["R_match_weightclass_rank"], fights["B_match_weightclass_rank"], strict=False)
    ]

    # Date back to ISO string to match the on-disk schema
    fights["date"] = fights["date"].dt.strftime("%Y-%m-%d")
    return fights

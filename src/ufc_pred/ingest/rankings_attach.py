"""Attach UFC rankings to scraped fight rows.

Uses martj42's rankings_history.csv (one snapshot per ranking update).
For each (fighter, weight class) we take the rank from the most recent
snapshot strictly prior to the fight date.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

from ufc_pred.ingest.identity import canonical, normalise
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
    if force or not RANKINGS_CACHE.exists() or time.time() - RANKINGS_CACHE.stat().st_mtime > 86400:
        r = requests.get(RANKINGS_URL, timeout=30)
        r.raise_for_status()
        import io

        check = pd.read_csv(io.BytesIO(r.content))
        if check.empty or not {"date", "fighter", "weightclass", "rank"} <= set(check.columns):
            raise ValueError("Invalid rankings download")
        temp = RANKINGS_CACHE.with_suffix(".tmp")
        temp.write_bytes(r.content)
        temp.replace(RANKINGS_CACHE)
    return RANKINGS_CACHE


def load_rankings(path: Path | None = None) -> pd.DataFrame:
    p = path or download_rankings()
    df = pd.read_csv(p, parse_dates=["date"])
    if path is None:
        observed = [
            pd.read_csv(f, parse_dates=["date"])
            for f in sorted((RANKINGS_CACHE.parent / "observed").glob("*.csv"))
        ]
        if observed:
            df = pd.concat([df, *observed], ignore_index=True).drop_duplicates(
                ["date", "weightclass", "fighter"], keep="last"
            )
    return df


def _rank_at(rankings: pd.DataFrame, fighter: str, weight_class: str, before: pd.Timestamp):
    """Rank in the latest complete snapshot strictly before the fight.

    Absence means unranked, including discontinued divisions. P4P uses the
    source's renamed men's/women's lists after the 2020 schema change.
    """
    if "_snapshot_lookup" not in rankings.attrs:
        lookup = {}
        for date, wc, name, rank in rankings[["date", "weightclass", "fighter", "rank"]].itertuples(
            index=False, name=None
        ):
            date = pd.Timestamp(date)
            lookup.setdefault(date, {})[(str(wc).strip(), normalise(canonical(name, wc)))] = rank
        rankings.attrs["_snapshot_lookup"] = lookup
        rankings.attrs["_snapshot_dates"] = pd.DatetimeIndex(sorted(lookup))
    dates = rankings.attrs["_snapshot_dates"]
    pos = dates.searchsorted(pd.Timestamp(before), side="left") - 1
    if pos < 0:
        return None
    snapshot = rankings.attrs["_snapshot_lookup"][dates[pos]]
    try:
        name = normalise(canonical(fighter, weight_class))
    except ValueError:
        return None
    classes = (
        [weight_class]
        if weight_class != "Pound-for-Pound"
        else ["Pound-for-Pound", "Men's Pound-for-Pound", "Women's Pound-for-Pound"]
    )
    values = [snapshot[(wc, name)] for wc in classes if (wc, name) in snapshot]
    if len(values) != 1 or pd.isna(values[0]):
        return None
    return int(values[0])


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
        return "neither" if int(r) == int(b) else "Red" if int(r) < int(b) else "Blue"

    fights["better_rank"] = [
        _better(r, b)
        for r, b in zip(fights["R_match_weightclass_rank"], fights["B_match_weightclass_rank"], strict=False)
    ]

    # Date back to ISO string to match the on-disk schema
    fights["date"] = fights["date"].dt.strftime("%Y-%m-%d")
    return fights


def capture_current_rankings(*, now=None, html=None):
    """Save an observed UFC.com snapshot; never backdate current rankings."""
    import json

    from bs4 import BeautifulSoup

    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    directory = RAW / "martj42_rankings" / "observed"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{now.date()}.csv"
    if path.exists():
        return path
    if html is None:
        response = requests.get("https://www.ufc.com/rankings", timeout=30)
        response.raise_for_status()
        html = response.text
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for section in soup.select(".view-grouping"):
        heading = section.select_one(".view-grouping-header")
        if heading is None:
            continue
        wc = heading.get_text(" ", strip=True).replace(" Top Rank", "").strip()
        if "Pound-for-Pound" not in wc:
            champion = section.select_one("caption h5 a")
            if champion:
                rows.append([str(now.date()), wc, champion.get_text(" ", strip=True), 0])
        for row in section.select("tbody tr"):
            rank = row.select_one(".views-field-weight-class-rank")
            fighter = row.select_one(".views-field-title a")
            if rank and fighter:
                rows.append(
                    [str(now.date()), wc, fighter.get_text(" ", strip=True), int(rank.get_text(strip=True))]
                )
    frame = pd.DataFrame(rows, columns=["date", "weightclass", "fighter", "rank"])
    if len(frame) < 150 or frame.weightclass.nunique() < 10:
        raise ValueError("Incomplete UFC.com rankings snapshot; refusing partial update")
    # Complete snapshots matter: a partial division scrape would unrank everyone else.
    frame.to_csv(path, index=False)
    path.with_suffix(".html").write_text(html)
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "observed_at_utc": now.isoformat(),
                "source": "https://www.ufc.com/rankings",
                "rows": len(frame),
            },
            indent=2,
        )
    )
    return path

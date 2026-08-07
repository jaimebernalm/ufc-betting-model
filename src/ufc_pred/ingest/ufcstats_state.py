"""Strictly pre-fight fighter state sourced from raw UFCStats history.

The wide Kaggle table stores pre-fight counters but periodically refreshed
profile averages.  A previous-row reconstruction therefore cannot update all
features, and newly appended rows can mix timestamps.  This module is the
single live/raw state source used by both the event scraper and inference.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from dataclasses import asdict
from difflib import get_close_matches

import pandas as pd
from bs4 import BeautifulSoup

from .ufcstats_client import UFCStatsClient
from .ufcstats_scraper import FighterStats, parse_fighter_page

FIGHTER_LIST_URL = "http://ufcstats.com/statistics/fighters?char={char}&page=all"


def _normalise_name(value: str) -> str:
    folded = "".join(
        c for c in unicodedata.normalize("NFKD", value.casefold()) if not unicodedata.combining(c)
    )
    parts = re.sub(r"[^a-z0-9]+", " ", folded).strip().split()
    if parts and parts[-1].rstrip(".") in {"jr", "sr", "ii", "iii", "iv"}:
        parts.pop()
    return " ".join(parts)


def fighter_stats_to_feature_dict(stats: FighterStats) -> dict[str, object]:
    """Map :class:`FighterStats` names to the wide-table side suffixes."""
    raw = asdict(stats)
    return {
        "current_lose_streak": raw["current_lose_streak"],
        "current_win_streak": raw["current_win_streak"],
        "draw": raw["draws"],
        "avg_SIG_STR_landed": raw["avg_sig_str_landed"],
        "avg_SIG_STR_pct": raw["avg_sig_str_pct"],
        "avg_SUB_ATT": raw["avg_sub_att"],
        "avg_TD_landed": raw["avg_td_landed"],
        "avg_TD_pct": raw["avg_td_pct"],
        "longest_win_streak": raw["longest_win_streak"],
        "losses": raw["losses"],
        "total_rounds_fought": raw["total_rounds_fought"],
        "total_title_bouts": raw["total_title_bouts"],
        "win_by_Decision_Majority": raw["win_by_dec_majority"],
        "win_by_Decision_Split": raw["win_by_dec_split"],
        "win_by_Decision_Unanimous": raw["win_by_dec_unanimous"],
        "win_by_KO/TKO": raw["win_by_ko"],
        "win_by_Submission": raw["win_by_sub"],
        "win_by_TKO_Doctor_Stoppage": raw["win_by_tko_doctor"],
        "wins": raw["wins"],
        "Stance": raw["stance"],
        "Height_cms": raw["height_cms"],
        "Reach_cms": raw["reach_cms"],
        "Weight_lbs": raw["weight_lbs"],
        "age": raw["age"],
    }


class UFCStatsStateSource:
    """Resolve fighters and compute strict pre-fight state from UFCStats.

    The object caches listing, profile and immutable fight-detail responses, so
    callers should share one instance across an entire card.
    """

    def __init__(
        self,
        client: UFCStatsClient | None = None,
        *,
        request_delay: float = 0.15,
        html_getter: Callable[[str], str] | None = None,
    ) -> None:
        self._owned_client = client is None and html_getter is None
        self._client = client or (
            UFCStatsClient(request_delay=request_delay) if html_getter is None else None
        )
        self._html_getter = html_getter
        self._html_cache: dict[str, str] = {}
        self._index_cache: dict[str, dict[str, str]] = {}
        self._state_cache: dict[tuple[str, str], dict[str, object]] = {}

    def close(self) -> None:
        if self._owned_client and self._client is not None:
            self._client.__exit__(None, None, None)

    def __enter__(self) -> UFCStatsStateSource:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _get(self, url: str) -> str:
        if url not in self._html_cache:
            if self._html_getter is not None:
                self._html_cache[url] = self._html_getter(url)
            elif self._client is not None:
                self._html_cache[url] = self._client.get(url)
            else:  # pragma: no cover - constructor makes this unreachable
                raise RuntimeError("No UFCStats HTML source configured")
        return self._html_cache[url]

    def _index_for_char(self, char: str) -> dict[str, str]:
        char = char.lower()
        if char in self._index_cache:
            return self._index_cache[char]

        soup = BeautifulSoup(self._get(FIGHTER_LIST_URL.format(char=char)), "html.parser")
        index: dict[str, str] = {}
        for row in soup.find_all("tr"):
            links = [a for a in row.find_all("a", href=True) if "fighter-details/" in a["href"]]
            if not links:
                continue
            visible = [a.get_text(" ", strip=True) for a in links if a.get_text(" ", strip=True)]
            if len(visible) < 2:
                continue
            full_name = f"{visible[0]} {visible[1]}".strip()
            index[_normalise_name(full_name)] = links[0]["href"]
        self._index_cache[char] = index
        return index

    def fighter_url(self, fighter: str) -> str:
        key = _normalise_name(fighter)
        parts = key.split()
        # Compound surnames are not alphabetized consistently ("Saint Denis"
        # may appear under S rather than D).  Try each name-token initial,
        # surname-first, before fuzzy matching the union.
        chars = list(dict.fromkeys(p[0] for p in reversed(parts) if p))
        candidates: dict[str, str] = {}
        for char in chars:
            index = self._index_for_char(char)
            if key in index:
                return index[key]
            candidates.update(index)
        matches = get_close_matches(key, list(candidates), n=1, cutoff=0.86)
        if not matches:
            raise ValueError(f"No UFCStats profile match for {fighter!r}")
        return candidates[matches[0]]

    def get_state(self, fighter: str, fight_date: pd.Timestamp) -> dict[str, object]:
        target = (
            pd.Timestamp(fight_date).tz_localize(None)
            if pd.Timestamp(fight_date).tzinfo
            else pd.Timestamp(fight_date)
        )
        cache_key = (_normalise_name(fighter), target.strftime("%Y-%m-%d"))
        if cache_key in self._state_cache:
            return self._state_cache[cache_key].copy()

        url = self.fighter_url(fighter)
        stats = parse_fighter_page(
            self._get(url),
            target.to_pydatetime(),
            fighter_url=url,
            fight_html_getter=self._get,
        )
        state = fighter_stats_to_feature_dict(stats)
        self._state_cache[cache_key] = state
        return state.copy()

    def __call__(self, fighter: str, fight_date: pd.Timestamp) -> dict[str, object]:
        return self.get_state(fighter, fight_date)

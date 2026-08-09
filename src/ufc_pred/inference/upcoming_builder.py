"""Build a synthetic upcoming-fight row from each fighter's latest stats.

Strategy: for each fighter, find their most recent fight in `fights.parquet`,
extract the side (R or B) they were on, and use those pre-fight cumulative
stats as the starting point. Then increment win/loss counters by the result
of that last fight (since those stats were pre-fight, not post-fight).
Age is advanced by the months elapsed since their last fight.

Ranks are populated via `rankings_attach.attach_ranks`.

Difference features (*_dif) are recomputed from the populated R_ / B_ pair.

The output row is missing only `skill_diff_mean` / `skill_diff_std`; those
are computed separately by `skill_for_upcoming.py` after the NUTS fit.
"""

from __future__ import annotations

from collections.abc import Callable
from difflib import get_close_matches

import numpy as np
import pandas as pd

from ufc_pred.ingest.rankings_attach import attach_ranks, load_rankings

_DIFF_PAIRS = [
    ("lose_streak_dif", "current_lose_streak"),
    ("win_streak_dif", "current_win_streak"),
    ("longest_win_streak_dif", "longest_win_streak"),
    ("win_dif", "wins"),
    ("loss_dif", "losses"),
    ("total_round_dif", "total_rounds_fought"),
    ("total_title_bout_dif", "total_title_bouts"),
    ("ko_dif", "win_by_KO/TKO"),
    ("sub_dif", "win_by_Submission"),
    ("height_dif", "Height_cms"),
    ("reach_dif", "Reach_cms"),
    ("age_dif", "age"),
    ("sig_str_dif", "avg_SIG_STR_landed"),
    ("avg_sub_att_dif", "avg_SUB_ATT"),
    ("avg_td_dif", "avg_TD_landed"),
]


# Manual feed-name → canonical-UFCstats-name aliases. The fuzzy matcher
# (cutoff 0.85) rejects these because of legal-name vs nickname or an appended
# country tag, even though they are the same fighter (verified against history).
_ALIASES = {
    "Beatriz Mesquita": "Bia Mesquita",  # legal name vs nickname; W-BW, Brazil
    "Andre (Bra) Lima": "Andre Lima",  # "(Bra)" disambiguation suffix; Flyweight
    "Vinicius De Oliveira Prestes De Matos": "Vinicius Oliveira",  # Kalshi full legal name
    "Sharabutdin Magomedov": "Shara Magomedov",  # legal name vs nickname; Middleweight
    "Abusupiyan Magomedov": "Abus Magomedov",  # legal name vs nickname; Middleweight
    "Cong Wang": "Wang Cong",  # Western vs Chinese name order; W, 5 fights in history
    "Yadier Delvalle": "Yadier del Valle",  # feed drops the space in "del Valle"; FW, 3 fights
    "Billy Goff": "Billy Ray Goff",  # feed drops the middle name; WW, 3 fights
    "Ravena Oliveira Morais": "Ravena Oliveira",  # feed appends 2nd surname; W-FLW, 3 fights
    # DANGEROUS FALSE POSITIVE, do not remove: difflib scores "Ty Cole Miller"
    # against "Cole Miller" (retired FW, 2010-2016) at 0.88 — above the 0.85
    # cutoff — so without this alias it silently resolves to the wrong fighter
    # and prices the bet on 12 fights of the wrong record. Correct man is
    # "Ty Miller", WW, debut 2026-01-24.
    "Ty Cole Miller": "Ty Miller",
}


def resolve_fighter_name(name: str, fights: pd.DataFrame) -> str:
    """Map a Polymarket-style name to the canonical UFCstats name."""
    name = _ALIASES.get(name, name)
    all_names = pd.unique(pd.concat([fights["R_fighter"], fights["B_fighter"]]).dropna())
    if name in all_names:
        return name
    matches = get_close_matches(name, list(all_names), n=1, cutoff=0.85)
    if not matches:
        raise ValueError(
            f"No fighter match for {name!r}. Closest: "
            f"{get_close_matches(name, list(all_names), n=3, cutoff=0.5)}"
        )
    return matches[0]


def _last_fighter_stats(fighter: str, fights: pd.DataFrame) -> tuple[dict, pd.Timestamp]:
    sub = fights[(fights["R_fighter"] == fighter) | (fights["B_fighter"] == fighter)]
    if sub.empty:
        raise ValueError(f"No historical fight found for {fighter!r}")
    row = sub.sort_values("date").iloc[-1]
    side = "R" if row["R_fighter"] == fighter else "B"
    stats = {}
    skip = {f"{side}_fighter", f"{side}_odds", f"{side}_ev"}
    for c in row.index:
        if c.startswith(f"{side}_") and c not in skip:
            stats[c[2:]] = row[c]
    won = (row["Winner"] == "Red" and side == "R") or (row["Winner"] == "Blue" and side == "B")
    if won:
        stats["wins"] = (stats.get("wins") or 0) + 1
        stats["current_win_streak"] = (stats.get("current_win_streak") or 0) + 1
        stats["current_lose_streak"] = 0
        stats["longest_win_streak"] = max(stats.get("longest_win_streak") or 0, stats["current_win_streak"])
    else:
        stats["losses"] = (stats.get("losses") or 0) + 1
        stats["current_lose_streak"] = (stats.get("current_lose_streak") or 0) + 1
        stats["current_win_streak"] = 0

    # Advance every counter that can be recovered from the previous fight.
    # The former implementation advanced only W/L and streaks, leaving live
    # rows several rounds and finishes behind the recorded-row training data.
    finish_round = pd.to_numeric(row.get("finish_round"), errors="coerce")
    if pd.notna(finish_round):
        stats["total_rounds_fought"] = (stats.get("total_rounds_fought") or 0) + int(finish_round)
    title_bout = row.get("title_bout", False)
    if pd.notna(title_bout) and bool(title_bout):
        stats["total_title_bouts"] = (stats.get("total_title_bouts") or 0) + 1

    if won:
        finish = str(row.get("finish") or "").strip()
        finish_counter = {
            "M-DEC": "win_by_Decision_Majority",
            "S-DEC": "win_by_Decision_Split",
            "U-DEC": "win_by_Decision_Unanimous",
            "KO/TKO": "win_by_KO/TKO",
            "SUB": "win_by_Submission",
            "Submission": "win_by_Submission",
            "TKO - Doctor's Stoppage": "win_by_TKO_Doctor_Stoppage",
        }.get(finish)
        if finish_counter:
            stats[finish_counter] = (stats.get(finish_counter) or 0) + 1

    # UFCStats career rates need the previous bout's landed/attempted totals,
    # which the 118-column wide table does not store.  Production supplies a
    # raw state_source below; this fallback intentionally leaves only those
    # five rates unchanged rather than pretending they were advanced.
    return stats, pd.Timestamp(row["date"])


def build_upcoming_row(
    fighter_a: str,
    fighter_b: str,
    fight_date: pd.Timestamp,
    weight_class: str,
    fights: pd.DataFrame,
    *,
    gender: str = "MALE",
    title_bout: bool = False,
    no_of_rounds: int = 3,
    rankings: pd.DataFrame | None = None,
    state_source: Callable[[str, pd.Timestamp], dict[str, object]] | None = None,
) -> pd.DataFrame:
    """Return a one-row DataFrame ready for `prepare()` + ensemble predict.

    `fighter_a` becomes the Red corner; `fighter_b` the Blue corner. Predictions
    return p(Red wins) = p(fighter_a wins).
    """
    fighter_a = resolve_fighter_name(fighter_a, fights)
    fighter_b = resolve_fighter_name(fighter_b, fights)

    # Only fights strictly before the target date may inform the features —
    # _last_fighter_stats folds the last fight's RESULT into wins/streaks, so
    # a same-day row would leak the outcome being predicted (replay hazard).
    fights = fights[pd.to_datetime(fights["date"]) < pd.Timestamp(fight_date)]

    if state_source is None:
        a_stats, a_last = _last_fighter_stats(fighter_a, fights)
        b_stats, b_last = _last_fighter_stats(fighter_b, fights)

        for stats, last in ((a_stats, a_last), (b_stats, b_last)):
            if stats.get("age") is not None and pd.notna(stats["age"]):
                years = (pd.Timestamp(fight_date) - last).days / 365.25
                stats["age"] = float(stats["age"]) + years
    else:
        # Exact pre-fight state from raw immutable bout history.  Age is
        # already computed from DOB at fight_date, so it must not be advanced.
        a_stats = state_source(fighter_a, pd.Timestamp(fight_date))
        b_stats = state_source(fighter_b, pd.Timestamp(fight_date))

    template = fights[fights["weight_class"] == weight_class].sort_values("date").iloc[-1].copy()
    row = template.copy()
    row["date"] = fight_date
    row["R_fighter"] = fighter_a
    row["B_fighter"] = fighter_b
    row["Winner"] = "Red"  # placeholder; ignored at inference
    row["weight_class"] = weight_class
    row["gender"] = gender
    row["title_bout"] = title_bout
    row["no_of_rounds"] = no_of_rounds
    row["empty_arena"] = False
    for k, v in a_stats.items():
        row[f"R_{k}"] = v
    for k, v in b_stats.items():
        row[f"B_{k}"] = v

    for diff_col, base in _DIFF_PAIRS:
        r, b = row.get(f"R_{base}"), row.get(f"B_{base}")
        # Training data convention is Blue minus Red (ufcstats_scraper computes
        # diff(blue, red); holds on ~100% of historical rows).
        row[diff_col] = (b - r) if pd.notna(r) and pd.notna(b) else np.nan

    row["skill_diff_mean"] = np.nan
    row["skill_diff_std"] = np.nan

    df = pd.DataFrame([row])
    df["date"] = pd.to_datetime(df["date"])
    rankings = rankings if rankings is not None else load_rankings()
    df = attach_ranks(df, rankings=rankings)
    df["date"] = pd.to_datetime(df["date"])
    return df

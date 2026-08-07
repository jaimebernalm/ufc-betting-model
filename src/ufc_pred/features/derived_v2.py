"""v2 derived features: cheap, no-scrape additions to v1.1's feature set.

All features are computed from the existing Kaggle table — no external data.
Two groups:

1. Layoff / activity (require a self-join on fighter name):
   - {R,B}_days_since_last_fight
   - {R,B}_fights_last_365d
   - {R,B}_fights_last_730d

2. Career ratios (pure column arithmetic on existing per-corner counts):
   - {R,B}_career_fights      = wins + losses + draws
   - {R,B}_win_rate           = wins / career_fights
   - {R,B}_finish_rate        = (KO/TKO + Submission) / wins

All values are strictly pre-fight (computed using only fights with date < the
current fight's date), so they're safe to use as features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NEW_NUMERIC_COLS = [
    "R_days_since_last_fight",
    "B_days_since_last_fight",
    "R_fights_last_365d",
    "B_fights_last_365d",
    "R_fights_last_730d",
    "B_fights_last_730d",
    "R_career_fights",
    "B_career_fights",
    "R_win_rate",
    "B_win_rate",
    "R_finish_rate",
    "B_finish_rate",
]


def _long_history(fights: pd.DataFrame) -> pd.DataFrame:
    """One row per (fighter, fight, side). Used for per-fighter date scans."""
    r = fights[["date", "R_fighter"]].rename(columns={"R_fighter": "fighter"}).copy()
    b = fights[["date", "B_fighter"]].rename(columns={"B_fighter": "fighter"}).copy()
    long = pd.concat([r, b], ignore_index=True)
    long = long.dropna(subset=["fighter", "date"])
    long = long.sort_values(["fighter", "date"], kind="mergesort").reset_index(drop=True)
    return long


def _layoff_and_activity(history: pd.DataFrame) -> pd.DataFrame:
    """Per (fighter, date) row: days_since_last_fight + fights_last_{365,730}d.

    All counts are STRICTLY prior to the current fight (no leakage).
    """
    history = history.copy()
    history["prev_date"] = history.groupby("fighter")["date"].shift(1)
    history["days_since_last_fight"] = (history["date"] - history["prev_date"]).dt.days

    fights_365 = np.zeros(len(history), dtype="int32")
    fights_730 = np.zeros(len(history), dtype="int32")
    for _, grp in history.groupby("fighter", sort=False):
        dates = grp["date"].to_numpy(dtype="datetime64[ns]")
        # For each row i, count prior rows j < i with dates[j] >= dates[i] - window.
        cutoff_365 = dates - np.timedelta64(365, "D")
        cutoff_730 = dates - np.timedelta64(730, "D")
        idx_365 = np.searchsorted(dates, cutoff_365, side="left")
        idx_730 = np.searchsorted(dates, cutoff_730, side="left")
        i_self = np.arange(len(dates))
        # Exclude the current row by subtracting 1 (prior fights only).
        fights_365[grp.index.values] = np.maximum(i_self - idx_365, 0)
        fights_730[grp.index.values] = np.maximum(i_self - idx_730, 0)
    history["fights_last_365d"] = fights_365
    history["fights_last_730d"] = fights_730
    return history


def compute(fights: pd.DataFrame) -> pd.DataFrame:
    """Return `fights` with NEW_NUMERIC_COLS appended.

    Idempotent-ish: if any new column already exists it will be overwritten.
    """
    f = fights.copy()
    f["date"] = pd.to_datetime(f["date"])

    history = _long_history(f)
    history = _layoff_and_activity(history)
    layoff = history[
        ["fighter", "date", "days_since_last_fight", "fights_last_365d", "fights_last_730d"]
    ].drop_duplicates(subset=["fighter", "date"])

    for side in ("R", "B"):
        fcol = f"{side}_fighter"
        merged = f[[fcol, "date"]].merge(
            layoff.rename(columns={"fighter": fcol}),
            on=[fcol, "date"],
            how="left",
            validate="many_to_one",
        )
        f[f"{side}_days_since_last_fight"] = merged["days_since_last_fight"].to_numpy()
        f[f"{side}_fights_last_365d"] = merged["fights_last_365d"].to_numpy()
        f[f"{side}_fights_last_730d"] = merged["fights_last_730d"].to_numpy()

    for side in ("R", "B"):
        wins = f[f"{side}_wins"].fillna(0).astype(float)
        losses = f[f"{side}_losses"].fillna(0).astype(float)
        draws = f[f"{side}_draw"].fillna(0).astype(float)
        finishes = f[f"{side}_win_by_KO/TKO"].fillna(0).astype(float) + f[f"{side}_win_by_Submission"].fillna(
            0
        ).astype(float)
        career = wins + losses + draws
        f[f"{side}_career_fights"] = career
        f[f"{side}_win_rate"] = np.where(career > 0, wins / career.replace(0, np.nan), np.nan)
        f[f"{side}_finish_rate"] = np.where(wins > 0, finishes / wins.replace(0, np.nan), np.nan)

    return f

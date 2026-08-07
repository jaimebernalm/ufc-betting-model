"""Which fights the live system can actually bet.

The backtest and the deployed runner do NOT see the same universe of fights,
and the difference is worth ~3 percentage points of ROI.

In a backtest a UFC debutant's fight already has a row in `fights.parquet`
carrying pre-fight stats (a 0-0 record), so the model scores it like any other
fight. Live, `inference.upcoming_builder.resolve_fighter_name` raises
"No fighter match" because that fighter is not in the historical name index at
all — the fight is dropped before features are ever built. Same fighter, two
code paths, and only one of them can place a bet.

Measured on the 1,141-fight test set (2026-08-02):

    all bets                    n=846   ROI +10.88%
    both fighters experienced   n=700   ROI  +7.79%
    at least one debutant       n=146   ROI +25.69%

40.8% of the backtest's total profit came from the slice live cannot reach, and
17.6% of test fights contain a debutant — which matches the 18.3% of live
captures that errored with "No fighter match".

Any evaluation used to make a deployment decision must call `deployable_mask`
and report the filtered number, otherwise it overstates the achievable edge.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["add_prior_fight_counts", "deployable_mask"]


def add_prior_fight_counts(
    fights: pd.DataFrame,
    *,
    date_col: str = "date",
    red_col: str = "R_fighter",
    blue_col: str = "B_fighter",
) -> pd.DataFrame:
    """Return a copy with `r_prior`, `b_prior` and `has_debut` columns.

    `r_prior` / `b_prior` count that corner's fights *strictly before* this
    fight's date, over the whole table. Counting appearances reads no outcomes,
    so this is safe under the walk-forward rule: it is the same information the
    live name index encodes (does this fighter exist in history yet?).

    Ties on the same date are counted as NOT prior, matching live behaviour —
    a fighter debuting earlier the same evening is still absent from the
    ingested history when the runner prices a later fight.
    """
    out = fights.copy()
    dates = pd.to_datetime(out[date_col])

    appearances: dict[str, list] = {}
    for fighter, date in zip(out[red_col], dates, strict=False):
        appearances.setdefault(fighter, []).append(date)
    for fighter, date in zip(out[blue_col], dates, strict=False):
        appearances.setdefault(fighter, []).append(date)
    for fighter in appearances:
        appearances[fighter] = np.sort(np.array(appearances[fighter], dtype="datetime64[ns]"))

    def _count(fighter, date) -> int:
        arr = appearances.get(fighter)
        if arr is None:
            return 0
        return int(np.searchsorted(arr, np.datetime64(date), side="left"))

    out["r_prior"] = [_count(f, d) for f, d in zip(out[red_col], dates, strict=False)]
    out["b_prior"] = [_count(f, d) for f, d in zip(out[blue_col], dates, strict=False)]
    out["has_debut"] = (out["r_prior"] == 0) | (out["b_prior"] == 0)
    return out


def deployable_mask(fights: pd.DataFrame, **kwargs) -> pd.Series:
    """Boolean mask of fights the live pipeline could actually have priced.

    True when BOTH fighters have at least one prior bout in the history table.
    Adds the prior-count columns first if they are not already present.
    """
    if "has_debut" not in fights.columns:
        fights = add_prior_fight_counts(fights, **kwargs)
    return ~fights["has_debut"]

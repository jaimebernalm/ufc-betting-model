"""Walk-forward Elo ratings (Task 2.1, ported from TennisPred elo_surface).

Two ratings per fighter, both strictly walk-forward (a fight's features are
the ratings *before* that fight; the outcome updates ratings only for later
fights):

- global Elo (all fights),
- weight-class Elo (only fights in that weight class — the analog of
  tennis's per-surface Elo; sparse, NaN until the fighter has fought in the
  class).

Tennis found ROI monotonically improving as K shrinks (adopted K=8). K is a
hyperparameter here — sweep it, don't trust the tennis constant.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

INIT = 1500.0
SCALE = 400.0


def _expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / SCALE))


def build_elo(fights: pd.DataFrame, k: float = 8.0) -> pd.DataFrame:
    """Per-fight pre-fight Elo features, keyed (date, R_fighter, B_fighter).

    Columns: R_elo, B_elo, elo_dif (R−B), R_wc_elo, B_wc_elo, wc_elo_dif,
    R_elo_n, B_elo_n (prior fight counts). Input must contain date,
    R_fighter, B_fighter, Winner, weight_class. Draws/NC rows should be
    filtered out by the caller (Winner in {Red, Blue}).
    """
    df = fights.sort_values("date", kind="stable").reset_index(drop=True)
    glob: dict[str, float] = {}
    wc: dict[tuple[str, str], float] = {}
    n_fights: dict[str, int] = {}

    rows = np.empty((len(df), 8))
    for i, (r, b, wclass, winner) in enumerate(
        zip(df["R_fighter"], df["B_fighter"], df["weight_class"], df["Winner"], strict=False)
    ):
        er, eb = glob.get(r, INIT), glob.get(b, INIT)
        wr, wb = wc.get((r, wclass), np.nan), wc.get((b, wclass), np.nan)
        rows[i] = (er, eb, er - eb, wr, wb, wr - wb, n_fights.get(r, 0), n_fights.get(b, 0))

        y = 1.0 if winner == "Red" else 0.0
        exp = _expected(er, eb)
        glob[r] = er + k * (y - exp)
        glob[b] = eb + k * ((1.0 - y) - (1.0 - exp))
        wr0 = wr if not np.isnan(wr) else INIT
        wb0 = wb if not np.isnan(wb) else INIT
        exp_w = _expected(wr0, wb0)
        wc[(r, wclass)] = wr0 + k * (y - exp_w)
        wc[(b, wclass)] = wb0 + k * ((1.0 - y) - (1.0 - exp_w))
        n_fights[r] = n_fights.get(r, 0) + 1
        n_fights[b] = n_fights.get(b, 0) + 1

    out = df[["date", "R_fighter", "B_fighter"]].copy()
    out[["R_elo", "B_elo", "elo_dif", "R_wc_elo", "B_wc_elo", "wc_elo_dif", "R_elo_n", "B_elo_n"]] = rows
    return out

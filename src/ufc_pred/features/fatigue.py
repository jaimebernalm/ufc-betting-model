"""Walk-forward fatigue / workload / damage features (Task 2.2).

Tennis's only Tier-2 keeper was the fatigue/rest family (+3.17pp val ROI):
*recent workload*, not just layoff, is what the market misprices. UFC
candidates, all computable strictly from a fighter's PRIOR fights:

- F1 workload: cage seconds over the last 1/2/3 fights, last fight went to
  decision, last fight was a 5-rounder.
- F2 damage: KO/TKO losses in the last 365/730 days.
- F3 layoff/change: days since last fight, age × layoff interaction,
  weight-class change vs previous fight.

Per-corner columns use R_/B_ prefixes and signed diffs use the `_dif`
suffix, so `static_v1._swap_red_blue` handles symmetry augmentation
automatically (rename + sign flip).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEC_FINISHES = {"U-DEC", "S-DEC", "M-DEC"}

F1_COLS = [
    "cage_secs_l1_dif",
    "cage_secs_l3_dif",
    "R_last_was_dec",
    "B_last_was_dec",
    "R_last_was_5r",
    "B_last_was_5r",
]
F2_COLS = [
    "R_ko_losses_365",
    "B_ko_losses_365",
    "R_ko_losses_730",
    "B_ko_losses_730",
    "ko_losses_730_dif",
]
F3_COLS = [
    "R_days_since_last",
    "B_days_since_last",
    "days_since_last_dif",
    "R_age_x_layoff",
    "B_age_x_layoff",
    "R_wc_changed",
    "B_wc_changed",
]
ALL_COLS = F1_COLS + F2_COLS + F3_COLS


def build_fatigue(fights: pd.DataFrame) -> pd.DataFrame:
    """Per-fight pre-fight fatigue features, keyed (date, R_fighter, B_fighter)."""
    df = fights.sort_values("date", kind="stable").reset_index(drop=True)
    # history per fighter: list of dicts of that fighter's PRIOR fights
    hist: dict[str, list[dict]] = {}

    feats = {
        c: np.full(len(df), np.nan)
        for c in [
            "R_cage_l1",
            "B_cage_l1",
            "R_cage_l3",
            "B_cage_l3",
            "R_last_was_dec",
            "B_last_was_dec",
            "R_last_was_5r",
            "B_last_was_5r",
            "R_ko_losses_365",
            "B_ko_losses_365",
            "R_ko_losses_730",
            "B_ko_losses_730",
            "R_days_since_last",
            "B_days_since_last",
            "R_wc_changed",
            "B_wc_changed",
        ]
    }

    dates = pd.to_datetime(df["date"])
    for i, row in enumerate(df.itertuples(index=False)):
        d = dates.iloc[i]
        for side, name in (("R", row.R_fighter), ("B", row.B_fighter)):
            h = hist.get(name, [])
            if h:
                last = h[-1]
                secs = [f["secs"] for f in h[-3:] if not np.isnan(f["secs"])]
                feats[f"{side}_cage_l1"][i] = last["secs"]
                feats[f"{side}_cage_l3"][i] = float(np.sum(secs)) if secs else np.nan
                feats[f"{side}_last_was_dec"][i] = float(last["was_dec"])
                feats[f"{side}_last_was_5r"][i] = float(last["was_5r"])
                feats[f"{side}_days_since_last"][i] = (d - last["date"]).days
                feats[f"{side}_wc_changed"][i] = float(last["wc"] != row.weight_class)
            feats[f"{side}_ko_losses_365"][i] = float(
                sum(1 for f in h if f["ko_loss"] and (d - f["date"]).days <= 365)
            )
            feats[f"{side}_ko_losses_730"][i] = float(
                sum(1 for f in h if f["ko_loss"] and (d - f["date"]).days <= 730)
            )

        # append this fight to both fighters' histories (post-feature)
        finish = row.finish if isinstance(row.finish, str) else ""
        rec_common = {
            "date": d,
            "secs": float(row.total_fight_time_secs) if pd.notna(row.total_fight_time_secs) else np.nan,
            "was_dec": finish in DEC_FINISHES,
            "was_5r": row.no_of_rounds == 5,
            "wc": row.weight_class,
        }
        r_won = row.Winner == "Red"
        hist.setdefault(row.R_fighter, []).append(
            {**rec_common, "ko_loss": (not r_won) and finish == "KO/TKO"}
        )
        hist.setdefault(row.B_fighter, []).append({**rec_common, "ko_loss": r_won and finish == "KO/TKO"})

    out = df[["date", "R_fighter", "B_fighter"]].copy()
    for c, v in feats.items():
        out[c] = v
    out["cage_secs_l1_dif"] = out["R_cage_l1"] - out["B_cage_l1"]
    out["cage_secs_l3_dif"] = out["R_cage_l3"] - out["B_cage_l3"]
    out["ko_losses_730_dif"] = out["R_ko_losses_730"] - out["B_ko_losses_730"]
    out["days_since_last_dif"] = out["R_days_since_last"] - out["B_days_since_last"]
    # age × layoff needs ages from the fights frame
    for side in ("R", "B"):
        age = pd.to_numeric(df[f"{side}_age"], errors="coerce").to_numpy(float)
        out[f"{side}_age_x_layoff"] = age * out[f"{side}_days_since_last"] / 365.25
    return out.drop(columns=["R_cage_l1", "B_cage_l1", "R_cage_l3", "B_cage_l3"])

"""v1 feature prep: take the Kaggle wide-format CSV and produce X, y.

Decisions baked in here:
- Target: y = 1 if Red corner wins, 0 if Blue.
- Draws (n=5) are dropped.
- Post-fight columns are dropped (would leak the answer).
- Odds columns are dropped: v1 measures pure-feature signal *without* the
  market crutch. We compare to the market baseline separately.
- Identifiers (fighter names, location strings, raw date) are dropped.
- Symmetry: every fight is added twice — once as-is, once with R/B swapped
  and label inverted. Prevents the model from learning a red-corner bias.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Anything in here is either post-fight (leaks the label) or an identifier.
POST_FIGHT_COLS = [
    "Winner",
    "finish",
    "finish_details",
    "finish_round",
    "finish_round_time",
    "total_fight_time_secs",
]

ODDS_COLS = [
    "R_odds",
    "B_odds",
    "R_ev",
    "B_ev",
    "r_dec_odds",
    "b_dec_odds",
    "r_sub_odds",
    "b_sub_odds",
    "r_ko_odds",
    "b_ko_odds",
]

ID_COLS = ["R_fighter", "B_fighter", "date", "location", "country"]

CATEGORICAL = ["weight_class", "gender", "R_Stance", "B_Stance", "better_rank"]


def _swap_red_blue(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with all R_*/B_* columns swapped and r_*/b_* odds swapped.

    Signed difference columns (`*_dif` = Blue − Red, `skill_diff_mean` =
    skill[R] − skill[B]) must be negated when the corners swap, or the
    augmented copies carry wrong-signed difs and the model learns an
    orientation-fragile mapping. Deployed ensembles trained before 2026-06-11
    predate this fix; inference symmetrizes over both orientations to
    compensate until the next retrain.
    """
    out = df.copy()
    rename = {}
    for c in out.columns:
        if c.startswith("R_"):
            rename[c] = "B_" + c[2:]
        elif c.startswith("B_"):
            rename[c] = "R_" + c[2:]
        elif c.startswith("r_"):
            rename[c] = "b_" + c[2:]
        elif c.startswith("b_"):
            rename[c] = "r_" + c[2:]
    out = out.rename(columns=rename)
    for c in out.columns:
        if c.endswith("_dif") or c.startswith("skill_diff_mean"):
            if pd.api.types.is_numeric_dtype(out[c]):
                out[c] = -out[c]
    if "better_rank" in out.columns:
        out["better_rank"] = out["better_rank"].replace({"Red": "Blue", "Blue": "Red"})
    return out


def prepare(
    df: pd.DataFrame,
    augment_symmetry: bool = True,
    one_hot: bool = True,
) -> tuple[pd.DataFrame, np.ndarray, pd.Series, list[str]]:
    """Return (X, y, dates, cat_features).

    X: feature dataframe.
    y: 1 if Red wins, 0 if Blue.
    dates: original fight dates (for diagnostics, not used as a feature).
    cat_features: list of categorical column names (empty when one_hot=True).

    When `one_hot=False`, categoricals are kept as strings (NaN → "__missing__")
    so CatBoost / similar models can handle them natively.
    """
    df = df[df["Winner"].isin(["Red", "Blue"])].copy()

    if augment_symmetry:
        flipped = _swap_red_blue(df)
        # After swap, the original "Red wins" becomes "Blue wins" → swap label too.
        flipped["Winner"] = flipped["Winner"].map({"Red": "Blue", "Blue": "Red"})
        df = pd.concat([df, flipped], ignore_index=True)

    y = (df["Winner"] == "Red").astype(int).to_numpy()
    dates = pd.to_datetime(df["date"])

    drop = set(POST_FIGHT_COLS) | set(ODDS_COLS) | set(ID_COLS)
    X = df.drop(columns=[c for c in drop if c in df.columns])

    cat_cols = [c for c in CATEGORICAL if c in X.columns]

    if one_hot:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True, dummy_na=True)
        X = X.apply(pd.to_numeric, errors="coerce")
        return X, y, dates, []

    # Keep categoricals as strings; coerce everything else to numeric.
    for c in cat_cols:
        X[c] = X[c].astype("object").where(X[c].notna(), "__missing__").astype(str)
    num_cols = [c for c in X.columns if c not in cat_cols]
    X[num_cols] = X[num_cols].apply(pd.to_numeric, errors="coerce")
    return X, y, dates, cat_cols

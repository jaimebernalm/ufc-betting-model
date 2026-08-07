"""Task 2.4 — segment the champion's betting ROI (pre-registered slices only).

Tennis found edge strongly non-uniform (favorites positive, longshots
negative). Slices pre-registered in TENNIS_PORTED_IDEAS.md §6 on 2026-07-09:
chosen-side price bucket (≥0.60 / 0.40–0.60 / <0.40), gender, weight-class
group, 5-rounder vs 3-rounder, title bout. No retraining — slices existing
ensemble bets on (1) val 2023 and (2) the Kalshi 2026 window. A filter is
deployable only if the sign agrees in BOTH windows.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd

from ufc_pred.backtest.kalshi_match import match_kalshi_to_fights
from ufc_pred.backtest.metrics import american_to_implied_prob
from ufc_pred.backtest.strategy_grid import ModelBundle, predict
from ufc_pred.features.static_v1 import _swap_red_blue
from ufc_pred.paths import MODELS, ROOT
from ufc_pred.utils.time_splits import TRAIN_END, VAL_END

LOG_PATH = ROOT / "experiments" / "2026-07-09_segment_roi.jsonl"
VAL_PREDS_CACHE = ROOT / "data/interim/val2023_champion_seed_preds.npz"
KALSHI_FEE = 0.07
EDGE_THR = 0.03

WC_GROUPS = {
    "Flyweight": "FLW-LW",
    "Bantamweight": "FLW-LW",
    "Featherweight": "FLW-LW",
    "Lightweight": "FLW-LW",
    "Welterweight": "WW-MW",
    "Middleweight": "WW-MW",
    "Light Heavyweight": "LHW-HW",
    "Heavyweight": "LHW-HW",
}


def eff_dec(price, fee_rate=KALSHI_FEE):
    return 1.0 / (np.asarray(price, float) * (1.0 + fee_rate * (1.0 - np.asarray(price, float))))


def bets_frame(p_model, y_red, price_red, price_blue, attrs: pd.DataFrame) -> pd.DataFrame:
    eR, eB = eff_dec(price_red), eff_dec(price_blue)
    ev_R = p_model * eR - 1.0
    ev_B = (1 - p_model) * eB - 1.0
    side_r = ev_R >= ev_B
    edge = np.where(side_r, ev_R, ev_B)
    mask = edge > EDGE_THR
    won = np.where(side_r, y_red == 1, y_red == 0)
    dec = np.where(side_r, eR, eB)
    df = attrs.loc[mask].copy()
    df["chosen_price"] = np.where(side_r, price_red, price_blue)[mask]
    df["pnl"] = np.where(won, dec - 1.0, -1.0)[mask]
    df["won"] = won[mask].astype(int)
    return df


def slice_table(bets: pd.DataFrame, window: str):
    bets = bets.copy()
    bets["price_bucket"] = pd.cut(
        bets["chosen_price"], [0, 0.40, 0.60, 1.0], labels=["p<0.40", "0.40-0.60", "p>=0.60"]
    )
    bets["wc_group"] = np.where(
        bets["gender"] == "FEMALE", "women", bets["weight_class"].map(WC_GROUPS).fillna("other")
    )
    bets["rounds"] = np.where(bets["no_of_rounds"] == 5, "5R", "3R")
    bets["title"] = np.where(bets["title_bout"].astype(bool), "title", "non-title")
    slices = {
        "price_bucket": bets["price_bucket"].astype(str),
        "gender": bets["gender"],
        "wc_group": bets["wc_group"],
        "rounds": bets["rounds"],
        "title": bets["title"],
    }
    rng = np.random.default_rng(0)
    rows = []
    for dim, ser in slices.items():
        for cell, g in bets.groupby(ser, observed=True):
            pnl = g["pnl"].to_numpy()
            n = len(pnl)
            if n == 0:
                continue
            idx = rng.integers(0, n, size=(5000, n))
            means = pnl[idx].mean(axis=1)
            row = {
                "window": window,
                "dim": dim,
                "cell": str(cell),
                "n_bets": n,
                "roi_pct": round(float(pnl.mean() * 100), 2),
                "ci95": [round(float(np.quantile(means, q) * 100), 2) for q in (0.025, 0.975)],
                "hit_rate": round(float(g["won"].mean()), 3),
                "ts": datetime.now(UTC).isoformat(),
            }
            rows.append(row)
            with LOG_PATH.open("a") as f:
                f.write(json.dumps(row) + "\n")
    t = pd.DataFrame(rows)
    print(t[["dim", "cell", "n_bets", "roi_pct", "ci95", "hit_rate"]].to_string(index=False))
    return t


def main():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fights = pd.read_parquet(ROOT / "data/processed/fights.parquet")
    fights = fights[fights["Winner"].isin(["Red", "Blue"])].copy()
    fights["date"] = pd.to_datetime(fights["date"])
    sk = pd.read_parquet(ROOT / "data/processed/skill_features_v3.parquet")
    sk["date"] = pd.to_datetime(sk["date"])
    fights = fights.merge(
        sk[["date", "R_fighter", "B_fighter", "skill_diff_mean", "skill_diff_std"]],
        on=["date", "R_fighter", "B_fighter"],
        how="left",
        validate="many_to_one",
    )
    attrs_cols = ["date", "weight_class", "gender", "no_of_rounds", "title_bout"]

    # --- val 2023 (cached 10-seed ensemble, no-vig settle + kalshi fee) ---
    print("=" * 70)
    print("VAL 2023 — ensemble bets @3%, BFO no-vig + Kalshi quadratic fee")
    print("=" * 70)
    val = fights[(fights["date"] > TRAIN_END) & (fights["date"] <= VAL_END)]
    val = val.dropna(subset=["R_odds", "B_odds"]).reset_index(drop=True)
    per_seed = np.load(VAL_PREDS_CACHE)["per_seed"]
    assert per_seed.shape[1] == len(val)
    p_model = per_seed.mean(axis=0)
    p_ri = american_to_implied_prob(val["R_odds"])
    p_bi = american_to_implied_prob(val["B_odds"])
    p_nv = np.asarray(p_ri / (p_ri + p_bi), float)
    y = (val["Winner"] == "Red").to_numpy(int)
    val_bets = bets_frame(p_model, y, p_nv, 1 - p_nv, val[attrs_cols])
    t_val = slice_table(val_bets, "val2023")

    # --- Kalshi 2026 window (deployed ensemble, symmetrized) ---
    print("\n" + "=" * 70)
    print("KALSHI 2026 — deployed ensemble bets @3%, T-90min prices + quad fee")
    print("=" * 70)
    meta, matched = match_kalshi_to_fights(fights)
    mirrored = _swap_red_blue(matched)
    per_seed_k = []
    for s in range(10):
        d = joblib.load(MODELS / f"v3_real_2025_11_30_seed{s}.joblib")
        b = ModelBundle(model=d["model"], columns=d["columns"], cat_features=d["cat_features"])
        per_seed_k.append(0.5 * (predict(b, matched) + 1.0 - predict(b, mirrored)))
    p_model_k = np.stack(per_seed_k).mean(axis=0)
    y_k = (matched["Winner"] == "Red").to_numpy(int)
    kal_bets = bets_frame(
        p_model_k,
        y_k,
        meta["kal_p_red"].to_numpy(float),
        meta["kal_p_blue"].to_numpy(float),
        matched[attrs_cols],
    )
    t_kal = slice_table(kal_bets, "kalshi2026")

    # --- both-window sign agreement ---
    print("\n— Cells with same-sign ROI in both windows (deployability check) —")
    m = t_val.merge(t_kal, on=["dim", "cell"], suffixes=("_val", "_kal"))
    m["same_sign"] = np.sign(m["roi_pct_val"]) == np.sign(m["roi_pct_kal"])
    print(
        m[["dim", "cell", "n_bets_val", "roi_pct_val", "n_bets_kal", "roi_pct_kal", "same_sign"]].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()

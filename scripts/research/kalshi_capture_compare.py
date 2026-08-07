"""3-way comparison: T-4h vs T-90min per-fight vs card-snapshot capture modes.

For each capture mode, runs the SAME matching + model evaluation pipeline
(taken from nb08/nb09) and prints flat-stake ROI + Kelly bankrolls.

Usage:
    PYTHONPATH=src .conda/bin/python scripts/kalshi_capture_compare.py
"""

from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv

REPO = Path(__file__).resolve().parent.parent
load_dotenv(REPO / ".env")

from ufc_pred.backtest.bet_eval import evaluate_bets, evaluate_bets_kelly
from ufc_pred.features.skill_v3_pipeline import OUTPUT as SKILL_V3_PARQUET
from ufc_pred.features.static_v1 import prepare
from ufc_pred.ingest.kaggle_mdabbert import HISTORY_PARQUET

APOSTROPHES = "'’ʼ`‘"


def deep_norm(s):
    if pd.isna(s) or s is None:
        return ""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    for ch in APOSTROPHES + "-.,":
        s = s.replace(ch, "")
    return re.sub(r"\s+", " ", s).strip().lower()


def last_token(s):
    s = deep_norm(s)
    return s.split()[-1] if s else ""


def load_kal(path: Path) -> pd.DataFrame:
    kal = pd.read_parquet(path)
    kal = kal.dropna(subset=["close_yes_price_a", "close_yes_price_b", "winner"]).copy()
    s = kal["close_yes_price_a"] + kal["close_yes_price_b"]
    keep = (
        (kal["close_yes_price_a"] >= 0.02)
        & (kal["close_yes_price_b"] >= 0.02)
        & (s >= 0.80)
        & (s <= 1.30)
        & ((kal["volume_a"] + kal["volume_b"]) >= 100)
    )
    kal = kal[keep].reset_index(drop=True)
    kal["fd_n"] = pd.to_datetime(kal["fight_date"]).dt.tz_localize(None).dt.normalize()
    kal["a_dn"] = kal["fighter_a"].map(deep_norm)
    kal["b_dn"] = kal["fighter_b"].map(deep_norm)
    kal["a_last"] = kal["fighter_a"].map(last_token)
    kal["b_last"] = kal["fighter_b"].map(last_token)
    return kal


def evaluate_capture(name: str, kal: pd.DataFrame, fights: pd.DataFrame, payload: dict, p_corrupt_fn) -> dict:
    fights_by_date = {d: g for d, g in fights.groupby("date")}

    def match_one(row):
        pools = []
        for off in (-1, 0, 1):
            d = row["fd_n"] + pd.Timedelta(days=off)
            if d in fights_by_date:
                pools.append(fights_by_date[d])
        if not pools:
            return None, None
        pool = pd.concat(pools)
        a, b = row["a_dn"], row["b_dn"]
        a_l, b_l = row["a_last"], row["b_last"]
        h = pool[((pool["R_dn"] == a) & (pool["B_dn"] == b)) | ((pool["R_dn"] == b) & (pool["B_dn"] == a))]
        if len(h) == 1:
            k = h.iloc[0]
            return k, k["R_dn"] == a
        h = pool[
            ((pool["R_last"] == a_l) & (pool["B_last"] == b_l))
            | ((pool["R_last"] == b_l) & (pool["B_last"] == a_l))
        ]
        if len(h) == 1:
            k = h.iloc[0]
            return k, k["R_last"] == a_l
        return None, None

    matches, kag_idx = [], []
    for _, r in kal.iterrows():
        k, a_is_red = match_one(r)
        if k is None:
            continue
        matches.append(
            {
                "date": k["date"],
                "a_is_red": a_is_red,
                "pa": r["close_yes_price_a"],
                "pb": r["close_yes_price_b"],
            }
        )
        kag_idx.append(k.name)
    if not matches:
        return {"name": name, "n_matched": 0}
    mf = fights.loc[kag_idx].reset_index(drop=True)
    X, _, _, _ = prepare(mf, augment_symmetry=False, one_hot=False)
    X = X.reindex(columns=payload["columns"], fill_value=None)
    for c in payload.get("cat_features", []):
        X[c] = X[c].fillna("__missing__").astype(str)
    p_red = payload["model"].predict_proba(X)[:, 1]
    p_corrupt = p_corrupt_fn(X)
    y_red = (mf["Winner"].to_numpy() == "Red").astype(int)

    def prob_to_american(p):
        p = float(np.clip(p, 0.02, 0.98))
        dec = 1.0 / p
        return (dec - 1.0) * 100.0 if dec >= 2.0 else -100.0 / (dec - 1.0)

    R_odds, B_odds = [], []
    for m in matches:
        pa, pb = m["pa"], m["pb"]
        if m["a_is_red"]:
            R_odds.append(prob_to_american(pa))
            B_odds.append(prob_to_american(pb))
        else:
            R_odds.append(prob_to_american(pb))
            B_odds.append(prob_to_american(pa))
    R = pd.Series(R_odds)
    B = pd.Series(B_odds)

    # Flat-stake ROI at edge=5%, Kalshi real fee (0.07 quadratic)
    r5 = evaluate_bets(
        p_red, y_red, R, B, edge_threshold=0.05, fee_rate=0.07, fee_model="kalshi", use_no_vig=False
    )

    # Bankrolls
    def sim(p, kelly, cap):
        return evaluate_bets_kelly(
            p,
            y_red,
            R,
            B,
            edge_threshold=0.03,
            fee_rate=0.07,
            fee_model="kalshi",
            use_no_vig=False,
            kelly_fraction=kelly,
            max_bet_fraction=cap,
            starting_bankroll=300.0,
        )

    A = sim(p_red, 0.10, 0.10)
    Bk = sim(p_red, 0.25, 1.00)
    C = sim(p_corrupt, 0.25, 1.00)

    # Spread
    sums = np.array([m["pa"] + m["pb"] for m in matches])

    return {
        "name": name,
        "n_matched": len(matches),
        "sum_mean": sums.mean(),
        "roi_pct": r5.roi_pct,
        "ci_lo": r5.ci95_roi_pct[0],
        "ci_hi": r5.ci95_roi_pct[1],
        "n_bets_flat": r5.n_bets,
        "hit": r5.hit_rate,
        "A_final": A["final_bankroll"],
        "A_n": A["n_bets"],
        "A_dd": A["max_drawdown_pct"],
        "B_final": Bk["final_bankroll"],
        "B_n": Bk["n_bets"],
        "B_dd": Bk["max_drawdown_pct"],
        "C_final": C["final_bankroll"],
        "C_n": C["n_bets"],
        "C_dd": C["max_drawdown_pct"],
    }


def main() -> int:
    fights = pd.read_parquet(HISTORY_PARQUET)
    fights = fights[fights["Winner"].isin(["Red", "Blue"])].copy()
    fights["date"] = pd.to_datetime(fights["date"])
    fights["R_dn"] = fights["R_fighter"].map(deep_norm)
    fights["B_dn"] = fights["B_fighter"].map(deep_norm)
    fights["R_last"] = fights["R_fighter"].map(last_token)
    fights["B_last"] = fights["B_fighter"].map(last_token)
    sk = pd.read_parquet(SKILL_V3_PARQUET)
    sk["date"] = pd.to_datetime(sk["date"])
    fights = fights.merge(
        sk[["date", "R_fighter", "B_fighter", "skill_diff_mean", "skill_diff_std"]],
        on=["date", "R_fighter", "B_fighter"],
        how="left",
        validate="many_to_one",
    )

    payload = joblib.load(REPO / "artifacts/models/v3_catboost_full2000_trainval.joblib")
    payload_c = joblib.load(REPO / "artifacts/models/v3_full2000_no_skill_corrupted_trainval.joblib")

    def p_corrupt_fn(X):
        return payload_c["model"].predict_proba(X)[:, 1]

    capture_modes = [
        (
            "T-4h per-fight (original)",
            REPO / "data/raw/kalshi/snapshots/historical_T-4h_perfight.parquet",
        ),
        (
            "T-90min per-fight (safe for main events)",
            REPO / "data/raw/kalshi/snapshots/historical_T-90min_perfight.parquet",
        ),
        (
            "T-30min per-fight (aggressive)",
            REPO / "data/raw/kalshi/snapshots/historical_T-30min_perfight.parquet",
        ),
        (
            "Card-snapshot (~30min pre-card)",
            REPO / "data/raw/kalshi/snapshots/historical_cardsnapshot.parquet",
        ),
    ]
    results = []
    for name, path in capture_modes:
        if not path.exists():
            print(f"[skip] {name}: {path.name} not found")
            continue
        print(f"\n=== {name} ===")
        kal = load_kal(path)
        print(f"  loaded {len(kal)} fights")
        r = evaluate_capture(name, kal, fights, payload, p_corrupt_fn)
        results.append(r)
        print(f"  matched: {r['n_matched']}  sum_mean: {r['sum_mean']:.3f}")
        print(
            f"  flat-stake ROI@5%: {r['roi_pct']:+.2f}%  CI95=[{r['ci_lo']:+.1f}, {r['ci_hi']:+.1f}]  n_bets={r['n_bets_flat']}"
        )
        print(f"  Account A ($300 → ${r['A_final']:,.2f},  n={r['A_n']}, maxDD={r['A_dd']:.1f}%)")
        print(f"  Account B ($300 → ${r['B_final']:,.2f},  n={r['B_n']}, maxDD={r['B_dd']:.1f}%)")
        print(f"  Account C ($300 → ${r['C_final']:,.2f},  n={r['C_n']}, maxDD={r['C_dd']:.1f}%)")

    # Side-by-side table
    print("\n" + "=" * 100)
    print("SIDE-BY-SIDE COMPARISON")
    print("=" * 100)
    if results:
        df = pd.DataFrame(results)
        print(
            df[
                [
                    "name",
                    "n_matched",
                    "sum_mean",
                    "n_bets_flat",
                    "roi_pct",
                    "ci_lo",
                    "ci_hi",
                    "A_final",
                    "B_final",
                    "C_final",
                ]
            ].to_string(index=False)
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

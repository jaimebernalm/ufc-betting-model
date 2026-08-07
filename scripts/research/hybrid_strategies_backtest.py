"""Hybrid strategy experiments on the Polymarket-matched test window.

Strategies (all ¼-K + no cap, $300 start, 2% fee, 3% edge threshold):
  1. baseline_rolling_real      : rolling-retrained real model (Account B from earlier)
  2. baseline_frozen_corrupted  : frozen v3_full2000_no_skill_corrupted_trainval
  3. baseline_rolling_corrupted : rolling-retrained corrupted (Account C from earlier)
  4. hybrid_frozen_side_rolling_stake:
         - Side chosen by frozen-corrupted (the model with the apparent edge)
         - Bet placed only if frozen-corrupted EV > threshold
         - Stake sized using rolling-real's p on the chosen side (¼-K)
         - Skip if rolling-real's full Kelly on that side is ≤ 0
  5. hybrid_consensus:
         - Bet only if frozen-corrupted, rolling-corrupted, and rolling-real ALL
           agree on the side AND all have EV > threshold
         - Stake sized using rolling-real's p (¼-K)
  6. hybrid_2of3_majority:
         - Bet whichever side gets 2 or more votes from
           {frozen-corrupted, rolling-corrupted, rolling-real}
         - Bet only if the majority side has avg EV > threshold across voters
         - Stake sized using rolling-real's p (¼-K)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

ROOT = Path(__file__).resolve().parents[1]

from ufc_pred.features.skill_v3_pipeline import OUTPUT as SKILL_V3_PARQUET
from ufc_pred.features.static_v1 import prepare
from ufc_pred.ingest.kaggle_mdabbert import HISTORY_PARQUET
from ufc_pred.models.wrappers import CorruptedSkillModel
from ufc_pred.paths import METRICS, MODELS
from ufc_pred.utils.time_splits import recency_weights

START = 300.0
FEE = 0.02
EDGE_THR = 0.03
KF = 0.25
CAP = 1.0


# Load
matched = pd.read_parquet(ROOT / "data/interim/polymarket_matched_to_kaggle_v2.parquet")
matched["date"] = pd.to_datetime(matched["date"])
fights = pd.read_parquet(HISTORY_PARQUET)
fights = fights[fights["Winner"].isin(["Red", "Blue"])].copy()
fights["date"] = pd.to_datetime(fights["date"])
sk = pd.read_parquet(SKILL_V3_PARQUET)
sk["date"] = pd.to_datetime(sk["date"])
fights = fights.merge(
    sk[["date", "R_fighter", "B_fighter", "skill_diff_mean", "skill_diff_std"]],
    on=["date", "R_fighter", "B_fighter"],
    how="left",
    validate="many_to_one",
)
key = ["date", "R_fighter", "B_fighter"]
matched_full = matched[key + ["polymarket_p_red", "polymarket_p_blue"]].merge(
    fights,
    on=key,
    how="left",
    validate="one_to_one",
)


def train_rolling(cutoff):
    df = fights[fights["date"] < cutoff].copy()
    X, y, d, cat = prepare(df, augment_symmetry=True, one_hot=False)
    n = len(X) // 2
    if len(X) == 2 * n and "skill_diff_mean" in X.columns:
        X = X.copy()
        X.loc[X.index[n:], "skill_diff_mean"] = -X.loc[X.index[n:], "skill_diff_mean"]
    w = recency_weights(d, reference_date=cutoff - pd.Timedelta(days=1))
    pool = Pool(X, y, cat_features=cat, weight=w)
    m = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        loss_function="Logloss",
        random_seed=0,
        verbose=False,
        allow_writing_files=False,
    )
    m.fit(pool, verbose=False)
    return m, list(X.columns), cat


def predict(mdl, cols, cat, df):
    X, _, _, _ = prepare(df, augment_symmetry=False, one_hot=False)
    X = X.reindex(columns=cols, fill_value=None)
    for c in cat:
        X[c] = X[c].fillna("__missing__").astype(str)
    return mdl.predict_proba(X)[:, 1]


# Frozen baselines (read-only)
frozen_real_payload = joblib.load(MODELS / "v3_catboost_full2000_trainval.joblib")
frozen_corr_payload = joblib.load(MODELS / "v3_full2000_no_skill_corrupted_trainval.joblib")


def compute_choice_and_stake(
    p_R_for_side,
    p_R_for_stake,
    polymarket_pR,
    polymarket_pB,
    edge_threshold=EDGE_THR,
    kf=KF,
    cap=CAP,
):
    """Decide whether to bet, on which side, with what stake fraction.
    Side is chosen by p_for_side; stake is computed from p_for_stake on that side.
    Returns (place_bet: bool, side: 'R' or 'B', stake_frac: float, decimal_odds_eff).
    """
    eR = 1 + (1 - FEE) * (1 / polymarket_pR - 1)
    eB = 1 + (1 - FEE) * (1 / polymarket_pB - 1)

    # Side selection via p_for_side
    evR_side = p_R_for_side * eR - 1
    evB_side = (1 - p_R_for_side) * eB - 1
    side_red = evR_side >= evB_side
    side = "R" if side_red else "B"
    side_ev = evR_side if side_red else evB_side
    if side_ev <= edge_threshold:
        return False, side, 0.0, 0.0

    # Stake calculation via p_for_stake on the chosen side
    dec_eff = eR if side_red else eB
    p_stake = p_R_for_stake if side_red else (1 - p_R_for_stake)
    b = dec_eff - 1
    q = 1 - p_stake
    fk = (b * p_stake - q) / b
    if fk <= 0:
        return False, side, 0.0, dec_eff
    sf = min(kf * fk, cap)
    return True, side, sf, dec_eff


def simulate_strategy(strategy, predictions_per_fight):
    """predictions_per_fight: list of dicts with all needed model probabilities and
    polymarket prices for each fight; the strategy function decides bet."""
    bankroll = START
    bets_made = 0
    bets_won = 0
    history = []
    for fight in predictions_per_fight:
        decision = strategy(fight)
        if decision is None:
            continue
        side, stake_frac, dec_eff, _ = decision
        stake = bankroll * stake_frac
        won = fight["winner"] == side
        realized = stake * (dec_eff - 1) if won else -stake
        bankroll += realized
        bets_made += 1
        if won:
            bets_won += 1
        history.append(
            {
                "date": fight["date"],
                "side": side,
                "stake": stake,
                "won": int(won),
                "realized": realized,
                "bankroll_after": bankroll,
            }
        )
    return {
        "final_bankroll": bankroll,
        "n_bets": bets_made,
        "n_wins": bets_won,
        "hit_rate": bets_won / bets_made if bets_made else 0.0,
        "history": history,
    }


# Strategy implementations -----------------------------------------------------


def strat_baseline_rolling_real(f):
    # Use rolling_real p for both side and stake
    bet, side, sf, dec = compute_choice_and_stake(f["p_rolling_real"], f["p_rolling_real"], f["pR"], f["pB"])
    return (side, sf, dec, "rolling_real") if bet else None


def strat_baseline_frozen_corrupted(f):
    bet, side, sf, dec = compute_choice_and_stake(f["p_frozen_corr"], f["p_frozen_corr"], f["pR"], f["pB"])
    return (side, sf, dec, "frozen_corr") if bet else None


def strat_baseline_rolling_corrupted(f):
    bet, side, sf, dec = compute_choice_and_stake(f["p_rolling_corr"], f["p_rolling_corr"], f["pR"], f["pB"])
    return (side, sf, dec, "rolling_corr") if bet else None


def strat_hybrid_frozen_side_rolling_stake(f):
    bet, side, sf, dec = compute_choice_and_stake(f["p_frozen_corr"], f["p_rolling_real"], f["pR"], f["pB"])
    return (side, sf, dec, "hybrid_frz_side_roll_stake") if bet else None


def strat_hybrid_consensus(f):
    """Bet only if all 3 models agree on side AND all have EV > threshold."""
    eR = 1 + (1 - FEE) * (1 / f["pR"] - 1)
    eB = 1 + (1 - FEE) * (1 / f["pB"] - 1)
    sides = {}
    for label, p in [
        ("fcorr", f["p_frozen_corr"]),
        ("rcorr", f["p_rolling_corr"]),
        ("rreal", f["p_rolling_real"]),
    ]:
        evR = p * eR - 1
        evB = (1 - p) * eB - 1
        side = "R" if evR >= evB else "B"
        ev = max(evR, evB)
        sides[label] = (side, ev)
    s_set = {v[0] for v in sides.values()}
    if len(s_set) != 1:
        return None
    side = next(iter(s_set))
    if any(v[1] <= EDGE_THR for v in sides.values()):
        return None
    # Stake via rolling_real
    dec = eR if side == "R" else eB
    p_stake = f["p_rolling_real"] if side == "R" else 1 - f["p_rolling_real"]
    b = dec - 1
    q = 1 - p_stake
    fk = (b * p_stake - q) / b
    if fk <= 0:
        return None
    sf = min(KF * fk, CAP)
    return (side, sf, dec, "hybrid_consensus")


def strat_hybrid_2of3(f):
    """Majority vote on side, bet if avg EV across voters > threshold."""
    eR = 1 + (1 - FEE) * (1 / f["pR"] - 1)
    eB = 1 + (1 - FEE) * (1 / f["pB"] - 1)
    votes_R = 0
    votes_B = 0
    evs_R = []
    evs_B = []
    for p in [f["p_frozen_corr"], f["p_rolling_corr"], f["p_rolling_real"]]:
        evR = p * eR - 1
        evB = (1 - p) * eB - 1
        if evR >= evB:
            votes_R += 1
            evs_R.append(evR)
        else:
            votes_B += 1
            evs_B.append(evB)
    if votes_R >= 2:
        side = "R"
        avg_ev = np.mean(evs_R)
    elif votes_B >= 2:
        side = "B"
        avg_ev = np.mean(evs_B)
    else:
        return None
    if avg_ev <= EDGE_THR:
        return None
    dec = eR if side == "R" else eB
    p_stake = f["p_rolling_real"] if side == "R" else 1 - f["p_rolling_real"]
    b = dec - 1
    q = 1 - p_stake
    fk = (b * p_stake - q) / b
    if fk <= 0:
        return None
    sf = min(KF * fk, CAP)
    return (side, sf, dec, "hybrid_2of3")


# Main loop -------------------------------------------------------------------

print("Building per-fight predictions across rolling months...")
predictions_all = []
month_starts = pd.date_range("2024-04-01", "2026-04-01", freq="MS")
for i in range(len(month_starts) - 1):
    m_s = month_starts[i]
    m_e = month_starts[i + 1]
    mf = matched_full[(matched_full["date"] >= m_s) & (matched_full["date"] < m_e)].reset_index(drop=True)
    if len(mf) == 0:
        continue

    m_real, cols, cat = train_rolling(m_s)
    m_corr_roll = CorruptedSkillModel(m_real, ["skill_diff_mean", "skill_diff_std"])

    p_rreal = predict(m_real, cols, cat, mf)
    p_rcorr = predict(m_corr_roll, cols, cat, mf)
    p_freal = predict(
        frozen_real_payload["model"],
        frozen_real_payload["columns"],
        frozen_real_payload.get("cat_features", []),
        mf,
    )
    p_fcorr = predict(
        frozen_corr_payload["model"],
        frozen_corr_payload["columns"],
        frozen_corr_payload.get("cat_features", []),
        mf,
    )

    for j in range(len(mf)):
        predictions_all.append(
            {
                "date": mf["date"].iloc[j],
                "R": mf["R_fighter"].iloc[j],
                "B": mf["B_fighter"].iloc[j],
                "winner": "R" if mf["Winner"].iloc[j] == "Red" else "B",
                "pR": mf["polymarket_p_red"].iloc[j],
                "pB": mf["polymarket_p_blue"].iloc[j],
                "p_rolling_real": p_rreal[j],
                "p_rolling_corr": p_rcorr[j],
                "p_frozen_real": p_freal[j],
                "p_frozen_corr": p_fcorr[j],
            }
        )
    print(f"  [{m_s.strftime('%Y-%m')}] n_fights={len(mf)}  predictions cached")

print(f"\nTotal fight-predictions: {len(predictions_all)}")
print("\nRunning strategies...")

strategies = {
    "baseline_rolling_real": strat_baseline_rolling_real,
    "baseline_rolling_corrupted": strat_baseline_rolling_corrupted,
    "baseline_frozen_corrupted": strat_baseline_frozen_corrupted,
    "hybrid_frozen_side_rolling_stake": strat_hybrid_frozen_side_rolling_stake,
    "hybrid_consensus": strat_hybrid_consensus,
    "hybrid_2of3": strat_hybrid_2of3,
}

results = {}
for name, fn in strategies.items():
    r = simulate_strategy(fn, predictions_all)
    results[name] = r

print()
print("=" * 90)
print(f"{'Strategy':38s} {'Final bankroll':>16s} {'n_bets':>8s} {'hit_rate':>10s}")
print("=" * 90)
for name, r in sorted(results.items(), key=lambda x: -x[1]["final_bankroll"]):
    print(f"{name:38s} ${r['final_bankroll']:>14,.2f}  {r['n_bets']:>8d}  {r['hit_rate']:>10.3f}")

out = {
    "run_at": datetime.now().isoformat(timespec="seconds"),
    "config": {"start": START, "fee": FEE, "edge_thr": EDGE_THR, "kelly": KF, "cap": CAP},
    "strategies": {n: {k: v for k, v in r.items() if k != "history"} for n, r in results.items()},
}
out_path = METRICS / "hybrid_strategies_backtest.json"
out_path.write_text(json.dumps(out, indent=2, default=float))
print(f"\nWrote {out_path}")

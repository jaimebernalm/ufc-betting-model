"""Direct validation of the ensemble-mean methodology on the 11-month
main eval window.

The seed-robust grid (notebook 07) ran SINGLE-SEED simulations 10 times and
reported the median bankroll. The deployment uses ENSEMBLE-MEAN predictions.
Those are not the same calculation. This script runs both on the same
training cutoff and the same eval window and reports them side-by-side:

  Cutoff:  2025-01-01  (analog of the deployment cutoff 2025-11-30, since
                       both are 6 months before their eval window starts)
  Eval :   2025-07-01 -> 2026-05-31  (435 Polymarket-matched fights)

Per-seed bankrolls (k=10) reproduce the all_accounts_seed_grid_bets.parquet
numbers. Ensemble bankrolls are the new measurement.

A passing result is: ensemble bankroll lands inside the seed range and close
to the seed median for every account.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

from ufc_pred.backtest.strategy_grid import (
    ACCOUNT_SPECS,
    predict,
    simulate_kelly,
    train_corrupted,
    train_real,
)
from ufc_pred.features.skill_v3_pipeline import OUTPUT as SKILL_V3_PARQUET
from ufc_pred.ingest.kaggle_mdabbert import HISTORY_PARQUET

CUTOFF = pd.Timestamp("2025-01-01")  # analog of deployment cutoff (6mo stale)
EVAL_START = pd.Timestamp("2025-07-01")
EVAL_END = pd.Timestamp("2026-05-31")
SEEDS = list(range(10))

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
eval_df = matched_full[(matched_full["date"] >= EVAL_START) & (matched_full["date"] <= EVAL_END)].reset_index(
    drop=True
)
print(f"Cutoff: {CUTOFF.date()}   Eval: {EVAL_START.date()} -> {EVAL_END.date()}   n={len(eval_df)}")
print(f"Seeds: {SEEDS}\n")

# Train + predict per seed
real_preds, corr_preds = [], []
t0 = time.time()
for s in SEEDS:
    r = train_real(fights, CUTOFF, seed=s)
    c = train_corrupted(fights, CUTOFF, seed=s)
    real_preds.append(predict(r, eval_df))
    corr_preds.append(predict(c, eval_df))
    print(f"  seed={s} trained + predicted   ({time.time() - t0:.0f}s elapsed)", flush=True)
real_preds = np.stack(real_preds)  # (n_seeds, n_fights)
corr_preds = np.stack(corr_preds)

print(f"\nDone training in {time.time() - t0:.0f}s\n")


# Per-seed bankrolls
def sim(p, kf, cap):
    f, b = simulate_kelly(p, eval_df, kelly_frac=kf, cap=cap)
    return f, len(b), sum(x["won"] for x in b)


print("=" * 86)
print("PER-SEED bankrolls (matches all_accounts_seed_grid for cutoff=2025-01-01)")
print("=" * 86)
per_seed = {a.name: [] for a in ACCOUNT_SPECS}
for s in SEEDS:
    row = []
    for acct in ACCOUNT_SPECS:
        p = corr_preds[s] if acct.model_kind == "corrupt" else real_preds[s]
        f, n, w = sim(p, acct.kelly_frac, acct.cap)
        per_seed[acct.name].append(
            {"seed": s, "final": f, "n_bets": n, "n_wins": w, "hit": w / n if n else 0.0}
        )
        row.append(f"{acct.name}=${f:>12,.0f}")
    print(f"  seed={s}  " + "  ".join(row))

# Ensemble
real_ens = real_preds.mean(axis=0)
corr_ens = corr_preds.mean(axis=0)
print(
    f"\nEnsemble pred std per fight (averaged): real={real_preds.std(axis=0).mean():.4f}  corr={corr_preds.std(axis=0).mean():.4f}"
)

# Per-account comparison
print("\n" + "=" * 86)
print("COMPARISON: ENSEMBLE vs PER-SEED-MEDIAN bankrolls")
print("=" * 86)
print(
    f"  {'Acct':<6s}{'ensemble':>14s}{'seed_min':>14s}{'seed_p25':>14s}{'seed_median':>14s}{'seed_p75':>14s}{'seed_max':>14s}{'ens_pos':>10s}"
)
for acct in ACCOUNT_SPECS:
    p = corr_ens if acct.model_kind == "corrupt" else real_ens
    f_ens, n_ens, w_ens = sim(p, acct.kelly_frac, acct.cap)
    seed_finals = np.array([r["final"] for r in per_seed[acct.name]])
    pct = (seed_finals < f_ens).mean() * 100  # ensemble percentile within the seed dist
    print(
        f"  {acct.name:<6s}${f_ens:>12,.0f}  ${seed_finals.min():>12,.0f}  "
        f"${np.percentile(seed_finals, 25):>12,.0f}  ${np.median(seed_finals):>12,.0f}  "
        f"${np.percentile(seed_finals, 75):>12,.0f}  ${seed_finals.max():>12,.0f}  "
        f"{pct:>7.0f}th"
    )

# Per-bet hit rate by account
print("\nPer-bet detail (ensemble run):")
for acct in ACCOUNT_SPECS:
    p = corr_ens if acct.model_kind == "corrupt" else real_ens
    f_ens, n_ens, w_ens = sim(p, acct.kelly_frac, acct.cap)
    seed_hits = np.array([r["hit"] for r in per_seed[acct.name]])
    seed_bets = np.array([r["n_bets"] for r in per_seed[acct.name]])
    print(
        f"  {acct.name}: ensemble {w_ens}/{n_ens} ({w_ens / n_ens:.3f})   "
        f"per-seed bets median={int(np.median(seed_bets))}   hit median={np.median(seed_hits):.3f}"
    )

print("\nIf the ensemble row lies between p25 and p75 of the seed distribution for")
print("every account, the deployment methodology behaves as the seed-robust grid predicted.")

"""Seed-robust strategy grid for all three deployment accounts.

Grid:
  - 10 seeds  (CatBoost random_seed in 0..9)
  - 8 strategies  (4 frozen cutoffs + 4 rolling cadences; see strategy_grid.STRATEGIES)
  - 3 accounts  (A: real 10%-K +10% cap, B: real 25%-K no cap, C: corrupt 25%-K no cap)
  - 11 eval months (2025-07 -> 2026-05)

For each (strategy, eval_month) we determine the cutoff from cutoff_for(),
train one real + one corrupted model per (cutoff, seed), score the month's
fights once per model, and feed the predictions into the per-account
Kelly simulator. Real predictions are shared between accounts A and B.

Outputs:
  artifacts/metrics/all_accounts_seed_grid.json
      Per-(account, strategy, seed) bankroll summary.
  data/interim/all_accounts_seed_grid_bets.parquet
      Per-bet records: (account, strategy, seed, eval_month, fight_id, won,
      pnl_flat, dec_odds, fk, bank_after).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

from ufc_pred.backtest.strategy_grid import (
    ACCOUNT_SPECS,
    START,
    STRATEGIES,
    cutoff_for,
    predict,
    simulate_kelly,
    train_corrupted,
    train_real,
    unique_cutoffs,
)
from ufc_pred.features.skill_v3_pipeline import OUTPUT as SKILL_V3_PARQUET
from ufc_pred.ingest.kaggle_mdabbert import HISTORY_PARQUET

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEEDS = list(range(10))
EVAL_START = pd.Timestamp("2025-07-01")
EVAL_END = pd.Timestamp("2026-05-31")

OUT_JSON = ROOT / "artifacts/metrics/all_accounts_seed_grid.json"
OUT_PARQUET = ROOT / "data/interim/all_accounts_seed_grid_bets.parquet"


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
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

eval_months: list[pd.Timestamp] = []
m = EVAL_START
while m <= EVAL_END:
    eval_months.append(m)
    m = (m + pd.offsets.MonthBegin(1)).normalize()

# Pre-slice each eval month once (avoids repeated boolean indexing)
month_slices: dict[pd.Timestamp, pd.DataFrame] = {}
for M in eval_months:
    M_end = (M + pd.offsets.MonthBegin(1)).normalize()
    month_slices[M] = matched_full[(matched_full["date"] >= M) & (matched_full["date"] < M_end)].reset_index(
        drop=True
    )

cuts = unique_cutoffs(STRATEGIES, eval_months)
print(f"Eval months: {len(eval_months)}  (n_fights total = {sum(len(s) for s in month_slices.values())})")
print(f"Strategies : {STRATEGIES}")
print(f"Seeds      : {SEEDS}")
print(f"Unique cutoffs ({len(cuts)}): {[c.strftime('%Y-%m-%d') for c in cuts]}")
print(f"Total trainings: {len(cuts)} cutoffs x {len(SEEDS)} seeds x 2 kinds = {len(cuts) * len(SEEDS) * 2}")
print()


# ---------------------------------------------------------------------------
# Run the grid: outer loop is (seed, kind, cutoff) so the model cache for the
# seed can be cleared at the end of each seed iteration to free memory.
# ---------------------------------------------------------------------------
all_bets: list[dict] = []
summary: list[dict] = []
t_start = time.time()

for seed in SEEDS:
    # Train all cutoffs at this seed for both kinds
    bundles_real: dict[pd.Timestamp, object] = {}
    bundles_corr: dict[pd.Timestamp, object] = {}
    t_seed = time.time()
    for ci, cutoff in enumerate(cuts):
        t0 = time.time()
        bundles_real[cutoff] = train_real(fights, cutoff, seed=seed)
        t1 = time.time()
        bundles_corr[cutoff] = train_corrupted(fights, cutoff, seed=seed)
        t2 = time.time()
        elapsed_seed = time.time() - t_seed
        total_elapsed = time.time() - t_start
        print(
            f"  seed={seed} cutoff={cutoff.date()} ({ci + 1}/{len(cuts)})  "
            f"real={t1 - t0:.1f}s  corr={t2 - t1:.1f}s  "
            f"seed_elapsed={elapsed_seed:.0f}s  total={total_elapsed / 60:.1f}m",
            flush=True,
        )

    # Cache predictions per (strategy, month, kind) so accounts A and B can
    # share real-model predictions for free.
    pred_cache: dict[tuple[str, pd.Timestamp, str], np.ndarray] = {}

    def get_pred(st: str, M: pd.Timestamp, kind: str) -> np.ndarray:
        ck = (st, M, kind)
        if ck not in pred_cache:
            cutoff = cutoff_for(st, M)
            bundle = bundles_corr[cutoff] if kind == "corrupt" else bundles_real[cutoff]
            pred_cache[ck] = predict(bundle, month_slices[M])
        return pred_cache[ck]

    # Simulate carry-over bankroll for each (account, strategy).
    for acct in ACCOUNT_SPECS:
        for st in STRATEGIES:
            bank = START
            n_total = 0
            w_total = 0
            for M in eval_months:
                sub = month_slices[M]
                if len(sub) == 0:
                    continue
                p = get_pred(st, M, acct.model_kind)
                final, bets = simulate_kelly(
                    p,
                    sub,
                    kelly_frac=acct.kelly_frac,
                    cap=acct.cap,
                    start=bank,
                )
                bank = final
                for b in bets:
                    b["eval_month"] = str(M.date())
                    b["account"] = acct.name
                    b["strategy"] = st
                    b["seed"] = seed
                    all_bets.append(b)
                n_total += len(bets)
                w_total += sum(b["won"] for b in bets)
            summary.append(
                {
                    "account": acct.name,
                    "strategy": st,
                    "seed": seed,
                    "final": bank,
                    "n_bets": n_total,
                    "n_wins": w_total,
                    "hit": (w_total / n_total) if n_total else 0.0,
                }
            )

    # Free per-seed memory before next seed
    # Free the trained bundles (large) without unbinding the names — the
    # closures above still reference them.
    bundles_real.clear()
    bundles_corr.clear()

    # Incremental save after each seed (in case of interruption)
    pd.DataFrame(all_bets).to_parquet(OUT_PARQUET, index=False)
    OUT_JSON.write_text(
        json.dumps(
            {
                "eval_window": [str(EVAL_START.date()), str(EVAL_END.date())],
                "seeds_completed": list(range(seed + 1)),
                "strategies": STRATEGIES,
                "accounts": [a.name for a in ACCOUNT_SPECS],
                "summary": summary,
            },
            indent=2,
            default=float,
        )
    )
    print(
        f"  >> seed {seed} complete.  total bets so far: {len(all_bets)}  "
        f"({(time.time() - t_start) / 60:.1f}m total)\n",
        flush=True,
    )


# Final summary
print()
print("=" * 78)
print("DONE")
print("=" * 78)
df = pd.DataFrame(summary)
for acct in [a.name for a in ACCOUNT_SPECS]:
    print(f"\nAccount {acct} — median final across seeds:")
    g = df[df["account"] == acct].groupby("strategy")["final"].median().sort_values(ascending=False)
    for st, med in g.items():
        print(f"  {st:18s} median ${med:>14,.2f}")

print(f"\nWrote {OUT_JSON}")
print(f"Wrote {OUT_PARQUET}  ({len(all_bets)} bet rows)")

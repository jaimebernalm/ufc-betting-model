"""Train the deployment ensemble for live deployment 2026-05-30.

Per DEPLOY.md:
  - Cutoff: 2025-11-30 (6 months stale at deployment)
  - 10 seeds (0..9), both kinds (real + corrupted)
  - Accounts A and B share the real ensemble; Account C uses corrupted of the
    same trained models.

Outputs 20 artifacts:
  artifacts/models/v3_real_2025_11_30_seed{0..9}.joblib
  artifacts/models/v3_corrupted_2025_11_30_seed{0..9}.joblib

Each artifact payload:
  {
    "model":         CatBoostClassifier (real) or CorruptedSkillModel (corrupt),
    "columns":       list[str],
    "cat_features":  list[str],
    "cutoff":        "2025-11-30",
    "seed":          int,
    "kind":          "real" | "corrupted",
    "n_train_rows":  int,
    "trained_at":    ISO timestamp,
  }

After training, runs a smoke validation: ensemble-mean predictions on the
Apr-May 2026 holdout window, simulate per-account Kelly, and report final
bankrolls + bet counts. The ensemble bankrolls should land near the seed
medians from notebook 07's grid (within seed-of-the-ensemble noise).
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

from ufc_pred.backtest.strategy_grid import (
    ACCOUNT_SPECS,
    START,
    predict,
    simulate_kelly,
    train_corrupted,
    train_real,
)
from ufc_pred.features.skill_v3_pipeline import OUTPUT as SKILL_V3_PARQUET
from ufc_pred.ingest.kaggle_mdabbert import HISTORY_PARQUET

CUTOFF = pd.Timestamp("2025-11-30")
SEEDS = list(range(10))
MODELS_DIR = ROOT / "artifacts/models"

# ---------------------------------------------------------------------------
# Load fights and verify training-data coverage
# ---------------------------------------------------------------------------
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

train_n = int((fights["date"] < CUTOFF).sum())
print(f"Cutoff: {CUTOFF.date()}")
print(f"Training rows (date < {CUTOFF.date()}): {train_n}")
print(f"Fights date range: {fights.date.min().date()} -> {fights.date.max().date()}")
print("Holdout window for smoke validation: 2026-04-01 -> 2026-05-31")
print(f"Seeds to train: {SEEDS}\n")


# ---------------------------------------------------------------------------
# Train + save
# ---------------------------------------------------------------------------
def save_bundle(bundle, kind: str, seed: int):
    path = MODELS_DIR / f"v3_{kind}_2025_11_30_seed{seed}.joblib"
    joblib.dump(
        {
            "model": bundle.model,
            "columns": bundle.columns,
            "cat_features": bundle.cat_features,
            "cutoff": str(CUTOFF.date()),
            "seed": seed,
            "kind": kind,
            "n_train_rows": train_n,
            "trained_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        },
        path,
    )
    return path


t_start = time.time()
real_bundles, corr_bundles = {}, {}
for s in SEEDS:
    t0 = time.time()
    rb = train_real(fights, CUTOFF, seed=s)
    t1 = time.time()
    cb = train_corrupted(fights, CUTOFF, seed=s)
    t2 = time.time()
    real_path = save_bundle(rb, "real", s)
    corr_path = save_bundle(cb, "corrupted", s)
    real_bundles[s] = rb
    corr_bundles[s] = cb
    print(
        f"  seed={s}  real {t1 - t0:5.1f}s  corrupted {t2 - t1:5.1f}s   "
        f"saved -> {real_path.name}, {corr_path.name}",
        flush=True,
    )

print(f"\nAll 20 artifacts trained + saved in {(time.time() - t_start):.0f}s")
print(
    f"Total disk: {sum(p.stat().st_size for p in MODELS_DIR.glob('v3_*_2025_11_30_seed*.joblib')) / 1e6:.1f} MB"
)


# ---------------------------------------------------------------------------
# Smoke validation: ensemble predictions on Apr-May 2026 holdout
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("SMOKE VALIDATION — ensemble on Apr-May 2026 holdout (clean OOS for cutoff 2025-11-30)")
print("=" * 78)

matched = pd.read_parquet(ROOT / "data/interim/polymarket_matched_to_kaggle_v2.parquet")
matched["date"] = pd.to_datetime(matched["date"])
key = ["date", "R_fighter", "B_fighter"]
matched_full = matched[key + ["polymarket_p_red", "polymarket_p_blue"]].merge(
    fights,
    on=key,
    how="left",
    validate="one_to_one",
)
holdout = matched_full[
    (matched_full["date"] >= pd.Timestamp("2026-04-01"))
    & (matched_full["date"] <= pd.Timestamp("2026-05-31"))
].reset_index(drop=True)
print(f"Holdout fights: {len(holdout)}\n")

# Ensemble mean predictions
p_real_each = np.stack([predict(real_bundles[s], holdout) for s in SEEDS])
p_corr_each = np.stack([predict(corr_bundles[s], holdout) for s in SEEDS])
p_real_mean = p_real_each.mean(axis=0)
p_corr_mean = p_corr_each.mean(axis=0)
print(
    f"Real    ensemble: pred mean={p_real_mean.mean():.3f}  std across seeds (mean)={p_real_each.std(axis=0).mean():.3f}"
)
print(
    f"Corrupt ensemble: pred mean={p_corr_mean.mean():.3f}  std across seeds (mean)={p_corr_each.std(axis=0).mean():.3f}\n"
)

# Account-level simulations using the ensemble mean
print(f"{'Acct':6s}{'Sizing':30s}{'Final bank':>16s}{'bets':>7s}{'wins':>6s}{'hit':>8s}{'ret':>10s}")
results = {}
for acct in ACCOUNT_SPECS:
    p = p_corr_mean if acct.model_kind == "corrupt" else p_real_mean
    final, bets = simulate_kelly(p, holdout, kelly_frac=acct.kelly_frac, cap=acct.cap)
    hit = sum(b["won"] for b in bets) / len(bets) if bets else 0.0
    results[acct.name] = {
        "final": final,
        "n_bets": len(bets),
        "n_wins": int(sum(b["won"] for b in bets)),
        "hit": hit,
        "ret": final / START,
    }
    print(
        f"{acct.name:6s}{f'KF={acct.kelly_frac} cap={acct.cap} ({acct.model_kind})':30s}"
        f"${final:>14,.2f}  {len(bets):>5d}  {sum(b['won'] for b in bets):>4d}  "
        f"{hit:>7.3f}  {final / START:>8.2f}x"
    )

# Also: per-seed simulations to compare with ensemble (spread = seed-variance the ensemble removes)
print("\nPer-seed bankrolls (for context — ensemble is the deployed prediction):")
for acct in ACCOUNT_SPECS:
    seed_finals = []
    for s in SEEDS:
        p = predict(corr_bundles[s] if acct.model_kind == "corrupt" else real_bundles[s], holdout)
        f, _ = simulate_kelly(p, holdout, kelly_frac=acct.kelly_frac, cap=acct.cap)
        seed_finals.append(f)
    sf = np.array(seed_finals)
    print(
        f"  {acct.name}: ensemble ${results[acct.name]['final']:>10,.0f}  | seed min ${sf.min():>10,.0f}  "
        f"median ${np.median(sf):>10,.0f}  max ${sf.max():>10,.0f}  spread {sf.max() / sf.min():.1f}x"
    )

# Save validation summary
out = ROOT / "artifacts/metrics/deployment_ensemble_smoke.json"
out.write_text(
    json.dumps(
        {
            "cutoff": str(CUTOFF.date()),
            "seeds": SEEDS,
            "holdout_window": ["2026-04-01", "2026-05-31"],
            "n_holdout_fights": int(len(holdout)),
            "ensemble_results": results,
            "n_train_rows": train_n,
            "trained_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        },
        indent=2,
        default=float,
    )
)
print(f"\nWrote {out}")
print("\nReady for deployment 2026-05-30.")

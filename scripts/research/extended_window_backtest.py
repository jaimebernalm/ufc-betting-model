"""Re-run the multi-freeze + hybrid backtests on the *extended* window now that
Apr-May 2026 fights and Polymarket prices are ingested.

Three slices reported:
  (1) full extended eval window: 2025-07-01 -> 2026-05-16  (~349+86 = ~435 bets)
  (2) Apr-May 2026 only: pure out-of-sample for every model trained on data
      ending 2026-03-28 (the previous skill features cutoff).
  (3) $300 forward sim on Apr-May 2026 alone, mirroring the 05/30 deployment
      plan (A: 10%-K + 10% cap real, B: 1/4-K no cap real, C: 1/4-K no cap
      fresh-frozen corrupted trained through 2026-03-28).

The "fresh freeze" for account C is trained on data <= 2026-03-28 so that
Apr-May 2026 is a clean OOS test of that exact deployment artifact.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

ROOT = Path(__file__).resolve().parents[1]

from ufc_pred.features.skill_v3_pipeline import OUTPUT as SKILL_V3_PARQUET
from ufc_pred.features.static_v1 import prepare
from ufc_pred.ingest.kaggle_mdabbert import HISTORY_PARQUET
from ufc_pred.models.wrappers import CorruptedSkillModel
from ufc_pred.utils.time_splits import recency_weights

FEE = 0.02
EDGE_THR = 0.03
START = 300.0

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
    fights, on=key, how="left", validate="one_to_one"
)


def train_real(cutoff):
    df = fights[fights["date"] < cutoff].copy()
    X, y, d, cat = prepare(df, augment_symmetry=True, one_hot=False)
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


def train_corrupted(cutoff):
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
    return CorruptedSkillModel(m, ["skill_diff_mean", "skill_diff_std"]), list(X.columns), cat


def predict(mdl, cols, cat, df):
    X, _, _, _ = prepare(df, augment_symmetry=False, one_hot=False)
    X = X.reindex(columns=cols, fill_value=None)
    for c in cat:
        X[c] = X[c].fillna("__missing__").astype(str)
    return mdl.predict_proba(X)[:, 1]


def sim(p_red_model, df, kf=0.25, cap=1.0, start=START, edge_thr=EDGE_THR):
    pR = df["polymarket_p_red"].to_numpy()
    pB = df["polymarket_p_blue"].to_numpy()
    eR = 1 + (1 - FEE) * (1 / pR - 1)
    eB = 1 + (1 - FEE) * (1 / pB - 1)
    p_R = p_red_model
    p_B = 1 - p_R
    ev_R = p_R * eR - 1
    ev_B = p_B * eB - 1
    br = ev_R >= ev_B
    cd = np.where(br, eR, eB)
    cp = np.where(br, p_R, p_B)
    won = np.where(br, df["Winner"].to_numpy() == "Red", df["Winner"].to_numpy() == "Blue").astype(int)
    ce = np.where(br, ev_R, ev_B)
    mask = ce > edge_thr
    bank = start
    n = 0
    w = 0
    pnl_log = []
    for i in range(len(p_R)):
        if not mask[i]:
            continue
        b = cd[i] - 1
        p = cp[i]
        q = 1 - p
        fk = (b * p - q) / b
        if fk <= 0:
            continue
        sf = min(kf * fk, cap)
        stake = bank * sf
        if won[i]:
            bank += stake * (cd[i] - 1)
            w += 1
        else:
            bank -= stake
        n += 1
        pnl_log.append(bank)
    return {
        "final": bank,
        "n_bets": n,
        "n_wins": w,
        "hit": w / n if n else 0.0,
        "ret": bank / start,
    }


# ============================================================================
# Slice 1: extended eval window 2025-07-01 -> 2026-05-16
# ============================================================================
EXT_START = pd.Timestamp("2025-07-01")
EXT_END = pd.Timestamp("2026-05-31")
ext = matched_full[(matched_full["date"] >= EXT_START) & (matched_full["date"] <= EXT_END)].reset_index(
    drop=True
)
print(f"Slice 1: extended window {EXT_START.date()} -> {EXT_END.date()}   n={len(ext)}")

# Multi-freeze on extended window
cutoffs = [
    ("2023-12-31_orig", pd.Timestamp("2024-01-01")),
    ("2024-06-30", pd.Timestamp("2024-07-01")),
    ("2024-12-31", pd.Timestamp("2025-01-01")),
    ("2025-06-30", pd.Timestamp("2025-07-01")),
    ("2026-03-28_fresh", pd.Timestamp("2026-03-29")),  # NEW: fresh freeze
]
print("\nTraining corrupted models at each cutoff...")
results_ext = []
preds_ext = {}
for label, c in cutoffs:
    cm, cols, cat = train_corrupted(c)
    p = predict(cm, cols, cat, ext)
    preds_ext[label] = p
    s = sim(p, ext, kf=0.25, cap=1.0)
    results_ext.append({"model": label, "cutoff": str(c.date()), **s})
    print(f"  [{label:22s}] ${START:.0f}->${s['final']:>12,.2f}  bets={s['n_bets']}  hit={s['hit']:.3f}")

# ============================================================================
# Slice 2: Apr-May 2026 alone (clean OOS for 2026-03-28 fresh freeze)
# ============================================================================
NEW_START = pd.Timestamp("2026-04-01")
NEW_END = pd.Timestamp("2026-05-31")
new = matched_full[(matched_full["date"] >= NEW_START) & (matched_full["date"] <= NEW_END)].reset_index(
    drop=True
)
print(f"\nSlice 2: NEW Apr-May 2026 only   n={len(new)}")

# Same set of corrupted freezes, evaluated only on Apr-May 2026
results_new_corr = []
for label, c in cutoffs:
    cm, cols, cat = train_corrupted(c)
    p = predict(cm, cols, cat, new)
    s = sim(p, new, kf=0.25, cap=1.0)
    results_new_corr.append({"model": "corrupt_" + label, **s})
    print(
        f"  corrupt [{label:22s}] ${START:.0f}->${s['final']:>10,.2f}  bets={s['n_bets']}  hit={s['hit']:.3f}"
    )

# Real models at the same cutoffs
print()
results_new_real = []
preds_real_apr = {}
for label, c in cutoffs:
    m, cols, cat = train_real(c)
    p = m.predict_proba(
        prepare(new, augment_symmetry=False, one_hot=False)[0]
        .reindex(columns=cols, fill_value=None)
        .pipe(lambda X: X.assign(**{cc: X[cc].fillna("__missing__").astype(str) for cc in cat}))
    )[:, 1]
    preds_real_apr[label] = p
    s = sim(p, new, kf=0.25, cap=1.0)
    results_new_real.append({"model": "real_" + label, **s})
    print(
        f"  real    [{label:22s}] ${START:.0f}->${s['final']:>10,.2f}  bets={s['n_bets']}  hit={s['hit']:.3f}"
    )

# ============================================================================
# Slice 3: the actual deployment trio on Apr-May 2026
#   A: 10%-Kelly + 10% cap, real model fresh-trained through 2026-03-28
#   B: 25%-Kelly + no cap,  real model fresh-trained through 2026-03-28
#   C: 25%-Kelly + no cap,  corrupted fresh-frozen through 2026-03-28
# ============================================================================
print("\n" + "=" * 78)
print("Slice 3: DEPLOYMENT TRIO on Apr-May 2026 (the dress rehearsal)")
print("=" * 78)
cutoff_deploy = pd.Timestamp("2026-03-29")

m_real, cols_r, cat_r = train_real(cutoff_deploy)
X = prepare(new, augment_symmetry=False, one_hot=False)[0].reindex(columns=cols_r, fill_value=None)
for c in cat_r:
    X[c] = X[c].fillna("__missing__").astype(str)
p_real_deploy = m_real.predict_proba(X)[:, 1]

m_corr, cols_c, cat_c = train_corrupted(cutoff_deploy)
p_corr_deploy = predict(m_corr, cols_c, cat_c, new)

sA = sim(p_real_deploy, new, kf=0.10, cap=0.10)
sB = sim(p_real_deploy, new, kf=0.25, cap=1.00)
sC = sim(p_corr_deploy, new, kf=0.25, cap=1.00)

print(f"\n{'Acct':6s}{'Strategy':40s}{'Final':>12s}{'Bets':>7s}{'Wins':>6s}{'Hit':>7s}{'Return':>10s}")
for tag, s, desc in [
    ("A", sA, "10%-K +10% cap | real fresh 2026-03-28"),
    ("B", sB, "25%-K no cap   | real fresh 2026-03-28"),
    ("C", sC, "25%-K no cap   | corrupt fresh 2026-03-28"),
]:
    print(
        f"{tag:6s}{desc:40s}${s['final']:>11,.2f}{s['n_bets']:>7d}{s['n_wins']:>6d}{s['hit']:>7.3f}{s['ret']:>10.3f}x"
    )

# Trio total bankroll
total = sA["final"] + sB["final"] + sC["final"]
print(f"\nTotal trio bankroll: $900 -> ${total:,.2f}  (return {total / 900:.3f}x)")

# Save artifact
out = {
    "extended_window": {
        "start": str(EXT_START.date()),
        "end": str(EXT_END.date()),
        "n_fights": int(len(ext)),
        "freezes": results_ext,
    },
    "new_apr_may_2026": {
        "start": str(NEW_START.date()),
        "end": str(NEW_END.date()),
        "n_fights": int(len(new)),
        "corrupted": results_new_corr,
        "real": results_new_real,
    },
    "deployment_trio_apr_may_2026": {
        "A": {"desc": "10%-K + 10% cap | real fresh 2026-03-28", **sA},
        "B": {"desc": "25%-K no cap   | real fresh 2026-03-28", **sB},
        "C": {"desc": "25%-K no cap   | corrupt fresh 2026-03-28", **sC},
        "total_final_bankroll": total,
    },
}
out_path = ROOT / "artifacts/metrics/extended_window_backtest.json"
out_path.write_text(json.dumps(out, indent=2, default=float))
print(f"\nWrote {out_path}")

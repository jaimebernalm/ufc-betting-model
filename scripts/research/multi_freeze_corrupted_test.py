"""Test: does the corrupted-frozen advantage come from the SPECIFIC seed=0 tree
structure trained on 2010-2023, or from a structural property of "freeze and use"?

Approach: train corrupted models with multiple cutoff dates, freeze each, and
evaluate all of them on the same shared late-test fight window (where they're
all genuinely out-of-sample). If the lucky pattern is structural, every fresh
freeze should produce a similar large advantage. If it's seed-specific luck,
only the original `v3_full2000_no_skill_corrupted_trainval` artifact will look
lucky and the others will be ordinary.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

ROOT = Path(__file__).resolve().parents[1]

from ufc_pred.features.skill_v3_pipeline import OUTPUT as SKILL_V3_PARQUET
from ufc_pred.features.static_v1 import prepare
from ufc_pred.ingest.kaggle_mdabbert import HISTORY_PARQUET
from ufc_pred.models.wrappers import CorruptedSkillModel
from ufc_pred.paths import METRICS
from ufc_pred.utils.time_splits import recency_weights

# ============================================================================
# Config
# ============================================================================
START = 300.0
FEE = 0.02
EDGE_THR = 0.03
KF = 0.25
CAP = 1.0

# Common evaluation window: late test period, where every freeze is genuinely
# out-of-sample for the LATEST cutoff (2025-06-30).
EVAL_START = pd.Timestamp("2025-07-01")
EVAL_END = pd.Timestamp("2026-04-01")

# Freeze cutoffs (model trained on data with date < cutoff)
FREEZE_CUTOFFS = [
    ("2023-12-31_orig", pd.Timestamp("2024-01-01")),  # the original — matches existing artifact
    ("2024-06-30", pd.Timestamp("2024-07-01")),
    ("2024-12-31", pd.Timestamp("2025-01-01")),
    ("2025-06-30", pd.Timestamp("2025-07-01")),
]


# ============================================================================
# Load data
# ============================================================================
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

eval_fights = matched_full[
    (matched_full["date"] >= EVAL_START) & (matched_full["date"] < EVAL_END)
].reset_index(drop=True)
print(f"Eval window: {EVAL_START.date()} → {(EVAL_END - pd.Timedelta(days=1)).date()}")
print(f"Polymarket-matched fights in eval window: {len(eval_fights)}")


# ============================================================================
# Train one corrupted model per cutoff
# ============================================================================
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
    cm = CorruptedSkillModel(m, ["skill_diff_mean", "skill_diff_std"])
    return cm, list(X.columns), cat, len(df)


def predict(mdl, cols, cat, df):
    X, _, _, _ = prepare(df, augment_symmetry=False, one_hot=False)
    X = X.reindex(columns=cols, fill_value=None)
    for c in cat:
        X[c] = X[c].fillna("__missing__").astype(str)
    return mdl.predict_proba(X)[:, 1]


def kelly_sim(p_R_model, df, kf=KF, cap=CAP):
    pR = df["polymarket_p_red"].to_numpy()
    pB = df["polymarket_p_blue"].to_numpy()
    eR = 1 + (1 - FEE) * (1 / pR - 1)
    eB = 1 + (1 - FEE) * (1 / pB - 1)
    p_R = p_R_model
    p_B = 1 - p_R
    ev_R = p_R * eR - 1
    ev_B = p_B * eB - 1
    br = ev_R >= ev_B
    ce = np.where(br, ev_R, ev_B)
    cd = np.where(br, eR, eB)
    cp = np.where(br, p_R, p_B)
    won = np.where(br, df["Winner"].to_numpy() == "Red", df["Winner"].to_numpy() == "Blue").astype(int)
    mask = ce > EDGE_THR

    bank = START
    n_bets = 0
    n_wins = 0
    side_history = []
    for i in range(len(p_R)):
        side_history.append("R" if br[i] else "B")
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
            n_wins += 1
        else:
            bank -= stake
        n_bets += 1
    return {
        "final_bankroll": bank,
        "n_bets": n_bets,
        "n_wins": n_wins,
        "hit_rate": n_wins / n_bets if n_bets else 0.0,
        "side_history": side_history,
    }


# Run each freeze
print("\nTraining frozen-corrupted models at each cutoff...")
models_info = []
predictions = {}
for label, cutoff in FREEZE_CUTOFFS:
    print(f"  [{label}] cutoff < {cutoff.date()}", end=" ")
    cm, cols, cat, n_train = train_corrupted(cutoff)
    p = predict(cm, cols, cat, eval_fights)
    predictions[label] = p
    sim = kelly_sim(p, eval_fights)
    models_info.append(
        {
            "label": label,
            "cutoff": str(cutoff.date()),
            "n_train": n_train,
            **{k: v for k, v in sim.items() if k != "side_history"},
            "side_history": sim["side_history"],
        }
    )
    print(
        f"  n_train={n_train}  ${START:.0f}→${sim['final_bankroll']:,.2f}  "
        f"n_bets={sim['n_bets']}  hit={sim['hit_rate']:.3f}"
    )


# ============================================================================
# Cross-check: do the different frozen models agree with each other?
# ============================================================================
print(f"\n{'=' * 80}")
print(f"CROSS-FREEZE COMPARISON on eval window ({len(eval_fights)} fights)")
print(f"{'=' * 80}")

# Final bankrolls side-by-side
print()
print(f"{'Freeze cutoff':25s} {'n_train':>8s} {'final_bankroll':>18s} {'n_bets':>8s} {'hit_rate':>10s}")
for m in models_info:
    print(
        f"{m['label']:25s} {m['n_train']:>8d} ${m['final_bankroll']:>16,.2f}  {m['n_bets']:>8d}  {m['hit_rate']:>10.3f}"
    )

# Pairwise side-agreement on eval fights
print()
print("--- Side-pick agreement between pairs of frozen models ---")
labels = [m["label"] for m in models_info]
hist_by_label = {m["label"]: m["side_history"] for m in models_info}
n_eval = len(eval_fights)
for i in range(len(labels)):
    for j in range(i + 1, len(labels)):
        a, b = labels[i], labels[j]
        agree = sum(1 for k in range(n_eval) if hist_by_label[a][k] == hist_by_label[b][k])
        print(f"  {a:25s} vs {b:25s}  agree {agree}/{n_eval}  ({agree / n_eval * 100:.1f}%)")

# Prediction correlation between all pairs
print()
print("--- Prediction correlations (Pearson r) ---")
for i in range(len(labels)):
    for j in range(i + 1, len(labels)):
        a, b = labels[i], labels[j]
        r = np.corrcoef(predictions[a], predictions[b])[0, 1]
        print(f"  {a:25s} vs {b:25s}  r={r:.4f}")

# Save
out = {
    "run_at": datetime.now().isoformat(timespec="seconds"),
    "eval_window": [str(EVAL_START.date()), str((EVAL_END - pd.Timedelta(days=1)).date())],
    "n_eval_fights": int(len(eval_fights)),
    "config": {"start": START, "fee": FEE, "kelly": KF, "cap": CAP, "edge_thr": EDGE_THR},
    "models": [{k: v for k, v in m.items() if k != "side_history"} for m in models_info],
}
out_path = METRICS / "multi_freeze_corrupted_test.json"
out_path.write_text(json.dumps(out, indent=2, default=float))
print(f"\nWrote {out_path}")

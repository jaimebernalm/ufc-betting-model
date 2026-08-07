"""Sweep frozen-corrupted models at many different cutoff dates.

All evaluated on the same shared OOS window 2025-07-01 -> 2026-05-31
(so every cutoff <= 2025-07-01 is genuinely out-of-sample for the whole
window). If the "frozen wins" pattern is robust, every cutoff should
do well. If 2024-01-01 was uniquely lucky, only it will stand out.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]

from ufc_pred.features.skill_v3_pipeline import OUTPUT as SKILL_V3_PARQUET
from ufc_pred.features.static_v1 import prepare
from ufc_pred.ingest.kaggle_mdabbert import HISTORY_PARQUET
from ufc_pred.models.wrappers import CorruptedSkillModel
from ufc_pred.utils.time_splits import recency_weights

FEE = 0.02
EDGE_THR = 0.03
START = 300.0
KF = 0.25
CAP = 1.0

CUTOFFS = [
    pd.Timestamp("2022-01-01"),
    pd.Timestamp("2022-07-01"),
    pd.Timestamp("2023-01-01"),
    pd.Timestamp("2023-07-01"),
    pd.Timestamp("2024-01-01"),  # original
    pd.Timestamp("2024-07-01"),
    pd.Timestamp("2025-01-01"),
    pd.Timestamp("2025-07-01"),
]

EVAL_START = pd.Timestamp("2025-07-01")
EVAL_END = pd.Timestamp("2026-05-31")

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

eval_df = matched_full[(matched_full["date"] >= EVAL_START) & (matched_full["date"] <= EVAL_END)].reset_index(
    drop=True
)
print(f"Eval window: {EVAL_START.date()} -> {EVAL_END.date()}  n={len(eval_df)}")


def train_corrupted(cutoff):
    df = fights[fights["date"] < cutoff].copy()
    X, y, d, cat = prepare(df, augment_symmetry=True, one_hot=False)
    n = len(X) // 2
    if len(X) == 2 * n and "skill_diff_mean" in X.columns:
        X = X.copy()
        X.loc[X.index[n:], "skill_diff_mean"] = -X.loc[X.index[n:], "skill_diff_mean"]
    w = recency_weights(d, reference_date=cutoff - pd.Timedelta(days=1))
    pool = Pool(X, y, cat_features=cat, weight=w)
    cb = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        loss_function="Logloss",
        random_seed=0,
        verbose=False,
        allow_writing_files=False,
    )
    cb.fit(pool, verbose=False)
    return (
        CorruptedSkillModel(cb, ["skill_diff_mean", "skill_diff_std"]),
        list(X.columns),
        cat,
        len(df),
    )


def predict(mdl, cols, cat, df):
    X, _, _, _ = prepare(df, augment_symmetry=False, one_hot=False)
    X = X.reindex(columns=cols, fill_value=None)
    for c in cat:
        X[c] = X[c].fillna("__missing__").astype(str)
    return mdl.predict_proba(X)[:, 1]


# Bet collection
pR = eval_df["polymarket_p_red"].to_numpy()
pB = eval_df["polymarket_p_blue"].to_numpy()
eR = 1 + (1 - FEE) * (1 / pR - 1)
eB = 1 + (1 - FEE) * (1 / pB - 1)
Y = (eval_df["Winner"].to_numpy() == "Red").astype(int)
fight_ids = [f"{r['date'].date()}|{r['R_fighter']}|{r['B_fighter']}" for _, r in eval_df.iterrows()]

all_bets = []
summary = []
predictions = {}
print("\nTraining frozen-corrupted at each cutoff...")
for c in CUTOFFS:
    label = c.strftime("%Y-%m-%d")
    mdl, cols, cat, n_train = train_corrupted(c)
    p_red = predict(mdl, cols, cat, eval_df)
    predictions[label] = p_red
    ev_R = p_red * eR - 1
    ev_B = (1 - p_red) * eB - 1
    br = ev_R >= ev_B
    chosen_dec = np.where(br, eR, eB)
    chosen_p = np.where(br, p_red, 1 - p_red)
    chosen_ev = np.where(br, ev_R, ev_B)
    won = np.where(br, Y == 1, Y == 0).astype(int)
    mask = chosen_ev > EDGE_THR

    # Bankroll sim
    bank = START
    n = 0
    w = 0
    for i in range(len(eval_df)):
        if not mask[i]:
            continue
        b = chosen_dec[i] - 1
        p = chosen_p[i]
        q = 1 - p
        fk = (b * p - q) / b
        if fk <= 0:
            continue
        stake = bank * min(KF * fk, CAP)
        if won[i]:
            bank += stake * (chosen_dec[i] - 1)
            w += 1
        else:
            bank -= stake
        n += 1
        all_bets.append(
            {
                "cutoff": label,
                "fight_id": fight_ids[i],
                "side": "R" if br[i] else "B",
                "won": int(won[i]),
                "dec_odds": float(chosen_dec[i]),
                "fk": float(fk),
                "pnl_flat": float(chosen_dec[i] - 1) if won[i] else -1.0,
            }
        )
    summary.append(
        {
            "cutoff": label,
            "n_train": n_train,
            "final": bank,
            "n_bets": n,
            "n_wins": w,
            "hit": w / n if n else 0.0,
            "roi_flat": np.mean([r["pnl_flat"] for r in all_bets if r["cutoff"] == label]) if n else 0.0,
        }
    )
    print(
        f"  {label}  n_train={n_train}  ${START:.0f}->${bank:>10,.2f}  bets={n:3d}  hit={w / n if n else 0:.3f}"
    )

bets_df = pd.DataFrame(all_bets)
print(f"\nTotal bet rows: {len(bets_df)}")


# Pairwise paired tests
def mcnemar_exact(b, c):
    if b + c == 0:
        return 1.0
    return binomtest(min(b, c), n=b + c, p=0.5, alternative="two-sided").pvalue


def boot_ci(diff, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


print("\n=== Pairwise tests: every frozen cutoff vs every other (paired on same fights) ===")
print(
    f"  {'A':<12s}{'B':<12s}{'paired_n':>10s}{'hitA':>7s}{'hitB':>7s}{'McP':>10s}{'pnl_diff':>10s}{'95% CI':>22s}"
)
labels = [c.strftime("%Y-%m-%d") for c in CUTOFFS]
pair_rows = []
for i in range(len(labels)):
    for j in range(i + 1, len(labels)):
        A, B = labels[i], labels[j]
        a = bets_df[bets_df["cutoff"] == A].set_index("fight_id")
        b = bets_df[bets_df["cutoff"] == B].set_index("fight_id")
        common = a.index.intersection(b.index)
        if len(common) == 0:
            continue
        a = a.loc[common].drop_duplicates()
        b = b.loc[common].drop_duplicates()
        common = a.index.intersection(b.index)
        a = a.loc[common]
        b = b.loc[common]
        bw = int(((a["won"] == 1) & (b["won"] == 0)).sum())
        cw = int(((a["won"] == 0) & (b["won"] == 1)).sum())
        p_mc = mcnemar_exact(bw, cw)
        diff = a["pnl_flat"].to_numpy() - b["pnl_flat"].to_numpy()
        mean, lo, hi = boot_ci(diff)
        sig = "*" if p_mc < 0.05 else ""
        pair_rows.append(
            {
                "A": A,
                "B": B,
                "n": int(len(common)),
                "hitA": float(a["won"].mean()),
                "hitB": float(b["won"].mean()),
                "mcp": p_mc,
                "pnl_diff": mean,
                "ci_lo": lo,
                "ci_hi": hi,
            }
        )
        print(
            f"  {A:<12s}{B:<12s}{len(common):>10d}{a['won'].mean():>7.3f}{b['won'].mean():>7.3f}"
            f"{p_mc:>10.4f}{mean:>+10.3f}  [{lo:>+5.3f},{hi:>+5.3f}] {sig}"
        )

# Prediction correlations
print("\n--- Prediction correlation matrix (Pearson r) ---")
print("              " + "  ".join([f"{lbl[2:7]:>7s}" for lbl in labels]))
for a in labels:
    row = [f"{np.corrcoef(predictions[a], predictions[b])[0, 1]:>7.3f}" for b in labels]
    print(f"  {a:<12s}" + "  ".join(row))

# How does S1 (2024-01-01) actually compare to the median?
median_hit = np.median([s["hit"] for s in summary])
median_roi = np.median([s["roi_flat"] for s in summary])
orig = next(s for s in summary if s["cutoff"] == "2024-01-01")
print(f"\nMedian across all cutoffs: hit={median_hit:.3f}  roi/bet={median_roi:+.3f}")
print(f"2024-01-01 (original):     hit={orig['hit']:.3f}  roi/bet={orig['roi_flat']:+.3f}")
print(
    f"Original vs median: hit Δ={orig['hit'] - median_hit:+.3f}  roi Δ={orig['roi_flat'] - median_roi:+.3f}"
)

out_path = ROOT / "artifacts/metrics/frozen_cutoff_sweep.json"
out_path.write_text(
    json.dumps(
        {
            "eval_window": [str(EVAL_START.date()), str(EVAL_END.date())],
            "n_eval_fights": int(len(eval_df)),
            "cutoffs": labels,
            "summary": summary,
            "pairs": pair_rows,
        },
        indent=2,
        default=float,
    )
)
print(f"\nWrote {out_path}")

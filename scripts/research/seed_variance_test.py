"""Pure seed variance baseline.

Train the SAME corrupted CatBoost (cutoff 2024-01-01) with 10 different
random seeds and evaluate each on the same Jul 2025 -> May 2026 window.
If seed-only variance is large, the cutoff-comparison bankroll spreads
we've been measuring are likely overinterpreted.

Also do a second pass with cutoff 2025-01-01 to see if seed variance
depends on cutoff.
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
SEEDS = list(range(10))
CUTOFFS_TO_TEST = [pd.Timestamp("2024-01-01"), pd.Timestamp("2025-01-01")]
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

pR = eval_df["polymarket_p_red"].to_numpy()
pB = eval_df["polymarket_p_blue"].to_numpy()
eR = 1 + (1 - FEE) * (1 / pR - 1)
eB = 1 + (1 - FEE) * (1 / pB - 1)
Y = (eval_df["Winner"].to_numpy() == "Red").astype(int)
fight_ids = [f"{r['date'].date()}|{r['R_fighter']}|{r['B_fighter']}" for _, r in eval_df.iterrows()]


def train_predict(cutoff, seed):
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
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
    )
    cb.fit(pool, verbose=False)
    mdl = CorruptedSkillModel(cb, ["skill_diff_mean", "skill_diff_std"])
    Xe, _, _, _ = prepare(eval_df, augment_symmetry=False, one_hot=False)
    Xe = Xe.reindex(columns=list(X.columns), fill_value=None)
    for c in cat:
        Xe[c] = Xe[c].fillna("__missing__").astype(str)
    return mdl.predict_proba(Xe)[:, 1]


def simulate(p_red):
    ev_R = p_red * eR - 1
    ev_B = (1 - p_red) * eB - 1
    br = ev_R >= ev_B
    chosen_dec = np.where(br, eR, eB)
    chosen_p = np.where(br, p_red, 1 - p_red)
    chosen_ev = np.where(br, ev_R, ev_B)
    won = np.where(br, Y == 1, Y == 0).astype(int)
    mask = chosen_ev > EDGE_THR
    bank = START
    n = 0
    w = 0
    bets = []
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
        bets.append(
            {
                "fight_id": fight_ids[i],
                "won": int(won[i]),
                "pnl_flat": float(chosen_dec[i] - 1) if won[i] else -1.0,
                "dec": float(chosen_dec[i]),
                "fk": float(fk),
            }
        )
    return bank, n, w, bets


def mcnemar_exact(b, c):
    if b + c == 0:
        return 1.0
    return binomtest(min(b, c), n=b + c, p=0.5, alternative="two-sided").pvalue


def boot_ci(diff, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


all_results = {}
for cutoff in CUTOFFS_TO_TEST:
    label = cutoff.strftime("%Y-%m-%d")
    print(f"\n{'=' * 78}\nCutoff: {label}\n{'=' * 78}")
    seed_results = []
    seed_bets = {}
    for seed in SEEDS:
        p_red = train_predict(cutoff, seed)
        bank, n, w, bets = simulate(p_red)
        seed_results.append(
            {
                "seed": seed,
                "final": bank,
                "n_bets": n,
                "n_wins": w,
                "hit": w / n if n else 0.0,
                "roi_flat": np.mean([b["pnl_flat"] for b in bets]) if bets else 0.0,
            }
        )
        seed_bets[seed] = pd.DataFrame(bets)
        print(
            f"  seed={seed}  ${START:.0f}->${bank:>12,.2f}  bets={n:3d}  hit={w / n if n else 0:.3f}  "
            f"roi/bet={seed_results[-1]['roi_flat']:+.3f}"
        )

    finals = np.array([r["final"] for r in seed_results])
    hits = np.array([r["hit"] for r in seed_results])
    rois = np.array([r["roi_flat"] for r in seed_results])
    print(f"\n  --- across {len(SEEDS)} seeds ---")
    print(
        f"  bankroll: min=${finals.min():,.0f}  median=${np.median(finals):,.0f}  "
        f"max=${finals.max():,.0f}  ratio max/min={finals.max() / finals.min():.2f}x"
    )
    print(
        f"  hit rate: min={hits.min():.3f}  median={np.median(hits):.3f}  max={hits.max():.3f}  spread={hits.max() - hits.min():+.3f}"
    )
    print(
        f"  roi/bet:  min={rois.min():+.3f}  median={np.median(rois):+.3f}  max={rois.max():+.3f}  spread={rois.max() - rois.min():+.3f}"
    )

    # Paired tests between seed 0 and each other seed
    print("\n  Paired tests vs seed=0:")
    base = seed_bets[0].set_index("fight_id")
    for s in SEEDS[1:]:
        other = seed_bets[s].set_index("fight_id")
        common = base.index.intersection(other.index)
        if len(common) == 0:
            continue
        a = base.loc[common].drop_duplicates()
        b = other.loc[common].drop_duplicates()
        common = a.index.intersection(b.index)
        a = a.loc[common]
        b = b.loc[common]
        bw = int(((a["won"] == 1) & (b["won"] == 0)).sum())
        cw = int(((a["won"] == 0) & (b["won"] == 1)).sum())
        p_mc = mcnemar_exact(bw, cw)
        diff = a["pnl_flat"].to_numpy() - b["pnl_flat"].to_numpy()
        mean, lo, hi = boot_ci(diff)
        print(
            f"    seed 0 vs seed {s}: paired_n={len(common)}  hitΔ={a['won'].mean() - b['won'].mean():+.3f}  "
            f"McP={p_mc:.4f}  pnl_diff={mean:+.3f}  CI [{lo:+.3f},{hi:+.3f}]"
        )

    all_results[label] = {
        "seeds": seed_results,
        "bank_min": float(finals.min()),
        "bank_max": float(finals.max()),
        "bank_median": float(np.median(finals)),
        "hit_spread": float(hits.max() - hits.min()),
        "roi_spread": float(rois.max() - rois.min()),
    }

out_path = ROOT / "artifacts/metrics/seed_variance_test.json"
out_path.write_text(json.dumps(all_results, indent=2, default=float))
print(f"\nWrote {out_path}")

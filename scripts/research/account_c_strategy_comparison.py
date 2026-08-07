"""Account C strategy comparison with paired significance tests.

Compares 5 candidate strategies for the corrupted model on the Jul2025 -> May2026
test window:
  S1 = frozen 2024-01-01 (original artifact, never retrained)
  S2 = rolling monthly
  S3 = rolling every 3 months
  S4 = rolling every 6 months
  S5 = rolling every 12 months

Paired McNemar + bootstrap PnL-diff CIs between every strategy pair on the
fights they BOTH bet on. Uses corrupted CatBoost identically across all
strategies; only the cutoff schedule differs.
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

EVAL_START = pd.Timestamp("2025-07-01")
EVAL_END = pd.Timestamp("2026-05-31")
eval_months = []
m = EVAL_START
while m <= EVAL_END:
    eval_months.append(m)
    m = (m + pd.offsets.MonthBegin(1)).normalize()

# Cache of trained corrupted models keyed by cutoff
_cache = {}


def get_corrupted(cutoff):
    if cutoff in _cache:
        return _cache[cutoff]
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
    _cache[cutoff] = (
        CorruptedSkillModel(cb, ["skill_diff_mean", "skill_diff_std"]),
        list(X.columns),
        cat,
    )
    return _cache[cutoff]


def predict(mdl_cols_cat, df):
    mdl, cols, cat = mdl_cols_cat
    X, _, _, _ = prepare(df, augment_symmetry=False, one_hot=False)
    X = X.reindex(columns=cols, fill_value=None)
    for c in cat:
        X[c] = X[c].fillna("__missing__").astype(str)
    return mdl.predict_proba(X)[:, 1]


# For each strategy, return the cutoff to use for predicting eval-month M
def cutoff_for(strategy, M):
    if strategy == "S1_frozen_2024_01":
        return pd.Timestamp("2024-01-01")
    if strategy == "S2_monthly":
        return M
    if strategy == "S3_3mo":
        # Retrain at Jul, Oct, Jan, Apr... anchored to first month of test window
        idx = (M.to_period("M") - EVAL_START.to_period("M")).n
        block = (idx // 3) * 3
        return (EVAL_START + pd.DateOffset(months=block)).normalize()
    if strategy == "S4_6mo":
        idx = (M.to_period("M") - EVAL_START.to_period("M")).n
        block = (idx // 6) * 6
        return (EVAL_START + pd.DateOffset(months=block)).normalize()
    if strategy == "S5_12mo":
        idx = (M.to_period("M") - EVAL_START.to_period("M")).n
        block = (idx // 12) * 12
        return (EVAL_START + pd.DateOffset(months=block)).normalize()
    raise ValueError(strategy)


STRATEGIES = ["S1_frozen_2024_01", "S2_monthly", "S3_3mo", "S4_6mo", "S5_12mo"]

# Print schedules for transparency
print("Retrain schedule per strategy:")
for s in STRATEGIES:
    cuts = sorted({cutoff_for(s, M) for M in eval_months})
    print(f"  {s:22s} cutoffs used: {[c.strftime('%Y-%m-%d') for c in cuts]}")
print()

# Collect per-bet records
all_rows = []
for M in eval_months:
    M_end = (M + pd.offsets.MonthBegin(1)).normalize()
    sub = matched_full[(matched_full["date"] >= M) & (matched_full["date"] < M_end)].reset_index(drop=True)
    if len(sub) == 0:
        continue
    pR = sub["polymarket_p_red"].to_numpy()
    pB = sub["polymarket_p_blue"].to_numpy()
    eR = 1 + (1 - FEE) * (1 / pR - 1)
    eB = 1 + (1 - FEE) * (1 / pB - 1)
    Y = (sub["Winner"].to_numpy() == "Red").astype(int)
    fight_ids = [f"{r['date'].date()}|{r['R_fighter']}|{r['B_fighter']}" for _, r in sub.iterrows()]
    print(f"{M.date()}  n={len(sub):2d}  ", end="")
    for s in STRATEGIES:
        cutoff = cutoff_for(s, M)
        p_red = predict(get_corrupted(cutoff), sub)
        ev_R = p_red * eR - 1
        ev_B = (1 - p_red) * eB - 1
        br = ev_R >= ev_B
        chosen_dec = np.where(br, eR, eB)
        chosen_p = np.where(br, p_red, 1 - p_red)
        chosen_ev = np.where(br, ev_R, ev_B)
        won = np.where(br, Y == 1, Y == 0).astype(int)
        mask = chosen_ev > EDGE_THR
        for i in range(len(sub)):
            if not mask[i]:
                continue
            # Skip if Kelly is negative (shouldn't happen with edge>thr but be safe)
            b = chosen_dec[i] - 1
            p = chosen_p[i]
            q = 1 - p
            fk = (b * p - q) / b
            if fk <= 0:
                continue
            all_rows.append(
                {
                    "eval_month": str(M.date()),
                    "fight_id": fight_ids[i],
                    "strategy": s,
                    "won": int(won[i]),
                    "dec_odds": float(chosen_dec[i]),
                    "fk": float(fk),
                    "pnl_flat": float(chosen_dec[i] - 1) if won[i] else -1.0,
                }
            )
        print(s.split("_")[0], end=" ")
    print()

bets = pd.DataFrame(all_rows)
print(f"\nTotal bet rows: {len(bets)}")

# Per-strategy bankroll simulation (Kelly with carry-over)
print("\n=== Per-strategy bankroll (¼-K, no cap, $300 start) ===")
results = []
for s in STRATEGIES:
    g = bets[bets["strategy"] == s].sort_values("eval_month")
    bank = START
    n = 0
    w = 0
    for _, row in g.iterrows():
        stake = bank * min(KF * row["fk"], CAP)
        if row["won"]:
            bank += stake * (row["dec_odds"] - 1)
            w += 1
        else:
            bank -= stake
        n += 1
    hit = w / n if n else 0.0
    roi_flat = g["pnl_flat"].mean() if len(g) else 0.0
    results.append({"strategy": s, "final": bank, "n_bets": n, "n_wins": w, "hit": hit, "roi_flat": roi_flat})
    print(f"  {s:22s}  ${START:.0f}->${bank:>10,.2f}  bets={n:3d}  hit={hit:.3f}  roi/bet={roi_flat:+.3f}")


# Paired significance tests between all pairs (McNemar + bootstrap PnL-diff)
def mcnemar_exact(b, c):
    if b + c == 0:
        return 1.0
    return binomtest(min(b, c), n=b + c, p=0.5, alternative="two-sided").pvalue


def boot_ci(diff, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


print("\n=== Paired significance (McNemar on side-pick win, bootstrap on per-bet flat PnL diff) ===")
print(f"  {'A vs B':<48s}{'paired_n':>10s}{'hitΔ':>8s}{'mcnemar_p':>12s}{'pnl_diff':>10s}{'95% CI':>22s}")
pair_results = []
for i in range(len(STRATEGIES)):
    for j in range(i + 1, len(STRATEGIES)):
        A, B = STRATEGIES[i], STRATEGIES[j]
        a = bets[bets["strategy"] == A].set_index("fight_id")
        b = bets[bets["strategy"] == B].set_index("fight_id")
        common = a.index.intersection(b.index)
        if len(common) == 0:
            continue
        # Dedup: a fight may appear once per strategy
        a = a.loc[common].drop_duplicates()
        b = b.loc[common].drop_duplicates()
        common = a.index.intersection(b.index)
        a = a.loc[common]
        b = b.loc[common]
        bw = int(((a["won"] == 1) & (b["won"] == 0)).sum())  # A right, B wrong
        cw = int(((a["won"] == 0) & (b["won"] == 1)).sum())  # A wrong, B right
        p_mc = mcnemar_exact(bw, cw)
        diff = a["pnl_flat"].to_numpy() - b["pnl_flat"].to_numpy()
        mean, lo, hi = boot_ci(diff)
        pair_results.append(
            {
                "A": A,
                "B": B,
                "paired_n": int(len(common)),
                "b": bw,
                "c": cw,
                "mcnemar_p": p_mc,
                "pnl_diff_mean": mean,
                "ci_lo": lo,
                "ci_hi": hi,
                "hit_A": float(a["won"].mean()),
                "hit_B": float(b["won"].mean()),
            }
        )
        print(
            f"  {A + ' vs ' + B:<48s}{len(common):>10d}{(a['won'].mean() - b['won'].mean()):>+8.3f}"
            f"{p_mc:>12.4f}{mean:>+10.3f}  [{lo:>+5.3f},{hi:>+5.3f}]"
        )

out = {
    "eval_window": [str(EVAL_START.date()), str(EVAL_END.date())],
    "strategies": STRATEGIES,
    "bankroll": results,
    "pairs": pair_results,
}
out_path = ROOT / "artifacts/metrics/account_c_strategy_comparison.json"
out_path.write_text(json.dumps(out, indent=2, default=float))
print(f"\nWrote {out_path}")

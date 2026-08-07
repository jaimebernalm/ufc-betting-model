"""Age sweep + significance test.

For every test month M in 2025-07 .. 2026-05, train corrupted *and* real
models at cutoffs M, M-1, M-3, M-6, M-12 months (so ages 0/1/3/6/12 when
predicting M). Predict month M, aggregate per-bet metrics by age across
the full test period, and run paired significance tests between age
buckets on the SAME fights.

Two tests:
  - McNemar (paired win/lose) for side-pick hit rate
  - Bootstrap CI on per-bet realized PnL diff (5000 resamples)
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
AGES = [0, 1, 3, 6, 12]  # months back from eval month start

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


def model_cache():
    cache = {}

    def get(cutoff, corrupted):
        ck = (cutoff, corrupted)
        if ck in cache:
            return cache[ck]
        df = fights[fights["date"] < cutoff].copy()
        X, y, d, cat = prepare(df, augment_symmetry=True, one_hot=False)
        if corrupted:
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
        mdl = CorruptedSkillModel(cb, ["skill_diff_mean", "skill_diff_std"]) if corrupted else cb
        cache[ck] = (mdl, list(X.columns), cat)
        return cache[ck]

    return get


get_model = model_cache()


def predict(mdl, cols, cat, df):
    X, _, _, _ = prepare(df, augment_symmetry=False, one_hot=False)
    X = X.reindex(columns=cols, fill_value=None)
    for c in cat:
        X[c] = X[c].fillna("__missing__").astype(str)
    return mdl.predict_proba(X)[:, 1]


# Collect per-bet rows tagged with (age, kind)
rows = []
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
    print(f"{M.date()}  n={len(sub)}  ", end="")
    for age in AGES:
        cutoff = (M - pd.DateOffset(months=age)).normalize()
        for kind in ("corrupt", "real"):
            mdl, cols, cat = get_model(cutoff, kind == "corrupt")
            p_red = predict(mdl, cols, cat, sub)
            ev_R = p_red * eR - 1
            ev_B = (1 - p_red) * eB - 1
            br = ev_R >= ev_B
            chosen_ev = np.where(br, ev_R, ev_B)
            chosen_dec = np.where(br, eR, eB)
            chosen_p = np.where(br, p_red, 1 - p_red)
            won = np.where(br, Y == 1, Y == 0).astype(int)
            mask = chosen_ev > EDGE_THR
            for i in range(len(sub)):
                if not mask[i]:
                    continue
                rows.append(
                    {
                        "eval_month": str(M.date()),
                        "fight_id": fight_ids[i],
                        "age_months": age,
                        "kind": kind,
                        "side": "R" if br[i] else "B",
                        "won": int(won[i]),
                        "p_model": float(chosen_p[i]),
                        "dec_odds": float(chosen_dec[i]),
                        "ev": float(chosen_ev[i]),
                        "pnl_flat": float(chosen_dec[i] - 1) if won[i] else -1.0,
                    }
                )
        print(f"a={age}", end=" ")
    print()

df = pd.DataFrame(rows)
print(f"\nTotal bet rows: {len(df)}")
df.to_parquet(ROOT / "data/interim/age_sweep_bets.parquet", index=False)

# Aggregate per (kind, age)
print("\n=== Aggregate per kind × age ===")
agg = (
    df.groupby(["kind", "age_months"])
    .agg(
        n_bets=("won", "size"),
        hit=("won", "mean"),
        roi_flat=("pnl_flat", "mean"),
    )
    .reset_index()
)
print(agg.to_string(index=False, formatters={"hit": "{:.3f}".format, "roi_flat": "{:+.3f}".format}))

# Significance tests: paired on same (fight_id, kind), age=0 vs age=K
print("\n=== Significance tests: paired age=0 vs age=K, same fight ===")


def mcnemar_exact(b, c):
    """b = age0 right & ageK wrong, c = age0 wrong & ageK right.
    Exact two-sided binomial under null p=0.5."""
    if b + c == 0:
        return 1.0
    return binomtest(min(b, c), n=b + c, p=0.5, alternative="two-sided").pvalue


def boot_ci(diff, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


for kind in ("corrupt", "real"):
    print(f"\n  --- {kind} ---")
    print(
        f"  {'age_K':>6s}{'paired_n':>10s}{'hit_0':>8s}{'hit_K':>8s}"
        f"{'b':>5s}{'c':>5s}{'mcnemar_p':>12s}"
        f"{'pnl_diff':>10s}{'95% CI':>22s}"
    )
    base = df[(df["kind"] == kind) & (df["age_months"] == 0)].set_index("fight_id")
    for K in AGES[1:]:
        other = df[(df["kind"] == kind) & (df["age_months"] == K)].set_index("fight_id")
        common = base.index.intersection(other.index)
        if len(common) == 0:
            continue
        b0 = base.loc[common]
        bK = other.loc[common]
        # McNemar
        b = int(((b0["won"] == 1) & (bK["won"] == 0)).sum())
        c = int(((b0["won"] == 0) & (bK["won"] == 1)).sum())
        p = mcnemar_exact(b, c)
        # Bootstrap PnL diff (age 0 minus age K)
        diff = b0["pnl_flat"].to_numpy() - bK["pnl_flat"].to_numpy()
        mean, lo, hi = boot_ci(diff)
        print(
            f"  {K:>6d}{len(common):>10d}{b0['won'].mean():>8.3f}{bK['won'].mean():>8.3f}"
            f"{b:>5d}{c:>5d}{p:>12.4f}"
            f"{mean:>+10.3f}  [{lo:>+5.3f},{hi:>+5.3f}]"
        )

out_path = ROOT / "artifacts/metrics/age_sweep_significance.json"
out_path.write_text(
    json.dumps(
        {
            "eval_window": [str(EVAL_START.date()), str(EVAL_END.date())],
            "ages": AGES,
            "n_bet_rows": int(len(df)),
            "aggregate": agg.to_dict(orient="records"),
        },
        indent=2,
        default=float,
    )
)
print(f"\nWrote {out_path}")

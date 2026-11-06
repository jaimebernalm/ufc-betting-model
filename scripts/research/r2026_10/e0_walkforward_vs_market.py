"""E0 — Does the stats model carry information beyond the closing line?

Yearly expanding-window walk-forward over the v2 candidate feature table
(point-in-time features rebuilt by the Sept-2026 audit). For each test year Y
we train only on fights dated before Jan 1 of Y and predict Y. The market
reference is the Kaggle/BestFightOdds closing moneyline, de-vigged
proportionally.

Models (all symmetric: corner-swap augmentation + symmetrised prediction):
  stats_v2     : the deployed recipe (2000 trees, depth 6, lr .05, l2 3, 4y half-life)
  stats_small  : a heavily regularised version of the same features
  resid        : CatBoost trained on the *residual* over the market
                 (baseline = logit(q_close)); only fights with odds

Output: artifacts/r2026_10/e0_oof.parquet (one row per fight-year prediction).
This is a diagnostic on historical years; it does not touch the shadow bundle.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from ufc_pred.features.static_v1 import _swap_red_blue, prepare
from ufc_pred.paths import ROOT
from ufc_pred.utils.time_splits import recency_weights

FEATURES = ROOT / "artifacts/shadow_candidate_2026_09_14_v2/training_features.parquet"
OUT = ROOT / "artifacts/r2026_10"


def american_to_prob(o):
    o = np.asarray(o, dtype=float)
    return np.where(o < 0, -o / (-o + 100.0), 100.0 / (o + 100.0))


def market_q(df: pd.DataFrame) -> np.ndarray:
    pr, pb = american_to_prob(df.R_odds), american_to_prob(df.B_odds)
    return pr / (pr + pb)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


RECIPES = {
    "stats_v2": dict(iterations=2000, learning_rate=0.05, depth=6, l2_leaf_reg=3),
    "stats_small": dict(iterations=400, learning_rate=0.03, depth=4, l2_leaf_reg=10, rsm=0.5),
    "resid": dict(iterations=300, learning_rate=0.03, depth=3, l2_leaf_reg=20, rsm=0.5),
}


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, recipe: str, seed: int, ref_date, threads: int):
    X, y, d, cat = prepare(train, augment_symmetry=True, one_hot=False)
    w = recency_weights(d, reference_date=ref_date)
    baseline = None
    if recipe == "resid":
        q = market_q(train)
        baseline = np.concatenate([logit(q), -logit(q)])
    model = CatBoostClassifier(
        **RECIPES[recipe],
        loss_function="Logloss",
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        thread_count=threads,
    )
    model.fit(Pool(X, y, cat_features=cat, weight=w, baseline=baseline))
    both = pd.concat([test, _swap_red_blue(test)], ignore_index=True)
    Xt, _, _, _ = prepare(both, augment_symmetry=False, one_hot=False)
    Xt = Xt.reindex(columns=model.feature_names_)
    for c in cat:
        Xt[c] = Xt[c].astype(str)
    raw = model.predict(Pool(Xt, cat_features=cat), prediction_type="RawFormulaVal")
    n = len(test)
    if recipe == "resid":
        qt = market_q(test)
        raw = raw + np.concatenate([logit(qt), -logit(qt)])
    p = 1 / (1 + np.exp(-raw))
    return (p[:n] + 1 - p[n:]) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2014-2026")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--recipes", default="stats_v2,stats_small,resid")
    args = ap.parse_args()
    y0, y1 = map(int, args.years.split("-"))
    df = pd.read_parquet(FEATURES)
    df = df[df.Winner.isin(["Red", "Blue"])].sort_values("date").reset_index(drop=True)
    rows = []
    for year in range(y0, y1 + 1):
        start = pd.Timestamp(f"{year}-01-01")
        end = pd.Timestamp(f"{year + 1}-01-01")
        train = df[df.date < start]
        test = df[(df.date >= start) & (df.date < end)].reset_index(drop=True)
        if test.empty:
            continue
        out = test[["date", "R_fighter", "B_fighter", "weight_class", "gender", "R_odds", "B_odds"]].copy()
        out["y"] = test.Winner.eq("Red").astype(int)
        out["q"] = np.where(test.R_odds.notna() & test.B_odds.notna(), market_q(test), np.nan)
        out["test_year"] = year
        out["n_train"] = len(train)
        for recipe in args.recipes.split(","):
            tr = train
            te = test
            if recipe == "resid":
                tr = train[train.R_odds.notna() & train.B_odds.notna()]
                mask = (test.R_odds.notna() & test.B_odds.notna()).to_numpy()
                te = test[mask].reset_index(drop=True)
            preds = []
            t0 = time.monotonic()
            for seed in range(args.seeds):
                preds.append(fit_predict(tr, te, recipe, seed, start - pd.Timedelta(days=1), args.threads))
            p = np.mean(preds, axis=0)
            if recipe == "resid":
                col = np.full(len(test), np.nan)
                col[mask] = p
                p = col
            out["p_" + recipe] = p
            print(
                f"{year} {recipe}: n_train={len(tr)} n_test={len(te)} {time.monotonic() - t0:.0f}s",
                flush=True,
            )
        rows.append(out)
        pd.concat(rows, ignore_index=True).to_parquet(OUT / "e0_oof.parquet", index=False)
    print("done")


if __name__ == "__main__":
    main()

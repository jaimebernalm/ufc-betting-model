"""Analyse E0 out-of-fold predictions against the closing line.

Reports, per test year and pooled:
  * log loss of market q, stats models and the residual model
  * walk-forward stacking: logit p = a*logit q + b*logit m + c, fitted on prior years only
  * walk-forward FLB recalibration of q alone (no model): logit p = a*logit q + c
  * flat-stake ROI at the actual book odds (vig included) for edge thresholds
Card-clustered bootstrap CIs (cards = fight dates) for the paired deltas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ufc_pred.paths import ROOT

OUT = ROOT / "artifacts/r2026_10"


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def ll(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def dec(o):
    o = np.asarray(o, float)
    return np.where(o < 0, 1 + 100 / -o, 1 + o / 100)


def card_boot(delta: np.ndarray, cards: np.ndarray, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    uc, inv = np.unique(cards, return_inverse=True)
    sums = np.bincount(inv, weights=delta)
    cnts = np.bincount(inv)
    stats = []
    for _ in range(n):
        idx = rng.integers(0, len(uc), len(uc))
        stats.append(sums[idx].sum() / cnts[idx].sum())
    return np.percentile(stats, [2.5, 97.5])


def walk_forward_stack(df, cols, min_years=3, C=1.0):
    """Fit logistic on prior test-years' OOF rows, apply to the current year."""
    out = np.full(len(df), np.nan)
    years = sorted(df.test_year.unique())
    for i, yr in enumerate(years):
        if i < min_years:
            continue
        tr = df[df.test_year < yr]
        te = df.test_year == yr
        X = np.column_stack([logit(tr[c]) for c in cols])
        # symmetric augmentation so no corner bias is learned
        X = np.vstack([X, -X])
        y = np.concatenate([tr.y, 1 - tr.y])
        lr = LogisticRegression(C=C, fit_intercept=False).fit(X, y)
        Xt = np.column_stack([logit(df.loc[te, c]) for c in cols])
        out[te.to_numpy()] = lr.predict_proba(Xt)[:, 1]
    return out


def roi_table(df, pcol, thresholds=(0.0, 0.02, 0.04, 0.06)):
    rows = []
    for t in thresholds:
        r, card, side_fav = [], [], []
        for _, x in df.iterrows():
            p = x[pcol]
            dr, db = dec(x.R_odds), dec(x.B_odds)
            evr, evb = p * dr - 1, (1 - p) * db - 1
            if max(evr, evb) <= t:
                continue
            if evr >= evb:
                r.append(x.y * dr - 1)
                side_fav.append(x.q >= 0.5)
            else:
                r.append((1 - x.y) * db - 1)
                side_fav.append(x.q < 0.5)
            card.append(str(x.date)[:10])
        r = np.array(r)
        if len(r) == 0:
            continue
        lo, hi = card_boot(r, np.array(card)) if len(r) > 5 else (np.nan, np.nan)
        rows.append(dict(model=pcol, ev_thr=t, bets=len(r), roi=r.mean(), lo=lo, hi=hi, fav_share=np.mean(side_fav)))
    return pd.DataFrame(rows)


def main():
    df = pd.read_parquet(OUT / "e0_oof.parquet")
    df = df[df.q.notna() & np.isfinite(df.q) & (df.R_odds.abs() >= 100) & (df.B_odds.abs() >= 100)].reset_index(drop=True)
    models = [c for c in df if c.startswith("p_")]
    df["p_flb"] = walk_forward_stack(df, ["q"])
    for m in [c for c in models if c != "p_resid"]:
        df[f"p_stack_{m[2:]}"] = walk_forward_stack(df, ["q", m])
    allp = [c for c in df if c.startswith("p_")]
    ev = df.dropna(subset=allp).reset_index(drop=True)
    cards = ev.date.astype(str).str[:10].to_numpy()
    print(f"rows with all predictions: {len(ev)}  years {ev.test_year.min()}-{ev.test_year.max()}")
    base = ll(ev.q, ev.y)
    print(f"{'model':22s} {'logloss':>8s} {'d_vs_q':>8s}  95% card CI")
    print(f"{'q (market)':22s} {base.mean():8.4f}")
    for c in allp:
        d = ll(ev[c], ev.y) - base
        lo, hi = card_boot(d, cards)
        print(f"{c:22s} {ll(ev[c], ev.y).mean():8.4f} {d.mean():+8.4f}  [{lo:+.4f}, {hi:+.4f}]")
    print("\nper year log loss")
    print(ev.assign(**{c: ll(ev[c], ev.y) for c in ["q"] + allp}).groupby("test_year")[["q"] + allp].mean().round(4).to_string())
    print("\nflat ROI at real book odds (vig included), bet the side with best EV above thr")
    tabs = pd.concat([roi_table(ev, c) for c in allp], ignore_index=True)
    print(tabs.round(4).to_string())
    tabs.to_csv(OUT / "e0_roi.csv", index=False)
    ev.to_parquet(OUT / "e0_eval.parquet", index=False)


if __name__ == "__main__":
    main()

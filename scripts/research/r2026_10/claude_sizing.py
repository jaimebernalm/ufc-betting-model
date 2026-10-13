"""Sizing study (separate from model selection): fractional Kelly with hard caps.

Equity is fixed at the start of each card (no intra-card lookahead: every stake on a card
is sized from the same pre-card equity and all results settle after the card).
Prices are the offered book odds (vig included) -- a proxy, not Kalshi execution.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ufc_pred.research.harness import OUT, american_to_decimal, load_base


def simulate(df: pd.DataFrame, pcol: str, frac: float, cap: float, card_cap: float, ev_thr: float, edge_ceiling: float):
    d = df.dropna(subset=[pcol, "q"]).sort_values("date")
    equity, peak, mdd, path = 1.0, 1.0, 0.0, []
    n_bets = 0
    for card, g in d.groupby("card", sort=True):
        stakes, rets = [], []
        for _, r in g.iterrows():
            p = r[pcol]
            dr, db = float(american_to_decimal(r.R_odds)), float(american_to_decimal(r.B_odds))
            evr, evb = p * dr - 1, (1 - p) * db - 1
            red = evr >= evb
            pp, dec, ev = (p, dr, evr) if red else (1 - p, db, evb)
            # plausibility ceiling: implausibly large disagreement with the line = probable model error
            mkt = r.q if red else 1 - r.q
            if ev <= ev_thr or (pp - mkt) > edge_ceiling:
                continue
            b = dec - 1
            kelly = max(0.0, (pp * b - (1 - pp)) / b)
            stakes.append(min(frac * kelly, cap))
            win = r.y == 1 if red else r.y == 0
            rets.append(b if win else -1.0)
        if not stakes:
            continue
        stakes = np.array(stakes)
        if stakes.sum() > card_cap:
            stakes *= card_cap / stakes.sum()
        equity *= 1 + float((stakes * np.array(rets)).sum())
        n_bets += len(stakes)
        peak = max(peak, equity)
        mdd = max(mdd, 1 - equity / peak)
        path.append(equity)
    return dict(model=pcol, frac=frac, cap=cap, card_cap=card_cap, ev_thr=ev_thr, ceiling=edge_ceiling,
                bets=n_bets, final=equity, max_dd=mdd, cards=len(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", nargs="+", required=True)
    ap.add_argument("--pcols", default="", help="comma list; default = first p_ column of each file")
    ap.add_argument("--min-year", type=int, default=2018)
    ap.add_argument("--out", default="sizing_dev.csv")
    args = ap.parse_args()
    base = load_base()[["date", "R_fighter", "B_fighter", "debut"]]
    rows = []
    for path in args.oof:
        o = pd.read_parquet(path)
        o = o.drop(columns=["debut"], errors="ignore").merge(base, on=["date", "R_fighter", "B_fighter"])
        o = o[~o.debut & (o.year >= args.min_year)]
        pcols = [c for c in args.pcols.split(",") if c in o] or [[c for c in o if c.startswith("p_")][0]]
        for pcol in pcols:
            for frac in [0.1, 0.25]:
                for cap in [0.02, 0.05]:
                    rows.append(simulate(o, pcol, frac, cap, card_cap=0.15, ev_thr=0.02, edge_ceiling=0.10))
            rows.append(simulate(o, pcol, 0.25, 1.0, card_cap=1.0, ev_thr=0.02, edge_ceiling=1.0))
    res = pd.DataFrame(rows)
    print(res.round(3).to_string(index=False))
    res.to_csv(OUT / "claude" / args.out, index=False)


if __name__ == "__main__":
    main()

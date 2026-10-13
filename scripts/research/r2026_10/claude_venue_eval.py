"""Venue validation on accredited Kalshi prices (part of the single holdout look).

Models are trained once on all fights before 2026-01-01 (the annual protocol's 2026 refit)
and scored on the accredited opportunities (artifacts/r2026_10/opportunities_kalshi.parquet).
Features: training_features v2 rows when the fight is in it (<= 2026-09-12), otherwise the
feature row archived in the shadow capture itself (same builder, same schema).

Anchor q options (reported separately):
  book   : BFO sportsbook consensus close (matched parquet), else Kaggle close in training_features
  kmid   : Kalshi mid at capture time, normalised
Bets at the captured ask with the Kalshi taker fee 0.07*p*(1-p) per contract.
"""

from __future__ import annotations

import argparse
import glob
import json

import numpy as np
import pandas as pd

from ufc_pred.research import anchored as A
from ufc_pred.research.harness import KEYS, OUT, bfo_exact, card_bootstrap_mean, load_base, logit, logloss

OPP = OUT / "opportunities_kalshi.parquet"
CAPTURES = "data/processed/shadow_candidate_2026_09_14_v2/captures/*.json"
BFO = OUT / "claude/bfo/bfo_matched.parquet"


def capture_rows(opp: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """Feature rows for opportunities missing from training_features, from the shadow captures."""
    have = opp.merge(base[KEYS], on=KEYS, how="left", indicator=True)
    missing = have[have._merge == "left_only"][KEYS]
    rows = []
    for f in glob.glob(CAPTURES):
        j = json.load(open(f))
        for r in (j.get("features") or {}).get("forward", []):
            rows.append(r)
    if not rows:
        return base.iloc[:0]
    cap = pd.DataFrame(rows)
    cap["date"] = pd.to_datetime(cap.date)
    out = []
    for _, m in missing.iterrows():
        hit = cap[(cap.date == m.date) & (cap.R_fighter == m.R_fighter) & (cap.B_fighter == m.B_fighter)]
        if len(hit) == 0:
            hit = cap[(cap.date == m.date) & (cap.R_fighter == m.B_fighter) & (cap.B_fighter == m.R_fighter)]
            if len(hit):
                from ufc_pred.features.static_v1 import _swap_red_blue

                hit = _swap_red_blue(hit.iloc[[0]])
                hit["R_fighter"], hit["B_fighter"] = m.R_fighter, m.B_fighter
        if len(hit):
            out.append(hit.iloc[[0]])
    return pd.concat(out, ignore_index=True) if out else base.iloc[:0]


def kalshi_bets(d: pd.DataFrame, pcol: str, thr: float) -> pd.DataFrame:
    fee_r = 0.07 * d.R_ask * (1 - d.R_ask)
    fee_b = 0.07 * d.B_ask * (1 - d.B_ask)
    p = d[pcol].to_numpy()
    ev_r = p / (d.R_ask + fee_r) - 1
    ev_b = (1 - p) / (d.B_ask + fee_b) - 1
    red = ev_r >= ev_b
    ev = np.where(red, ev_r, ev_b)
    cost = np.where(red, d.R_ask + fee_r, d.B_ask + fee_b)
    win = np.where(red, d.y, 1 - d.y)
    ret = win / cost - 1
    price = np.where(red, d.R_ask, d.B_ask)
    bet = (ev > thr) & np.isfinite(p)
    return pd.DataFrame({"card": d.card, "bet": bet, "ret": ret, "price": price, "ev": ev})


def summarize(d: pd.DataFrame, pcol: str, thr: float) -> dict:
    b = kalshi_bets(d, pcol, thr)
    t = b[b.bet]
    pnl = np.where(b.bet, b.ret, 0.0)
    out = dict(model=pcol, thr=thr, opps=len(b), bets=len(t), pnl_per_opp=float(pnl.mean()))
    if len(t) >= 5:
        out["roi"] = float(t.ret.mean())
        out["roi_lo"], out["roi_hi"] = card_bootstrap_mean(t.ret.to_numpy(), t.card.to_numpy())
        out["fav_share"] = float((t.price >= 0.5).mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", default="R1,R3,G0,MF")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--perf", default=None)
    args = ap.parse_args()
    opp = pd.read_parquet(OPP)
    base = load_base(fill_bfo_from="2024-01-01")
    feats = pd.concat([base, capture_rows(opp, base)], ignore_index=True)
    if args.perf:
        for p in args.perf.split(","):
            feats = feats.merge(pd.read_parquet(p), on=KEYS, how="left")
    # anchors
    bfo = bfo_exact() if BFO.exists() else None
    d = opp.merge(feats.drop(columns=["y", "card", "year"], errors="ignore"), on=KEYS, how="left")
    d["q_book"] = d.q
    if bfo is not None and "q_close_consensus_R" in bfo:
        d = d.merge(bfo[KEYS + ["q_close_consensus_R"]], on=KEYS, how="left")
        d["q_book"] = d.q_close_consensus_R.fillna(d.q)
    print(f"opportunities {len(d)}; with features {d.R_wins.notna().sum()}; with book q {d.q_book.notna().sum()}")
    train = base[base.date < pd.Timestamp("2026-01-01")].reset_index(drop=True)
    fns = {
        "R1": A.make_resid_fn(logit_q_feature=True),
        "R3": A.make_resid_fn(logit_q_feature=True, flb_baseline=True),
        "G0": A.make_glm_resid_fn(),
        "MF": A.make_flb_fn(),
    }
    rows = []
    for anchor in ["q_book", "q_kalshi_mid"]:
        t = d[d[anchor].notna() & d.R_wins.notna()].copy()
        t["q"] = t[anchor]
        t["has_odds"] = True
        t["Winner"] = np.where(t.y == 1, "Red", "Blue")
        t["p_anchor"] = t.q
        t["p_old"] = t.old_p_R
        for name in args.cands.split(","):
            preds = [fns[name](train, t.reset_index(drop=True), s) for s in range(args.seeds)]
            t["p_" + name] = np.mean(preds, axis=0)
        for c in ["p_anchor", "p_old"] + ["p_" + n for n in args.cands.split(",")]:
            for thr in [0.0, 0.02, 0.04]:
                r = summarize(t, c, thr)
                r.update(anchor=anchor, ll=float(logloss(t[c], t.y).mean()), ll_kmid=float(logloss(t.q_kalshi_mid, t.y).mean()))
                rows.append(r)
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(res.round(4).to_string(index=False))
    res.to_csv(OUT / "claude/venue_kalshi.csv", index=False)


if __name__ == "__main__":
    main()

"""Corrected multi-venue gate.

Fixes an error in the first pass: `polymarket_matched_to_kaggle_v2.parquet`'s
R_odds/B_odds are NOT sportsbook lines — they are the Polymarket probabilities
re-expressed as American odds (exact to 1e-16, zero overround). Comparing them
to Polymarket prices compared Polymarket to itself. Real book odds come from
fights.parquet (mean overround 1.043).

Adds the two tests that separate venue effect from era effect:
  - Polymarket restricted to the Kalshi window
  - head-to-head on fights priced by BOTH venues
"""

import joblib
import numpy as np
import pandas as pd

from ufc_pred.backtest.bet_eval import _effective_decimal
from ufc_pred.features.skill_v3_pipeline import OUTPUT as SKILL_V3_PARQUET
from ufc_pred.features.static_v1 import prepare
from ufc_pred.ingest.kaggle_mdabbert import HISTORY_PARQUET
from ufc_pred.paths import INTERIM, MODELS

RNG = np.random.default_rng(7)
EDGE_THR = 0.03


def imp(o):
    o = np.asarray(o, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.where(o < 0, -o / (-o + 100.0), 100.0 / (o + 100.0))
    return np.where(np.isfinite(v), v, np.nan)


def novig(r_odds, b_odds):
    ir, ib = imp(r_odds), imp(b_odds)
    tot = ir + ib
    return ir / tot, ib / tot


def gate(label, d, price_r, price_b, p_red, fee_rate, fee_model, n_boot=10000):
    price_r, price_b = np.asarray(price_r, float), np.asarray(price_b, float)
    p_red = np.asarray(p_red, float)
    ok = (
        np.isfinite(price_r)
        & np.isfinite(price_b)
        & np.isfinite(p_red)
        & (price_r >= 0.02)
        & (price_b >= 0.02)
        & (price_r <= 0.98)
        & (price_b <= 0.98)
    )
    d, price_r, price_b, p_red = d[ok], price_r[ok], price_b[ok], p_red[ok]
    if len(d) == 0:
        print(f"{label:56s} EMPTY")
        return None
    eff_r = _effective_decimal(1.0 / price_r, fee_rate, fee_model)
    eff_b = _effective_decimal(1.0 / price_b, fee_rate, fee_model)
    edge_r, edge_b = p_red - price_r, (1 - p_red) - price_b
    bet_red = edge_r >= edge_b
    m = np.where(bet_red, edge_r, edge_b) >= EDGE_THR
    if m.sum() == 0:
        print(f"{label:56s} n={len(d):4d}  NO BETS")
        return None
    won = np.where(bet_red, d["Winner"].to_numpy() == "Red", d["Winner"].to_numpy() == "Blue")
    eff = np.where(bet_red, eff_r, eff_b)
    pnl = np.where(won, eff - 1.0, -1.0)[m]
    boot = pnl[RNG.integers(0, len(pnl), (n_boot, len(pnl)))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    sig = " **" if lo > 0 or hi < 0 else ""
    print(
        f"{label:56s} n={len(d):4d} bets={m.sum():4d} hit={100 * won[m].mean():5.1f}% "
        f"ROI={100 * pnl.mean():+7.2f}% CI95=({100 * lo:+6.1f},{100 * hi:+6.1f}){sig}"
    )
    return pnl


# ---------------------------------------------------------------- rebuild data
f = pd.read_parquet(HISTORY_PARQUET)
f = f[f["Winner"].isin(["Red", "Blue"])].reset_index(drop=True)
f["date"] = pd.to_datetime(f["date"]).dt.tz_localize(None)
sk = pd.read_parquet(SKILL_V3_PARQUET)
sk["date"] = pd.to_datetime(sk["date"]).dt.tz_localize(None)
f = f.merge(
    sk[["date", "R_fighter", "B_fighter", "skill_diff_mean", "skill_diff_std"]],
    on=["date", "R_fighter", "B_fighter"],
    how="left",
    validate="many_to_one",
)

seen = {}
rp = []
bp = []
for _, row in f.sort_values("date").iterrows():
    rp.append(seen.get(row["R_fighter"], 0))
    bp.append(seen.get(row["B_fighter"], 0))
    seen[row["R_fighter"]] = seen.get(row["R_fighter"], 0) + 1
    seen[row["B_fighter"]] = seen.get(row["B_fighter"], 0) + 1
idx = f.sort_values("date").index
f["r_prior"] = pd.Series(rp, index=idx).reindex(f.index)
f["b_prior"] = pd.Series(bp, index=idx).reindex(f.index)
f["is_debut"] = (f["r_prior"] == 0) | (f["b_prior"] == 0)

tv = joblib.load(MODELS / "v3_catboost_full2000_trainval.joblib")
X, _, _, _ = prepare(f, augment_symmetry=False, one_hot=False)
X = X.reindex(columns=tv["columns"], fill_value=None)
for c in tv.get("cat_features", []):
    X[c] = X[c].fillna("__missing__").astype(str)
f["p_tv"] = tv["model"].predict_proba(X)[:, 1]

poly = pd.read_parquet(INTERIM / "polymarket_matched_to_kaggle_v2.parquet")
poly["date"] = pd.to_datetime(poly["date"]).dt.tz_localize(None)
P = poly[["date", "R_fighter", "B_fighter", "polymarket_p_red", "polymarket_p_blue"]].merge(
    f[["date", "R_fighter", "B_fighter", "Winner", "R_odds", "B_odds", "is_debut", "p_tv"]],
    on=["date", "R_fighter", "B_fighter"],
    how="inner",
)

kal = pd.read_parquet("/tmp/kalshi_union_best.parquet")
kal = kal[kal["_ok"] & kal["canon_a"].notna() & kal["canon_b"].notna()].copy()
kal["date"] = pd.to_datetime(kal["fight_date"]).dt.tz_localize(None)
a = f.merge(
    kal,
    left_on=["date", "R_fighter", "B_fighter"],
    right_on=["date", "canon_a", "canon_b"],
    how="inner",
)
a["kal_R"], a["kal_B"] = a["close_yes_price_a"], a["close_yes_price_b"]
b = f.merge(
    kal,
    left_on=["date", "R_fighter", "B_fighter"],
    right_on=["date", "canon_b", "canon_a"],
    how="inner",
)
b["kal_R"], b["kal_B"] = b["close_yes_price_b"], b["close_yes_price_a"]
K = pd.concat([a, b], ignore_index=True).drop_duplicates(["date", "R_fighter", "B_fighter"])

Pn, Kn = P[~P["is_debut"]], K[~K["is_debut"]]
W0, W1 = "2026-01-24", "2026-05-16"  # window where both venues exist

print("=" * 122)
print("A. LONG-HISTORY POLYMARKET  (trainval model, out-of-sample from 2024-01)")
print("=" * 122)
gate(
    "Polymarket price | debut-excl | poly fee 2%",
    Pn,
    Pn["polymarket_p_red"],
    Pn["polymarket_p_blue"],
    Pn["p_tv"],
    0.02,
    "polymarket",
)
gate(
    "Polymarket price | debut-excl | Kalshi fee 7%",
    Pn,
    Pn["polymarket_p_red"],
    Pn["polymarket_p_blue"],
    Pn["p_tv"],
    0.07,
    "kalshi",
)
gate(
    "Polymarket price | ALL fights | poly fee 2%",
    P,
    P["polymarket_p_red"],
    P["polymarket_p_blue"],
    P["p_tv"],
    0.02,
    "polymarket",
)

print("\n" + "=" * 122)
print("B. REAL SPORTSBOOK no-vig on the SAME rows (odds from fights.parquet, overround 1.043)")
print("=" * 122)
pr, pb = novig(Pn["R_odds"], Pn["B_odds"])
gate("Sportsbook no-vig | debut-excl | poly fee 2%", Pn, pr, pb, Pn["p_tv"], 0.02, "polymarket")
gate("Sportsbook no-vig | debut-excl | Kalshi fee 7%", Pn, pr, pb, Pn["p_tv"], 0.07, "kalshi")

print("\n" + "=" * 122)
print("C. ERA vs VENUE — same model, same protocol, matched calendar window")
print("=" * 122)
Pw = Pn[(Pn["date"] >= W0) & (Pn["date"] <= W1)]
Kw = Kn[(Kn["date"] >= W0) & (Kn["date"] <= W1)]
gate(
    f"Polymarket {W0}..{W1}",
    Pw,
    Pw["polymarket_p_red"],
    Pw["polymarket_p_blue"],
    Pw["p_tv"],
    0.02,
    "polymarket",
)
gate(f"Kalshi     {W0}..{W1}", Kw, Kw["kal_R"], Kw["kal_B"], Kw["p_tv"], 0.07, "kalshi")
gate(
    "Polymarket BEFORE 2026-01-24",
    Pn[Pn["date"] < W0],
    Pn[Pn["date"] < W0]["polymarket_p_red"],
    Pn[Pn["date"] < W0]["polymarket_p_blue"],
    Pn[Pn["date"] < W0]["p_tv"],
    0.02,
    "polymarket",
)
gate(
    "Kalshi     AFTER  2026-05-16",
    Kn[Kn["date"] > W1],
    Kn[Kn["date"] > W1]["kal_R"],
    Kn[Kn["date"] > W1]["kal_B"],
    Kn[Kn["date"] > W1]["p_tv"],
    0.07,
    "kalshi",
)

print("\n" + "=" * 122)
print("D. HEAD-TO-HEAD — identical fights priced by BOTH venues")
print("=" * 122)
H = Kn.merge(
    P[["date", "R_fighter", "B_fighter", "polymarket_p_red", "polymarket_p_blue"]],
    on=["date", "R_fighter", "B_fighter"],
    how="inner",
)
print(f"fights priced by both venues: {len(H)}")
if len(H):
    gate(
        "  same fights | Polymarket price",
        H,
        H["polymarket_p_red"],
        H["polymarket_p_blue"],
        H["p_tv"],
        0.02,
        "polymarket",
    )
    gate("  same fights | Kalshi price", H, H["kal_R"], H["kal_B"], H["p_tv"], 0.07, "kalshi")
    d = (H["kal_R"] - H["polymarket_p_red"]).astype(float)
    print(
        f"  price gap (Kalshi - Poly) on R side: mean {d.mean():+.4f}  "
        f"median {d.median():+.4f}  MAE {d.abs().mean():.4f}"
    )
    gate(
        "  same fights | Kalshi price, poly fee 2%",
        H,
        H["kal_R"],
        H["kal_B"],
        H["p_tv"],
        0.02,
        "polymarket",
    )

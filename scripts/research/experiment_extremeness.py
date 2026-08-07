"""Does prediction EXTREMENESS alone explain Account C's advantage?

Account C (corrupted model) produces more extreme probabilities than the real
model and posts bigger simulated bankrolls. Hypothesis to test: sharpening the
real model's probabilities toward the extremes (temperature < 1 on logits)
replicates C's results — i.e. C's advantage is an overconfidence/Kelly-sizing
interaction, not better information.

Setup mirrors scripts/all_accounts_seed_grid.py: cutoff 2025-01-01 (deployed
analog), eval window 2025-07-01 -> 2026-05-31, Polymarket matched prices,
FEE=0.02, EDGE_THR=0.03. Models are retrained with the FIXED augmentation
(2026-06-11), so absolute numbers differ slightly from the stored grid.

Sizing: quarter-Kelly, no per-bet cap (the B/C account config), simulated
card-open (all bets on a card sized off the card-opening bankroll) with a
100% total-exposure cap — the realistic regime from the 2026-06-11 audit.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

import argparse

from ufc_pred.backtest.strategy_grid import (  # noqa: E402
    EDGE_THR,
    FEE,
    predict,
    train_corrupted,
    train_real,
)
from ufc_pred.features.skill_v3_pipeline import OUTPUT as SKILL_V3_PARQUET  # noqa: E402
from ufc_pred.ingest.kaggle_mdabbert import HISTORY_PARQUET  # noqa: E402

_ap = argparse.ArgumentParser()
_ap.add_argument("--window", choices=["main", "earlier"], default="main")
_args = _ap.parse_args()
if _args.window == "main":
    CUTOFF = pd.Timestamp("2025-01-01")
    EVAL_START, EVAL_END = pd.Timestamp("2025-07-01"), pd.Timestamp("2026-05-31")
else:
    CUTOFF = pd.Timestamp("2024-01-01")
    EVAL_START, EVAL_END = pd.Timestamp("2024-07-01"), pd.Timestamp("2025-06-30")
SEEDS = range(5)
TEMPS = [1.0, 1.25, 1.5, 2.0, 3.0]  # logit multipliers; >1 = sharper
KFRAC, START = 0.25, 300.0
print(
    f"window={_args.window}  cutoff={CUTOFF.date()}  eval {EVAL_START.date()} -> {EVAL_END.date()}",
    flush=True,
)

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
ev = matched[key + ["polymarket_p_red", "polymarket_p_blue"]].merge(
    fights,
    on=key,
    how="left",
    validate="one_to_one",
)
ev = ev[(ev["date"] >= EVAL_START) & (ev["date"] <= EVAL_END)].sort_values("date").reset_index(drop=True)
print(f"eval fights: {len(ev)}  {ev['date'].min().date()} -> {ev['date'].max().date()}", flush=True)

pR = ev["polymarket_p_red"].to_numpy()
pB = ev["polymarket_p_blue"].to_numpy()
eR = 1 + (1 - FEE) * (1 / pR - 1)
eB = 1 + (1 - FEE) * (1 / pB - 1)
Y = (ev["Winner"].to_numpy() == "Red").astype(int)
dates = ev["date"].dt.strftime("%Y-%m-%d").to_numpy()


def sharpen(p: np.ndarray, t: float) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p)) * t
    return 1.0 / (1.0 + np.exp(-z))


def simulate(p_red: np.ndarray) -> dict:
    ev_R = p_red * eR - 1
    ev_B = (1 - p_red) * eB - 1
    br = ev_R >= ev_B
    dec = np.where(br, eR, eB)
    p_ch = np.where(br, p_red, 1 - p_red)
    evc = np.where(br, ev_R, ev_B)
    won = np.where(br, Y == 1, Y == 0).astype(int)
    mask = evc > EDGE_THR
    b = dec - 1.0
    fk = (b * p_ch - (1 - p_ch)) / b
    mask &= fk > 0

    frac = KFRAC * fk[mask]
    mult = np.where(won[mask] == 1, b[mask], -1.0)
    d = dates[mask]
    bank = START
    for day in pd.unique(d):
        ix = d == day
        f_, m_ = frac[ix], mult[ix]
        scale = min(1.0, 1.0 / f_.sum()) if f_.sum() > 0 else 1.0
        bank += bank * np.sum(f_ * scale * m_)
        bank = max(bank, 0.0)

    pnl_flat = np.where(won[mask] == 1, b[mask], -1.0)
    pc = np.clip(p_red, 1e-6, 1 - 1e-6)
    ll = float(-np.mean(Y * np.log(pc) + (1 - Y) * np.log(1 - pc)))
    return {
        "n_bets": int(mask.sum()),
        "flat_roi_pct": float(pnl_flat.mean() * 100),
        "hit": float(won[mask].mean()),
        "final_capped": float(bank),
        "log_loss": ll,
        "extremeness": float(np.mean(np.abs(p_red - 0.5))),
    }


rows = []
for seed in SEEDS:
    print(f"seed {seed}: training real + corrupted ...", flush=True)
    bre = train_real(fights, CUTOFF, seed=seed)
    bco = train_corrupted(fights, CUTOFF, seed=seed)
    p_real = predict(bre, ev)
    p_corr = predict(bco, ev)
    for t in TEMPS:
        rows.append({"seed": seed, "variant": f"real T={t}", **simulate(sharpen(p_real, t))})
        rows.append({"seed": seed, "variant": f"corr T={t}", **simulate(sharpen(p_corr, t))})

df = pd.DataFrame(rows)
agg = df.groupby("variant").median(numeric_only=True).drop(columns="seed")
agg = agg.loc[[f"real T={t}" for t in TEMPS] + [f"corr T={t}" for t in TEMPS]]
pd.set_option("display.width", 160)
print("\nMedians over 5 seeds (quarter-Kelly uncapped, card-open + 100% exposure cap):")
print(agg.round(3).to_string())
out = ROOT / f"data/interim/experiment_extremeness_{_args.window}.parquet"
df.to_parquet(out, index=False)
print(f"\nper-seed rows -> {out}")

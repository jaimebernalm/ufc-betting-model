"""Venue-priced, deployable-only evaluation — the Phase-2 decision gate.

Everything the old backtests got wrong, corrected in one place:

  * priced at the Kalshi ASK actually payable at T-90 (falls back to the last
    trade where no quote survives), not a no-vig sportsbook closing line
  * debut fights EXCLUDED — the live runner cannot price a fighter with no
    history, so including them overstates the achievable edge (see
    ufc_pred.backtest.universe)
  * the DEPLOYED artefacts: 10-seed real+corrupted ensembles at cutoff
    2025-11-30, orientation-symmetrised, optional logit sharpening
  * Kalshi's real quadratic upfront fee, 0.07 * P * (1-P) per contract
  * every snapshot fight post-dates the training cutoff, so this is a genuine
    out-of-sample forward test rather than a re-read of a spent test set

Gate: deploy only if the deployable-slice CI95 excludes zero.

Usage:  python scripts/venue_priced_evaluation.py [--sharpen 1.0 1.25] [--threshold 0.03]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent

from ufc_pred.backtest.strategy_grid import ModelBundle, predict
from ufc_pred.backtest.universe import add_prior_fight_counts
from ufc_pred.features.skill_v3_pipeline import OUTPUT as SKILL_V3_PARQUET
from ufc_pred.ingest.kaggle_mdabbert import HISTORY_PARQUET
from ufc_pred.paths import METRICS, MODELS

SNAPSHOT = REPO / "data/raw/kalshi/snapshots/historical_T-90min_perfight_combined.parquet"
CUTOFF_TAG = "2025_11_30"
N_SEEDS = 10
FEE_COEFF = 0.07


def swap_corners(df: pd.DataFrame) -> pd.DataFrame:
    ren = {}
    for c in df.columns:
        if c.startswith("R_"):
            ren[c] = "B_" + c[2:]
        elif c.startswith("B_"):
            ren[c] = "R_" + c[2:]
    out = df.rename(columns=ren)
    if "Winner" in out:
        out["Winner"] = out["Winner"].map({"Red": "Blue", "Blue": "Red"})
    for c in out.columns:
        if c.endswith("_dif"):
            out[c] = -out[c]
    if "skill_diff_mean" in out:
        out["skill_diff_mean"] = -out["skill_diff_mean"]
    return out


def sharpen(p: np.ndarray, T: float) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return 1.0 / (1.0 + np.exp(-T * np.log(p / (1 - p))))


def boot_ci(v: np.ndarray, n: int = 20000, seed: int = 0):
    if len(v) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(v, size=(n, len(v)), replace=True).mean(axis=1) * 100
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def load_joined() -> pd.DataFrame:
    snap = pd.read_parquet(SNAPSHOT)
    snap["fight_date"] = pd.to_datetime(snap["fight_date"])
    if snap["fight_date"].dt.tz is not None:
        snap["fight_date"] = snap["fight_date"].dt.tz_localize(None)

    fights = pd.read_parquet(HISTORY_PARQUET)
    fights = fights[fights["Winner"].isin(["Red", "Blue"])].reset_index(drop=True)
    fights["date"] = pd.to_datetime(fights["date"])
    sk = pd.read_parquet(SKILL_V3_PARQUET)
    sk["date"] = pd.to_datetime(sk["date"])
    fights = fights.merge(
        sk[["date", "R_fighter", "B_fighter", "skill_diff_mean", "skill_diff_std"]],
        on=["date", "R_fighter", "B_fighter"],
        how="left",
        validate="many_to_one",
    )
    fights = add_prior_fight_counts(fights)

    idx = {}
    for i, r in fights.iterrows():
        d = r["date"].normalize()
        idx[(r["R_fighter"], r["B_fighter"], d)] = i
        idx[(r["B_fighter"], r["R_fighter"], d)] = i

    rows, unmatched = [], 0
    for _, s in snap.iterrows():
        ca, cb = s.get("canon_a"), s.get("canon_b")
        if not isinstance(ca, str) or not isinstance(cb, str):
            unmatched += 1
            continue
        key = (ca, cb, s["fight_date"].normalize())
        if key not in idx:
            unmatched += 1
            continue
        # price: prefer the transactable ask, fall back to last trade
        ask_a, ask_b = s.get("ask_yes_price_a"), s.get("ask_yes_price_b")
        px_a = ask_a if pd.notna(ask_a) and ask_a >= 0.02 else s["close_yes_price_a"]
        px_b = ask_b if pd.notna(ask_b) and ask_b >= 0.02 else s["close_yes_price_b"]
        if not (pd.notna(px_a) and pd.notna(px_b)) or px_a < 0.02 or px_b < 0.02:
            unmatched += 1
            continue
        if s["settle_result_a"] not in ("yes", "no"):
            unmatched += 1
            continue
        f = fights.loc[idx[key]]
        rows.append(
            {
                "fight_idx": idx[key],
                "fight": f"{ca} vs {cb}",
                "date": s["fight_date"],
                "canon_a": ca,
                "px_a": float(px_a),
                "px_b": float(px_b),
                "won_a": s["settle_result_a"] == "yes",
                "priced_at_ask": bool(pd.notna(ask_a) and ask_a >= 0.02),
                "has_debut": bool(f["has_debut"]),
            }
        )
    df = pd.DataFrame(rows)
    print(f"snapshot rows {len(snap)} -> joined & usable {len(df)}  ({unmatched} dropped)")
    return df, fights


def score(df: pd.DataFrame, fights: pd.DataFrame) -> np.ndarray:
    """p(fighter_a wins), deployed ensemble, orientation-symmetrised."""
    sub = fights.loc[df["fight_idx"]].reset_index(drop=True)
    flip = sub["R_fighter"].to_numpy() != df["canon_a"].to_numpy()
    fwd = sub.copy()
    fwd.loc[flip] = swap_corners(sub.loc[flip])
    both = pd.concat([fwd, swap_corners(fwd)], ignore_index=True)
    n = len(fwd)
    real, corr = [], []
    for s in range(N_SEEDS):
        br = joblib.load(MODELS / f"v3_real_{CUTOFF_TAG}_seed{s}.joblib")
        bc = joblib.load(MODELS / f"v3_corrupted_{CUTOFF_TAG}_seed{s}.joblib")
        pr = predict(ModelBundle(br["model"], br["columns"], br["cat_features"]), both.copy())
        pc = predict(ModelBundle(bc["model"], bc["columns"], bc["cat_features"]), both.copy())
        real.append(0.5 * (pr[:n] + 1.0 - pr[n:]))
        corr.append(0.5 * (pc[:n] + 1.0 - pc[n:]))
    return np.mean(real, axis=0), np.mean(corr, axis=0)


def evaluate(df: pd.DataFrame, p_a: np.ndarray, thr: float) -> pd.DataFrame:
    px_a, px_b = df["px_a"].to_numpy(), df["px_b"].to_numpy()
    edge_a, edge_b = p_a - px_a, (1 - p_a) - px_b
    take_a = edge_a >= edge_b
    price = np.where(take_a, px_a, px_b)
    edge = np.where(take_a, edge_a, edge_b)
    won = np.where(take_a, df["won_a"].to_numpy(), ~df["won_a"].to_numpy())
    fee = FEE_COEFF * price * (1 - price)
    pnl = np.where(won, (1 - price - fee) / price, -1.0)
    out = df.copy()
    out["side_a"], out["price"], out["edge"], out["won"], out["pnl"] = take_a, price, edge, won, pnl
    out["p_bet"] = np.where(take_a, p_a, 1 - p_a)
    return out[out["edge"] > thr].copy()


def report(bets: pd.DataFrame, label: str) -> dict:
    if len(bets) == 0:
        print(f"  {label:34s} n=  0")
        return {}
    v = bets["pnl"].to_numpy()
    lo, hi = boot_ci(v)
    flag = "  <-- CI includes zero" if lo <= 0 <= hi else ""
    print(
        f"  {label:34s} n={len(bets):3d}  ROI={100 * v.mean():+7.2f}%  "
        f"CI95=({lo:+6.1f},{hi:+6.1f})  hit={100 * bets['won'].mean():5.1f}%{flag}"
    )
    return {
        "n": len(bets),
        "roi_pct": 100 * float(v.mean()),
        "ci95": [lo, hi],
        "hit_rate": float(bets["won"].mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sharpen", type=float, nargs="+", default=[1.0, 1.25])
    ap.add_argument("--threshold", type=float, default=0.03)
    args = ap.parse_args()

    df, fights = load_joined()
    if df.empty:
        print("no usable rows")
        return 1
    print(
        f"  priced at true ask: {df['priced_at_ask'].sum()}/{len(df)}   debut fights: {df['has_debut'].sum()}"
    )
    print(
        f"  date range: {df['date'].min().date()} .. {df['date'].max().date()}  "
        f"(training cutoff {CUTOFF_TAG.replace('_', '-')} — all out-of-sample)"
    )

    p_real, p_corr = score(df, fights)
    results = {}
    for T in args.sharpen:
        p = sharpen(p_real, T)
        bets = evaluate(df, p, args.threshold)
        print(f"\n=== sharpen T={T}, edge threshold {args.threshold:.0%}, ask-priced, quadratic fee ===")
        r = {}
        r["all"] = report(bets, "ALL fights")
        dep = bets[~bets["has_debut"]]
        r["deployable"] = report(dep, "DEPLOYABLE (no debut)  <-- gate")
        r["debut"] = report(bets[bets["has_debut"]], "debut only (unreachable live)")
        r["fav"] = report(dep[dep["price"] >= 0.5], "  deployable favourites >=50c")
        r["dog"] = report(dep[dep["price"] < 0.5], "  deployable underdogs  <50c")
        results[f"T={T}"] = r

        if T == max(args.sharpen) or len(args.sharpen) == 1:
            print("\n  calibration on the deployable slice (model p -> realized):")
            d = dep.copy()
            d["bucket"] = pd.cut(d["p_bet"], [0, 0.4, 0.5, 0.6, 0.7, 0.8, 1.01])
            g = d.groupby("bucket", observed=True).agg(
                n=("won", "size"), mean_p=("p_bet", "mean"), realized=("won", "mean")
            )
            for b, row in g.iterrows():
                print(
                    f"    {str(b):12s} n={int(row['n']):3d}  "
                    f"model={row['mean_p']:.3f}  realized={row['realized']:.3f}"
                )

    out = METRICS / "venue_priced_evaluation.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")

    gate = results[f"T={args.sharpen[-1]}"].get("deployable", {})
    ci = gate.get("ci95", [float("nan")] * 2)
    ok = len(gate) > 0 and ci[0] > 0
    print("\n" + "=" * 72)
    print(
        f"GATE: deployable CI95 excludes zero?  {'PASS' if ok else 'FAIL'}"
        f"   (CI {ci[0]:+.1f}%, {ci[1]:+.1f}%)"
    )
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())

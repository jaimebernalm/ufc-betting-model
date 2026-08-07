"""Model-free anchor / line-shopping strategy, ported from TennisPred (Task 1.1).

Strategy: take the sportsbook no-vig probability (Kaggle BestFightOdds-sourced
`R_odds`/`B_odds`) as the truth estimate; bet the Kalshi side whose T-90min
pre-fight price offers EV above a threshold after Kalshi's verified quadratic
taker fee (0.07 × P × (1−P), finding #33). The UFC analogue of Kaunitz, Zhong
& Kreiner (2017) — see ../TennisPred/scripts/anchor_strategy.py for the tennis
findings this ports.

No trained model → the full Kalshi-priced history (2026-01 →, limited by
Kaggle outcome coverage) is a legitimate evaluation sample. Thresholds, odds
buckets, and variants were pre-registered in TENNIS_PORTED_IDEAS.md §6 on
2026-07-09 before results were looked at.

Caveat vs tennis: tennis's edge came from best-of-20-books line shopping;
here there is exactly one bettable venue, so the honest tennis comparison row
is the AVG-anchored variant (+4.07% only with the odds cap).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from ufc_pred.backtest.kalshi_match import match_kalshi_to_fights
from ufc_pred.backtest.metrics import american_to_implied_prob
from ufc_pred.paths import ROOT

LOG_PATH = ROOT / "experiments" / "2026-07-09_anchor_strategy.jsonl"
KALSHI_FEE = 0.07

# Pre-registered (TENNIS_PORTED_IDEAS.md §6): chosen-side Kalshi price buckets.
BUCKETS = [("p>=0.50", 0.50, 1.01), ("0.33-0.50", 0.33, 0.50), ("p<0.33", 0.0, 0.33)]


def kalshi_eff_dec(price: np.ndarray, fee_rate: float = KALSHI_FEE) -> np.ndarray:
    """Effective decimal odds buying YES at `price`, quadratic taker fee upfront.

    Per $1 of total capital (cost + fee): eff = 1 / (P × (1 + fee×(1−P))).
    """
    return 1.0 / (price * (1.0 + fee_rate * (1.0 - price)))


def build_dataset(anchor: str = "bfo") -> pd.DataFrame:
    """anchor="bfo": BFO no-vig as truth (pre-registered 1.1).
    anchor="poly": Polymarket pre-fight prob as truth (amendment 1.1b)."""
    fights = pd.read_parquet(ROOT / "data/processed/fights.parquet")
    fights = fights[fights["Winner"].isin(["Red", "Blue"])].copy()
    meta, matched = match_kalshi_to_fights(fights)
    df = pd.concat(
        [
            meta.reset_index(drop=True),
            matched[
                [
                    "R_fighter",
                    "B_fighter",
                    "R_odds",
                    "B_odds",
                    "Winner",
                    "weight_class",
                    "gender",
                    "title_bout",
                    "no_of_rounds",
                ]
            ].reset_index(drop=True),
        ],
        axis=1,
    )
    if anchor == "bfo":
        df = df.dropna(subset=["R_odds", "B_odds"]).reset_index(drop=True)
        p_r = american_to_implied_prob(df["R_odds"])
        p_b = american_to_implied_prob(df["B_odds"])
        df["anchor_p_red"] = np.asarray(p_r / (p_r + p_b), float)
    elif anchor == "poly":
        pm = pd.read_parquet(ROOT / "data/interim/polymarket_matched_to_kaggle_v2.parquet")
        pm["date"] = pd.to_datetime(pm["date"])
        df = df.merge(
            pm[["date", "R_fighter", "B_fighter", "polymarket_p_red", "polymarket_p_blue"]],
            on=["date", "R_fighter", "B_fighter"],
            how="inner",
            validate="one_to_one",
        )
        tot = df["polymarket_p_red"] + df["polymarket_p_blue"]
        df["anchor_p_red"] = (df["polymarket_p_red"] / tot).to_numpy(float)
    else:
        raise ValueError(anchor)
    df["y_red"] = (df["Winner"] == "Red").astype(int)
    return df


def build_dataset_poly_bettable() -> pd.DataFrame:
    """Amendment 1.1c: anchor = BFO no-vig, bettable = Polymarket pre-fight
    price, window 2024-04 → 2026-03 (both sources present)."""
    pm = pd.read_parquet(ROOT / "data/interim/polymarket_matched_to_kaggle_v2.parquet")
    pm["date"] = pd.to_datetime(pm["date"])
    # pm's own R_odds/B_odds are DERIVED FROM the Polymarket prices (legacy
    # compat in rebuild_polymarket_matched_v2.py) — join the real BFO
    # moneylines from fights.parquet instead.
    fights = pd.read_parquet(ROOT / "data/processed/fights.parquet")
    fights["date"] = pd.to_datetime(fights["date"])
    df = pm.drop(columns=["R_odds", "B_odds"]).merge(
        fights[
            [
                "date",
                "R_fighter",
                "B_fighter",
                "R_odds",
                "B_odds",
                "weight_class",
                "gender",
                "title_bout",
                "no_of_rounds",
            ]
        ],
        on=["date", "R_fighter", "B_fighter"],
        how="inner",
        validate="one_to_one",
    )
    df = df.dropna(subset=["R_odds", "B_odds", "polymarket_p_red", "polymarket_p_blue"]).copy()
    p_r = american_to_implied_prob(df["R_odds"])
    p_b = american_to_implied_prob(df["B_odds"])
    df["anchor_p_red"] = np.asarray(p_r / (p_r + p_b), float)
    df["kal_p_red"] = df["polymarket_p_red"]  # reuse run_variant plumbing
    df["kal_p_blue"] = df["polymarket_p_blue"]
    df["y_red"] = (df["Winner"] == "Red").astype(int)
    return df.reset_index(drop=True)


def run_variant(
    df: pd.DataFrame,
    name: str,
    thr: float,
    price_lo: float = 0.0,
    price_hi: float = 1.01,
    fee_rate: float = KALSHI_FEE,
    log: bool = True,
) -> dict:
    p = df["anchor_p_red"].to_numpy()
    y = df["y_red"].to_numpy()
    kr = df["kal_p_red"].to_numpy()
    kb = df["kal_p_blue"].to_numpy()
    eff_R = kalshi_eff_dec(kr, fee_rate)
    eff_B = kalshi_eff_dec(kb, fee_rate)
    ev_R = p * eff_R - 1.0
    ev_B = (1.0 - p) * eff_B - 1.0
    side_r = ev_R >= ev_B
    edge = np.where(side_r, ev_R, ev_B)
    chosen_price = np.where(side_r, kr, kb)
    chosen_eff = np.where(side_r, eff_R, eff_B)

    bets = (edge > thr) & (chosen_price >= price_lo) & (chosen_price < price_hi)
    won = np.where(side_r, y == 1, y == 0)[bets]
    pnl = np.where(won, chosen_eff[bets] - 1.0, -1.0)
    n = int(bets.sum())
    roi = float(pnl.mean() * 100) if n else float("nan")
    se = float(pnl.std(ddof=1) / np.sqrt(n) * 100) if n > 1 else float("nan")
    row = {
        "name": name,
        "thr": thr,
        "price_bucket": [price_lo, price_hi],
        "fee_rate": fee_rate,
        "n_bets": n,
        "n_eligible": int(len(df)),
        "roi_pct": round(roi, 3) if n else None,
        "ci95_halfwidth": round(1.96 * se, 3) if n > 1 else None,
        "hit_rate": round(float(won.mean()), 3) if n else None,
        "mean_edge_pct": round(float(edge[bets].mean() * 100), 2) if n else None,
        "n_side_red": int((np.where(side_r, 1, 0)[bets]).sum()),
        "ts": datetime.now(UTC).isoformat(),
    }
    if log:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a") as f:
            f.write(json.dumps(row) + "\n")
    ci = f"±{1.96 * se:5.2f}" if n > 1 else "      "
    print(f"{name:38s} thr={thr:5.3f} n={n:4d} ROI={roi:7.2f}% {ci} hit={row['hit_rate']}")
    return row


def _describe(df: pd.DataFrame, label: str):
    gap = df["anchor_p_red"] - df["kal_p_red"] / (df["kal_p_red"] + df["kal_p_blue"])
    print(f"{len(df)} fights, {label} ({df['date'].min().date()} → {df['date'].max().date()})")
    print(
        f"anchor − bettable-mid gap: mean {gap.mean():+.4f}, "
        f"median {gap.median():+.4f}, |gap| median {gap.abs().median():.4f}\n"
    )


def main() -> int:
    print("=" * 74)
    print("1.1  BFO-anchored, Kalshi bettable (pre-registered)")
    print("=" * 74)
    df = build_dataset("bfo")
    _describe(df, "BFO anchor → Kalshi T-90min")
    print("— Trustworthy-bettor curve —")
    for thr in (0.0, 0.01, 0.02, 0.03, 0.05):
        run_variant(df, "anchor_bfo_kalshi", thr)
    print("\n— Pre-registered odds buckets at thr=0.02 —")
    for bname, lo, hi in BUCKETS:
        run_variant(df, f"anchor_bfo_kalshi_{bname}", 0.02, price_lo=lo, price_hi=hi)

    print("\n" + "=" * 74)
    print("1.1b Polymarket-anchored, Kalshi bettable (amendment, cross-venue)")
    print("=" * 74)
    dfb = build_dataset("poly")
    _describe(dfb, "Polymarket anchor → Kalshi T-90min")
    for thr in (0.0, 0.01, 0.02, 0.03, 0.05):
        run_variant(dfb, "anchor_poly_kalshi", thr)
    print("\n— Buckets at thr=0.02 —")
    for bname, lo, hi in BUCKETS:
        run_variant(dfb, f"anchor_poly_kalshi_{bname}", 0.02, price_lo=lo, price_hi=hi)

    print("\n" + "=" * 74)
    print("1.1c BFO-anchored, Polymarket bettable, 2024-04 → 2026-03 (amendment)")
    print("      (model-free ⇒ spent-test window usable; fee 0.03 quadratic)")
    print("=" * 74)
    dfc = build_dataset_poly_bettable()
    _describe(dfc, "BFO anchor → Polymarket pre-fight")
    for thr in (0.0, 0.01, 0.02, 0.03, 0.05):
        run_variant(dfc, "anchor_bfo_poly", thr, fee_rate=0.03)
    print("\n— Buckets at thr=0.02 —")
    for bname, lo, hi in BUCKETS:
        run_variant(dfc, f"anchor_bfo_poly_{bname}", 0.02, price_lo=lo, price_hi=hi, fee_rate=0.03)
    print("\n— Half-year breakdown, thr=0.02 (sign consistency) —")
    dfc["half"] = (
        dfc["date"].dt.year.astype(str) + "H" + ((dfc["date"].dt.quarter > 2).astype(int) + 1).astype(str)
    )
    for h, g in dfc.groupby("half"):
        run_variant(g.reset_index(drop=True), f"anchor_bfo_poly_{h}", 0.02, fee_rate=0.03)

    print("\n— Capped variant p ≥ 0.33 (tennis dec ≤ 3.0 analog; the two upper")
    print("  pre-registered buckets pooled — flag: pooling is post-hoc) —")
    for thr in (0.01, 0.02, 0.03, 0.05):
        run_variant(dfc, "anchor_bfo_poly_cap33", thr, price_lo=0.33, fee_rate=0.03)
    print("  half-year consistency at thr=0.02:")
    for h, g in dfc.groupby("half"):
        run_variant(
            g.reset_index(drop=True),
            f"anchor_bfo_poly_cap33_{h}",
            0.02,
            price_lo=0.33,
            fee_rate=0.03,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

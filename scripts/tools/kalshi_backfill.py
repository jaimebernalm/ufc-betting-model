"""Pull all settled Kalshi UFC fights since N months ago into a parquet."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

REPO = Path(__file__).resolve().parent.parent
load_dotenv(REPO / ".env")
os.environ.setdefault("KALSHI_ENV", "prod")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, default=18)
    ap.add_argument("--out", type=Path, default=REPO / "data/raw/kalshi/historical.parquet")
    ap.add_argument(
        "--include-non-ufc",
        action="store_true",
        help="Include Netflix-MMA-Special and other non-UFC events",
    )
    ap.add_argument(
        "--buffer-min",
        type=int,
        default=240,
        help="Settlement buffer in minutes (per-fight capture; default 240 = ~4h pre-fight)",
    )
    args = ap.parse_args()

    from ufc_pred.ingest.kalshi_client import KalshiClient
    from ufc_pred.ingest.kalshi_history import build_polymarket_cutoffs, iter_settled_fights
    from ufc_pred.ingest.kalshi_name_match import build_name_index, match_name

    since = pd.Timestamp.now(tz="UTC").normalize() - pd.DateOffset(months=args.months)
    print(f"Backfilling Kalshi settled fights since {since.date()} (ufc_only={not args.include_non_ufc})")

    # Load Polymarket historical for cross-referenced cutoffs (gold standard
    # pre-fight close detection). Falls back to gap-detect heuristic where
    # no Polymarket match exists.
    pm_path = REPO / "data/raw/polymarket/historical_2024-04-13_to_2026-05-24.parquet"
    pm_cutoffs = None
    if pm_path.exists():
        poly = pd.read_parquet(pm_path)
        pm_cutoffs = build_polymarket_cutoffs(poly)
        print(f"Loaded {len(poly)} Polymarket rows → {len(pm_cutoffs)} cutoff keys")
    else:
        print(f"WARN: no Polymarket parquet at {pm_path}; using gap-detect heuristic only")

    t0 = time.time()
    rows = []
    print(f"Settlement buffer = {args.buffer_min} minutes")
    for hf in iter_settled_fights(
        client=KalshiClient(),
        since=since,
        ufc_only=not args.include_non_ufc,
        sleep_s=0.05,
        polymarket_cutoffs=pm_cutoffs,
        settlement_buffer_minutes=args.buffer_min,
    ):
        rows.append(hf)
        if len(rows) % 50 == 0:
            print(f"  ... {len(rows)} fights ({time.time() - t0:.0f}s)")
    df = pd.DataFrame([h.__dict__ for h in rows])
    print(f"\nFetched {len(df)} fights in {time.time() - t0:.0f}s")
    if df.empty:
        return 1
    df = df.sort_values("fight_date", ascending=True).reset_index(drop=True)

    # Reconcile names against fights.parquet
    fights = pd.read_parquet(REPO / "data/processed/fights.parquet")
    idx = build_name_index(fights)
    ma = df["fighter_a"].map(lambda n: match_name(n, None, idx))
    mb = df["fighter_b"].map(lambda n: match_name(n, None, idx))
    df["canon_a"] = [m.canonical for m in ma]
    df["canon_b"] = [m.canonical for m in mb]
    df["match_score_a"] = [m.score for m in ma]
    df["match_score_b"] = [m.score for m in mb]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    matched = (df["canon_a"].notna() & df["canon_b"].notna()).sum()
    print(f"  match rate (both fighters canonical): {matched}/{len(df)} ({100 * matched / len(df):.1f}%)")
    print(f"  date range: {df['fight_date'].min().date()} → {df['fight_date'].max().date()}")
    print(
        f"  with closing price A+B: {(df['close_yes_price_a'].notna() & df['close_yes_price_b'].notna()).sum()}/{len(df)}"
    )
    print(f"  with winner labeled:   {df['winner'].notna().sum()}/{len(df)}")
    try:
        print(f"  saved → {args.out.relative_to(REPO)}")
    except ValueError:
        print(f"  saved → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

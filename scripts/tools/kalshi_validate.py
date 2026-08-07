"""Validate kalshi_card (live) + name reconciliation + kalshi_history (backfill).

Usage:
    PYTHONPATH=src .conda/bin/python scripts/kalshi_validate.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

REPO = Path(__file__).resolve().parent.parent
load_dotenv(REPO / ".env")
os.environ.setdefault("KALSHI_ENV", "prod")


def main() -> int:
    from ufc_pred.inference.kalshi_card import fetch_next_card
    from ufc_pred.ingest.kalshi_client import KalshiClient
    from ufc_pred.ingest.kalshi_history import iter_settled_fights
    from ufc_pred.ingest.kalshi_name_match import build_name_index, match_name

    client = KalshiClient()
    fights = pd.read_parquet(REPO / "data/processed/fights.parquet")
    name_idx = build_name_index(fights)
    print(f"fights.parquet: {len(fights):,} rows, {len(name_idx['all_names']):,} unique fighters")

    # 1) LIVE: tonight's card via kalshi_card.fetch_next_card
    print("\n=== [1] LIVE: tonight's card ===")
    card = fetch_next_card(client=client, target_date=pd.Timestamp.now(tz="UTC"))
    if not card:
        print("  No fights today via target_date; falling back to next soonest card.")
        card = fetch_next_card(client=client)
    print(f"  {len(card)} fights on {card[0].fight_date.date() if card else 'n/a'}")

    print(f"\n  {'fight':<48} {'best_ask_A':>11} {'depth5c_A':>10}  {'best_ask_B':>11} {'depth5c_B':>10}")
    print("  " + "-" * 96)
    for f in card:
        ba_a = f.asks_a[0][0] if f.asks_a else None
        ba_b = f.asks_b[0][0] if f.asks_b else None
        d_a = sum(q for p, q in f.asks_a if ba_a and p <= ba_a + 0.05)
        d_b = sum(q for p, q in f.asks_b if ba_b and p <= ba_b + 0.05)
        label = f"{f.fighter_a[:22]} vs {f.fighter_b[:22]}"
        print(f"  {label:<48} {ba_a or 0:>11.2f} {d_a:>10.0f}  {ba_b or 0:>11.2f} {d_b:>10.0f}")

    # 2) NAME RECONCILIATION on the live card
    print("\n=== [2] Name reconciliation: Kalshi -> fights.parquet ===")
    print(f"  {'kalshi':<28} {'canonical':<28} {'score':>6}  {'method'}")
    print("  " + "-" * 80)
    unmatched = 0
    for f in card:
        for n in (f.fighter_a, f.fighter_b):
            m = match_name(n, f.fight_date, name_idx)
            tag = "✓" if m.canonical else "✗"
            if not m.canonical:
                unmatched += 1
            print(f"  {tag} {n:<26} {(m.canonical or '— no match —'):<28} {m.score:>6.1f}  {m.method}")
    print(f"\n  Unmatched: {unmatched}/{2 * len(card)}  ({'OK' if unmatched == 0 else 'investigate above'})")

    # 3) HISTORICAL backfill — last 60 days
    print("\n=== [3] Historical backfill (last 60 days) ===")
    since = pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=60)
    hist_rows = []
    for hf in iter_settled_fights(client=client, since=since, sleep_s=0.05):
        hist_rows.append(hf)
        if len(hist_rows) >= 200:
            break  # safety cap
    df = pd.DataFrame([h.__dict__ for h in hist_rows])
    if df.empty:
        print("  no settled fights in window")
        return 0
    df = df.sort_values("fight_date", ascending=False)
    print(f"  {len(df)} settled fights since {since.date()}")
    print(f"  date range: {df['fight_date'].min().date()} -> {df['fight_date'].max().date()}")
    print(f"  with closing yes-price on A: {df['close_yes_price_a'].notna().sum()}/{len(df)}")
    print(f"  with settlement result:      {df['winner'].notna().sum()}/{len(df)}")
    print(f"  median vol_a (24h fp units): {df['volume_a'].median():.0f}")
    print(f"  median OI_a   (contracts):   {df['open_interest_a'].median():.0f}")

    # Name reconciliation on historical
    print("\n  Match historical fighter names -> canonical:")
    match_a = df["fighter_a"].map(lambda n: match_name(n, None, name_idx))
    match_b = df["fighter_b"].map(lambda n: match_name(n, None, name_idx))
    df["canon_a"] = [m.canonical for m in match_a]
    df["canon_b"] = [m.canonical for m in match_b]
    df["score_a"] = [m.score for m in match_a]
    df["score_b"] = [m.score for m in match_b]
    matched = (df["canon_a"].notna() & df["canon_b"].notna()).sum()
    print(f"  both fighters matched: {matched}/{len(df)}")
    bad = df[df["canon_a"].isna() | df["canon_b"].isna()]
    if len(bad):
        print("\n  Sample unmatched (need manual review):")
        for _, r in bad.head(10).iterrows():
            print(
                f"    {r['fight_date'].date()}  {r['fighter_a']!r:<28}->{r['canon_a']!r:<28}  vs  {r['fighter_b']!r:<28}->{r['canon_b']!r}"
            )

    # Sanity: a few rows
    print("\n  Sample (most recent 5 fights):")
    cols = [
        "fight_date",
        "fighter_a",
        "fighter_b",
        "close_yes_price_a",
        "close_yes_price_b",
        "winner",
        "volume_a",
        "volume_b",
    ]
    print(df[cols].head(5).to_string(index=False))

    # Save artifact
    out = REPO / "data/raw/kalshi"
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / f"historical_{since.date()}_to_{pd.Timestamp.now().date()}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\n  saved -> {out_path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

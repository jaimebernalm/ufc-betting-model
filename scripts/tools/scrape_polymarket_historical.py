"""Scrape Polymarket closing odds for already-resolved UFC fights.

Default window: 2026-03-29 (day after test-set end) through today.

Usage:
    PYTHONPATH=src .conda/bin/python scripts/scrape_polymarket_historical.py
    PYTHONPATH=src .conda/bin/python scripts/scrape_polymarket_historical.py 2026-03-29 2026-05-24

Writes:
    data/raw/polymarket/historical_{start}_to_{end}.parquet
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime

from ufc_pred.ingest.polymarket import scrape_historical
from ufc_pred.paths import RAW

OUT_DIR = RAW / "polymarket"
DEFAULT_START = "2026-03-29"


def main() -> None:
    start = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_START
    end = sys.argv[2] if len(sys.argv) > 2 else datetime.now(UTC).strftime("%Y-%m-%d")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Scraping closed UFC markets {start} → {end} ...")
    df = scrape_historical(start, end)

    if df.empty:
        print("No closed UFC fight markets in window.")
        return

    out = OUT_DIR / f"historical_{start}_to_{end}.parquet"
    df.to_parquet(out, index=False)

    n_with_price = df["closing_price_a"].notna().sum()
    print(f"\nWrote {len(df)} fights ({n_with_price} with closing price) → {out}")
    print(
        df[["fight_date", "fighter_a", "fighter_b", "closing_price_a", "winner", "volume"]].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()

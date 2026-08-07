"""Scrape Polymarket UFC odds and write a daily snapshot parquet.

Usage:
    PYTHONPATH=src .conda/bin/python scripts/scrape_polymarket.py

Writes:
    data/raw/polymarket/YYYY-MM-DD.parquet   (today's snapshot)
    data/raw/polymarket/latest.parquet       (overwritten each run)
"""

from __future__ import annotations

from datetime import UTC, datetime

from ufc_pred.ingest.polymarket import scrape
from ufc_pred.paths import RAW

OUT_DIR = RAW / "polymarket"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = scrape()
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    snapshot = OUT_DIR / f"{today}.parquet"
    latest = OUT_DIR / "latest.parquet"

    if df.empty:
        print("No open UFC fight markets returned by Polymarket.")
        return

    df.to_parquet(snapshot, index=False)
    df.to_parquet(latest, index=False)

    print(f"Wrote {len(df)} fights → {snapshot}")
    print(
        df[["fight_date", "fighter_a", "fighter_b", "p_a", "p_b", "volume", "liquidity"]].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()

"""Snapshot Kalshi fills into a local, append-only store.

Why: Kalshi's API drops old portfolio history within days of settlement, so
fills must be captured while a card is live (or shortly after). This store is
the ground truth for what was ACTUALLY bet — vs bet_notifications/, which
record what was RECOMMENDED. The two can differ (manual bets with no alert,
partial fills, user-adjusted sizes); per-account (A/B/C virtual ledger)
attribution reconciles actual fills pro-rata against the recommendation's
per-account share ratios (1:2:6 default for off-alert bets).

Store: data/processed/kalshi_fills.json — {fill_id: fill_dict}, merged on
every run; existing entries are never overwritten. Idempotent, read-only
against Kalshi (a single paginated GET), safe to call every runner tick.

CLI:  python scripts/save_fills.py [-v]
"""

from __future__ import annotations

import argparse
import json

from dotenv import load_dotenv

from ufc_pred.ingest.kalshi_client import KalshiClient
from ufc_pred.paths import PROCESSED, ROOT

load_dotenv(ROOT / ".env")

FILLS_PATH = PROCESSED / "kalshi_fills.json"
PAGE_LIMIT = 200
MAX_PAGES = 25  # safety bound; 5000 fills per run is far beyond one card


def fetch_fills(client: KalshiClient | None = None) -> list[dict]:
    """Fetch all currently-visible fills, newest first (paginated)."""
    client = client or KalshiClient()
    out: list[dict] = []
    cursor = None
    for _ in range(MAX_PAGES):
        params: dict = {"limit": PAGE_LIMIT}
        if cursor:
            params["cursor"] = cursor
        r = client.request("GET", "/portfolio/fills", params=params)
        fills = r.get("fills", []) or []
        out.extend(fills)
        cursor = r.get("cursor")
        if not cursor or not fills:
            break
    else:
        raise RuntimeError("Fill pagination exceeded MAX_PAGES; refusing an incomplete snapshot")
    return out


def save_fills(client: KalshiClient | None = None, *, verbose: bool = False) -> int:
    """Merge fresh fills into the store. Returns the number of NEW fills."""
    import fcntl

    fresh = fetch_fills(client)
    FILLS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FILLS_PATH.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        # A corrupt archive must remain visible and must block accounting.
        store = json.loads(FILLS_PATH.read_text()) if FILLS_PATH.exists() else {}
        new = 0
        for fill in fresh:
            fid = fill.get("fill_id") or fill.get("id") or fill.get("trade_id")
            if fid is None:
                raise ValueError("Fill without stable venue ID; cannot guarantee deduplication")
            if fid not in store:
                store[fid] = fill
                new += 1
        if new:
            temp = FILLS_PATH.with_suffix(".tmp")
            temp.write_text(json.dumps(store, indent=1))
            temp.replace(FILLS_PATH)
        if verbose:
            print(f"[fills] {len(fresh)} visible, {new} new, {len(store)} archived")
        return new


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verbose", "-v", action="store_true", default=True)
    args = ap.parse_args()
    n = save_fills(verbose=args.verbose)
    print(f"saved {n} new fill(s) -> {FILLS_PATH}")


if __name__ == "__main__":
    main()

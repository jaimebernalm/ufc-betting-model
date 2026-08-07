"""Read-only Kalshi API smoke test for tonight's UFC card.

Verifies:
  1. Auth works (signed requests succeed)
  2. We can discover today's UFC card under KXUFCFIGHT
  3. For each fight: prices + liquidity + orderbook depth

Usage:
    PYTHONPATH=src .conda/bin/python scripts/kalshi_smoke_test.py
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")


def _as_float(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", choices=["prod", "demo"], help="Override KALSHI_ENV")
    ap.add_argument("--date", help="YYYY-MM-DD; default today (UTC). Searches event tickers for this date.")
    ap.add_argument(
        "--all-upcoming",
        action="store_true",
        help="Show all open KXUFCFIGHT events, not just one date",
    )
    args = ap.parse_args()
    if args.env:
        os.environ["KALSHI_ENV"] = args.env

    from ufc_pred.ingest.kalshi_client import KalshiClient

    c = KalshiClient()
    print(f"== Kalshi smoke test ({c.config.env}) ==")
    print(f"   base_url: {c.config.base_url}")
    print(f"   key_id:   {c.config.key_id[:8]}...{c.config.key_id[-4:]}")

    print("\n[1] /exchange/status")
    print(f"    {c.get_exchange_status()}")

    print("\n[2] /portfolio/balance")
    bal = c.get_balance()
    print(
        f"    ${bal.get('balance', 0) / 100:.2f}  (portfolio_value=${bal.get('portfolio_value', 0) / 100:.2f})"
    )

    target_date = args.date or datetime.utcnow().strftime("%Y-%m-%d")
    yr, mo, day = target_date.split("-")
    months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    date_tag = f"{yr[2:]}{months[int(mo) - 1]}{int(day):02d}"
    print(f"\n[3] /events?series_ticker=KXUFCFIGHT (looking for {date_tag} in ticker)")
    r = c.request("GET", "/events", params={"series_ticker": "KXUFCFIGHT", "limit": 200})
    all_evs = r.get("events", [])
    if args.all_upcoming:
        evs = all_evs
    else:
        evs = [e for e in all_evs if date_tag in e.get("event_ticker", "")]
    print(f"    {len(all_evs)} total events under series; {len(evs)} match {date_tag}")
    if not evs:
        print(f"    No fights on {target_date}. Use --all-upcoming to see everything, or --date YYYY-MM-DD.")
        return 0

    print(f"\n[4] Fights on {target_date}:\n")
    print(
        f"{'FIGHT':<48} {'A:bid/ask':>11} {'A_vol':>8} {'A_OI':>8} {'A_depth':>9}  |  {'B:bid/ask':>11} {'B_vol':>8} {'B_OI':>8} {'B_depth':>9}"
    )
    print("-" * 160)

    total_open_orders_value = 0.0
    for ev in evs:
        et = ev.get("event_ticker")
        title = ev.get("sub_title") or ev.get("title", "")
        mkts = c.list_markets(event_ticker=et, limit=10).get("markets", [])
        if len(mkts) != 2:
            print(f"  {title}: unexpected market count {len(mkts)}; skipping")
            continue

        rows = []
        for m in mkts:
            t = m["ticker"]
            name = m.get("yes_sub_title") or t.split("-")[-1]
            yb = _as_float(m.get("yes_bid_dollars"))
            ya = _as_float(m.get("yes_ask_dollars"))
            vol = _as_float(m.get("volume_24h_fp"), 0) or _as_float(m.get("volume_fp"), 0) or 0
            oi = _as_float(m.get("open_interest_fp"), 0) or 0
            # Orderbook: to BUY yes, we lift the no resting orders.
            # Convert (no_price p, qty) -> ask for yes at (1-p) with qty.
            ob = c.get_orderbook(t, depth=50)
            yes_asks = sorted([(round(1 - p, 4), q) for (p, q) in ob["no"]])
            # Depth = sum of qty within 5¢ of best ask
            best_ask = yes_asks[0][0] if yes_asks else None
            depth_5c = sum(q for (p, q) in yes_asks if best_ask is not None and p <= best_ask + 0.05)
            rows.append(
                {
                    "name": name,
                    "yb": yb,
                    "ya": ya,
                    "vol": vol,
                    "oi": oi,
                    "best_ask": best_ask,
                    "depth_5c": depth_5c,
                    "yes_asks": yes_asks[:5],
                }
            )

        a, b = rows
        fight_label = f"{a['name'][:22]} vs {b['name'][:22]}"

        def fmt(r):
            ba = f"{r['yb']:.2f}/{r['ya']:.2f}" if r["yb"] is not None and r["ya"] is not None else "  -  "
            return f"{ba:>11} {r['vol']:>8.0f} {r['oi']:>8.0f} {r['depth_5c']:>9.0f}"

        print(f"{fight_label:<48} {fmt(a)}  |  {fmt(b)}")
        total_open_orders_value += a["depth_5c"] * (a["best_ask"] or 0) + b["depth_5c"] * (b["best_ask"] or 0)

    print("\n[5] Sanity:")
    print(
        f"    Total $-value of resting offers within 5¢ of best ask across the card: ${total_open_orders_value:,.0f}"
    )
    print("    (Rough proxy for total fillable size at decent prices)")
    print(
        "\n[done] Auth, market discovery, and orderbook all working. Next step: wire to model + Kelly sizing."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

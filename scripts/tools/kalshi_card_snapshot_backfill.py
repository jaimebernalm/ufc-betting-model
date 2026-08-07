"""Card-snapshot backfill: one Kalshi capture timestamp per FIGHT NIGHT.

Simulates a realistic deployment: you check Kalshi prices ONCE, about 30
minutes before the first prelim starts, then use those prices for every
bet on that card. Models a "set-and-forget" deployment without
intra-card price refreshes.

For each card (= group of fights on the same date):
  1. Find the earliest fight-end timestamp across all markets on that card
     (= when the first prelim concluded — that fight started ~20 min earlier).
  2. card_snapshot_ts = first_fight_end - (fight_buffer + 30 min)
     fight_buffer absorbs typical first-fight duration + walkouts (~20-30 min);
     +30 min places us 30 min BEFORE the cage door closed for that first fight.
  3. For each market on the card, capture last trade BEFORE card_snapshot_ts.

Saves to data/raw/kalshi/snapshots/historical_cardsnapshot.parquet.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

REPO = Path(__file__).resolve().parent.parent
load_dotenv(REPO / ".env")
os.environ.setdefault("KALSHI_ENV", "prod")


def find_fight_end(client, ticker: str, page_size: int = 1000, max_pages: int = 20) -> pd.Timestamp | None:
    """Find approximate fight end time = first trade NOT at settlement price
    (walking newest-first). Same logic as fetch_pre_fight_price step 1.
    """
    r = client.request("GET", "/markets/trades", params={"ticker": ticker, "limit": 1})
    newest = r.get("trades", [])
    if not newest:
        return None
    try:
        settle_price = float(newest[0]["yes_price_dollars"])
    except (KeyError, ValueError, TypeError):
        return None
    cursor = None
    for _ in range(max_pages):
        params = {"ticker": ticker, "limit": page_size}
        if cursor:
            params["cursor"] = cursor
        r = client.request("GET", "/markets/trades", params=params)
        trades = r.get("trades", [])
        if not trades:
            return None
        for t in trades:
            try:
                yp = float(t["yes_price_dollars"])
            except (KeyError, ValueError, TypeError):
                continue
            if abs(yp - settle_price) > 0.005:
                ts = pd.Timestamp(t["created_time"])
                if ts.tz is None:
                    ts = ts.tz_localize("UTC")
                return ts
        cursor = r.get("cursor")
        if not cursor:
            return None
    return None


def trade_before(client, ticker: str, cutoff_ts: pd.Timestamp) -> tuple[float | None, pd.Timestamp | None]:
    """Get the most recent trade with timestamp < cutoff_ts."""
    r = client.request(
        "GET",
        "/markets/trades",
        params={"ticker": ticker, "limit": 1, "max_ts": int(cutoff_ts.timestamp())},
    )
    pre = r.get("trades", [])
    if not pre:
        return None, None
    try:
        ts = pd.Timestamp(pre[0]["created_time"])
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        yp = float(pre[0]["yes_price_dollars"])
        return yp, ts
    except (KeyError, ValueError, TypeError):
        return None, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, default=24)
    ap.add_argument(
        "--first-fight-buffer-min",
        type=int,
        default=30,
        help="Minutes before first fight cage-door close (default 30)",
    )
    ap.add_argument(
        "--first-fight-duration-min",
        type=int,
        default=20,
        help="Estimated first-prelim duration+walkouts in minutes (default 20)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO / "data/raw/kalshi/snapshots/historical_cardsnapshot.parquet",
    )
    args = ap.parse_args()

    from ufc_pred.ingest.kalshi_client import KalshiClient
    from ufc_pred.ingest.kalshi_history import _is_real_ufc, _parse_ticker_date
    from ufc_pred.ingest.kalshi_name_match import build_name_index, match_name

    client = KalshiClient()
    since = pd.Timestamp.now(tz="UTC").normalize() - pd.DateOffset(months=args.months)
    print(
        f"Card-snapshot backfill: capture all markets {args.first_fight_buffer_min}min "
        f"before first fight (assuming first-fight duration ~{args.first_fight_duration_min}min)"
    )
    print(f"Since: {since.date()}\n")

    # Phase 1: discover all settled UFC events grouped by card (date)
    print("Phase 1: discovering settled events...")
    cards: dict = defaultdict(list)  # date -> [event_dict, ...]
    cursor = None
    while True:
        params = {"series_ticker": "KXUFCFIGHT", "status": "settled", "limit": 200}
        if cursor:
            params["cursor"] = cursor
        r = client.request("GET", "/events", params=params)
        evs = r.get("events", [])
        if not evs:
            break
        keep_paging = False
        for ev in evs:
            fd = _parse_ticker_date(ev.get("event_ticker", ""))
            if fd is None:
                continue
            if fd < since:
                continue
            keep_paging = True
            if not _is_real_ufc(ev):
                continue
            cards[fd].append(ev)
        cursor = r.get("cursor")
        if not cursor or not keep_paging:
            break
    print(f"  Found {sum(len(v) for v in cards.values())} events across {len(cards)} cards\n")

    # Phase 2: for each card, find the earliest fight-end across its markets
    # to compute card_snapshot_ts.
    print("Phase 2: computing card snapshot timestamps...")
    card_snapshot: dict[pd.Timestamp, pd.Timestamp] = {}
    card_markets: dict[pd.Timestamp, list] = {}  # date -> [(event, [markets])]
    t0 = time.time()
    for i, (date, events) in enumerate(sorted(cards.items())):
        ends: list[pd.Timestamp] = []
        cd_markets = []
        for ev in events:
            mkts = client.list_markets(event_ticker=ev["event_ticker"], status="settled", limit=10).get(
                "markets", []
            )
            if len(mkts) != 2:
                continue
            # We only need one fight-end from the card to anchor — use the
            # market with the smallest yes_bid_dollars (lopsided fights settle
            # fast). Actually simpler: try both markets, take earliest fight_end.
            cd_markets.append((ev, mkts))
            # Sample fight_end from market_a only (faster). For card-snapshot
            # we just need an approximate earliest, not per-market precision.
            fe = find_fight_end(client, mkts[0]["ticker"])
            if fe is not None:
                ends.append(fe)
        if not ends:
            continue
        first_fight_end = min(ends)
        snapshot = first_fight_end - pd.Timedelta(
            minutes=args.first_fight_duration_min + args.first_fight_buffer_min
        )
        card_snapshot[date] = snapshot
        card_markets[date] = cd_markets
        if (i + 1) % 5 == 0:
            print(
                f"  ... {i + 1}/{len(cards)} cards ({time.time() - t0:.0f}s)  "
                f"latest: {date.date()} snapshot={snapshot}"
            )

    # Phase 3: pull each market's price as of card_snapshot
    print(
        f"\nPhase 3: pulling prices at card snapshots ({sum(len(v) for v in card_markets.values())} markets)..."
    )
    rows = []
    fights = pd.read_parquet(REPO / "data/processed/fights.parquet")
    idx = build_name_index(fights)
    t0 = time.time()
    n_done = 0
    for date, ev_markets in card_markets.items():
        snap = card_snapshot[date]
        for ev, mkts in ev_markets:
            m_a, m_b = mkts[0], mkts[1]
            name_a = m_a.get("yes_sub_title") or m_a["ticker"].split("-")[-1]
            name_b = m_b.get("yes_sub_title") or m_b["ticker"].split("-")[-1]
            res_a = m_a.get("result")
            res_b = m_b.get("result")
            winner = "A" if res_a == "yes" else ("B" if res_b == "yes" else None)
            p_a, ts_a = trade_before(client, m_a["ticker"], snap)
            p_b, ts_b = trade_before(client, m_b["ticker"], snap)
            close = pd.Timestamp(m_a.get("close_time") or m_b.get("close_time") or date)
            if close.tz is None:
                close = close.tz_localize("UTC")
            ma = match_name(name_a, None, idx)
            mb = match_name(name_b, None, idx)
            rows.append(
                {
                    "event_ticker": ev["event_ticker"],
                    "event_title": ev.get("title", ""),
                    "sub_title": ev.get("sub_title", ""),
                    "fight_date": date,
                    "close_time": close,
                    "card_snapshot_ts": snap,
                    "fighter_a": name_a,
                    "fighter_b": name_b,
                    "ticker_a": m_a["ticker"],
                    "ticker_b": m_b["ticker"],
                    "close_yes_price_a": p_a,
                    "close_yes_price_b": p_b,
                    "capture_ts_a": ts_a,
                    "capture_ts_b": ts_b,
                    "settle_result_a": res_a,
                    "settle_result_b": res_b,
                    "winner": winner,
                    "volume_a": float(m_a.get("volume_fp") or 0) or None,
                    "volume_b": float(m_b.get("volume_fp") or 0) or None,
                    "open_interest_a": float(m_a.get("open_interest_fp") or 0) or None,
                    "open_interest_b": float(m_b.get("open_interest_fp") or 0) or None,
                    "canon_a": ma.canonical,
                    "canon_b": mb.canonical,
                    "match_score_a": ma.score,
                    "match_score_b": mb.score,
                }
            )
            n_done += 1
            if n_done % 25 == 0:
                print(f"  ... {n_done} markets ({time.time() - t0:.0f}s)")

    df = pd.DataFrame(rows).sort_values("fight_date").reset_index(drop=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    # Stats
    have_both = (df["close_yes_price_a"].notna() & df["close_yes_price_b"].notna()).sum()
    s = df["close_yes_price_a"].fillna(0) + df["close_yes_price_b"].fillna(0)
    usable = (
        (df["close_yes_price_a"] >= 0.02) & (df["close_yes_price_b"] >= 0.02) & (s >= 0.80) & (s <= 1.30)
    ).sum()
    print(f"\nDone: {len(df)} markets across {len(card_snapshot)} cards")
    print(f"  with both prices captured:  {have_both}/{len(df)}")
    print(f"  usable (sensible sum):      {usable}/{len(df)}")
    print(f"  saved → {args.out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

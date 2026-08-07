"""Pull the next UFC card from Kalshi and return per-fight order books.

Mirrors `polymarket_card.fetch_next_card` so the rest of the prediction
pipeline (skill fitting, ensemble inference, Kelly sizing) is venue-agnostic.

Kalshi UFC moneyline lives under series `KXUFCFIGHT`. Each fight is one
*event* with two binary *markets* (one per fighter, mutually exclusive),
e.g. `KXUFCFIGHT-26JUN06MUHBON` -> `-MUH` and `-BON`.

Order book translation (Kalshi -> Polymarket-style CardFight):
  Kalshi's per-market orderbook has `yes_dollars` (resting buy orders on YES)
  and `no_dollars` (resting buy orders on NO). For fighter A's market:
    bids_a (offers to BUY YES on A) <- market_A.yes_dollars
    asks_a (offers to SELL YES on A) <- market_A.no_dollars mirrored: (1-p, qty)
  Same for fighter B from market B.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from ufc_pred.ingest.kalshi_client import KalshiClient

_MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
_TICKER_DATE_RE = re.compile(r"-(\d{2})([A-Z]{3})(\d{2})")  # e.g. -26JUN06


def _parse_ticker_date(event_ticker: str) -> pd.Timestamp | None:
    m = _TICKER_DATE_RE.search(event_ticker)
    if not m:
        return None
    yy, mon, dd = m.group(1), m.group(2), m.group(3)
    try:
        mo = _MONTHS.index(mon) + 1
    except ValueError:
        return None
    return pd.Timestamp(year=2000 + int(yy), month=mo, day=int(dd), tz="UTC")


@dataclass
class CardFight:
    """Compatible with polymarket_card.CardFight (same field names + types)."""

    event_title: str
    fight_date: pd.Timestamp
    fighter_a: str
    fighter_b: str
    token_a: str  # Kalshi market ticker for fighter A (e.g. ...-MUH)
    token_b: str  # Kalshi market ticker for fighter B
    market_id: str  # Kalshi event ticker
    asks_a: list[tuple[float, float]]  # asks ascending by price
    asks_b: list[tuple[float, float]]
    bids_a: list[tuple[float, float]]  # bids descending by price
    bids_b: list[tuple[float, float]]
    volume: float | None
    liquidity: float | None


def _market_books(client: KalshiClient, ticker: str) -> tuple[list, list]:
    """Return (asks, bids) for one Kalshi market in Polymarket-style coords.

    asks: list of (price, qty) where price is the cost to BUY one $1 YES contract
    bids: list of (price, qty) where price is what someone will PAY for one YES contract
    """
    ob = client.get_orderbook(ticker, depth=50)
    # bids = direct yes-side: someone is bidding price p for a YES contract
    bids = sorted([(float(p), float(q)) for (p, q) in ob["yes"]], reverse=True)
    # asks = mirror the no-side: a NO bid at p is equivalent to a YES ask at (1-p)
    asks = sorted([(round(1.0 - float(p), 4), float(q)) for (p, q) in ob["no"]])
    return asks, bids


def fetch_next_card(
    *,
    client: KalshiClient | None = None,
    target_date: pd.Timestamp | None = None,
    max_days_ahead: int = 14,
) -> list[CardFight]:
    """Return per-fight orderbooks for the next UFC card on Kalshi.

    If `target_date` is given, returns fights whose event-ticker date matches
    that day (e.g. today's card). Otherwise picks the soonest date with any
    open KXUFCFIGHT events within `max_days_ahead`.
    """
    c = client or KalshiClient()
    r = c.request("GET", "/events", params={"series_ticker": "KXUFCFIGHT", "limit": 200})
    events = r.get("events", [])

    # Bucket events by their ticker-encoded date
    by_date: dict[pd.Timestamp, list[dict]] = {}
    for ev in events:
        d = _parse_ticker_date(ev.get("event_ticker", ""))
        if d is None:
            continue
        by_date.setdefault(d, []).append(ev)

    if not by_date:
        return []

    now = pd.Timestamp.now(tz="UTC").normalize()
    cutoff = now + pd.Timedelta(days=max_days_ahead)

    if target_date is not None:
        td = (
            pd.Timestamp(target_date).tz_localize("UTC")
            if pd.Timestamp(target_date).tz is None
            else pd.Timestamp(target_date)
        )
        td = td.normalize()
        card_events = by_date.get(td, [])
    else:
        upcoming = sorted(d for d in by_date if now <= d <= cutoff)
        if not upcoming:
            return []
        card_events = by_date[upcoming[0]]

    fights: list[CardFight] = []
    for ev in card_events:
        et = ev["event_ticker"]
        ev_title = ev.get("title", "")
        fight_date = _parse_ticker_date(et) or pd.Timestamp.now(tz="UTC")
        mkts = c.list_markets(event_ticker=et, limit=10).get("markets", [])
        if len(mkts) != 2:
            continue
        m_a, m_b = mkts[0], mkts[1]
        fighter_a = m_a.get("yes_sub_title") or m_a["ticker"].split("-")[-1]
        fighter_b = m_b.get("yes_sub_title") or m_b["ticker"].split("-")[-1]
        asks_a, bids_a = _market_books(c, m_a["ticker"])
        asks_b, bids_b = _market_books(c, m_b["ticker"])

        def _as_f(x, default=None):
            try:
                return float(x)
            except (TypeError, ValueError):
                return default

        vol = (_as_f(m_a.get("volume_fp"), 0) or 0) + (_as_f(m_b.get("volume_fp"), 0) or 0)
        liq = (_as_f(m_a.get("liquidity_dollars"), 0) or 0) + (_as_f(m_b.get("liquidity_dollars"), 0) or 0)

        fights.append(
            CardFight(
                event_title=ev_title,
                fight_date=fight_date,
                fighter_a=fighter_a,
                fighter_b=fighter_b,
                token_a=m_a["ticker"],
                token_b=m_b["ticker"],
                market_id=et,
                asks_a=asks_a,
                asks_b=asks_b,
                bids_a=bids_a,
                bids_b=bids_b,
                volume=vol if vol > 0 else None,
                liquidity=liq if liq > 0 else None,
            )
        )
    return fights

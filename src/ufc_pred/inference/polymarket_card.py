"""Pull the next UFC card from Polymarket and return per-fight order books."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd
import requests

from ufc_pred.ingest.polymarket import (
    GAMMA_BASE,
    HTTP_TIMEOUT,
    fetch_order_book,
    iter_h2h_markets,
)


@dataclass
class CardFight:
    event_title: str
    fight_date: pd.Timestamp
    fighter_a: str
    fighter_b: str
    token_a: str
    token_b: str
    market_id: str
    asks_a: list[tuple[float, float]]
    asks_b: list[tuple[float, float]]
    bids_a: list[tuple[float, float]]
    bids_b: list[tuple[float, float]]
    volume: float | None
    liquidity: float | None


def _book_to_levels(book: dict, key: str, reverse: bool) -> list[tuple[float, float]]:
    levels = [(float(o["price"]), float(o["size"])) for o in book.get(key, [])]
    return sorted(levels, reverse=reverse)


def fetch_next_card(
    *,
    session: requests.Session | None = None,
    max_days_ahead: int = 14,
) -> list[CardFight]:
    """Return the soonest UFC card's fights with live order books.

    Looks across all open UFC events; groups by event; returns fights from
    the event with the earliest end date within `max_days_ahead`.
    """
    sess = session or requests.Session()
    now = pd.Timestamp.now(tz="UTC")
    cutoff = now + pd.Timedelta(days=max_days_ahead)

    r = sess.get(
        f"{GAMMA_BASE}/events",
        params={
            "closed": "false",
            "tag_slug": "ufc",
            "limit": 100,
            "order": "endDate",
            "ascending": "true",
        },
        timeout=HTTP_TIMEOUT,
    )
    r.raise_for_status()
    events = r.json()
    if not events:
        return []

    in_window = []
    for e in events:
        end = e.get("endDate")
        if not end:
            continue
        ts = pd.Timestamp(end)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        if now < ts <= cutoff:
            in_window.append((ts, e))
    if not in_window:
        return []
    in_window.sort(key=lambda x: x[0])
    earliest_ts = in_window[0][0]
    # All events whose endDate matches the same card (within 24h of earliest)
    card_events = [e for ts, e in in_window if (ts - earliest_ts) <= pd.Timedelta(hours=24)]

    fights: list[CardFight] = []
    for event in card_events:
        for market in iter_h2h_markets(event, include_closed=False):
            outcomes = json.loads(market.get("outcomes") or "[]")
            tokens = json.loads(market.get("clobTokenIds") or "[]")
            if len(outcomes) != 2 or len(tokens) != 2:
                continue
            end_raw = market.get("endDate") or event.get("endDate")
            fight_date = pd.Timestamp(end_raw)
            if fight_date.tzinfo is None:
                fight_date = fight_date.tz_localize("UTC")
            book_a = fetch_order_book(tokens[0], session=sess)
            book_b = fetch_order_book(tokens[1], session=sess)
            fights.append(
                CardFight(
                    event_title=event.get("title", ""),
                    fight_date=fight_date,
                    fighter_a=outcomes[0],
                    fighter_b=outcomes[1],
                    token_a=str(tokens[0]),
                    token_b=str(tokens[1]),
                    market_id=str(market.get("id")),
                    asks_a=_book_to_levels(book_a, "asks", reverse=False),
                    asks_b=_book_to_levels(book_b, "asks", reverse=False),
                    bids_a=_book_to_levels(book_a, "bids", reverse=True),
                    bids_b=_book_to_levels(book_b, "bids", reverse=True),
                    volume=float(market["volume"]) if market.get("volume") else None,
                    liquidity=float(market["liquidity"]) if market.get("liquidity") else None,
                )
            )
    return fights

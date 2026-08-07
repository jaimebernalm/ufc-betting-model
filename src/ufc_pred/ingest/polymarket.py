"""Polymarket UFC odds scraper.

Pulls open UFC fight-night events from the public Gamma API, extracts the
moneyline market for each fight, and returns one row per fight with both
fighters' implied probabilities.

Gamma API: https://gamma-api.polymarket.com (public, no auth).
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import pandas as pd
import requests

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"
UFC_TAG_SLUG = "ufc"
PAGE_LIMIT = 100
HTTP_TIMEOUT = 30
CLOSING_WINDOW_DAYS = 7  # how far back to look for the last pre-fight trade
NON_MONEYLINE_OUTCOMES = {
    frozenset({"Yes", "No"}),
    frozenset({"Over", "Under"}),
}


@dataclass(frozen=True)
class FightOdds:
    event_id: str
    event_title: str
    event_slug: str
    fight_date: pd.Timestamp
    fighter_a: str
    fighter_b: str
    p_a: float
    p_b: float
    best_bid_a: float | None
    best_ask_a: float | None
    last_trade_a: float | None
    volume: float | None
    liquidity: float | None
    market_id: str
    market_slug: str
    scraped_at: pd.Timestamp


def _parse_maybe_json_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return []
    return []


def _to_float(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_utc_ts(value) -> pd.Timestamp | None:
    if value in (None, "", "null"):
        return None
    try:
        ts = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(ts):
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _resolve_cutoff(market: dict, event: dict, fight_date: pd.Timestamp) -> pd.Timestamp:
    """Best-effort 'last second of legitimate trading' timestamp.

    Preference order:
      1. `gameStartTime` on market or event (per-fight format, exact).
      2. `closedTime` / `umaEndDate` minus 6h (per-card format — these are UMA
         resolution times; subtracting 6h lands roughly before card start).
      3. fight_date minus 6h.
    """
    for key in ("gameStartTime",):
        ts = _parse_utc_ts(market.get(key) or event.get(key))
        if ts is not None:
            return ts
    for key in ("closedTime", "umaEndDate"):
        ts = _parse_utc_ts(market.get(key) or event.get(key))
        if ts is not None:
            return ts - pd.Timedelta(hours=6)
    return fight_date - pd.Timedelta(hours=6)


def fetch_ufc_events(
    *,
    closed: bool = False,
    end_date_min: str | None = None,
    end_date_max: str | None = None,
    session: requests.Session | None = None,
) -> list[dict]:
    """Page through UFC-tagged events on Gamma, optionally restricted by end date."""
    sess = session or requests.Session()
    out: list[dict] = []
    offset = 0
    while True:
        params: dict[str, object] = {
            "closed": str(closed).lower(),
            "limit": PAGE_LIMIT,
            "offset": offset,
            "tag_slug": UFC_TAG_SLUG,
            "order": "endDate",
            "ascending": "true",
        }
        if end_date_min:
            params["end_date_min"] = end_date_min
        if end_date_max:
            params["end_date_max"] = end_date_max
        r = sess.get(f"{GAMMA_BASE}/events", params=params, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        out.extend(batch)
        if len(batch) < PAGE_LIMIT:
            break
        offset += PAGE_LIMIT
        time.sleep(0.2)
    return out


SETTLED_LOW = 0.005  # below this = post-resolution arbitrage trade
SETTLED_HIGH = 0.995  # above this = post-resolution arbitrage trade


def fetch_order_book(
    token_id: str,
    *,
    session: requests.Session | None = None,
) -> dict:
    """Return the live CLOB order book for `token_id`.

    Response shape: {"bids": [{"price": "...", "size": "..."}, ...],
                     "asks": [{"price": "...", "size": "..."}, ...]}.
    Bids sorted desc by price, asks sorted asc by price.
    """
    sess = session or requests.Session()
    r = sess.get(f"{CLOB_BASE}/book", params={"token_id": token_id}, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()


def ask_depth(book: dict, within_cents: int = 3) -> float:
    """USD depth on the ask side within `within_cents` of the best ask."""
    asks = sorted([(float(o["price"]), float(o["size"])) for o in book.get("asks", [])])
    if not asks:
        return 0.0
    top = asks[0][0]
    return sum(p * s for p, s in asks if p <= top + within_cents / 100)


def fetch_closing_price(
    token_id: str,
    fight_date: pd.Timestamp,
    *,
    session: requests.Session | None = None,
    window_days: int = CLOSING_WINDOW_DAYS,
) -> tuple[float, pd.Timestamp] | None:
    """Return (price, ts) of the closing odds for `token_id` before `fight_date`.

    Strategy: pull all CLOB history in [fight_date - window_days, fight_date],
    then walk backward through the points dropping any whose price is in the
    "settled" range [<0.005, >0.995] — those are post-resolution arbitrage
    trades that snap to 0/1 after the market resolves. The last non-settled
    point is the genuine pre-fight closing line.

    For the per-card event format (no `gameStartTime`), the caller passes a
    coarse cutoff and this walk-back logic handles the rest.
    """
    sess = session or requests.Session()
    end_ts = int(fight_date.timestamp())
    start_ts = end_ts - window_days * 86400
    params = {
        "market": token_id,
        "startTs": start_ts,
        "endTs": end_ts,
        "fidelity": 60,
    }
    r = sess.get(f"{CLOB_BASE}/prices-history", params=params, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    history = r.json().get("history") or []
    if not history:
        return None
    for point in reversed(history):
        p = float(point["p"])
        if SETTLED_LOW < p < SETTLED_HIGH:
            return p, pd.Timestamp(point["t"], unit="s", tz="UTC")
    # Entire window is settled-range — fall back to last point so caller still
    # gets *something* (heavy-favorite closing lines can legitimately be >0.99,
    # though it's rare on Polymarket).
    last = history[-1]
    return float(last["p"]), pd.Timestamp(last["t"], unit="s", tz="UTC")


def _find_moneyline_market(event: dict, *, include_closed: bool = False) -> dict | None:
    """First H2H moneyline market in this event (legacy single-fight events)."""
    for m in iter_h2h_markets(event, include_closed=include_closed):
        return m
    return None


def iter_h2h_markets(event: dict, *, include_closed: bool = False) -> Iterator[dict]:
    """Yield every head-to-head market in an event.

    Polymarket has two UFC event formats:
      - per-fight event: one event = one fight, one H2H market.
      - per-card event (e.g. "UFC 300"): one event = the whole card, with
        multiple H2H markets inside, one per bout on the card.

    A market is H2H if it has two outcomes that are NOT Yes/No or Over/Under
    and the question mentions " vs". Yields raw market dicts.
    """
    for m in event.get("markets") or []:
        if not include_closed and (m.get("closed") or not m.get("active")):
            continue
        outcomes = _parse_maybe_json_list(m.get("outcomes"))
        if len(outcomes) != 2:
            continue
        if frozenset(outcomes) in NON_MONEYLINE_OUTCOMES:
            continue
        question = m.get("question", "") or ""
        # Per-card events: each H2H market's question is "Card: A vs B"; per-fight
        # events: the moneyline market's question equals the event title which
        # also contains " vs.". The check on `question` filters out side-bet
        # markets with two fighter-style outcomes that aren't moneylines.
        if " vs" not in question.lower():
            continue
        yield m


def parse_event(event: dict, *, scraped_at: pd.Timestamp) -> FightOdds | None:
    title = event.get("title", "")
    # Skip meta markets like "Who will X fight next?" — they aren't head-to-head fights.
    if " vs." not in title and " vs " not in title:
        return None
    market = _find_moneyline_market(event)
    if market is None:
        return None
    outcomes = _parse_maybe_json_list(market.get("outcomes"))
    prices = _parse_maybe_json_list(market.get("outcomePrices"))
    if len(outcomes) != 2 or len(prices) != 2:
        return None
    p_a = _to_float(prices[0])
    p_b = _to_float(prices[1])
    if p_a is None or p_b is None:
        return None

    end_date_raw = event.get("endDate") or market.get("endDate")
    try:
        fight_date = pd.Timestamp(end_date_raw).tz_convert("UTC")
    except (TypeError, ValueError, AttributeError):
        fight_date = pd.Timestamp(end_date_raw, tz="UTC") if end_date_raw else pd.NaT

    return FightOdds(
        event_id=str(event.get("id")),
        event_title=title,
        event_slug=event.get("slug", ""),
        fight_date=fight_date,
        fighter_a=outcomes[0],
        fighter_b=outcomes[1],
        p_a=p_a,
        p_b=p_b,
        best_bid_a=_to_float(market.get("bestBid")),
        best_ask_a=_to_float(market.get("bestAsk")),
        last_trade_a=_to_float(market.get("lastTradePrice")),
        volume=_to_float(market.get("volume")),
        liquidity=_to_float(market.get("liquidity")),
        market_id=str(market.get("id")),
        market_slug=market.get("slug", ""),
        scraped_at=scraped_at,
    )


def parse_events(events: Iterable[dict], *, scraped_at: pd.Timestamp) -> Iterator[FightOdds]:
    for e in events:
        row = parse_event(e, scraped_at=scraped_at)
        if row is not None:
            yield row


def to_dataframe(rows: Iterable[FightOdds]) -> pd.DataFrame:
    df = pd.DataFrame([r.__dict__ for r in rows])
    if df.empty:
        return df
    return df.sort_values("fight_date").reset_index(drop=True)


def scrape(*, session: requests.Session | None = None) -> pd.DataFrame:
    scraped_at = pd.Timestamp.now(tz="UTC")
    events = fetch_ufc_events(session=session)
    return to_dataframe(parse_events(events, scraped_at=scraped_at))


def scrape_historical(
    start_date: str,
    end_date: str | None = None,
    *,
    session: requests.Session | None = None,
    progress: bool = True,
) -> pd.DataFrame:
    """Scrape closed UFC fight markets in [start_date, end_date] with closing odds.

    For each resolved fight, looks up the last CLOB trade before the fight
    started and records it as the closing implied probability for fighter A.
    Fighter B's closing prob is 1 − price_a (binary market, no-vig by
    construction on Polymarket).

    Dates are ISO strings, e.g. "2026-03-29".
    """
    sess = session or requests.Session()
    scraped_at = pd.Timestamp.now(tz="UTC")

    end_iso = (end_date or scraped_at.strftime("%Y-%m-%d")) + "T23:59:59Z"
    start_iso = start_date + "T00:00:00Z"

    events = fetch_ufc_events(
        closed=True,
        end_date_min=start_iso,
        end_date_max=end_iso,
        session=sess,
    )

    rows: list[dict] = []
    # Each event may contain >1 H2H market (UFC 300 has 7). Flatten to
    # (event, market) pairs across both per-fight and per-card formats.
    head_to_heads: list[tuple[dict, dict]] = []
    for e in events:
        for m in iter_h2h_markets(e, include_closed=True):
            head_to_heads.append((e, m))

    for i, (event, market) in enumerate(head_to_heads, start=1):
        outcomes = _parse_maybe_json_list(market.get("outcomes"))
        token_ids = _parse_maybe_json_list(market.get("clobTokenIds"))
        if len(outcomes) != 2 or len(token_ids) != 2:
            continue
        # Use the market's endDate when set (per-fight events); for per-card
        # events the event endDate is shared across all bouts and is a usable
        # proxy for the fight date.
        end_date_raw = market.get("endDate") or event.get("endDate")
        try:
            fight_date = pd.Timestamp(end_date_raw)
            if fight_date.tzinfo is None:
                fight_date = fight_date.tz_localize("UTC")
            else:
                fight_date = fight_date.tz_convert("UTC")
        except (TypeError, ValueError):
            continue

        # Cutoff for "before this fight started": prefer per-fight gameStartTime,
        # then UMA resolution (closedTime/umaEndDate) minus a 6h buffer (so we
        # land before the fight rather than after), then endDate minus 6h.
        cutoff = _resolve_cutoff(market, event, fight_date)

        try:
            closing = fetch_closing_price(str(token_ids[0]), cutoff, session=sess)
        except requests.HTTPError as exc:
            if progress:
                print(
                    f"[{i}/{len(head_to_heads)}] HTTP {exc.response.status_code} "
                    f"on {outcomes[0]} vs {outcomes[1]}"
                )
            closing = None
        time.sleep(0.15)  # be polite to CLOB

        resolved_prices = _parse_maybe_json_list(market.get("outcomePrices"))
        winner = None
        if len(resolved_prices) == 2:
            try:
                winner = outcomes[0] if float(resolved_prices[0]) >= 0.5 else outcomes[1]
            except (TypeError, ValueError):
                winner = None

        row = {
            "event_id": str(event.get("id")),
            "event_title": event.get("title", ""),
            "fight_date": fight_date,
            "fighter_a": outcomes[0],
            "fighter_b": outcomes[1],
            "closing_price_a": closing[0] if closing else None,
            "closing_price_b": (1.0 - closing[0]) if closing else None,
            "closing_ts": closing[1] if closing else None,
            "winner": winner,
            "volume": _to_float(market.get("volume")),
            "liquidity": _to_float(market.get("liquidity")),
            "market_id": str(market.get("id")),
            "market_slug": market.get("slug", ""),
            "token_id_a": str(token_ids[0]),
            "token_id_b": str(token_ids[1]),
            "scraped_at": scraped_at,
        }
        rows.append(row)

        if progress and i % 10 == 0:
            print(f"  scraped {i}/{len(head_to_heads)}")

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("fight_date").reset_index(drop=True)

"""Detect whether a Kalshi UFC market's fight has concluded — BEFORE Kalshi
flips the market status to finalized/determined (which can lag by hours).

Kalshi UFC markets keep trading after the fight ends: arbitrageurs pin the
price to ~0.99 (winner) / ~0.01 (loser) within seconds of the result. We treat
a market as *provisionally resolved* when:

  1. the newest trade is at an extreme price (>= PIN_THRESHOLD on either side),
  2. the consecutive run of extreme trades started AFTER the scheduled fight
     start (guards against 97-99c favorites being "pinned" before the fight
     even begins — e.g. Barez was 0.99 pre-fight),
  3. the pin has been stable for >= stable_sec (guards against momentary
     mid-fight blips: a near-finish can spike the price briefly).

The definitive settlement status (finalized/determined + result) is checked
first and always wins.

NOTE (observed 2026-06-06 card): Kalshi now CLOSES UFC markets around fight
end and settles ~5-8 minutes later (settlement_timer ~300s) — there may be no
post-fight pin trades at all. For that regime we report ("closed", None) once
the market status is "closed" after the scheduled fight start and a short
grace period has passed: the fight is over but the result isn't official yet.
Callers can use "closed" as a capture trigger but must NOT book PnL from it.
"""

from __future__ import annotations

import pandas as pd

PIN_THRESHOLD = 0.97  # price at/above this (or at/below 1-this) = pinned
PIN_STABLE_SEC = 90  # pin must hold this long ("wait 1 minute or a bit more")
MAX_TRADE_PAGES = 3  # paginate up to 3x1000 trades hunting the pin start
PAGE_SIZE = 1000
CLOSED_GRACE_SEC = 120  # market closed early -> wait this long before trusting
# it as "fight over" (settlement follows ~5 min later)


def _ts(trade: dict) -> pd.Timestamp | None:
    try:
        ts = pd.Timestamp(trade["created_time"])
        return ts.tz_localize("UTC") if ts.tz is None else ts
    except (KeyError, TypeError, ValueError):
        return None


def _price(trade: dict) -> float | None:
    try:
        return float(trade["yes_price_dollars"])
    except (KeyError, TypeError, ValueError):
        return None


def market_resolution(
    client,
    ticker: str,
    *,
    fight_start_utc: pd.Timestamp | None = None,
    now_utc: pd.Timestamp | None = None,
    pin_threshold: float = PIN_THRESHOLD,
    stable_sec: int = PIN_STABLE_SEC,
) -> tuple[str, bool] | None:
    """Resolution state of one Kalshi market.

    Returns:
      ("settled", yes_won)      -- Kalshi has finalized/determined the market
      ("closed", None)          -- trading closed after the fight started and
                                   CLOSED_GRACE_SEC passed: fight is over,
                                   result pending. Trigger-only — no PnL.
      ("provisional", yes_won)  -- price pinned to the extreme post-fight-start
                                   and stable for >= stable_sec
      None                      -- not resolved (or not confidently resolvable)
    """
    now = now_utc or pd.Timestamp.now(tz="UTC")

    # 1) Definitive settlement status always wins.
    try:
        m = client.get_market(ticker)
    except Exception:
        return None
    m = m.get("market", m)
    if m.get("status", "") in ("finalized", "determined") and m.get("result", "") in ("yes", "no"):
        return ("settled", m.get("result") == "yes")

    # 1b) Current Kalshi regime: market closes around fight end, settles ~5min
    # later. status "closed" after the scheduled start = fight over.
    if m.get("status", "") == "closed" and fight_start_utc is not None:
        fs = pd.Timestamp(fight_start_utc)
        if fs.tz is None:
            fs = fs.tz_localize("UTC")
        close_ts = None
        try:
            close_ts = pd.Timestamp(m["close_time"])
            if close_ts.tz is None:
                close_ts = close_ts.tz_localize("UTC")
        except (KeyError, TypeError, ValueError):
            pass
        if (
            now > fs
            and close_ts is not None
            and close_ts > fs
            and (now - close_ts).total_seconds() >= CLOSED_GRACE_SEC
        ):
            return ("closed", None)

    # 2) Provisional: walk trades newest-first looking for a stable pin.
    pin_side: bool | None = None  # True = YES pinned high, False = NO winning
    pin_start: pd.Timestamp | None = None
    run_complete = False
    cursor = None
    for _ in range(MAX_TRADE_PAGES):
        params: dict = {"ticker": ticker, "limit": PAGE_SIZE}
        if cursor:
            params["cursor"] = cursor
        try:
            r = client.request("GET", "/markets/trades", params=params)
        except Exception:
            return None
        trades = r.get("trades", [])
        if not trades:
            run_complete = True  # exhausted history; run extends to market open
            break
        for t in trades:  # newest -> oldest
            p = _price(t)
            ts = _ts(t)
            if p is None or ts is None:
                continue
            if pin_side is None:
                if p >= pin_threshold:
                    pin_side = True
                elif p <= 1.0 - pin_threshold:
                    pin_side = False
                else:
                    return None  # newest trade not extreme -> not resolved
                pin_start = ts
                continue
            on_pin = (p >= pin_threshold) if pin_side else (p <= 1.0 - pin_threshold)
            if on_pin:
                pin_start = ts
            else:
                run_complete = True
                break
        if run_complete:
            break
        cursor = r.get("cursor")
        if not cursor:
            run_complete = True
            break

    if pin_side is None or pin_start is None:
        return None
    if not run_complete:
        # Pin run longer than MAX_TRADE_PAGES of trades and we never saw it
        # start — can't prove it began after the fight started. Decline.
        return None
    if fight_start_utc is not None:
        fs = pd.Timestamp(fight_start_utc)
        if fs.tz is None:
            fs = fs.tz_localize("UTC")
        if pin_start <= fs:
            return None  # pinned since before the fight (heavy favorite, not a result)
    if (now - pin_start).total_seconds() < stable_sec:
        return None  # too fresh; could be a mid-fight blip
    return ("provisional", pin_side)

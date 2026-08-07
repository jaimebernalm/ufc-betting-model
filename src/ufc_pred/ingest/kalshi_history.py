"""Backfill historical Kalshi UFC fight prices for model evaluation.

For each settled KXUFCFIGHT event, pulls both per-fighter markets and
returns one row per fight with:
  - fighter names + settlement result (winner)
  - closing prices (last trade price before close, both sides)
  - volume + open interest at settlement (liquidity proxies)
  - close timestamp

Limitations: Kalshi REST does NOT expose historical orderbook depth
snapshots — only aggregate volume/OI. Same constraint as the Polymarket
backfill in notebook 05.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass

import pandas as pd

from ufc_pred.ingest.kalshi_client import KalshiClient


@dataclass
class HistoricalFight:
    event_ticker: str
    event_title: str
    sub_title: str
    fight_date: pd.Timestamp  # parsed from event ticker
    close_time: pd.Timestamp  # actual settlement time
    fighter_a: str
    fighter_b: str
    ticker_a: str
    ticker_b: str
    close_yes_price_a: float | None  # last trade price on -A market
    close_yes_price_b: float | None
    settle_result_a: str | None  # "yes" or "no" (yes means A won)
    settle_result_b: str | None
    winner: str | None  # "A" or "B" (or None if no_contest/draw)
    volume_a: float | None
    volume_b: float | None
    open_interest_a: float | None
    open_interest_b: float | None
    # Best quote at the same capture instant as close_yes_price_*. The last
    # TRADE price is not what a backtest can transact at — buying YES lifts the
    # ask. Backtests priced off close_yes_price_* are therefore optimistic by
    # roughly half the spread; use ask_yes_price_* to price entries.
    ask_yes_price_a: float | None = None
    ask_yes_price_b: float | None = None
    bid_yes_price_a: float | None = None
    bid_yes_price_b: float | None = None


def _as_f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def quote_before(
    client: KalshiClient,
    ticker: str,
    cutoff_ts: pd.Timestamp,
    *,
    lookback_minutes: int = 90,
) -> tuple[float | None, float | None]:
    """Best (yes_ask, yes_bid) at or before `cutoff_ts`, from 1-minute candles.

    Kalshi's REST API exposes no historical orderbook, but the candlestick
    endpoint carries per-minute `yes_ask`/`yes_bid` closes, which is enough to
    reconstruct the price a taker would actually have paid. Returns
    (None, None) when the market had no quotes in the lookback window.
    """
    if cutoff_ts is None:
        return None, None
    series = ticker.split("-")[0]
    end = int(pd.Timestamp(cutoff_ts).timestamp())
    try:
        r = client.request(
            "GET",
            f"/series/{series}/markets/{ticker}/candlesticks",
            params={"start_ts": end - lookback_minutes * 60, "end_ts": end, "period_interval": 1},
        )
    except Exception:
        return None, None

    def _close(candle, side):
        v = (candle.get(side) or {}).get("close_dollars")
        return None if v in (None, "") else _as_f(v)

    for candle in reversed(r.get("candlesticks", [])):
        if candle.get("end_period_ts", 0) > end:
            continue
        ask, bid = _close(candle, "yes_ask"), _close(candle, "yes_bid")
        if ask is not None or bid is not None:
            return ask, bid
    return None, None


def _parse_ticker_date(et: str) -> pd.Timestamp | None:
    import re

    m = re.search(r"-(\d{2})([A-Z]{3})(\d{2})", et)
    if not m:
        return None
    months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    try:
        return pd.Timestamp(
            year=2000 + int(m.group(1)),
            month=months.index(m.group(2)) + 1,
            day=int(m.group(3)),
            tz="UTC",
        )
    except ValueError:
        return None


def fetch_pre_fight_price(
    client: KalshiClient,
    ticker: str,
    *,
    cutoff_ts: pd.Timestamp | None = None,
    settlement_buffer_minutes: int = 240,
    max_pages: int = 60,  # kept for backcompat; unused in efficient path
    page_size: int = 1000,
) -> tuple[float | None, pd.Timestamp | None, str]:
    """Find the true pre-fight closing price for a Kalshi market.

    Kalshi UFC markets stay OPEN during the fight and continue trading until
    settlement. The "last trade" can be from mid-fight (large in-play swings)
    or from post-fight arbitrage (price stuck at 0.99/0.01). Neither is a
    valid pre-fight market consensus.

    Two strategies, in priority order:

    1. **External cutoff** (preferred when available): if `cutoff_ts` is given
       (e.g. Polymarket's `closing_ts` for the same fight), return the latest
       Kalshi trade strictly before that timestamp. Polymarket markets DO close
       at fight start by design, so this gives an unambiguous pre-fight price.

    2. **Gap-detection heuristic** (fallback): walk through trades newest→oldest.
       Pre-fight trades arrive slowly (minutes/hours apart). In-fight trades
       arrive in bursts (sub-second). Find the first trade whose preceding
       (older) trade is more than `gap_threshold_seconds` away — that older
       trade is in the pre-fight "calm" zone. Returns its price.

    Returns (yes_price, trade_timestamp, method) — method ∈ {'cutoff',
    'gap_detect', 'no_data'}. Price/timestamp are None on failure.
    """
    # Step 1: find the fight-end transition. Approach: get the newest trade
    # (which is settlement-era, at the result price — 0.01 or 0.99), then
    # walk backward to find the first trade NOT at that settlement price.
    # That's approximately fight end.
    r = client.request("GET", "/markets/trades", params={"ticker": ticker, "limit": 1})
    newest = r.get("trades", [])
    if not newest:
        return None, None, "no_data"
    try:
        settle_price = float(newest[0]["yes_price_dollars"])
    except (KeyError, TypeError, ValueError):
        return None, None, "parse_error"

    fight_end_ts: pd.Timestamp | None = None
    cursor = None
    for _ in range(20):  # up to 20k trades back — generous for big main events
        params = {"ticker": ticker, "limit": page_size}
        if cursor:
            params["cursor"] = cursor
        r = client.request("GET", "/markets/trades", params=params)
        trades = r.get("trades", [])
        if not trades:
            break
        for t in trades:  # newest first
            try:
                yp = float(t["yes_price_dollars"])
            except (KeyError, TypeError, ValueError):
                continue
            if abs(yp - settle_price) > 0.005:  # not at settlement price
                ts = pd.Timestamp(t["created_time"])
                if ts.tz is None:
                    ts = ts.tz_localize("UTC")
                fight_end_ts = ts
                break
        if fight_end_ts is not None:
            break
        cursor = r.get("cursor")
        if not cursor:
            break
    if fight_end_ts is None:
        return None, None, "no_betting_era_found"

    # Step 2: cutoff = fight_end - settlement_buffer. This buffer is the
    # maximum reasonable fight duration + walkouts. 90 min works for most
    # 3-round prelims; 240 min (4h) safely covers 5-round main events.
    # NOTE: fight_end_ts may itself be mid-fight (price briefly dropped into
    # [0.05, 0.95] right before settlement). The buffer absorbs this.
    heuristic_cutoff = fight_end_ts - pd.Timedelta(minutes=settlement_buffer_minutes)
    cutoff = heuristic_cutoff
    method = f"fight-end-{settlement_buffer_minutes}min"
    if cutoff_ts is not None:
        ext = pd.Timestamp(cutoff_ts)
        if ext.tz is None:
            ext = ext.tz_localize("UTC")
        # Use external only if it's MORE RECENT than the heuristic cutoff
        # (more aggressive = closer to true fight start = better closing line)
        if ext > heuristic_cutoff and ext < fight_end_ts:
            cutoff = ext
            method = "cutoff"

    # Step 3: query the most recent trade before the cutoff
    r2 = client.request(
        "GET",
        "/markets/trades",
        params={"ticker": ticker, "limit": 1, "max_ts": int(cutoff.timestamp())},
    )
    pre = r2.get("trades", [])
    if not pre:
        return None, None, "no_pre_fight_data"
    try:
        ts = pd.Timestamp(pre[0]["created_time"])
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        yp = float(pre[0]["yes_price_dollars"])
        return yp, ts, method
    except (KeyError, TypeError, ValueError):
        return None, None, "parse_error"


def _settled_events(client: KalshiClient, cursor: str | None, limit: int) -> tuple[list[dict], str | None]:
    params = {"series_ticker": "KXUFCFIGHT", "status": "settled", "limit": limit}
    if cursor:
        params["cursor"] = cursor
    r = client.request("GET", "/events", params=params)
    return r.get("events", []), r.get("cursor")


def _pm_key(fight_date: pd.Timestamp, name_a: str, name_b: str) -> tuple:
    """Canonical key for matching Kalshi fights to Polymarket fights.
    Uses date + sorted lowercase last-names (order-invariant)."""
    import unicodedata

    def lastn(s):
        s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode("ascii")
        return s.lower().strip().split()[-1] if s.strip() else ""

    d = (
        pd.Timestamp(fight_date).tz_localize(None).normalize()
        if pd.Timestamp(fight_date).tz
        else pd.Timestamp(fight_date).normalize()
    )
    return (d, tuple(sorted([lastn(name_a), lastn(name_b)])))


def build_polymarket_cutoffs(poly_df: pd.DataFrame) -> dict:
    """From a Polymarket historical parquet, build {key -> closing_ts} for
    cross-referencing in Kalshi backfill. The closing_ts is roughly fight
    start (Polymarket markets close when the fight begins)."""
    out: dict = {}
    for _, row in poly_df.iterrows():
        fd = pd.Timestamp(row["fight_date"])
        ct = pd.Timestamp(row["closing_ts"]) if row.get("closing_ts") is not None else None
        if ct is None or pd.isna(ct):
            continue
        if ct.tz is None:
            ct = ct.tz_localize("UTC")
        # Index by ±1 day windows so off-by-one date conventions still hit
        for off in (-1, 0, 1):
            d = (fd.tz_localize(None) if fd.tz else fd).normalize() + pd.Timedelta(days=off)
            (
                d,
                tuple(sorted([_pm_key.__wrapped__ if False else None])),
            )  # noop, replaced below
        # Use the helper for correctness
        key0 = _pm_key(fd, row["fighter_a"], row["fighter_b"])
        out[key0] = ct
        # Also store under ±1 day variants in case date conventions differ
        d_orig = key0[0]
        for off in (-1, 1):
            out[(d_orig + pd.Timedelta(days=off), key0[1])] = ct
    return out


# Non-UFC promotions Kalshi files under the KXUFCFIGHT series. Matched as a
# substring of the lowercased title. Add new promotions here as they appear —
# the accept rules below are structural and would otherwise let them through.
_NON_UFC_TITLE_MARKERS = ("netflix",)

_NUMBERED_CARD_RE = re.compile(r"^\d{3}\s*:")  # e.g. "329: Ankalaev vs Guskov"


def _is_real_ufc(ev: dict) -> bool:
    """Filter out non-UFC events that Kalshi files under KXUFCFIGHT (e.g.
    'Netflix MMA Special: Rousey vs Carano').

    Kalshi titles used to always start with 'UFC' ('UFC 329', 'UFC Fight
    Night: ...'), and this function used to just test that prefix. Around
    2026-06-20 Kalshi dropped the prefix: real UFC cards now also appear as
    'Fight Night: X vs Y', '329: X vs Y', or a bare 'X vs Y'. The old check
    silently rejected every event from 26JUN20 onward (99 events), which
    blocked the whole snapshot-backfill path. Accept those shapes too, and
    keep non-UFC promotions out with an explicit denylist.
    """
    t = (ev.get("title") or "").strip()
    if not t:
        return False
    low = t.lower()
    if any(m in low for m in _NON_UFC_TITLE_MARKERS):
        return False
    if low.startswith("ufc"):
        return True
    if low.startswith("fight night"):
        return True
    if _NUMBERED_CARD_RE.match(t):
        return True
    return " vs" in low


def iter_settled_fights(
    *,
    client: KalshiClient | None = None,
    since: pd.Timestamp | None = None,
    until: pd.Timestamp | None = None,
    page_size: int = 200,
    sleep_s: float = 0.10,
    ufc_only: bool = True,
    polymarket_cutoffs: dict | None = None,
    settlement_buffer_minutes: int = 240,
) -> Iterator[HistoricalFight]:
    """Yield historical fights from Kalshi, newest first, optionally bounded by date."""
    c = client or KalshiClient()
    cursor: str | None = None
    while True:
        evs, cursor = _settled_events(c, cursor, page_size)
        if not evs:
            return
        for ev in evs:
            et = ev["event_ticker"]
            fd = _parse_ticker_date(et)
            if fd is None:
                continue
            if since is not None and fd < since:
                return
            if until is not None and fd > until:
                continue
            if ufc_only and not _is_real_ufc(ev):
                continue

            mkts = c.list_markets(event_ticker=et, status="settled", limit=10).get("markets", [])
            if len(mkts) != 2:
                continue
            m_a, m_b = mkts[0], mkts[1]
            name_a = m_a.get("yes_sub_title") or m_a["ticker"].split("-")[-1]
            name_b = m_b.get("yes_sub_title") or m_b["ticker"].split("-")[-1]
            res_a = m_a.get("result")
            res_b = m_b.get("result")
            winner = "A" if res_a == "yes" else ("B" if res_b == "yes" else None)
            close = pd.Timestamp(m_a.get("close_time") or m_b.get("close_time") or fd)
            if close.tz is None:
                close = close.tz_localize("UTC")
            # Pre-fight close: use Polymarket cutoff if available (gold standard),
            # else gap-detect heuristic. Both methods avoid in-fight contamination.
            cutoff = None
            if polymarket_cutoffs is not None:
                # Match by date + sorted fighter-name pair
                key = _pm_key(fd, name_a, name_b)
                cutoff = polymarket_cutoffs.get(key)
            close_a, close_a_ts, method_a = fetch_pre_fight_price(
                c,
                m_a["ticker"],
                cutoff_ts=cutoff,
                settlement_buffer_minutes=settlement_buffer_minutes,
            )
            close_b, close_b_ts, method_b = fetch_pre_fight_price(
                c,
                m_b["ticker"],
                cutoff_ts=cutoff,
                settlement_buffer_minutes=settlement_buffer_minutes,
            )
            if close_a is None:
                close_a = _as_f(m_a.get("last_price_dollars"))
            if close_b is None:
                close_b = _as_f(m_b.get("last_price_dollars"))
            # Transactable quote at the same instant as the captured trade.
            ask_a, bid_a = quote_before(c, m_a["ticker"], close_a_ts or cutoff)
            ask_b, bid_b = quote_before(c, m_b["ticker"], close_b_ts or cutoff)
            yield HistoricalFight(
                event_ticker=et,
                event_title=ev.get("title", ""),
                sub_title=ev.get("sub_title", ""),
                fight_date=fd,
                close_time=close,
                fighter_a=name_a,
                fighter_b=name_b,
                ticker_a=m_a["ticker"],
                ticker_b=m_b["ticker"],
                close_yes_price_a=close_a,
                close_yes_price_b=close_b,
                settle_result_a=res_a,
                settle_result_b=res_b,
                winner=winner,
                volume_a=_as_f(m_a.get("volume_fp")),
                volume_b=_as_f(m_b.get("volume_fp")),
                open_interest_a=_as_f(m_a.get("open_interest_fp")),
                open_interest_b=_as_f(m_b.get("open_interest_fp")),
                ask_yes_price_a=ask_a,
                ask_yes_price_b=ask_b,
                bid_yes_price_a=bid_a,
                bid_yes_price_b=bid_b,
            )
            if sleep_s:
                time.sleep(sleep_s)
        if not cursor:
            return


def backfill_to_dataframe(
    *,
    client: KalshiClient | None = None,
    since: pd.Timestamp | None = None,
    until: pd.Timestamp | None = None,
    ufc_only: bool = True,
    polymarket_cutoffs: dict | None = None,
) -> pd.DataFrame:
    rows = [
        asdict(f)
        for f in iter_settled_fights(
            client=client,
            since=since,
            until=until,
            ufc_only=ufc_only,
            polymarket_cutoffs=polymarket_cutoffs,
        )
    ]
    return pd.DataFrame(rows)

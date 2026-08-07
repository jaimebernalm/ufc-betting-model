"""Scrape UFC.com for per-fight start times on tonight's card.

UFC.com publishes per-section broadcast start times (Main Card, Prelims,
Early Prelims) but not per-fight times. Per-fight start times are derived
by taking the section start + 30 min * fight_index (reversed: main event
goes LAST chronologically but is listed FIRST on the page).

Two entry points:
  fetch_today_event()                -> dict | None  (event metadata)
  fetch_card_schedule(event_url)     -> list[FightSchedule]
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

UFC_BASE = "https://www.ufc.com"
EVENTS_URL = f"{UFC_BASE}/events"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36"
)

REPO_ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = REPO_ROOT / "data" / "raw" / "ufc_schedule" / "_cache"
AUDIT_DIR = REPO_ROOT / "data" / "raw" / "ufc_schedule"
CACHE_TTL_SEC = 600  # 10 minutes per PLAN_AUTO_BET §"DO NOT cache past 10 min"

# Per-fight spacing within a section. PLAN_AUTO_BET specifies 30 min.
FIGHT_SPACING_MIN = 30

# Operator's local timezone (CT/MA).
LOCAL_TZ = "US/Eastern"


@dataclass
class FightSchedule:
    fighter_a: str  # red corner display name "First Last"
    fighter_b: str  # blue corner display name "First Last"
    start_time_utc: pd.Timestamp
    card_position: str  # "main_card" | "prelim" | "early_prelim"
    weight_class: str | None
    title_bout: bool = False
    no_of_rounds: int = 3


# ---------------------------------------------------------------------------
# HTTP + cache
# ---------------------------------------------------------------------------


def _fetch_html(url: str, *, force_refresh: bool = False) -> str:
    """GET `url` with a 10-min disk cache (PLAN_AUTO_BET requirement)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", url)
    cache_path = CACHE_DIR / f"{safe}.html"
    if not force_refresh and cache_path.exists():
        age = time.time() - cache_path.stat().st_mtime
        if age < CACHE_TTL_SEC:
            return cache_path.read_text(encoding="utf-8")
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
    r.raise_for_status()
    cache_path.write_text(r.text, encoding="utf-8")
    return r.text


# ---------------------------------------------------------------------------
# Events listing -> today's event
# ---------------------------------------------------------------------------


def _parse_events_listing(html: str) -> list[dict]:
    """Extract upcoming events from /events. Returns a list of:
    {url, title, main_card_ts, prelims_ts}.
    """
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for art in soup.select("article.c-card-event--result"):
        info = art.select_one(".c-card-event--result__info") or art
        title_link = info.select_one(".c-card-event--result__headline a")
        if not title_link:
            continue
        href = title_link.get("href") or ""
        if not href.startswith("/event/"):
            continue
        date_div = info.select_one(".c-card-event--result__date") or art
        attrs = date_div.attrs
        try:
            main_ts = int(attrs.get("data-main-card-timestamp", "") or 0)
            prelims_ts = int(attrs.get("data-prelims-card-timestamp", "") or 0) or main_ts
        except ValueError:
            continue
        if main_ts == 0:
            continue
        out.append(
            {
                "url": UFC_BASE + href,
                "slug": href.removeprefix("/event/"),
                "title": title_link.get_text(strip=True),
                "main_card_ts": main_ts,
                "prelims_ts": prelims_ts,
            }
        )
    return out


def fetch_today_event(now_utc: pd.Timestamp | None = None) -> dict | None:
    """Return today's UFC event metadata, or None if no event tonight.

    "Today" = the EDT/EST calendar date that the main card starts on. This
    matches how UFC fans (and Kalshi tickers) refer to a card. An event whose
    main card starts at 8pm EDT June 6 is "the June 6 card" even though it
    crosses midnight UTC.

    Returned dict shape:
      {url, slug, title, main_card_ts, prelims_ts, date}
    where `date` is a tz-naive pd.Timestamp of the EDT date.
    """
    now = now_utc or pd.Timestamp.now(tz="UTC")
    html = _fetch_html(EVENTS_URL)
    events = _parse_events_listing(html)
    today_local_date = now.tz_convert(LOCAL_TZ).normalize().tz_localize(None)

    today_events = []
    for ev in events:
        main_utc = pd.Timestamp(ev["main_card_ts"], unit="s", tz="UTC")
        ev_local_date = main_utc.tz_convert(LOCAL_TZ).normalize().tz_localize(None)
        if ev_local_date == today_local_date:
            today_events.append((main_utc, ev))

    if not today_events:
        return None

    today_events.sort(key=lambda t: t[0])
    main_utc, ev = today_events[0]
    return {**ev, "date": today_local_date}


# ---------------------------------------------------------------------------
# Event page -> per-fight schedule
# ---------------------------------------------------------------------------


def _parse_section_fights(
    section, *, section_start_utc: pd.Timestamp, card_position: str
) -> list[FightSchedule]:
    """Parse one card section (main-card / prelims-card / early-prelims).

    UFC.com lists fights in display order (main event first). Chronologically
    the main event goes LAST, so we reverse before applying the +30min stride.
    """
    fights = section.select("div.c-listing-fight")
    parsed: list[tuple[str, str, str | None, bool]] = []

    def _corner_name(corner) -> str:
        """Some fighters have <span given>First</span><span family>Last</span>,
        others just have plain text inside the corner link. Handle both."""
        given = corner.select_one(".c-listing-fight__corner-given-name")
        family = corner.select_one(".c-listing-fight__corner-family-name")
        if given or family:
            parts = [
                given.get_text(strip=True) if given else "",
                family.get_text(strip=True) if family else "",
            ]
            return " ".join(p for p in parts if p).strip()
        return " ".join(corner.get_text(strip=True).split())

    for f in fights:
        red = f.select_one(".c-listing-fight__corner-name--red")
        blue = f.select_one(".c-listing-fight__corner-name--blue")
        if not (red and blue):
            continue
        fighter_a = _corner_name(red)
        fighter_b = _corner_name(blue)
        # The `details-content__class` div contains "Welterweight Bout".
        wc_el = f.select_one(".details-content__class") or f.select_one(".c-listing-fight__class-text")
        class_text = wc_el.get_text(" ", strip=True) if wc_el else ""
        title_bout = "title" in class_text.casefold()
        wc = class_text
        for token in ("UFC ", " Interim", " Title", " Bout"):
            wc = wc.replace(token, "")
        wc = " ".join(wc.split()) or None
        if not fighter_a or not fighter_b:
            continue
        parsed.append((fighter_a, fighter_b, wc, title_bout))

    # Reverse: page lists main event first; main event fights LAST.
    parsed = list(reversed(parsed))
    return [
        FightSchedule(
            fighter_a=a,
            fighter_b=b,
            start_time_utc=section_start_utc + pd.Timedelta(minutes=FIGHT_SPACING_MIN * i),
            card_position=card_position,
            weight_class=wc,
            title_bout=title_bout,
            # UFC title fights and the last-listed main-card bout (the main
            # event after reversal) are scheduled for five rounds.
            no_of_rounds=(5 if title_bout or (card_position == "main_card" and i == len(parsed) - 1) else 3),
        )
        for i, (a, b, wc, title_bout) in enumerate(parsed)
    ]


def fetch_card_schedule(
    event_url: str,
    *,
    main_card_ts: int | None = None,
    prelims_ts: int | None = None,
) -> list[FightSchedule]:
    """Scrape per-fight start times from a UFC event page.

    Section start times (main_card_ts, prelims_ts) should be passed in from
    `fetch_today_event` (the events-listing page). When omitted, falls back to
    a heuristic: 5pm ET prelims, 8pm ET main card (PLAN_AUTO_BET §edge case 1).
    Early prelims default to prelims_start - 60 min.

    Returns fights in chronological order.
    """
    html = _fetch_html(event_url)
    soup = BeautifulSoup(html, "html.parser")

    if main_card_ts and prelims_ts:
        main_start = pd.Timestamp(main_card_ts, unit="s", tz="UTC")
        prelim_start = pd.Timestamp(prelims_ts, unit="s", tz="UTC")
    else:
        # Heuristic fallback. Use the event page's hero timestamp if present,
        # else assume 5pm/8pm EDT today.
        today_local = pd.Timestamp.now(tz=LOCAL_TZ).normalize()
        prelim_start = (today_local + pd.Timedelta(hours=17)).tz_convert("UTC")
        main_start = (today_local + pd.Timedelta(hours=20)).tz_convert("UTC")
    early_start = prelim_start - pd.Timedelta(hours=1)

    out: list[FightSchedule] = []
    for section_id, start_utc, label in [
        ("early-prelims", early_start, "early_prelim"),
        ("prelims-card", prelim_start, "prelim"),
        ("main-card", main_start, "main_card"),
    ]:
        section = soup.find(id=section_id)
        if not section:
            continue
        out.extend(_parse_section_fights(section, section_start_utc=start_utc, card_position=label))
    return out


# ---------------------------------------------------------------------------
# Live fight results
# ---------------------------------------------------------------------------


@dataclass
class FightResult:
    fighter_red: str
    fighter_blue: str
    winner: str | None  # display name of the winner, None if undecided
    is_draw_or_nc: bool  # draw / no-contest — decided but no winner
    method: str | None  # "Decision - Unanimous", "Submission", ... if shown


def fetch_card_results(event_url: str, *, force_refresh: bool = True) -> list[FightResult]:
    """Parse per-fight RESULTS from a UFC.com event page.

    UFC.com updates the event page DURING the card: finished fights get
    `c-listing-fight__outcome--win/--loss` badges on each corner plus
    Round/Time/Method. This is the fastest public "fight is over" signal we
    have (Kalshi markets freeze in-fight and closures can batch late).

    force_refresh=True bypasses the 10-min disk cache — live results must be
    fresh; a stale page just means "no result yet", which is safe but slow.
    """
    html = _fetch_html(event_url, force_refresh=force_refresh)
    soup = BeautifulSoup(html, "html.parser")
    out: list[FightResult] = []
    for block in soup.select(".c-listing-fight"):
        names: dict[str, str | None] = {}
        outcomes: dict[str, str | None] = {}
        for corner in ("red", "blue"):
            nm = block.select_one(f".c-listing-fight__corner-name--{corner}")
            names[corner] = nm.get_text(" ", strip=True) if nm else None
            body = block.select_one(f".c-listing-fight__corner-body--{corner}")
            badge = body.select_one("[class*=outcome--]") if body else None
            outcome = None
            if badge:
                cls = " ".join(badge.get("class", []))
                if "outcome--win" in cls:
                    outcome = "win"
                elif "outcome--loss" in cls:
                    outcome = "loss"
                elif "outcome--draw" in cls or "outcome--no-contest" in cls:
                    outcome = "draw_nc"
            outcomes[corner] = outcome
        if not names["red"] or not names["blue"]:
            continue
        wins = [c for c in ("red", "blue") if outcomes[c] == "win"]
        draw_nc = any(outcomes[c] == "draw_nc" for c in ("red", "blue"))
        winner = names[wins[0]] if len(wins) == 1 else None
        method = None
        res_el = block.select_one(".c-listing-fight__result-text.method") or block.select_one(
            "[class*=result-text][class*=method]"
        )
        if res_el:
            method = res_el.get_text(" ", strip=True) or None
        out.append(
            FightResult(
                fighter_red=names["red"],
                fighter_blue=names["blue"],
                winner=winner,
                is_draw_or_nc=draw_nc and not winner,
                method=method,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


def write_audit_log(date: pd.Timestamp, event: dict, schedule: list[FightSchedule]) -> Path:
    """Persist the scraped schedule for replay + audit (PLAN_AUTO_BET §scraper notes)."""
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    path = AUDIT_DIR / f"{date.strftime('%Y-%m-%d')}.json"
    payload = {
        "scraped_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "event": {k: (v.isoformat() if isinstance(v, pd.Timestamp) else v) for k, v in event.items()},
        "fights": [{**asdict(f), "start_time_utc": f.start_time_utc.isoformat()} for f in schedule],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# CLI: `python -m ufc_pred.ingest.ufc_schedule`
# ---------------------------------------------------------------------------


def _cli():
    import argparse

    ap = argparse.ArgumentParser(description="Scrape today's UFC card schedule.")
    ap.add_argument("--url", help="Bypass /events, scrape this event URL directly.")
    ap.add_argument("--no-cache", action="store_true", help="Force re-fetch (ignore 10-min cache).")
    ap.add_argument(
        "--audit",
        action="store_true",
        help="Write parsed schedule to data/raw/ufc_schedule/<date>.json.",
    )
    args = ap.parse_args()

    if args.no_cache:
        # Wipe cache before fetching.
        if CACHE_DIR.exists():
            for p in CACHE_DIR.glob("*.html"):
                p.unlink()

    if args.url:
        event = {
            "url": args.url,
            "slug": args.url.rstrip("/").rsplit("/", 1)[-1],
            "title": "(direct)",
            "main_card_ts": None,
            "prelims_ts": None,
            "date": pd.Timestamp.now(tz=LOCAL_TZ).normalize().tz_localize(None),
        }
    else:
        event = fetch_today_event()
        if not event:
            print("No UFC event today.")
            return

    print(f"Event: {event['title']}")
    print(f"URL:   {event['url']}")
    if event.get("main_card_ts"):
        print(f"Prelims: {pd.Timestamp(event['prelims_ts'], unit='s', tz='UTC').tz_convert(LOCAL_TZ)}")
        print(f"Main:    {pd.Timestamp(event['main_card_ts'], unit='s', tz='UTC').tz_convert(LOCAL_TZ)}")

    schedule = fetch_card_schedule(
        event["url"],
        main_card_ts=event.get("main_card_ts"),
        prelims_ts=event.get("prelims_ts"),
    )
    print(f"\n{len(schedule)} fights scraped:")
    for f in schedule:
        local = f.start_time_utc.tz_convert(LOCAL_TZ)
        print(
            f"  [{f.card_position:12s}] {local.strftime('%H:%M %Z')}  "
            f"{f.fighter_a:30s} vs {f.fighter_b:30s}  ({f.weight_class})"
        )

    if args.audit:
        path = write_audit_log(event["date"], event, schedule)
        print(f"\nWrote audit log: {path}")


if __name__ == "__main__":
    _cli()

"""BestFightOdds (BFO) scraper + parsers for historical UFC moneylines.

Politeness: single thread, >= MIN_INTERVAL s between network requests, every raw
page cached gzip-compressed under data/raw/bfo/{events,fighters}/<slug>.html.gz,
so re-runs never re-download.

Page facts (verified 2026-10-13):
* Event page: JSON-LD ``SportsEvent`` carries ``name`` and ``startDate``.
  The ``<table class="odds-table">`` header has ``<th data-b=<book_id>>`` with the
  book name; body moneyline cells are ``<td class="but-sg" data-li="[book,pos,mu]">``
  and every fighter row has a ``but-si`` button cell ``data-li="[pos,mu]"``.
  Only *currently listed* books are shown as columns, so events before those books
  existed have empty cells (no per-book history for old events).
* Fighter page: ``team-stats-table`` with, per matchup, two rows (page owner first,
  opponent second), each with Open, Closing-range low, Closing-range high and the
  matchup id in ``data-li="[mu, pos]"`` of the sparkline cell. The event link sits
  on the main row and the date on the opponent row ("Sep 12th 2026").
"""

from __future__ import annotations

import gzip
import html as htmllib
import json
import re
import time
from datetime import datetime
from pathlib import Path

import requests

from ufc_pred.paths import RAW

BASE = "https://www.bestfightodds.com"
SITEMAP = f"{BASE}/sitemap-events.xml"
CACHE = RAW / "bfo"
UA = "Mozilla/5.0 (research; contact: local)"
MIN_INTERVAL = 1.1

_session = None
_last_request = 0.0


def _read_with_deadline(resp, deadline: float) -> bytes:
    """Read a streamed body, aborting if the whole transfer exceeds ``deadline`` s.

    ``requests`` timeouts are per-socket-read, so a trickling connection can
    otherwise hang forever (observed once on BFO).
    """
    t0, chunks = time.monotonic(), []
    for chunk in resp.iter_content(65536):
        chunks.append(chunk)
        if time.monotonic() - t0 > deadline:
            resp.close()
            raise requests.Timeout(f"body read exceeded {deadline}s")
    return b"".join(chunks)


class BlockedError(RuntimeError):
    """Raised when BFO persistently refuses requests (403/429)."""


# ---------------------------------------------------------------- fetching
def _cache_path(url: str) -> Path:
    path = url.replace(BASE, "").strip("/")
    if path.startswith("events/"):
        return CACHE / "events" / f"{path.split('/', 1)[1]}.html.gz"
    if path.startswith("fighters/"):
        return CACHE / "fighters" / f"{path.split('/', 1)[1]}.html.gz"
    return CACHE / "misc" / (re.sub(r"[^A-Za-z0-9._-]+", "_", path) + ".gz")


def is_cached(url: str) -> bool:
    return _cache_path(url).exists()


def fetch(url: str, refresh: bool = False, max_tries: int = 5) -> str:
    """Return page text, from cache when available; otherwise polite GET."""
    global _session, _last_request
    path = _cache_path(url)
    if path.exists() and not refresh:
        return gzip.decompress(path.read_bytes()).decode("utf-8")
    if _session is None:
        _session = requests.Session()
        _session.headers["User-Agent"] = UA
    blocked = 0
    for attempt in range(max_tries):
        wait = MIN_INTERVAL - (time.monotonic() - _last_request)
        if wait > 0:
            time.sleep(wait)
        _last_request = time.monotonic()
        try:
            resp = _session.get(url, timeout=(10, 30), stream=True)
            body = _read_with_deadline(resp, deadline=90)
        except requests.RequestException:
            time.sleep(5 * 2**attempt)
            continue
        if resp.status_code == 200:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(gzip.compress(body))
            return body.decode(resp.encoding or "utf-8", errors="replace")
        if resp.status_code == 404:
            raise FileNotFoundError(url)
        if resp.status_code in (403, 429):
            blocked += 1
        if resp.status_code in (403, 429) or resp.status_code >= 500:
            time.sleep(10 * 2**attempt)
            continue
        raise RuntimeError(f"HTTP {resp.status_code} for {url}")
    if blocked:
        raise BlockedError(f"{blocked} blocked responses for {url}")
    raise RuntimeError(f"giving up on {url}")


# ---------------------------------------------------------------- event list
_EXCLUDE = ("ufc-bjj", "dana-white", "contender-series", "dwcs", "road-to-ufc",
            "fight-pass-invitational")
# note: "ufc-freedom-fights-250" is the real 2026-06-14 White House card, so it is kept


def is_ufc_card(slug: str) -> bool:
    s = slug.lower()
    if any(x in s for x in _EXCLUDE):
        return False
    return s.startswith("ufc") or s.startswith("noche-ufc")


def list_event_urls(ufc_only: bool = True, refresh: bool = False) -> list[dict]:
    """[{slug, url, lastmod}] from the events sitemap (lastmod ~= event date)."""
    xml = fetch(SITEMAP, refresh=refresh)
    out = []
    for loc, lastmod in re.findall(r"<loc>([^<]+)</loc>\s*<lastmod>([^<]*)</lastmod>", xml):
        slug = loc.rstrip("/").rsplit("/", 1)[-1]
        if ufc_only and not is_ufc_card(slug):
            continue
        out.append({"slug": slug, "url": loc, "lastmod": lastmod})
    return out


# ---------------------------------------------------------------- parsing
def _text(fragment: str) -> str:
    return htmllib.unescape(re.sub(r"<[^>]+>", "", fragment)).strip()


def _american(s: str):
    s = s.strip().replace("−", "-")
    if re.fullmatch(r"[+-]?\d+", s):
        return int(s)
    return None


def parse_event(page: str) -> dict:
    """Parse an event page -> {'meta': {...}, 'rows': [...]} (one row per fighter x book).

    Fighters with no line from any book still get a row with book_id None so the
    matchup/fighter slug list is complete.
    """
    meta = {"event_title": None, "event_date": None, "header_date": None}
    m = re.search(r'<script type="application/ld\+json">(.*?)</script>', page, re.S)
    if m:
        try:
            ld = json.loads(m.group(1))
            meta["event_title"] = ld.get("name")
            meta["event_date"] = ld.get("startDate")
        except json.JSONDecodeError:
            pass
    m = re.search(r'<div class="table-header"><a href="[^"]*"><h1>(.*?)</h1>', page)
    if m and not meta["event_title"]:
        meta["event_title"] = _text(m.group(1)).removesuffix(" Odds")
    m = re.search(r'table-header-date">([^<]*)<', page)
    if m:
        meta["header_date"] = m.group(1).strip()

    start = page.find('<table class="odds-table">')
    if start < 0:
        return {"meta": meta, "rows": []}
    end = page.find("</table>", start)
    table = page[start:end]
    thead, _, tbody = table.partition("<tbody")
    books = {}
    for bid, inner in re.findall(r'<th scope="col" data-b="(\d+)">(.*?)</th>', thead, re.S):
        books[int(bid)] = _text(inner.split("<br", 1)[0])

    rows = []
    for tr in re.split(r"<tr\b", tbody)[1:]:
        head = tr[: tr.find(">")]
        if 'class="pr"' in head:
            continue
        fm = re.search(r'href="/fighters/([^"]+)"><span[^>]*>([^<]*)</span>', tr)
        sm = re.search(r'class="button-cell but-si" data-li="\[(\d),(\d+)\]"', tr)
        if not fm or not sm:
            continue
        slug, name = fm.group(1), _text(fm.group(2))
        pos, mu = int(sm.group(1)), int(sm.group(2))
        cells = re.findall(
            r'class="but-sg" data-li="\[(\d+),(\d),(\d+)\]"><span[^>]*>([^<]*)</span>', tr
        )
        base = {"matchup_id": mu, "fighter_name": name, "fighter_slug": slug, "fighter_pos": pos}
        got = False
        for bid, p, m_id, odds in cells:
            val = _american(odds)
            if val is None:
                continue
            got = True
            rows.append({**base, "book_id": int(bid), "book_name": books.get(int(bid)),
                         "american_odds": val})
        if not got:
            rows.append({**base, "book_id": None, "book_name": None, "american_odds": None})
    # opponent name per matchup
    names = {(r["matchup_id"], r["fighter_pos"]): r["fighter_name"] for r in rows}
    for r in rows:
        r["opponent_name"] = names.get((r["matchup_id"], 3 - r["fighter_pos"]))
    meta["books"] = books
    return {"meta": meta, "rows": rows}


def _parse_bfo_date(s: str):
    s = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", s.strip())
    for fmt in ("%b %d %Y", "%B %d %Y"):
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            pass
    return None


def parse_fighter(page: str) -> list[dict]:
    """Parse a fighter page's odds history -> one row per (matchup, side).

    Both sides of every matchup are on the page, so a single fighter page yields
    opening lines for the owner *and* each opponent.
    """
    start = page.find('<table class="team-stats-table"')
    if start < 0:
        return []
    end = page.find("</table>", start)
    body = page[start:end].partition("<tbody>")[2]
    out = []
    cur_event = None
    pending = None  # main row awaiting its opponent row
    for tr in re.split(r"<tr\b", body)[1:]:
        if "event-header" in tr[:80]:
            em = re.search(r'href="/events/([^"]+)">(.*?)</a>\s*([^<]*)<', tr)
            cur_event = {"event_slug": em.group(1), "event_title": _text(em.group(2)),
                         "event_date": _parse_bfo_date(em.group(3))} if em else None
            continue
        fm = re.search(r'class="oppcell"><a href="/fighters/([^"]+)">([^<]*)</a>', tr)
        if not fm:
            continue
        lines = [_american(x) for x in re.findall(
            r'<td class="moneyline"[^>]*>(?:<span[^>]*>([^<]*)</span>)?', tr)]
        lines = (lines + [None, None, None])[:3]
        rec = {"fighter_slug": fm.group(1), "fighter_name": _text(fm.group(2)),
               "open_odds": lines[0], "close_low": lines[1], "close_high": lines[2]}
        if 'class="main-row"' in tr[:40]:
            mm = re.search(r'data-li="\[(\d+),(\d)\]"', tr)
            ev = re.search(r'class="item-non-mobile" scope="row"[^>]*><a href="/events/([^"]+)"', tr)
            rec["matchup_id"] = int(mm.group(1)) if mm else None
            rec["fighter_pos"] = int(mm.group(2)) if mm else None
            rec.update(cur_event or {})
            if ev:
                rec["event_slug"] = ev.group(1)
            pending = rec
            continue
        if pending is None:
            continue
        dm = re.search(r'color: #767676">([^<]*)<', tr)
        date = _parse_bfo_date(dm.group(1)) if dm else pending.get("event_date")
        owner = pending
        opp = {**rec, "matchup_id": owner["matchup_id"],
               "fighter_pos": (3 - owner["fighter_pos"]) if owner["fighter_pos"] else None,
               "event_slug": owner.get("event_slug"), "event_title": owner.get("event_title")}
        owner["event_date"] = opp["event_date"] = date or owner.get("event_date")
        owner["opponent_name"], opp["opponent_name"] = opp["fighter_name"], owner["fighter_name"]
        owner["opponent_slug"], opp["opponent_slug"] = opp["fighter_slug"], owner["fighter_slug"]
        out.extend([owner, opp])
        pending = None
    return out


# ---------------------------------------------------------------- odds math
def american_to_prob(a):
    """Implied probability (with vig) from American odds; NaN-safe."""
    if a is None or a != a:
        return float("nan")
    a = float(a)
    return 100.0 / (a + 100.0) if a > 0 else -a / (-a + 100.0)


def devig_pair(a_r, a_b):
    """Proportional de-vig of a two-way American pair -> P(R)."""
    pr, pb = american_to_prob(a_r), american_to_prob(a_b)
    s = pr + pb
    return pr / s if s == s and s > 0 else float("nan")

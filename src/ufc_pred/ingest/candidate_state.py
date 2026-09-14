"""Versioned candidate inputs: frozen histories plus strictly prior new UFC bouts.

Static profiles and the audited external-history cohort never refresh silently.
New identities require a new reviewed input version. Every live response is
archived by content hash, so the exact observations can be replayed offline.
"""

import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

from .identity import PROFILE_URLS, canonical, normalise
from .strict_history import RawHistoryStateSource
from .ufcstats_scraper import (
    _fight_seconds,
    _is_ufc_history_event,
    _parse_int,
    _parse_landed_attempted,
    parse_date,
)
from .ufcstats_state import UFCStatsStateSource

TOTALS = ["sig_landed", "sig_attempted", "td_landed", "td_attempted", "sub_attempted", "seconds"]


def new_ufc_entries(html, fighter, url, after, before, get):
    """Read only completed UFC bouts after the frozen export and before target day.

    Missing totals/identity in a newly completed bout block prediction rather
    than silently treating unobserved statistics as zero.
    """
    soup = BeautifulSoup(html, "html.parser")
    header = soup.select_one(".b-content__title-highlight")
    actual = header.get_text(" ", strip=True) if header else ""
    scoped = (
        fighter in ("Bruno Silva [FLW]", "Bruno Silva [MW]")
        and url == PROFILE_URLS[fighter]
        and normalise(actual) == "bruno silva"
    )
    if not scoped and normalise(canonical(actual)) != normalise(fighter):
        raise ValueError(f"Profile identity mismatch for {fighter}: {actual}")
    entries, seen = [], set()
    for row in soup.select("tr.b-fight-details__table-row"):
        cols = row.select("p.b-fight-details__table-text")
        if len(cols) < 17 or not _is_ufc_history_event(cols[11].get_text(" ", strip=True)):
            continue
        outcome = cols[0].get_text(strip=True).lower()
        if outcome not in ("win", "loss", "draw", "nc"):
            continue
        date = pd.Timestamp(parse_date(cols[12].get_text(strip=True))).normalize()
        if not after < date < before:
            continue
        link = row.find("a", href=re.compile(r"fight-details/"))
        detail_url = row.get("data-link") or (link.get("href") if link else None)
        if not detail_url:
            raise ValueError(f"Missing bout detail for {fighter} on {date.date()}")
        if detail_url in seen:
            raise ValueError(f"Duplicate bout detail: {detail_url}")
        seen.add(detail_url)
        detail = BeautifulSoup(get(detail_url), "html.parser")
        body = detail.find("tbody", class_="b-fight-details__table-body")
        tr = body.find("tr") if body else None
        cells = tr.find_all("td", recursive=False) if tr else []
        if len(cells) < 8:
            raise ValueError(f"Missing totals: {detail_url}")
        urls = [a.get("href", "").rstrip("/") for a in cells[0].find_all("a", href=True)]
        side = urls.index(url.rstrip("/"))  # raises on wrong identity
        sig = cells[2].find_all("p")[side].get_text(strip=True)
        td = cells[5].find_all("p")[side].get_text(strip=True)
        sub = cells[7].find_all("p")[side].get_text(strip=True)
        if (
            not re.fullmatch(r"\d+\s+of\s+\d+", sig)
            or not re.fullmatch(r"\d+\s+of\s+\d+", td)
            or not sub.isdigit()
        ):
            raise ValueError(f"Incomplete totals: {detail_url}")
        sl, sa = _parse_landed_attempted(sig)
        tl, ta = _parse_landed_attempted(td)
        seconds = _fight_seconds(cols[15].get_text(strip=True), cols[16].get_text(strip=True))
        if seconds <= 0:
            raise ValueError(f"Invalid elapsed time: {detail_url}")
        method = cols[13].get_text(strip=True)
        if method == "KO/TKO" and re.search(
            r"Method:\s*TKO\s*-\s*Doctor", detail.get_text(" ", strip=True), re.I
        ):
            method = "TKO - Doctor's Stoppage"
        entries.append(
            dict(
                key=normalise(fighter),
                date=date,
                outcome={"win": "W", "loss": "L", "draw": "D", "nc": "NC"}[outcome],
                rounds=int(cols[15].get_text(strip=True)),
                method=method,
                title=bool(row.find("img", src=re.compile("belt"))),
                sig_landed=sl,
                sig_attempted=sa,
                td_landed=tl,
                td_attempted=ta,
                sub_attempted=_parse_int(sub),
                seconds=seconds,
            )
        )
    # Profile pages are newest first; raw export accumulation is chronological.
    return sorted(entries, key=lambda r: r["date"])


class CandidateStateSource(RawHistoryStateSource):
    def __init__(self, contract_path, *, live=False, html_getter=None, capture_dir=None):
        contract = json.loads(Path(contract_path).read_text())
        self.profiles = contract["profiles"]
        for profile in self.profiles.values():
            profile["DOB"] = pd.to_datetime(profile.get("DOB"), errors="coerce")
        self.entries = contract["entries"]
        for row in self.entries:
            row["date"] = pd.Timestamp(row["date"])
        self.last_event = pd.Timestamp(contract["last_ufc_event"])
        self._build_states(self.entries)
        self._by_key = {}
        for row in self.entries:
            self._by_key.setdefault(row["key"], []).append(row)
        self.live = live
        self._live_cache = {}
        self.capture_dir = Path(capture_dir) if capture_dir else None
        self.observations = {}
        self.client = UFCStatsStateSource(html_getter=html_getter) if live else None

    def _get(self, url):
        html = self.client._get(url)
        digest = hashlib.sha256(html.encode()).hexdigest()
        self.observations[url] = digest
        if self.capture_dir:
            self.capture_dir.mkdir(parents=True, exist_ok=True)
            path = self.capture_dir / (digest + ".html")
            if not path.exists():
                path.write_text(html)
        return html

    def __call__(self, fighter, fight_date):
        fighter = canonical(fighter)
        target = pd.Timestamp(fight_date).tz_localize(None).normalize()
        key = normalise(fighter)
        if key not in self.profiles:
            raise ValueError(
                f"Identity outside frozen candidate profiles: {fighter}; input version review required"
            )
        state = super().__call__(fighter, target)
        if not self.live or target <= self.last_event:
            return state
        cache_key = (key, str(target.date()))
        if cache_key not in self._live_cache:
            url = self.profiles[key]["URL"]
            extra = new_ufc_entries(self._get(url), fighter, url, self.last_event, target, self._get)
            accumulator = RawHistoryStateSource.__new__(RawHistoryStateSource)
            accumulator.profiles = self.profiles
            accumulator._build_states(self._by_key.get(key, []) + extra)
            self._live_cache[cache_key] = accumulator(fighter, target)
        return self._live_cache[cache_key].copy()

    def close(self):
        if self.client:
            self.client.close()


def state_source_for_models(models_dir=None, *, capture_dir=None):
    """Keep default CLI/recommendation inputs paired with the selected model bundle."""
    from ufc_pred.paths import CONFIGS, ROOT

    if models_dir is None:
        config = json.loads((CONFIGS / "inference.json").read_text())
        models_dir = ROOT / config["model_bundle"]
    bundle = Path(models_dir).parent
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    contract = bundle / "candidate_inputs.json"
    if contract.exists() or manifest.get("input_constructor") == "candidate_v2":
        return CandidateStateSource(contract, live=True, capture_dir=capture_dir)
    return UFCStatsStateSource()

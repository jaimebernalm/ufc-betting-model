"""Discover new UFC events from ufcstats.com and append them to ufc-master.csv."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

from .kaggle_mdabbert import HISTORY_CSV
from .rankings_attach import attach_ranks, load_rankings
from .ufcstats_client import UFCStatsClient
from .ufcstats_scraper import scrape_event

COMPLETED_EVENTS_URL = "http://ufcstats.com/statistics/events/completed?page=all"


@dataclass
class EventListing:
    date: datetime
    name: str
    url: str


def list_completed_events(client: UFCStatsClient) -> list[EventListing]:
    html = client.get(COMPLETED_EVENTS_URL)
    bs = BeautifulSoup(html, "html.parser")
    out: list[EventListing] = []
    seen: set[str] = set()
    for a in bs.find_all("a", href=re.compile(r"event-details/")):
        href = a["href"]
        if href in seen:
            continue
        seen.add(href)
        row = a.find_parent("tr")
        if row is None:
            continue
        span = row.find("span", {"class": "b-statistics__date"})
        if span is None:
            continue
        try:
            dt = datetime.strptime(span.get_text(strip=True), "%B %d, %Y")
        except ValueError:
            continue
        out.append(EventListing(dt, a.get_text(strip=True), href))
    return out


def latest_master_date(csv_path: Path = HISTORY_CSV) -> datetime:
    df = pd.read_csv(csv_path, usecols=["date"], parse_dates=["date"])
    return df["date"].max().to_pydatetime()


def update_master(
    csv_path: Path = HISTORY_CSV,
    *,
    since: datetime | None = None,
    dry_run: bool = False,
) -> dict:
    """Scrape any events newer than the cutoff and append to ufc-master.csv.

    Args:
        csv_path: Master CSV path to update.
        since: Lower bound (exclusive). Defaults to the max date already in the file.
        dry_run: If True, list the events to scrape but do not modify the CSV.
    """
    if since is None:
        since = latest_master_date(csv_path)
    today = datetime.now()

    with UFCStatsClient() as client:
        events = list_completed_events(client)
        events = [e for e in events if e.date > since and e.date <= today]
        events.sort(key=lambda e: e.date)

        summary = {
            "cutoff": since.strftime("%Y-%m-%d"),
            "new_events": [
                {"date": e.date.strftime("%Y-%m-%d"), "name": e.name, "url": e.url} for e in events
            ],
            "rows_appended": 0,
        }
        if dry_run or not events:
            return summary

        rankings = load_rankings()
        master = pd.read_csv(csv_path)
        master_cols = master.columns.tolist()

        new_rows = []
        for e in events:
            print(f"[scrape] {e.date:%Y-%m-%d}  {e.name}")
            df = scrape_event(e.url, client=client)
            df = attach_ranks(df, rankings=rankings)
            for c in master_cols:
                if c not in df.columns:
                    df[c] = None
            df = df[master_cols]
            new_rows.append(df)
            print(f"  → {len(df)} fights scraped")

        appended = pd.concat(new_rows, ignore_index=True)
        out = pd.concat([master, appended], ignore_index=True)
        out.to_csv(csv_path, index=False)
        summary["rows_appended"] = int(len(appended))
        return summary

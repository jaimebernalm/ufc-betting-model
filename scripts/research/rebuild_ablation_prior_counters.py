"""Recover changed record fields from official prior-only full fighter histories.

Candidate universe: fighters whose original versus UFC-only counters differ.
Keep the frozen UFC-only values when identity cannot be verified; report every
failure. Uses no profile rate snapshots and no target/future result in features.
"""

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from ablate_predictive_degradation import OUT, datasets
from bs4 import BeautifulSoup

from ufc_pred.ingest.identity import PROFILE_URLS, canonical, normalise
from ufc_pred.ingest.strict_history import COUNTERS, RawHistoryStateSource
from ufc_pred.ingest.ufcstats_client import UFCStatsClient
from ufc_pred.ingest.ufcstats_scraper import _is_ufc_history_event, parse_date
from ufc_pred.paths import ROOT

_, _, new, old = datasets()
source = RawHistoryStateSource(ROOT / "data/raw/ufcstats_export")
names = set()
for side in ["R", "B"]:
    cols = [side + "_" + c for c in COUNTERS]
    changed = ~np.isclose(new[cols].to_numpy(float), old[cols].to_numpy(float), equal_nan=True).all(1)
    names.update(new.loc[changed, side + "_fighter"])
folder = OUT / "official_checks"
folder.mkdir(exist_ok=True)
local = threading.local()


def fetch(name):
    profile = source.profiles[normalise(name)]
    url = profile["URL"]
    path = folder / (hashlib.sha256(url.encode()).hexdigest() + ".html")
    try:
        if not path.exists():
            if not hasattr(local, "client"):
                local.client = UFCStatsClient(request_delay=2.0)
            text = local.client.get(url)
            path.write_text(text)
        soup = BeautifulSoup(path.read_text(), "html.parser")
        header = soup.select_one(".b-content__title-highlight")
        actual = header.get_text(" ", strip=True) if header else ""
        # The scoped Bruno names use verified distinct URLs.
        verified_scoped_bruno = (
            name in ("Bruno Silva [FLW]", "Bruno Silva [MW]")
            and url == PROFILE_URLS[name]
            and normalise(actual) == "bruno silva"
        )
        if not verified_scoped_bruno and normalise(canonical(actual)) != normalise(name):
            raise ValueError("Identity mismatch: " + actual)
        extra = []
        for row in soup.select("tr.b-fight-details__table-row"):
            cols = row.select("p.b-fight-details__table-text")
            if len(cols) < 17:
                continue
            event = cols[11].get_text(" ", strip=True)
            if _is_ufc_history_event(event):
                continue
            try:
                date = pd.Timestamp(parse_date(cols[12].get_text(strip=True)))
                outcome = cols[0].get_text(strip=True).lower()
                if outcome not in ["win", "loss", "draw", "nc"]:
                    continue
                extra.append(
                    {
                        "date": str(date.date()),
                        "outcome": {"win": "W", "loss": "L", "draw": "D", "nc": "NC"}[outcome],
                        "rounds": int(cols[15].get_text(strip=True)),
                        "method": cols[13].get_text(strip=True),
                        "title": bool(row.find("img", src=lambda x: x and "belt" in x)),
                        "event": event,
                    }
                )
            except (ValueError, TypeError):
                continue
        return name, {"url": url, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "extra": extra}
    except Exception as e:
        return name, {"url": url, "error": str(e)}


registry_path = OUT / "prior_external_histories.json"
registry = json.loads(registry_path.read_text()) if registry_path.exists() else {}
pending = sorted(names - set(registry))
with ThreadPoolExecutor(max_workers=4) as pool:
    futures = [pool.submit(fetch, n) for n in pending]
    for future in as_completed(futures):
        name, record = future.result()
        registry[name] = record
        registry_path.write_text(json.dumps(registry, indent=2))
        if len(registry) % 25 == 0:
            print("profiles", len(registry), "/", len(names), flush=True)

# UFC bouts remain from the verified raw source; add only prior external bouts.
events = pd.read_csv(ROOT / "data/raw/ufcstats_export/ufc_event_details.csv")
events["EVENT"] = events.EVENT.str.strip()
events["date"] = pd.to_datetime(events.DATE)
results = pd.read_csv(ROOT / "data/raw/ufcstats_export/ufc_fight_results.csv")
results["EVENT"] = results.EVENT.str.strip()
results = results.merge(events[["EVENT", "date"]], on="EVENT", validate="many_to_one")
by_fighter = {n: [] for n, r in registry.items() if r.get("extra")}
import re

for r in results.itertuples(index=False):
    pair = re.split(r"\s+vs\.?\s+", r.BOUT.strip(), maxsplit=1)
    outcomes = str(r.OUTCOME).split("/")
    if len(pair) != 2 or len(outcomes) != 2:
        continue
    for fighter, outcome in zip(pair, outcomes, strict=True):
        name = canonical(fighter, r.WEIGHTCLASS)
        if name in by_fighter:
            by_fighter[name].append(
                {
                    "date": r.date,
                    "outcome": outcome.strip(),
                    "rounds": int(r.ROUND),
                    "method": r.METHOD.strip(),
                    "title": "title" in str(r.WEIGHTCLASS).lower(),
                    "event": r.EVENT,
                }
            )

method_map = {
    "M-DEC": "win_by_Decision_Majority",
    "S-DEC": "win_by_Decision_Split",
    "U-DEC": "win_by_Decision_Unanimous",
    "KO/TKO": "win_by_KO/TKO",
    "SUB": "win_by_Submission",
    "TKO - Doctor's Stoppage": "win_by_TKO_Doctor_Stoppage",
    "Submission": "win_by_Submission",
    "Decision - Majority": "win_by_Decision_Majority",
    "Decision - Split": "win_by_Decision_Split",
    "Decision - Unanimous": "win_by_Decision_Unanimous",
}
output = new.copy()
for name, ufc in by_fighter.items():
    history = pd.DataFrame(ufc + registry[name]["extra"])
    history["date"] = pd.to_datetime(history.date)
    history = history.sort_values("date", kind="stable")
    dates = []
    states = []
    state = dict.fromkeys(COUNTERS, 0)
    running = 0
    for date, day in history.groupby("date", sort=True):
        for r in day.itertuples(index=False):
            state["total_rounds_fought"] += r.rounds
            state["total_title_bouts"] += int(r.title)
            if r.outcome == "W":
                state["wins"] += 1
                state["current_win_streak"] += 1
                state["current_lose_streak"] = 0
                running += 1
                state["longest_win_streak"] = max(state["longest_win_streak"], running)
                if r.method in method_map:
                    state[method_map[r.method]] += 1
            elif r.outcome == "L":
                state["losses"] += 1
                state["current_lose_streak"] += 1
                state["current_win_streak"] = 0
                running = 0
            elif r.outcome == "D":
                state["draw"] += 1
                running = 0
        dates.append(date)
        states.append(dict(state))
    dates = pd.DatetimeIndex(dates)
    for side in ["R", "B"]:
        for i in output.index[output[side + "_fighter"].eq(name)]:
            # Strictly before target day, including for same-day tournaments.
            pos = dates.searchsorted(output.loc[i, "date"], side="left") - 1
            values = states[pos] if pos >= 0 else dict.fromkeys(COUNTERS, 0)
            for c, value in values.items():
                output.loc[i, side + "_" + c] = value
from ufc_pred.inference.upcoming_builder import _DIFF_PAIRS

for diff, base in _DIFF_PAIRS:
    if base in COUNTERS:
        output[diff] = output["B_" + base] - output["R_" + base]
output.to_parquet(OUT / "strict_extended_counters.parquet", index=False)
coverage = {
    "candidate_fighters": len(names),
    "fetched": len(registry),
    "with_external_history": len(by_fighter),
    "errors": {n: r["error"] for n, r in registry.items() if "error" in r},
    "scope": "fighters with changed record fields; others retain UFC-only baseline; no target/future bouts included",
    "external_events": sum(len(r.get("extra", [])) for r in registry.values()),
}
(OUT / "external_counter_coverage.json").write_text(json.dumps(coverage, indent=2))
print(json.dumps(coverage, indent=2), flush=True)

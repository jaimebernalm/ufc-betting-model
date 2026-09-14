"""Independent official-parser checks, with cached evidence and scope comparison."""

import hashlib
import json

import pandas as pd
from ablate_predictive_degradation import OUT

from ufc_pred.ingest import ufcstats_scraper
from ufc_pred.ingest.identity import normalise
from ufc_pred.ingest.strict_history import RawHistoryStateSource
from ufc_pred.ingest.ufcstats_client import UFCStatsClient
from ufc_pred.ingest.ufcstats_state import UFCStatsStateSource
from ufc_pred.paths import ROOT

cases = [
    ("Arnold Allen", "2026-01-24"),
    ("Jean Silva", "2026-01-24"),
    ("Waldo Cortes Acosta", "2026-09-12"),
    ("Curtis Blaydes", "2026-09-12"),
    ("Anthony Wint", "2026-08-22"),
    ("Terrance Chatman", "2026-08-22"),
    ("Brendan Allen", "2026-06-06"),
    ("Guilherme Pat", "2026-04-04"),
]
raw = RawHistoryStateSource(ROOT / "data/raw/ufcstats_export")
cache = OUT / "official_checks"
cache.mkdir(exist_ok=True)
http = UFCStatsClient()
urls = {}


def get(url):
    key = hashlib.sha256(url.encode()).hexdigest()
    p = cache / (key + ".html")
    if not p.exists():
        p.write_text(http.get(url))
    urls[url] = {"file": str(p.relative_to(ROOT)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    return p.read_text()


out = []
for name, date in cases:
    row = {"fighter": name, "date": date}
    try:
        live = UFCStatsStateSource(html_getter=get)
        live.fighter_url = lambda name: raw.profiles[normalise(name)]["URL"]
        current = live.get_state(name, pd.Timestamp(date))
        expected = raw(name, date)
        row["ufc_scope_mismatches"] = {
            c: {"raw": str(expected[c]), "official": str(current[c])}
            for c in expected
            if not ((pd.isna(expected[c]) and pd.isna(current[c])) or expected[c] == current[c])
        }
        original = ufcstats_scraper._is_ufc_history_event
        try:
            ufcstats_scraper._is_ufc_history_event = lambda name: True
            all_source = UFCStatsStateSource(html_getter=get)
            all_source.fighter_url = live.fighter_url
            all_state = all_source.get_state(name, pd.Timestamp(date))
        finally:
            ufcstats_scraper._is_ufc_history_event = original
        row["additional_competition_prior_information"] = {
            c: {"ufc_only": str(current[c]), "all_official_history": str(all_state[c])}
            for c in current
            if not ((pd.isna(current[c]) and pd.isna(all_state[c])) or current[c] == all_state[c])
        }
    except Exception as e:
        row["error"] = str(e)
    out.append(row)
    (OUT / "independent_source_checks.json").write_text(json.dumps(out, indent=2))
    (cache / "manifest.json").write_text(json.dumps(urls, indent=2))
    print(
        name,
        "mismatches",
        row.get("ufc_scope_mismatches"),
        "extra fields",
        len(row.get("additional_competition_prior_information", {})),
        row.get("error", ""),
        flush=True,
    )
http.__exit__(None, None, None)

"""Verify a reviewable parser fix against archived official HTML, without changing shadow."""

import difflib
import hashlib
import json
import sys
import types

import pandas as pd
from ablate_predictive_degradation import OUT

from ufc_pred.ingest import ufcstats_state
from ufc_pred.ingest.strict_history import RawHistoryStateSource
from ufc_pred.paths import ROOT

path = ROOT / "src/ufc_pred/ingest/ufcstats_scraper.py"
before = path.read_text()
needle = """        method = cols[13].get_text(strip=True)
        if result == "win":"""
replacement = """        method = cols[13].get_text(strip=True)
        # The profile summary labels medical stoppages as KO/TKO. The bout
        # detail has the specific method used by the historical raw export.
        if result == "win" and method == "KO/TKO" and fight_html_getter is not None:
            detail_url = row.get("data-link")
            if not detail_url:
                link = row.find("a", href=re.compile(r"fight-details/"))
                detail_url = link.get("href") if link is not None else None
            if detail_url:
                detail = BeautifulSoup(fight_html_getter(detail_url), "html.parser")
                if re.search(r"Method:\\s*TKO\\s*-\\s*Doctor", detail.get_text(" ", strip=True), re.I):
                    method = "TKO - Doctor's Stoppage"
        if result == "win":"""
assert before.count(needle) == 1
after = before.replace(needle, replacement)
(OUT / "proposed_doctor_parser_fix.patch").write_text(
    "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile="a/src/ufc_pred/ingest/ufcstats_scraper.py",
            tofile="b/src/ufc_pred/ingest/ufcstats_scraper.py",
        )
    )
)
module = types.ModuleType("ufc_pred.ingest._proposed_doctor_fix")
module.__package__ = "ufc_pred.ingest"
sys.modules[module.__name__] = module
exec(compile(after, str(path), "exec"), module.__dict__)
ufcstats_state.parse_fighter_page = module.parse_fighter_page
raw = RawHistoryStateSource(ROOT / "data/raw/ufcstats_export")
from ufc_pred.ingest.identity import normalise


def get(url):
    p = OUT / "official_checks" / (hashlib.sha256(url.encode()).hexdigest() + ".html")
    return p.read_text()


result = []
for name, date in [
    ("Jean Silva", "2026-01-24"),
    ("Curtis Blaydes", "2026-09-12"),
    ("Arnold Allen", "2026-01-24"),
]:
    source = ufcstats_state.UFCStatsStateSource(html_getter=get)
    source.fighter_url = lambda name: raw.profiles[normalise(name)]["URL"]
    actual = source.get_state(name, pd.Timestamp(date))
    expected = raw(name, date)
    diffs = {
        c: [str(expected[c]), str(actual[c])]
        for c in expected
        if not ((pd.isna(expected[c]) and pd.isna(actual[c])) or expected[c] == actual[c])
    }
    assert not diffs, (name, diffs)
    result.append({"fighter": name, "date": date, "mismatches": diffs})
(OUT / "proposed_doctor_fix_validation.json").write_text(json.dumps(result, indent=2))
print("Proposed patch verified on three independent official profiles; frozen runtime unchanged")

"""Verify missing static attributes against official pages; isolate research data.

Publication timestamps are unavailable: this is a source-quality sensitivity,
not proof that all static values were known at every past forecast time.
"""

import hashlib
import json

import pandas as pd
from ablate_predictive_degradation import OUT
from bs4 import BeautifulSoup

from ufc_pred.ingest.identity import canonical, normalise
from ufc_pred.ingest.strict_history import RawHistoryStateSource, _measurement
from ufc_pred.ingest.ufcstats_client import UFCStatsClient
from ufc_pred.paths import ROOT

source = RawHistoryStateSource(ROOT / "data/raw/ufcstats_export")
names = sorted(pd.read_csv(OUT / "lost_profile_fields.csv").fighter.unique())
folder = OUT / "official_checks"
folder.mkdir(exist_ok=True)
results = {}
with UFCStatsClient() as client:
    for name in names:
        url = source.profiles[normalise(name)]["URL"]
        path = folder / (hashlib.sha256(url.encode()).hexdigest() + ".html")
        try:
            if not path.exists():
                path.write_text(client.get(url))
            soup = BeautifulSoup(path.read_text(), "html.parser")
            header = soup.select_one(".b-content__title-highlight")
            actual = header.get_text(" ", strip=True) if header else ""
            if normalise(canonical(actual)) != normalise(name):
                raise ValueError(f"Identity mismatch: {actual!r}")
            info = {}
            for li in soup.select("li.b-list__box-list-item"):
                text = li.get_text(" ", strip=True)
                if ":" in text:
                    key, value = text.split(":", 1)
                    info[key.strip().lower()] = value.strip()
            height = _measurement(info.get("height", ""), "height")
            reach = _measurement(info.get("reach", ""), "reach")
            if pd.isna(reach):
                reach = height
            dob = pd.to_datetime(info.get("dob", ""), errors="coerce")
            stance = info.get("stance")
            results[name] = {
                "url": url,
                "html_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "Height_cms": height,
                "Reach_cms": reach,
                "Stance": None if stance in ["", None, "--"] else stance,
                "DOB": None if pd.isna(dob) else str(dob.date()),
            }
        except Exception as e:
            results[name] = {"error": str(e), "url": url}
        (OUT / "completed_profile_attributes.json").write_text(json.dumps(results, indent=2))
        print(len(results), name, results[name].get("error", "verified"), flush=True)

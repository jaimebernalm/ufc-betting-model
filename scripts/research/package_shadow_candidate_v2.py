"""Package candidate inputs without refitting or modifying previous research artifacts."""

import hashlib
import json
import shutil

import pandas as pd

from ufc_pred.ingest.candidate_state import TOTALS
from ufc_pred.ingest.identity import normalise
from ufc_pred.ingest.strict_history import RawHistoryStateSource
from ufc_pred.paths import ROOT

OUT = ROOT / "artifacts/shadow_candidate_2026_09_14_v2"
ABL = ROOT / "artifacts/ablation_2026_09_14"
OLD = ROOT / "artifacts/corrected_2026_09_14"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if (OUT / "manifest.json").exists():
        raise RuntimeError("Release already frozen; create another version")
    OUT.mkdir(exist_ok=True)
    source = RawHistoryStateSource(ROOT / "data/raw/ufcstats_export")
    for name, extra in json.loads((ABL / "completed_profile_attributes.json").read_text()).items():
        if "error" in extra:
            continue
        profile = source.profiles[normalise(name)]
        for attr in ["Height_cms", "Reach_cms", "Stance", "DOB"]:
            if pd.isna(profile.get(attr)) and pd.notna(extra.get(attr)):
                profile[attr] = extra[attr]
    external = json.loads((ABL / "prior_external_histories.json").read_text())
    entries = list(source.entries)
    for name, record in external.items():
        for row in record.get("extra", []):
            entries.append({**dict.fromkeys(TOTALS, 0), **row, "key": normalise(name)})
    contract = {
        "version": "shadow-candidate-v2-2026-09-14",
        "last_ufc_event": str(source.last_event.date()),
        "profiles": source.profiles,
        "entries": entries,
        "external_cohort": sorted(external),
        "policy": {
            "rates": "Frozen UFC bout totals plus newly completed UFC bouts strictly before target day",
            "counters": "Same cumulative fold; add frozen audited external histories with zero rate totals",
            "profiles": "Frozen raw profiles, only missing height/reach/stance/DOB completed from verified sources",
            "future_external_bouts": "Not admitted in this version",
            "unknown_identity": "Capture error and skip; requires a reviewed input version",
            "future_ufc_bouts": "Official identity and complete totals required; archive source HTML",
        },
        "provenance": {
            str(p.relative_to(ROOT)): sha(p)
            for p in [
                *sorted((ROOT / "data/raw/ufcstats_export").glob("*.csv")),
                ROOT / "data/raw/ufcstats_verified_profiles/profiles.json",
                ABL / "completed_profile_attributes.json",
                ABL / "prior_external_histories.json",
            ]
        },
    }
    (OUT / "candidate_inputs.json").write_text(json.dumps(contract, default=str, indent=2))
    shutil.copytree(ABL / "candidate/models", OUT / "models", dirs_exist_ok=True)
    for src, name in [
        (ABL / "profiles_plus_prior_external_counters.parquet", "training_features.parquet"),
        (OLD / "verified_history.parquet", "verified_history.parquet"),
        (ABL / "completed_profile_attributes.json", "completed_profile_attributes.json"),
        (ABL / "prior_external_histories.json", "prior_external_histories.json"),
        (ABL / "candidate/manifest.json", "research_candidate_manifest.json"),
    ]:
        shutil.copy2(src, OUT / name)
    print("Packaged", len(source.profiles), "profiles and", len(entries), "fighter-bout records")


if __name__ == "__main__":
    main()

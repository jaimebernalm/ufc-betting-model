"""Validate deployed candidate inputs against frozen training and archived future observations."""

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from evaluate_corrected_system import predictions
from package_shadow_candidate_v2 import ABL, OUT

from ufc_pred.inference.upcoming_builder import _DIFF_PAIRS, build_upcoming_row
from ufc_pred.ingest.candidate_state import CandidateStateSource
from ufc_pred.ingest.identity import normalise
from ufc_pred.ingest.rankings_attach import load_rankings
from ufc_pred.ingest.strict_history import RawHistoryStateSource, apply_state_features
from ufc_pred.ingest.ufcstats_state import UFCStatsStateSource
from ufc_pred.paths import ROOT


def get(url):
    return (ABL / "official_checks" / (hashlib.sha256(url.encode()).hexdigest() + ".html")).read_text()


def compare(expected, actual):
    mismatches = {}
    for c in expected.keys():
        a, b = expected[c], actual[c]
        if not ((pd.isna(a) and pd.isna(b)) or a == b):
            mismatches[c] = [str(a), str(b)]
    return mismatches


def main():
    source = CandidateStateSource(OUT / "candidate_inputs.json")
    expected = pd.read_parquet(OUT / "training_features.parquet")
    ranks = load_rankings()
    actual = apply_state_features(expected, source, ranks)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
    print("All", len(expected), "historical rows reproduce exactly", flush=True)
    # Independent live row assembly includes derived fields/ranks and no-debut rules.
    fields = [
        s + "_" + c for s in ["R", "B"] for c in source(expected.iloc[-1].R_fighter, expected.iloc[-1].date)
    ]
    fields += [d for d, b in _DIFF_PAIRS]
    fields += [c for c in expected if c.endswith("_rank") or c == "better_rank"]
    count = 0
    for _, row in expected.iloc[::67].iterrows():
        prior = expected[expected.date < row.date]
        known = set(prior.R_fighter) | set(prior.B_fighter)
        if not {row.R_fighter, row.B_fighter} <= known:
            continue
        built = build_upcoming_row(
            row.R_fighter,
            row.B_fighter,
            row.date,
            row.weight_class,
            expected,
            gender=row.gender,
            title_bout=bool(row.title_bout),
            no_of_rounds=int(row.no_of_rounds),
            rankings=ranks,
            state_source=source,
        ).iloc[0]
        diff = compare(row[fields], built)
        assert not diff, (row.R_fighter, diff)
        count += 1
    print("Live assembly samples:", count, flush=True)
    pred = predictions(actual[actual.date >= pd.Timestamp("2025-11-30")], OUT / "models")
    reference = np.load(ABL / "all_variant_predictions.npz")["profiles_plus_prior_external_counters"]
    error = max(float(np.max(abs(pred[k] - reference[i]))) for i, k in enumerate(["real", "corrupted"]))
    assert error < 1e-12
    print("Ensemble prediction max error:", error, flush=True)
    raw = RawHistoryStateSource(ROOT / "data/raw/ufcstats_export")
    doctor, prospective = [], []
    for name, date in [
        ("Jean Silva", "2026-01-24"),
        ("Curtis Blaydes", "2026-09-12"),
        ("Arnold Allen", "2026-01-24"),
    ]:
        live = UFCStatsStateSource(html_getter=get)
        live.fighter_url = lambda n: raw.profiles[normalise(n)]["URL"]
        diff = compare(raw(name, date), live(name, pd.Timestamp(date)))
        assert not diff, (name, diff)
        doctor.append({"fighter": name, "date": date, "mismatches": diff})
        # Remove the recent UFC bouts from the baseline. Production appending
        # must reconstruct the full frozen answer using only archived web pages.
        contract = json.loads((OUT / "candidate_inputs.json").read_text())
        cutoff = pd.Timestamp("2023-12-31")
        key = normalise(name)
        removed = [
            r
            for r in contract["entries"]
            if r["key"] == key and pd.Timestamp(r["date"]) > cutoff and "event" not in r
        ]
        contract["entries"] = [
            r
            for r in contract["entries"]
            if not (r["key"] == key and pd.Timestamp(r["date"]) > cutoff and "event" not in r)
        ]
        contract["last_ufc_event"] = str(cutoff.date())
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "contract.json"
            p.write_text(json.dumps(contract))
            serving = CandidateStateSource(p, live=True, html_getter=get, capture_dir=Path(td) / "sources")
            got = serving(name, pd.Timestamp(date))
            diff = compare(source(name, date), got)
            assert not diff, (name, diff)
            archived = list((Path(td) / "sources").glob("*.html"))
            assert archived
            serving.close()
        prospective.append(
            {
                "fighter": name,
                "date": date,
                "removed_baseline_records": len(removed),
                "prior_bouts_to_rebuild": sum(pd.Timestamp(r["date"]) < pd.Timestamp(date) for r in removed),
                "mismatches": diff,
                "archived_responses": len(archived),
            }
        )
    report = {
        "historical_rows": len(expected),
        "historical_feature_cells": int(expected.size),
        "historical_mismatches": 0,
        "live_builder_samples": count,
        "prediction_eval_rows": len(reference[0]),
        "prediction_max_absolute_error": error,
        "doctor_parser_official_fixtures": doctor,
        "prospective_append_official_fixtures": prospective,
    }
    if (OUT / "manifest.json").exists():
        assert report == json.loads((OUT / "validation.json").read_text())
    else:
        (OUT / "validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

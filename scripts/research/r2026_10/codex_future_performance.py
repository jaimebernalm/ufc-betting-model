"""Export frozen B1/B2 features for scheduled fights from an explicit as-of day.

Example (research checkout only):
  PYTHONPATH=src python scripts/research/r2026_10/codex_future_performance.py \
    --targets /path/to/scheduled_fights.parquet --as-of 2026-10-13 --run-id paper_20261013

Targets contain date, R_fighter, B_fighter and weight_class. No outcomes or odds
are read. Source history is an existing three-CSV UFCStats snapshot; this command
does not fetch data, refresh the frozen archive, or change live state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from ufc_pred.paths import ROOT
from ufc_pred.research.perf_adjusted import FEATURE_COLUMNS as B2_COLUMNS
from ufc_pred.research.perf_features import FEATURE_COLUMNS as B1_COLUMNS
from ufc_pred.research.perf_features import KEY_COLUMNS, load_bouts
from ufc_pred.research.perf_future import build_future_performance

TARGET_FIELDS = [*KEY_COLUMNS, "weight_class"]
SOURCE_FILES = ("ufc_event_details.csv", "ufc_fight_results.csv", "ufc_fight_stats.csv")
OUTPUT_ROOT = ROOT / "artifacts/r2026_10/codex/future"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_targets(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        targets = pd.read_parquet(path, columns=TARGET_FIELDS)
    elif path.suffix.lower() == ".csv":
        targets = pd.read_csv(path, usecols=TARGET_FIELDS)
    else:
        raise ValueError("Targets must be a CSV or Parquet file")
    # Parquet dates stay as supplied. CSV dates are parsed after reading only the
    # four scheduled-fight fields; the core validates midnight and timezone.
    if path.suffix.lower() == ".csv":
        if not targets.date.astype(str).str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
            raise ValueError("CSV dates must use calendar YYYY-MM-DD, without times")
        targets["date"] = pd.to_datetime(targets.date, format="%Y-%m-%d", errors="raise")
    if targets.empty:
        raise ValueError("No scheduled fights supplied")
    return targets[TARGET_FIELDS]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--as-of", required=True, help="Exclusive history cutoff, YYYY-MM-DD")
    parser.add_argument("--run-id", required=True, help="New immutable artifact directory name")
    parser.add_argument("--source-dir", type=Path, default=ROOT / "data/raw/ufcstats_export")
    parser.add_argument("--b1-only", action="store_true")
    args = parser.parse_args()
    if ROOT.name != "ufc-pred-research":
        raise RuntimeError("Future performance export is restricted to the research checkout")
    branch = subprocess.check_output(["git", "-C", str(ROOT), "branch", "--show-current"], text=True).strip()
    if branch != "research/robustez-modelo-2026-10-13":
        raise RuntimeError(f"Unexpected research branch: {branch}")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.run_id):
        raise ValueError("run-id must contain 1-64 letters, digits, underscores or hyphens")
    destination = OUTPUT_ROOT / args.run_id
    if OUTPUT_ROOT.is_symlink() or not destination.resolve().is_relative_to(ROOT.resolve()):
        raise RuntimeError("Future artifacts must remain inside the research checkout")
    if destination.exists():
        raise FileExistsError("This run-id already exists; choose a new one to preserve its snapshot")
    inputs = [args.targets, *(args.source_dir / name for name in SOURCE_FILES)]
    sources = {str(path.resolve()): sha256(path) for path in inputs}
    code = {
        str(path.relative_to(ROOT)): sha256(path)
        for path in (
            Path(__file__),
            ROOT / "src/ufc_pred/research/perf_future.py",
            ROOT / "src/ufc_pred/research/perf_features.py",
            ROOT / "src/ufc_pred/research/perf_adjusted.py",
            ROOT / "src/ufc_pred/ingest/identity.py",
            ROOT / "configs/fighter_aliases.json",
        )
    }
    targets = read_targets(args.targets)
    observations, exclusions = load_bouts(args.source_dir)
    frame, diagnostics = build_future_performance(
        observations, targets, args.as_of, include_adjusted=not args.b1_only,
    )
    expected = [*KEY_COLUMNS, *B1_COLUMNS, *([] if args.b1_only else B2_COLUMNS)]
    if list(frame.columns) != expected:
        raise AssertionError("Future feature columns differ from the frozen B1/B2 schema")
    pd.testing.assert_frame_equal(frame[list(KEY_COLUMNS)], targets[list(KEY_COLUMNS)])
    if not np.isfinite(frame.drop(columns=list(KEY_COLUMNS)).to_numpy(dtype=float)).all():
        raise AssertionError("Nonfinite future feature values")
    # A changing data snapshot cannot be bound to a trustworthy provenance hash.
    if sources != {str(path.resolve()): sha256(path) for path in inputs}:
        raise RuntimeError("An input changed during export; retry with an immutable source snapshot")
    if any(sha256(ROOT / path) != digest for path, digest in code.items()):
        raise RuntimeError("Feature code changed during export; retry after the recipe is frozen")
    destination.mkdir(parents=True, exist_ok=False)
    feature_path = destination / "performance.parquet"
    exclusion_path = destination / "source_exclusions.parquet"
    target_path = destination / "scheduled_targets.parquet"
    frame.to_parquet(feature_path, index=False)
    exclusions.to_parquet(exclusion_path, index=False)
    targets.to_parquet(target_path, index=False)
    manifest = {
        "status": "complete_research_only",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "branch": branch,
        "as_of": args.as_of,
        "target_fields_read": TARGET_FIELDS,
        "target_labels_read": False,
        "target_rows": len(targets),
        "first_target_date": str(targets.date.min()),
        "last_target_date": str(targets.date.max()),
        "include_adjusted": not args.b1_only,
        "source_hashes": sources,
        "code_hashes": code,
        "runtime": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__,
                    "sklearn": sklearn.__version__},
        "feature_columns": expected[len(KEY_COLUMNS):],
        "diagnostics": diagnostics,
        "source_exclusions_by_reason": exclusions.reason.value_counts().to_dict(),
        "artifacts": {p.name: sha256(p) for p in (feature_path, exclusion_path, target_path)},
        "limitations": [
            "As-of filtering of a retrospective archive does not establish original publication availability.",
            "Check source freshness and align the base-model history before interpreting future predictions.",
            "No model training, prediction, stake selection, orders or changes to the existing watchdog occur.",
        ],
    }
    # Written last: consumers must require this completed manifest before use.
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print(json.dumps({"features": str(feature_path), "manifest": str(manifest_path),
                      "rows": len(frame), "features_count": len(expected) - len(KEY_COLUMNS)}))


if __name__ == "__main__":
    main()

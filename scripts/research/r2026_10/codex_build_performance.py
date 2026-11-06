"""Build research-only prior-performance features without reading target labels.

Run from the research worktree with PYTHONPATH=src and a separate/read-only
Python runtime. This command never modifies the frozen candidate or its caches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ufc_pred.paths import ROOT
from ufc_pred.research.perf_features import (
    FEATURE_COLUMNS,
    KEY_COLUMNS,
    PerformanceRecipe,
    build_smoothed_features,
    load_bouts,
)

SOURCE = ROOT / "data/raw/ufcstats_export"
TARGET = ROOT / "artifacts/shadow_candidate_2026_09_14_v2/training_features.parquet"
OUTPUT = ROOT / "artifacts/r2026_10/codex"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    paths = [
        TARGET,
        SOURCE / "ufc_event_details.csv",
        SOURCE / "ufc_fight_results.csv",
        SOURCE / "ufc_fight_stats.csv",
        ROOT / "configs/fighter_aliases.json",
        ROOT / "src/ufc_pred/ingest/identity.py",
        ROOT / "src/ufc_pred/research/perf_features.py",
        Path(__file__),
    ]
    adjusted = ROOT / "src/ufc_pred/research/perf_adjusted.py"
    if adjusted.exists():
        paths.append(adjusted)
    return {str(p.relative_to(ROOT)): sha256(p) for p in paths}


def prepare_destination(path: Path, overwrite: bool) -> None:
    if not path.resolve().is_relative_to(OUTPUT.resolve()):
        raise ValueError("Output must stay inside the Codex research artifact directory")
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Refusing to replace {path}; use --overwrite to preserve and supersede it")
        previous = path.with_name(f"{path.stem}.previous-{sha256(path)[:16]}{path.suffix}")
        if not previous.exists():
            path.rename(previous)


def save_parquet(frame: pd.DataFrame, path: Path, overwrite: bool) -> None:
    prepare_destination(path, overwrite)
    temporary = path.with_suffix(".building.parquet")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def validate_features(frame: pd.DataFrame, targets: pd.DataFrame, expected: tuple | None = None) -> dict:
    keys = list(KEY_COLUMNS)
    pd.testing.assert_frame_equal(frame[keys].reset_index(drop=True), targets[keys].reset_index(drop=True))
    if frame.duplicated(keys).any():
        raise ValueError("Duplicate target fight keys")
    features = [c for c in frame if c not in keys]
    if expected is not None and features != list(expected):
        raise ValueError("Feature schema differs from the frozen recipe")
    if not features or not np.isfinite(frame[features].to_numpy(dtype=float)).all():
        raise ValueError("Feature matrix must be nonempty and finite")
    if any(
        not (c.startswith(("R_perf_", "B_perf_")) or c.startswith("perf_") and c.endswith("_dif"))
        for c in features
    ):
        raise ValueError("Unexpected feature column; metadata must stay out of the model matrix")
    return {
        "rows": len(frame),
        "feature_columns": features,
        "keys_exact_and_order_preserved": True,
        "first_target_date": str(frame.date.min()),
        "last_target_date": str(frame.date.max()),
        "finite": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("b1", "b2", "both", "benchmark-b2"), default="b1")
    parser.add_argument(
        "--overwrite", action="store_true", help="Preserve old artifacts by content hash before replacing"
    )
    args = parser.parse_args()
    if ROOT.name != "ufc-pred-research":
        raise RuntimeError("This research builder must run against the isolated research checkout")
    branch = subprocess.check_output(["git", "-C", str(ROOT), "branch", "--show-current"], text=True).strip()
    if branch != "research/robustez-modelo-2026-10-13":
        raise RuntimeError(f"Unexpected research branch: {branch}")
    if OUTPUT.is_symlink() or not OUTPUT.resolve().is_relative_to(ROOT.resolve()):
        raise RuntimeError("Research artifacts must not resolve outside the research checkout")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    # Labels, odds, and existing model predictions are deliberately not loaded.
    targets = pd.read_parquet(TARGET, columns=[*KEY_COLUMNS, "weight_class"])
    observations, exclusions = load_bouts(SOURCE)
    print(
        f"Parsed {observations.bout_id.nunique()} bouts / {len(observations)} fighter observations; {len(exclusions)} exclusions",
        flush=True,
    )
    hashes = source_hashes()
    recipe = PerformanceRecipe()
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "mode": args.mode,
        "status": "research_only_not_deployment_ready",
        "branch": branch,
        "source_hashes": hashes,
        "target_fields_read": [*KEY_COLUMNS, "weight_class"],
        "target_labels_read": False,
        "temporal_policy": "B1 strictly prior calendar days; B2 strictly prior target-month start",
        "b1_recipe": asdict(recipe),
        "n_observations": len(observations),
        "n_bouts": int(observations.bout_id.nunique()),
        "exclusions_by_reason": exclusions.reason.value_counts().to_dict(),
        "source_first_date": str(observations.date.min()),
        "source_last_date": str(observations.date.max()),
        "artifacts": {},
        "limitations": [
            "No new model training, outcome scoring, or claim of profitability is performed by this export.",
            "Raw bout statistics are retrospective records; their original publication times are not archived.",
            "Future-use parity requires the same paired-statistics schema; the frozen live parser is not changed.",
            "Ambiguous joins and invalid/missing bout statistics are excluded and recorded, never changed to zeros.",
        ],
    }
    if args.mode in ("b1", "both"):
        frame = build_smoothed_features(observations, targets, recipe)
        info = validate_features(frame, targets, FEATURE_COLUMNS)
        output = OUTPUT / "perf_features.parquet"
        save_parquet(frame, output, args.overwrite)
        info["sha256"] = sha256(output)
        info["missing_by_year"] = (
            frame.assign(year=frame.date.dt.year)
            .groupby("year")[["R_perf_missing", "B_perf_missing"]]
            .sum()
            .to_dict()
        )
        manifest["artifacts"][output.name] = info
        save_parquet(exclusions, OUTPUT / "perf_exclusions.parquet", args.overwrite)
        print(f"B1 ready: {len(frame)} rows, {len(FEATURE_COLUMNS)} numeric features", flush=True)
    if args.mode in ("b2", "both", "benchmark-b2"):
        from ufc_pred.research.perf_adjusted import build_adjusted_features, fit_adjusted_state

        if args.mode == "benchmark-b2":
            diagnostics = []
            for month in ("2016-01-01", "2020-01-01", "2023-01-01"):
                cutoff = pd.Timestamp(month)
                tick = time.monotonic()
                state = fit_adjusted_state(observations[observations.date < cutoff], cutoff)
                diagnostics.append(
                    {
                        "month": month,
                        "wall_seconds": time.monotonic() - tick,
                        "diagnostics": state.diagnostics,
                    }
                )
                print(f"B2 benchmark {month}: {diagnostics[-1]['wall_seconds']:.2f}s", flush=True)
            manifest["benchmark"] = diagnostics
        else:
            frame, diagnostics = build_adjusted_features(observations, targets, return_diagnostics=True)
            info = validate_features(frame, targets)
            output = OUTPUT / "perf_adjusted.parquet"
            save_parquet(frame, output, args.overwrite)
            info["sha256"] = sha256(output)
            manifest["artifacts"][output.name] = info
            save_parquet(diagnostics, OUTPUT / "perf_adjusted_fits.parquet", args.overwrite)
            print(
                f"B2 ready: {len(frame)} rows, {len(frame.columns) - len(KEY_COLUMNS)} numeric features",
                flush=True,
            )
    manifest["wall_seconds"] = time.monotonic() - start
    manifest_path = OUTPUT / f"perf_{args.mode}_manifest.json"
    prepare_destination(manifest_path, args.overwrite)
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print(f"Manifest: {manifest_path}; total {manifest['wall_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()

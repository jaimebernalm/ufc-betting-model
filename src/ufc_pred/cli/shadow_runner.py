"""Frozen corrected ensemble in shadow mode: captures only, no orders or alerts.

Uses the existing event-triggered schedule, an independent history/idempotency
namespace, and immutable decision records. Future results are evaluated later;
no strategy parameter is adapted after a win or loss.
"""

import argparse
import fcntl
import hashlib
import json
import traceback
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from ufc_pred.paths import PROCESSED, ROOT

BUNDLE = ROOT / "artifacts/shadow_candidate_2026_09_14_v2"
SHADOW = PROCESSED / "shadow_candidate_2026_09_14_v2"


def verify_bundle():
    manifest = json.loads((BUNDLE / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        # Mutable external data have their training vintage in the manifest;
        # only model/training artifacts and frozen policy must stay unchanged.
        if manifest.get("input_constructor") == "candidate_v2" or name.startswith(
            (str(BUNDLE.relative_to(ROOT)) + "/", "configs/")
        ):
            actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            if actual != digest:
                raise ValueError(f"Frozen bundle changed: {name}")
    runtime = json.loads((BUNDLE / "runtime_manifest.json").read_text())
    for name, digest in runtime.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Frozen shadow runtime changed: {name}; create a new version explicitly")
    return manifest


def refresh_history():
    from ufc_pred.ingest.identity import canonicalize_frame
    from ufc_pred.ingest.rankings_attach import capture_current_rankings
    from ufc_pred.ingest.ufcstats_update import update_master

    today = str(datetime.now(UTC).date())
    marker = SHADOW / "data_refresh.json"
    csv = SHADOW / "history.csv"
    if marker.exists() and json.loads(marker.read_text()).get("date") == today:
        return
    if not csv.exists():
        pd.read_parquet(BUNDLE / "verified_history.parquet").to_csv(csv, index=False)
    summary = update_master(
        csv_path=csv,
        completed_before=datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None),
    )
    capture_current_rankings()
    history = canonicalize_frame(pd.read_csv(csv, parse_dates=["date"]))
    history.to_parquet(SHADOW / "history.parquet", index=False)
    marker.write_text(
        json.dumps(
            {"date": today, "summary": summary, "history_last_date": str(history.date.max().date())}, indent=2
        )
    )


def main():
    global BUNDLE, SHADOW
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--once",
        action="store_true",
        help="Capture all current-card fights; default preserves watchdog timing",
    )
    parser.add_argument(
        "--check", action="store_true", help="Verify frozen artifacts only, without network or captures"
    )
    parser.add_argument("--bundle", type=Path, help="Explicit frozen bundle path")
    parser.add_argument("--shadow-dir", type=Path, help="Independent capture directory")
    args = parser.parse_args()
    if bool(args.bundle) != bool(args.shadow_dir):
        parser.error("--bundle and --shadow-dir must be supplied together to isolate versions")
    if args.bundle:
        BUNDLE = args.bundle.resolve()
    if args.shadow_dir:
        SHADOW = args.shadow_dir.resolve()
    SHADOW.mkdir(parents=True, exist_ok=True)
    with (SHADOW / "runner.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0
        now = datetime.now(UTC).isoformat()
        try:
            verify_bundle()
            if args.check:
                print("Frozen shadow bundle verified")
                return 0
            refresh_history()
            from ufc_pred.cli import bet_runner as runner
            from ufc_pred.ingest.kalshi_client import KalshiClient
            from ufc_pred.ops.shadow_evaluation import project

            protocol = json.loads((BUNDLE / "shadow_protocol.json").read_text())
            project(SHADOW, protocol, KalshiClient())

            class PaperClient(KalshiClient):
                def get_balance(self):
                    ledger = project(SHADOW, protocol)
                    amount = max(0.0, sum(ledger["cash"].values()))
                    return {"balance_dollars": str(amount), "balance": amount * 100}

            from ufc_pred.ingest.candidate_state import state_source_for_models

            sources = []

            def source_factory():
                source = state_source_for_models(BUNDLE / "models", capture_dir=SHADOW / "sources")
                sources.append(source)
                return source

            runner.KalshiClient = PaperClient
            runner.FIGHTS_PATH = SHADOW / "history.parquet"
            runner.NOTIF_DIR = SHADOW / "captures"
            runner.IDEMPOTENCY_PATH = SHADOW / "idempotency.json"
            runner.TICKER_CACHE_PATH = SHADOW / "ticker_cache.json"
            runner.BANKROLLS_PATH = SHADOW / "bankroll_reference.json"
            # This is a reference for capture sizing, never a real-money ledger.
            # Prospective growth is evaluated using explicit execution scenarios.
            if not runner.BANKROLLS_PATH.exists():
                anchor = json.loads((BUNDLE / "shadow_protocol.json").read_text())["reference_bankrolls"]
                runner.BANKROLLS_PATH.write_text(json.dumps(anchor, indent=2))
            runner.notify = lambda *a, **kw: None

            def schedule_record(date, event, schedule):
                from dataclasses import asdict

                directory = SHADOW / "schedules" / str(pd.Timestamp(date).date())
                directory.mkdir(parents=True, exist_ok=True)
                path = directory / (datetime.now(UTC).strftime("%H%M%S%f") + ".json")
                path.write_text(
                    json.dumps(
                        {"event": event, "fights": [asdict(f) for f in schedule]}, default=str, indent=2
                    )
                )
                return path

            runner.write_audit_log = schedule_record
            original = runner.write_record

            def record(date, key, payload):
                payload["bundle_sha256"] = hashlib.sha256((BUNDLE / "manifest.json").read_bytes()).hexdigest()
                payload["history_sha256"] = hashlib.sha256(
                    (SHADOW / "history.parquet").read_bytes()
                ).hexdigest()
                payload["decision_recorded_at_utc"] = datetime.now(UTC).isoformat()
                payload["input_observations"] = {
                    url: digest
                    for source in sources
                    for url, digest in getattr(source, "observations", {}).items()
                }
                payload["execution"] = "shadow observation only; no actual fill implied"
                return original(date, key, payload)

            runner.write_record = record
            rc = runner.run(
                SimpleNamespace(
                    once=args.once,
                    watchdog=not args.once,
                    dry_run=True,
                    shadow=True,
                    models_dir=BUNDLE / "models",
                    state_source_factory=source_factory,
                    time_offset_min=0,
                    verbose=False,
                )
            )
            status = {"at_utc": now, "status": "ok" if rc == 0 else "blocked", "exit_code": rc}
        except Exception as exc:
            status = {"at_utc": now, "status": "error", "error": str(exc)}
            traceback.print_exc()
            rc = 1
        (SHADOW / "status.json").write_text(json.dumps(status, indent=2))
        return rc


if __name__ == "__main__":
    raise SystemExit(main())

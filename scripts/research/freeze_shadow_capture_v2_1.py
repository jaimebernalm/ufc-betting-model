"""Freeze capture runtime v2.1: same v2 models, new capture code; no trading actions.

v2 is retired as a decision model, but its runtime is still the capture pipeline
the market-anchored paper logger reads. The frozen v2 runtime blocks capture on
any code change, so the capture fixes ship as an explicit new version:
  * a corner whose Kalshi name does not resolve falls back to UFC.com's spelling
  * in shadow mode every fight is captured at T-60 of its scheduled start

Writes the new bundle, seeds the new shadow directory with the v2 history and
prepares both LaunchAgents (capture watchdog and anchored-shadow logger). It
does not install them.
"""

import hashlib
import json
import platform
import plistlib
import shutil
from datetime import UTC, datetime
from importlib.metadata import distributions
from pathlib import Path

from ufc_pred.paths import PROCESSED, ROOT

OLD = ROOT / "artifacts/shadow_candidate_2026_09_14_v2"
OUT = ROOT / "artifacts/shadow_capture_2026_11_05_v2_1"
OLD_SHADOW = PROCESSED / "shadow_candidate_2026_09_14_v2"
SHADOW = PROCESSED / "shadow_capture_2026_11_05_v2_1"
LOGGER_OUT = PROCESSED / "anchored_shadow_2026_10_13"
# Regenerated below; everything else in the v2 bundle is copied byte for byte.
REGENERATED = {"manifest.json", "runtime_manifest.json", "shadow_protocol.json", "environment.json",
               "com.ufcbet.watchdog.plist"}
REGENERATED_DIRS = {"runtime_snapshot", "reproduction"}
CONFIGS = ["configs/inference.json", "configs/fighter_aliases.json", "configs/fight_result_corrections.json"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if (OUT / "manifest.json").exists():
        raise RuntimeError("Already frozen; create another version")
    old = json.loads((OLD / "manifest.json").read_text())
    for name, digest in old["files"].items():
        if name.startswith(str(OLD.relative_to(ROOT)) + "/"):
            assert sha(ROOT / name) == digest, f"v2 bundle modified: {name}"
    for path in sorted(OLD.rglob("*")):
        rel = path.relative_to(OLD)
        if path.is_dir() or rel.parts[0] in REGENERATED_DIRS or str(rel) in REGENERATED:
            continue
        (OUT / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, OUT / rel)
    for model in (OLD / "models").glob("*.joblib"):
        assert sha(model) == sha(OUT / "models" / model.name)

    now = datetime.now(UTC).isoformat()
    protocol = json.loads((OLD / "shadow_protocol.json").read_text())
    protocol.update(
        version="2026-11-05-capture-v2-1",
        frozen_at_utc=now,
        previous_bundle=str(OLD.relative_to(ROOT)),
        decision_model=(
            "Legacy: v2 recommendations and paper ledger are kept for continuity only. "
            "The candidate under study is the market-anchored paper logger "
            "(ufc_pred.research.anchored_shadow), which reads these captures."
        ),
        capture_changes=[
            "Kalshi name that does not resolve falls back to the UFC.com name of the same corner",
            "Shadow mode captures every fight at T-60 of its scheduled start (scheduled_lead)",
        ],
        promotion="Shadow observation only; promotion to live requires a separate decision",
    )
    (OUT / "shadow_protocol.json").write_text(json.dumps(protocol, indent=2))

    config_path = ROOT / "configs/inference.json"
    config = json.loads(config_path.read_text())
    config.update(model_bundle=str((OUT / "models").relative_to(ROOT)), mode="shadow")
    config_path.write_text(json.dumps(config, indent=2) + "\n")

    for rel in [
        "scripts/research/freeze_shadow_capture_v2_1.py",
        "tests/test_capture_v2_1.py",
        "docs/CAPTURA_V2_1_2026_11_05.md",
    ]:
        copy = OUT / "reproduction" / rel
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_bytes((ROOT / rel).read_bytes())
    (OUT / "environment.json").write_text(
        json.dumps(
            {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "packages": sorted((d.metadata["Name"], d.version) for d in distributions()),
            },
            indent=2,
        )
    )
    # Research code is not imported by the capture runtime and must stay editable
    # without blocking capture, so it is left out of the frozen runtime.
    runtime = {}
    for path in sorted((ROOT / "src/ufc_pred").rglob("*.py")):
        rel = path.relative_to(ROOT)
        if rel.parts[2] == "research":
            continue
        runtime[str(rel)] = sha(path)
        copy = OUT / "runtime_snapshot" / rel
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_bytes(path.read_bytes())
    (OUT / "runtime_manifest.json").write_text(json.dumps(runtime, indent=2))

    python = str(ROOT / ".conda/bin/python")
    watchdog = plistlib.loads((OLD / "com.ufcbet.watchdog.plist").read_bytes())
    watchdog["ProgramArguments"] = [python, "-m", "ufc_pred.cli.shadow_runner",
                                    "--bundle", str(OUT), "--shadow-dir", str(SHADOW)]
    watchdog["StandardOutPath"] = watchdog["StandardErrorPath"] = str(SHADOW / "runner.log")
    (OUT / "com.ufcbet.watchdog.plist").write_bytes(plistlib.dumps(watchdog))
    logger = {
        "Label": "com.ufcbet.anchored-shadow",
        "ProgramArguments": [python, "-m", "ufc_pred.research.anchored_shadow", "run",
                             "--captures", str(SHADOW / "captures")],
        "EnvironmentVariables": {"PYTHONPATH": str(ROOT / "src")},
        "WorkingDirectory": str(ROOT),
        "StartInterval": 60,
        "RunAtLoad": True,
        "StandardOutPath": str(LOGGER_OUT / "runner.log"),
        "StandardErrorPath": str(LOGGER_OUT / "runner.log"),
    }
    (OUT / "com.ufcbet.anchored-shadow.plist").write_bytes(plistlib.dumps(logger))

    manifest = {
        **{k: old[k] for k in ["policy", "input_contract"]},
        "version": "shadow-capture-v2-1-2026-11-05",
        "frozen_at_utc": now,
        "deployment_ready": True,
        "mode": "shadow",
        "live_trading_enabled": False,
        "input_constructor": "candidate_v2",
        "pending_for_shadow": [],
        "previous_bundle": str(OLD.relative_to(ROOT)),
        "model_binaries": "Byte-identical copies of the v2 bundle models",
        "files": {str(p.relative_to(ROOT)): sha(p) for p in sorted(OUT.rglob("*")) if p.is_file()},
    }
    for rel in CONFIGS:
        manifest["files"][rel] = sha(ROOT / rel)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))

    SHADOW.mkdir(parents=True, exist_ok=True)
    if not (SHADOW / "history.csv").exists():
        shutil.copy2(OLD_SHADOW / "history.csv", SHADOW / "history.csv")
    print("Frozen", len(manifest["files"]), "artifacts and", len(runtime), "runtime modules at", now)
    print("LaunchAgents prepared in", OUT)


if __name__ == "__main__":
    main()

"""Freeze the validated candidate as a distinct shadow release; no trading actions."""

import json
import platform
import plistlib
from datetime import UTC, datetime
from importlib.metadata import distributions
from pathlib import Path

from package_shadow_candidate_v2 import ABL, OLD, OUT, sha

from ufc_pred.paths import ROOT


def main():
    if (OUT / "manifest.json").exists():
        raise RuntimeError("Already frozen; create another version")
    validation = json.loads((OUT / "validation.json").read_text())
    assert validation["historical_rows"] == 7425 and validation["historical_mismatches"] == 0
    assert validation["prediction_max_absolute_error"] == 0
    assert all(not row["mismatches"] for row in validation["prospective_append_official_fixtures"])
    reference = json.loads((ABL / "candidate/manifest.json").read_text())
    for file, digest in reference["files"].items():
        assert sha(ROOT / file) == digest, f"Research candidate modified: {file}"
    for model in (OUT / "models").glob("*.joblib"):
        assert sha(model) == sha(ABL / "candidate/models" / model.name)
    now = datetime.now(UTC).isoformat()
    protocol = json.loads((OLD / "shadow_protocol.json").read_text())
    protocol.update(
        version="2026-09-14-candidate-v2",
        frozen_at_utc=now,
        previous_bundle=str(OLD.relative_to(ROOT)),
        input_policy=json.loads((OUT / "candidate_inputs.json").read_text())["policy"],
        promotion="Shadow observation only; promotion to live requires a separate decision",
        review_rule="Review after BOTH at least 12 completed cards and 150 resolved eligible fights; no retuning between results",
    )
    (OUT / "shadow_protocol.json").write_text(json.dumps(protocol, indent=2))
    config_path = ROOT / "configs/inference.json"
    config = json.loads(config_path.read_text())
    config.update(model_bundle=str((OUT / "models").relative_to(ROOT)), mode="shadow")
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    shadow = ROOT / "data/processed/shadow_candidate_2026_09_14_v2"
    shadow.mkdir(exist_ok=True)
    plist_path = Path.home() / "Library/LaunchAgents/com.ufcbet.watchdog.plist"
    plist = plistlib.loads(plist_path.read_bytes())
    plist["ProgramArguments"] = [
        str(ROOT / ".conda/bin/python"),
        "-m",
        "ufc_pred.cli.shadow_runner",
        "--bundle",
        str(OUT),
        "--shadow-dir",
        str(shadow),
    ]
    plist["StandardOutPath"] = plist["StandardErrorPath"] = str(shadow / "runner.log")
    (OUT / "com.ufcbet.watchdog.plist").write_bytes(plistlib.dumps(plist))
    for path in [
        ROOT / "scripts/research/package_shadow_candidate_v2.py",
        ROOT / "scripts/research/validate_shadow_candidate_v2.py",
        ROOT / "scripts/research/freeze_shadow_candidate_v2.py",
        ROOT / "tests/test_candidate_state.py",
        ROOT / "docs/ACTIVACION_SOMBRA_CANDIDATA_V2_2026_09_14.md",
    ]:
        copy = OUT / "reproduction" / path.relative_to(ROOT)
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_bytes(path.read_bytes())
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
    runtime = {}
    for path in sorted((ROOT / "src/ufc_pred").rglob("*.py")):
        runtime[str(path.relative_to(ROOT))] = sha(path)
        copy = OUT / "runtime_snapshot" / path.relative_to(ROOT)
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_bytes(path.read_bytes())
    (OUT / "runtime_manifest.json").write_text(json.dumps(runtime, indent=2))
    manifest = {
        **{k: reference[k] for k in ["policy", "input_contract"]},
        "version": "shadow-candidate-v2-2026-09-14",
        "frozen_at_utc": now,
        "deployment_ready": True,
        "mode": "shadow",
        "live_trading_enabled": False,
        "input_constructor": "candidate_v2",
        "pending_for_shadow": [],
        "previous_bundle": str(OLD.relative_to(ROOT)),
        "model_binaries": "Exact research candidate copies, including original research_only provenance metadata",
        "probability_roundtrip_max_error": 0.0,
        "files": {str(p.relative_to(ROOT)): sha(p) for p in sorted(OUT.rglob("*")) if p.is_file()},
    }
    for path in [
        config_path,
        ROOT / "configs/fighter_aliases.json",
        ROOT / "configs/fight_result_corrections.json",
    ]:
        manifest["files"][str(path.relative_to(ROOT))] = sha(path)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("Frozen", len(manifest["files"]), "artifacts and", len(runtime), "runtime modules at", now)
    print("LaunchAgent prepared at", OUT / "com.ufcbet.watchdog.plist")


if __name__ == "__main__":
    main()

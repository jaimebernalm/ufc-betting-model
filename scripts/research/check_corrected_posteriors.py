"""Match persisted skill features to the posterior files actually producing them."""

import hashlib
import json
import pickle

import numpy as np
import pandas as pd

from ufc_pred.features.skill_v3 import skill_diff_for_fights
from ufc_pred.paths import PROCESSED, ROOT

out = ROOT / "artifacts/corrected_2026_09_14"
features = pd.read_parquet(out / "skill_features.parquet")
features["date"] = pd.to_datetime(features.date)
report = []
for month, group in features.groupby(features.date.dt.to_period("M")):
    if group.skill_diff_mean.isna().all():
        report.append({"month": str(month), "status": "all_NaN"})
        continue
    matches = []
    for p in sorted((PROCESSED / "skill_posteriors").glob(f"{month}-01_*.pkl")):
        with p.open("rb") as f:
            payload = pickle.load(f)
        if "_diagnostics" not in payload["samples"]:
            continue
        calculated = skill_diff_for_fights(group, payload["samples"], payload["index"])
        if np.allclose(
            calculated[["skill_diff_mean", "skill_diff_std"]],
            group[["skill_diff_mean", "skill_diff_std"]],
            equal_nan=True,
            atol=1e-10,
            rtol=1e-10,
        ):
            matches.append(
                {
                    "file": str(p.relative_to(ROOT)),
                    "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                    "diagnostics": payload["samples"]["_diagnostics"],
                }
            )
    report.append({"month": str(month), "status": "matched" if matches else "UNMATCHED", "matches": matches})
summary = {
    "months": len(report),
    "unmatched": [r["month"] for r in report if r["status"] == "UNMATCHED"],
    "max_rhat": max(m["diagnostics"]["max_rhat"] for r in report for m in r.get("matches", [])),
    "min_ess": min(m["diagnostics"]["min_ess"] for r in report for m in r.get("matches", [])),
    "months_with_divergences": sorted(
        {r["month"] for r in report for m in r.get("matches", []) if m["diagnostics"]["divergences"] > 0}
    ),
    "detail": report,
}
(out / "posterior_validation.json").write_text(json.dumps(summary, indent=2))
print({k: v for k, v in summary.items() if k != "detail"})

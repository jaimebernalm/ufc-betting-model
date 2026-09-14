"""Package the research candidate in the standard model format; do not deploy."""

import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from ablate_predictive_degradation import CUTOFF, FIX, OUT
from evaluate_corrected_system import predictions

from ufc_pred.models.wrappers import CorruptedSkillModel
from ufc_pred.paths import ROOT

name = "profiles_plus_prior_external_counters"
folder = OUT / "candidate"
models = folder / "models"
models.mkdir(parents=True, exist_ok=True)
features = OUT / (name + ".parquet")
f = pd.read_parquet(features)
feature_sha = hashlib.sha256(features.read_bytes()).hexdigest()
for seed in range(10):
    model = joblib.load(OUT / name / f"seed{seed}.joblib")
    meta = {
        "columns": model.feature_names_,
        "cat_features": [model.feature_names_[i] for i in model.get_cat_feature_indices()],
        "cutoff": str(CUTOFF.date()),
        "seed": seed,
        "n_train_rows": int((f.date < CUTOFF).sum()),
        "training_sha256": feature_sha,
        "feature_version": "ablation-prior-external-counters-2026-09-14",
        "research_only": True,
    }
    for kind, m in [
        ("real", model),
        ("corrupted", CorruptedSkillModel(model, ["skill_diff_mean", "skill_diff_std"])),
    ]:
        joblib.dump({**meta, "model": m}, models / f"v3_{kind}_2025_11_30_seed{seed}.joblib")
actual = predictions(f[f.date >= CUTOFF], models)
expected = np.load(OUT / "all_variant_predictions.npz")[name]
diff = max(float(np.max(abs(actual[k] - expected[i]))) for i, k in enumerate(["real", "corrupted"]))
assert diff < 1e-12
policy = json.loads((FIX / "manifest.json").read_text())["policy"]
files = [
    features,
    OUT / "completed_profile_attributes.json",
    OUT / "prior_external_histories.json",
    OUT / "external_counter_coverage.json",
    *models.glob("*.joblib"),
]
manifest = {
    "policy": policy,
    "deployment_ready": False,
    "mode": "research_only",
    "probability_roundtrip_max_error": diff,
    "input_contract": {
        "rates": "strict prior UFC-only",
        "counters": "strict prior UFC plus verified external history for 796 audited fighters",
        "profiles": "frozen original raw profiles with verified missing static attributes completed",
        "rankings_skill": "same corrected frozen source/recipe",
    },
    "pending_for_shadow": [
        "implement and validate this input contract in live constructor",
        "resolve verified doctor-stoppage parser discrepancy",
        "create a new frozen runtime/prospective protocol version",
    ],
    "files": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
}
(folder / "manifest.json").write_text(json.dumps(manifest, indent=2))
print("20 research bundles packaged; roundtrip error", diff, "; existing shadow unchanged")

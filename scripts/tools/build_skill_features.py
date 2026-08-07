"""CLI entrypoint: build data/processed/skill_features_v3.parquet.

Run once. Slow (~30 min on CPU). Output is consumed by baseline_v3.
"""

from __future__ import annotations

from ufc_pred.features.skill_v3_pipeline import build

if __name__ == "__main__":
    stats = build()
    for k, v in stats.items():
        print(f"{k}: {v}")

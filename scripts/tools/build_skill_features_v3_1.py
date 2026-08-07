"""CLI: build data/processed/skill_features_v3_1.parquet.

Long-running (~5-9 hours on CPU). Writes a partial parquet every 12 months so
interrupted runs leave recoverable progress.
"""

from __future__ import annotations

from ufc_pred.features.skill_v3_1_pipeline import build

if __name__ == "__main__":
    stats = build()
    for k, v in stats.items():
        print(f"{k}: {v}")

"""v3: v1.1 CatBoost + Bayesian skill features (skill_diff_mean, skill_diff_std).

Identical CatBoost config to v1.1. The ONLY change is two new columns joined
from the scalar Bradley-Terry skill posterior (built by
`scripts/tools/build_skill_features.py`).

No hyperparameter tuning between versions — a clean read on "did skill features
help" requires the same model config as v1.1.
"""

from __future__ import annotations

from ufc_pred.features.joins import join_skill_v3
from ufc_pred.features.skill_v3_pipeline import OUTPUT as SKILL_PARQUET
from ufc_pred.models._report import main
from ufc_pred.models._spec import ModelSpec, Recipe

VERSION = "v3_catboost_skill"


def _coverage(*, X_train, X_val, model, **_) -> dict:
    """Skill-feature join coverage — a silent drop here would quietly gut v3."""
    return {
        "skill_features": {
            "parquet": str(SKILL_PARQUET),
            "train_coverage": float(X_train["skill_diff_mean"].notna().mean()),
            "val_coverage": float(X_val["skill_diff_mean"].notna().mean()),
        }
    }


def _report(run: dict) -> list[str]:
    sk = run["skill_features"]
    return [f"skill cov: train={sk['train_coverage']:.3f}  val={sk['val_coverage']:.3f}"]


SPEC = ModelSpec(
    version=VERSION,
    recipe=Recipe(
        name="v3 (scalar Bayesian skill)",
        join=join_skill_v3,
        flip_columns=("skill_diff_mean",),
    ),
    early_stopping=True,
    diagnostics=_coverage,
    extra_report=_report,
)


if __name__ == "__main__":
    main(SPEC)

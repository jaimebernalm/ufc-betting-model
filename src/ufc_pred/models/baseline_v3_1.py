"""v3.1: v1.1 CatBoost + time-varying Bayesian skill features.

Identical CatBoost config to v1.1 / v3. The only change vs v3 is the *source*
of the skill columns: a random-walk posterior that lets skill drift over a
career, instead of v3's single scalar per fighter. Motivated by the long-layoff
failure mode, where a static skill estimate stays stale.
"""

from __future__ import annotations

from ufc_pred.features.joins import join_skill_v3_1
from ufc_pred.features.skill_v3_1_pipeline import OUTPUT as SKILL_PARQUET
from ufc_pred.models._report import main
from ufc_pred.models._spec import ModelSpec, Recipe

VERSION = "v3_1_catboost_timevarying_skill"


def _coverage(*, X_train, X_val, model, **_) -> dict:
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
        name="v3.1 (time-varying Bayesian skill)",
        join=join_skill_v3_1,
        flip_columns=("skill_diff_mean",),
    ),
    early_stopping=True,
    diagnostics=_coverage,
    extra_report=_report,
)


if __name__ == "__main__":
    main(SPEC)

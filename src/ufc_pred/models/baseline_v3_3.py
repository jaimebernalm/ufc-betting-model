"""v3.3 stacked skill: CatBoost with BOTH skill feature versions.

Feeds 4 columns into CatBoost:
  skill_diff_mean_v3,   skill_diff_std_v3     (scalar Bradley-Terry, from v3)
  skill_diff_mean_v3_1, skill_diff_std_v3_1   (time-varying random walk, v3.1)

v3 was better on aggregate log_loss; v3.1 fixed the long-layoff failure mode.
Stacking tests whether the tree model can pick up the layoff signal from v3.1
without losing v3's per-fight signal. CatBoost config identical to v1.1/v3/v3.1.
"""

from __future__ import annotations

from ufc_pred.features.joins import join_skill_stacked
from ufc_pred.models._report import main
from ufc_pred.models._spec import ModelSpec, Recipe

VERSION = "v3_3_catboost_stacked_skill"


def _skill_importances(*, X_train, X_val, model, **_) -> dict:
    """Which of the 4 skill columns CatBoost actually leaned on."""
    importances = dict(zip(X_train.columns, model.feature_importances_, strict=False))
    return {"skill_feature_importances": {k: float(v) for k, v in importances.items() if "skill_" in k}}


def _report(run: dict) -> list[str]:
    lines = ["skill feature importances:"]
    for k, v in sorted(run["skill_feature_importances"].items(), key=lambda kv: -kv[1]):
        lines.append(f"  {k:30s}  {v:.3f}")
    return lines


SPEC = ModelSpec(
    version=VERSION,
    recipe=Recipe(
        name="v3.3 (stacked scalar + time-varying skill)",
        join=join_skill_stacked,
        flip_columns=("skill_diff_mean_v3", "skill_diff_mean_v3_1"),
    ),
    early_stopping=True,
    diagnostics=_skill_importances,
    extra_report=_report,
)


if __name__ == "__main__":
    main(SPEC)

"""v7.1 — 10-seed ensemble of v3 (scalar Bayesian skill), no early stopping.

Same training protocol as v7 (10 seeds, no early stopping) applied to the v3
architecture instead of v3.3. Motivation: v3 single beat v7 (a v3.3 ensemble)
on val ROI, so this tests whether the right architecture plus ensemble
robustness beats either alone.

Because a single CatBoost seed produces a very wide spread in final bankroll,
the deployed model averages `predict_proba` over seeds 0-9 rather than trusting
any one of them.
"""

from __future__ import annotations

from ufc_pred.features.joins import join_skill_v3
from ufc_pred.models._report import main
from ufc_pred.models._spec import ModelSpec, Recipe

VERSION = "v7_1_catboost_ensemble_v3"
ENSEMBLE_SIZE = 10

SPEC = ModelSpec(
    version=VERSION,
    recipe=Recipe(
        name="v3 (scalar Bayesian skill only)",
        join=join_skill_v3,
        flip_columns=("skill_diff_mean",),
    ),
    ensemble_size=ENSEMBLE_SIZE,
    early_stopping=False,
    betting_eval=True,
)


if __name__ == "__main__":
    main(SPEC)

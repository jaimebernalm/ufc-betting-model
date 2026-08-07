"""v7 — 10-seed ensemble of v3.3 (stacked Bayesian skill), no early stopping.

Two changes from the v3.3 baseline, both deliberate:

1. **No early stopping.** v3.3 fit with `eval_set=val_pool, use_best_model=True`,
   which tunes the tree count against validation log-loss — the same set later
   used to judge betting return. That biases the model toward log-loss and away
   from profit. Every rung from here on trains to a fixed iteration count and is
   selected on the betting backtest instead.

2. **Ensembling over 10 seeds.** Identical data and hyperparameters differing
   only in `random_seed` produced an order-of-magnitude spread in final
   bankroll, so a single seed is not a model — it is a sample. Averaging
   `predict_proba` over seeds 0-9 lands deployment near the median.
"""

from __future__ import annotations

from ufc_pred.features.joins import join_skill_stacked
from ufc_pred.models._report import main
from ufc_pred.models._spec import ModelSpec, Recipe

VERSION = "v7_catboost_ensemble"
ENSEMBLE_SIZE = 10

SPEC = ModelSpec(
    version=VERSION,
    recipe=Recipe(
        name="v3.3 (stacked scalar + time-varying skill)",
        join=join_skill_stacked,
        flip_columns=("skill_diff_mean_v3", "skill_diff_mean_v3_1"),
    ),
    ensemble_size=ENSEMBLE_SIZE,
    early_stopping=False,
    betting_eval=True,
)


if __name__ == "__main__":
    main(SPEC)

"""Declarative description of one rung of the model ladder.

The ladder's whole point is that versions differ in *exactly one* dimension at
a time (see the project methodology: no hyperparameter tuning between versions,
or the comparison means nothing). Encoding each version as a `ModelSpec` makes
that discipline checkable — the diff between two versions is the diff between
two specs, and nothing else.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

# Shared CatBoost configuration. Every CatBoost rung from v1.1 onward uses
# these exact values; a version that changed them would not be comparable.
BASE_CATBOOST_PARAMS: dict[str, Any] = {
    "iterations": 2000,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3,
    "loss_function": "Logloss",
    "eval_metric": "Logloss",
    "verbose": False,
    "allow_writing_files": False,
}


@dataclass(frozen=True)
class Recipe:
    """What extra columns a version attaches to the fight table."""

    name: str
    #: Applied to the raw fight table. `None` means "static features only".
    join: Callable[[pd.DataFrame], pd.DataFrame] | None = None
    #: Join before the Red/Blue winner filter instead of after. v2's derived
    #: features were computed on the unfiltered table; preserved for exactness.
    join_before_winner_filter: bool = False
    #: Signed-difference columns to negate on the symmetry-augmented half.
    flip_columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelSpec:
    """One rung of the ladder."""

    version: str
    recipe: Recipe
    #: Overrides merged onto BASE_CATBOOST_PARAMS.
    param_overrides: dict[str, Any] = field(default_factory=dict)
    #: >1 trains one member per seed 0..n-1 and averages predict_proba.
    ensemble_size: int = 1
    #: Early stopping fits against the validation set. This biases the model
    #: toward val log-loss and away from val betting return — it is a leak with
    #: respect to model selection, and the later rungs deliberately disable it.
    early_stopping: bool = False
    #: Also run the betting backtest on val and record ROI per fee scenario.
    betting_eval: bool = False
    #: Extra per-version diagnostics: (X_train, X_val, model) -> {section: data}.
    diagnostics: Callable[..., dict[str, Any]] | None = None
    #: Report lines printed after the split summary.
    extra_report: Callable[[dict], Sequence[str]] | None = None

    def catboost_params(self, seed: int = 0) -> dict[str, Any]:
        params = dict(BASE_CATBOOST_PARAMS)
        if self.early_stopping:
            params["early_stopping_rounds"] = 100
        params["random_seed"] = seed
        params.update(self.param_overrides)
        return params


#: Fee scenarios used by `betting_eval`: (label, use_no_vig, fee_rate).
BETTING_SCENARIOS: tuple[tuple[str, bool, float], ...] = (
    ("sportsbook_with_vig", False, 0.0),
    ("no_vig_no_fee", True, 0.0),
    ("kalshi_like", True, 0.07),
)

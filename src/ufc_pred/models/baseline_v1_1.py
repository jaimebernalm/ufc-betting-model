"""v1.1 baseline: CatBoost on the same static features as v1.

Same train/val split, same recency weights, same symmetry augmentation. Only
the estimator and its categorical handling change, so the v1 → v1.1 delta reads
purely as "does gradient boosting beat logistic regression here".

This is the reference config every later CatBoost rung inherits — see
`models._spec.BASE_CATBOOST_PARAMS`.
"""

from __future__ import annotations

from ufc_pred.models._report import main
from ufc_pred.models._spec import ModelSpec, Recipe

VERSION = "v1_1_catboost"


def _data_dates(*, fights, **_) -> dict:
    return {
        "data_dates": {
            "history_first": str(fights["date"].min().date()),
            "history_last": str(fights["date"].max().date()),
        }
    }


SPEC = ModelSpec(
    version=VERSION,
    recipe=Recipe(name="v1 static features only"),
    early_stopping=True,
    diagnostics=_data_dates,
)


if __name__ == "__main__":
    main(SPEC)

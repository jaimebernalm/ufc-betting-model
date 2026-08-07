"""v2: v1.1 CatBoost + derived no-scrape features (layoff, activity, career ratios).

Identical CatBoost config to v1.1. The ONLY change is the 10 numeric columns
appended by `features.derived_v2.compute`.

Keep-or-kill bar: must improve val log_loss by ≥0.5% relative vs v1.1 without
worsening ECE. Failing it would mean the Kaggle feature set is saturated and
the UFCStats scrape needs to target the genuinely missing columns (defense %,
control time, knockdowns).
"""

from __future__ import annotations

from ufc_pred.features.derived_v2 import NEW_NUMERIC_COLS
from ufc_pred.features.joins import join_derived
from ufc_pred.models._report import main
from ufc_pred.models._spec import ModelSpec, Recipe

VERSION = "v2_catboost_derived"


def _coverage(*, X_train, X_val, model, **_) -> dict:
    return {
        "derived_features": {
            "columns": NEW_NUMERIC_COLS,
            "train_coverage": {
                c: float(X_train[c].notna().mean()) for c in NEW_NUMERIC_COLS if c in X_train.columns
            },
            "val_coverage": {
                c: float(X_val[c].notna().mean()) for c in NEW_NUMERIC_COLS if c in X_val.columns
            },
        }
    }


def _report(run: dict) -> list[str]:
    cov = run["derived_features"]["train_coverage"]
    avg = sum(cov.values()) / len(cov) if cov else 0.0
    return [f"derived feature avg train coverage: {avg:.3f}"]


SPEC = ModelSpec(
    version=VERSION,
    recipe=Recipe(
        name="v2 (derived no-scrape features)",
        join=join_derived,
        join_before_winner_filter=True,
    ),
    early_stopping=True,
    diagnostics=_coverage,
    extra_report=_report,
)


if __name__ == "__main__":
    main(SPEC)

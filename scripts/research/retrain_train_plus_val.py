"""Retrain v3_full2000 on train+val combined, for test-set evaluation.

Same recipe as `scripts/retrain_no_early_stopping.py` (v3 variant):
  - v3 scalar Bayesian skill features (already leak-free per-month walk-forward)
  - CatBoost, 2000 iters, lr=0.05, depth=6, l2=3, seed=0, no early stopping
  - symmetry augmentation with sign-flip on skill_diff_mean
  - exponential recency weights, half-life 4y, reference = VAL_END (2023-12-31)

Also wraps the trained model with CorruptedSkillModel for the diagnostic
counterpart (matches build_corrupted_and_v1_1_full2000.py).

Outputs:
  artifacts/models/v3_catboost_full2000_trainval.joblib
  artifacts/models/v3_full2000_no_skill_corrupted_trainval.joblib
  artifacts/metrics/v3_catboost_full2000_trainval.json
  artifacts/metrics/v3_full2000_no_skill_corrupted_trainval.json
"""

from __future__ import annotations

import json
from datetime import datetime

import joblib
import pandas as pd
from catboost import CatBoostClassifier, Pool

from ufc_pred.features.skill_v3_pipeline import OUTPUT as SKILL_V3_PARQUET
from ufc_pred.features.static_v1 import prepare
from ufc_pred.ingest.kaggle_mdabbert import HISTORY_PARQUET
from ufc_pred.models.wrappers import CorruptedSkillModel
from ufc_pred.paths import METRICS, MODELS
from ufc_pred.utils.time_splits import VAL_END, recency_weights, split


def _join_v3(fights: pd.DataFrame) -> pd.DataFrame:
    sk = pd.read_parquet(SKILL_V3_PARQUET)
    sk["date"] = pd.to_datetime(sk["date"])
    return fights.merge(
        sk[["date", "R_fighter", "B_fighter", "skill_diff_mean", "skill_diff_std"]],
        on=["date", "R_fighter", "B_fighter"],
        how="left",
        validate="many_to_one",
    )


def _augment_skill_sign(X: pd.DataFrame) -> pd.DataFrame:
    """Sign-flip skill_diff_mean on the augmented (R/B-swapped) half."""
    n = len(X) // 2
    if len(X) != 2 * n:
        return X
    X = X.copy()
    second = X.index[n:]
    if "skill_diff_mean" in X.columns:
        X.loc[second, "skill_diff_mean"] = -X.loc[second, "skill_diff_mean"]
    return X


def main():
    fights = pd.read_parquet(HISTORY_PARQUET)
    fights = fights[fights["Winner"].isin(["Red", "Blue"])].copy()
    fights["date"] = pd.to_datetime(fights["date"])
    fights = _join_v3(fights)
    splits = split(fights)

    # Combine train + val for retraining.
    train_plus_val = pd.concat([splits.train, splits.val], ignore_index=True)
    print(
        f"Train+val combined: n={len(train_plus_val)} fights "
        f"({splits.train['date'].min().date()} → {splits.val['date'].max().date()})"
    )

    X_tv, y_tv, d_tv, cat_features = prepare(train_plus_val, augment_symmetry=True, one_hot=False)
    X_tv = _augment_skill_sign(X_tv)

    # Reference = VAL_END so val fights get weight ~1.0 (they're the most recent
    # training data) and train fights decay relative to that.
    sample_weight = recency_weights(d_tv, reference_date=VAL_END)
    pool = Pool(X_tv, y_tv, cat_features=cat_features, weight=sample_weight)

    m = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        loss_function="Logloss",
        eval_metric="Logloss",
        random_seed=0,
        verbose=False,
        allow_writing_files=False,
    )
    print("Fitting CatBoost (2000 iters, no early stopping, seed=0)...")
    m.fit(pool, verbose=False)
    print("  done.")

    # Save the real model.
    real_version = "v3_catboost_full2000_trainval"
    real_path = MODELS / f"{real_version}.joblib"
    payload = {
        "model": m,
        "columns": list(X_tv.columns),
        "one_hot": False,
        "cat_features": cat_features,
    }
    joblib.dump(payload, real_path)
    (METRICS / f"{real_version}.json").write_text(
        json.dumps(
            {
                "version": real_version,
                "trained_at": datetime.now().isoformat(timespec="seconds"),
                "note": "v3_full2000 architecture retrained on train+val combined for "
                "test-set evaluation. Skill features unchanged (walk-forward "
                "already covers test months). Reference date for recency "
                "weights = VAL_END (2023-12-31).",
                "n_train_fights_pre_augment": int(len(train_plus_val)),
                "n_train_rows_post_augment": int(len(X_tv)),
                "recipe": {
                    "iterations": 2000,
                    "learning_rate": 0.05,
                    "depth": 6,
                    "l2_leaf_reg": 3,
                    "seed": 0,
                    "early_stopping": False,
                    "symmetry_augmentation": True,
                    "recency_half_life_years": 4.0,
                    "recency_reference_date": str(VAL_END.date()),
                },
                "model_path": str(real_path),
            },
            indent=2,
            default=float,
        )
    )
    print(f"  saved → {real_path}")

    # Build corrupted wrapper.
    skill_cols = ["skill_diff_mean", "skill_diff_std"]
    wrapped = CorruptedSkillModel(m, skill_cols)

    corrupt_version = "v3_full2000_no_skill_corrupted_trainval"
    corrupt_path = MODELS / f"{corrupt_version}.joblib"
    joblib.dump(
        {
            "model": wrapped,
            "columns": list(X_tv.columns),
            "one_hot": False,
            "cat_features": cat_features,
        },
        corrupt_path,
    )
    (METRICS / f"{corrupt_version}.json").write_text(
        json.dumps(
            {
                "version": corrupt_version,
                "trained_at": datetime.now().isoformat(timespec="seconds"),
                "note": "v3_full2000_trainval wrapped to force NaN on skill columns at "
                "inference. Diagnostic counterpart to the train+val champion.",
                "wrapped_columns_set_to_nan": skill_cols,
                "wraps": real_version,
                "model_path": str(corrupt_path),
            },
            indent=2,
            default=float,
        )
    )
    print(f"  saved → {corrupt_path}")


if __name__ == "__main__":
    main()

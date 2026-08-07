"""v1.2 = v1.1 CatBoost + isotonic calibration.

Diagnostic to see how much of v1.1's log loss is miscalibration vs ranking error.

Protocol:
1. Keep v1.1's trained model as the ranker (already saved).
2. Refit CatBoost with 5-fold CV on un-augmented training data (fixed iteration
   count = v1.1's tree_count_; no early stopping inside folds, no val leak).
3. Use those OOF predictions as the input to fit `IsotonicRegression` against
   actual labels. The calibrator never sees val.
4. At inference, take v1.1's raw probability and pass it through the calibrator.
5. Compare raw v1.1 vs calibrated v1.2 on val.
"""

from __future__ import annotations

import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold

from ufc_pred.backtest.metrics import evaluate, market_no_vig_prob_red
from ufc_pred.features.static_v1 import prepare
from ufc_pred.ingest.kaggle_mdabbert import HISTORY_PARQUET
from ufc_pred.paths import METRICS, MODELS
from ufc_pred.utils.time_splits import TEST_END, TRAIN_END, VAL_END, recency_weights, split

VERSION = "v1_2_catboost_isotonic"
N_FOLDS = 5


def _fold_model(iterations: int) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=iterations,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        loss_function="Logloss",
        random_seed=0,
        verbose=False,
        allow_writing_files=False,
    )


def train_and_eval() -> dict:
    # 1. Load v1.1 ranker
    v1_1_path = MODELS / "v1_1_catboost.joblib"
    v1_1 = joblib.load(v1_1_path)
    ranker = v1_1["model"]
    cat_features = v1_1["cat_features"]
    columns = v1_1["columns"]
    best_iter = int(ranker.tree_count_)

    # 2. OOF predictions on un-augmented train (no symmetry; calibration cares
    # about true frequencies, and the swapped duplicates would leak across folds).
    fights = pd.read_parquet(HISTORY_PARQUET)
    fights = fights[fights["Winner"].isin(["Red", "Blue"])].copy()
    splits = split(fights)
    X_tr, y_tr, d_tr, _ = prepare(splits.train, augment_symmetry=False, one_hot=False)
    X_tr = X_tr.reindex(columns=columns, fill_value=None)
    for c in cat_features:
        X_tr[c] = X_tr[c].fillna("__missing__").astype(str)
    w_tr = recency_weights(d_tr)

    oof = np.zeros(len(y_tr))
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
    for _fold, (tr_idx, va_idx) in enumerate(kf.split(X_tr)):
        pool_tr = Pool(X_tr.iloc[tr_idx], y_tr[tr_idx], cat_features=cat_features, weight=w_tr[tr_idx])
        m = _fold_model(iterations=best_iter)
        m.fit(pool_tr)
        oof[va_idx] = m.predict_proba(X_tr.iloc[va_idx])[:, 1]

    # 3. Fit isotonic on OOF (with recency weights so 2010 fights don't dominate)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(oof, y_tr, sample_weight=w_tr)

    # 4. Apply to val
    X_val, y_val, _, _ = prepare(splits.val, augment_symmetry=False, one_hot=False)
    X_val = X_val.reindex(columns=columns, fill_value=None)
    for c in cat_features:
        X_val[c] = X_val[c].fillna("__missing__").astype(str)

    raw_val = ranker.predict_proba(X_val)[:, 1]
    calib_val = iso.transform(raw_val)

    raw_metrics = evaluate(y_val, raw_val, label="val_raw_v1_1")
    calib_metrics = evaluate(y_val, calib_val, label="val")

    sub = splits.val.dropna(subset=["R_odds", "B_odds"])
    market_metrics = evaluate(
        (sub["Winner"] == "Red").astype(int).to_numpy(),
        market_no_vig_prob_red(sub),
        label="val_market",
    )

    # train metrics for the ranker stay informational
    raw_train = ranker.predict_proba(X_tr)[:, 1]
    train_metrics = evaluate(y_tr, iso.transform(raw_train), label="train")

    MODELS.mkdir(parents=True, exist_ok=True)
    METRICS.mkdir(parents=True, exist_ok=True)
    model_path = MODELS / f"{VERSION}.joblib"
    joblib.dump(
        {
            "ranker_path": str(v1_1_path),
            "isotonic": iso,
            "columns": columns,
            "cat_features": cat_features,
            "one_hot": False,
        },
        model_path,
    )

    run = {
        "version": VERSION,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "splits": {
            "train_end": str(TRAIN_END.date()),
            "val_end": str(VAL_END.date()),
            "test_end": str(TEST_END.date()),
            "n_train": int(len(X_tr)),
            "n_val": int(len(splits.val)),
            "n_folds": N_FOLDS,
        },
        "data_dates": {
            "history_first": str(fights["date"].min().date()),
            "history_last": str(fights["date"].max().date()),
        },
        "calibration": {
            "iterations_per_fold": best_iter,
            "n_oof_predictions": int(len(oof)),
        },
        "metrics": {
            "train": train_metrics,
            "val": calib_metrics,
            "val_raw_v1_1": raw_metrics,
            "val_market": market_metrics,
        },
        "model_path": str(model_path),
    }
    (METRICS / f"{VERSION}.json").write_text(json.dumps(run, indent=2))
    return run


def _print_run(run: dict) -> None:
    print(f"version: {run['version']}  (trained {run['trained_at']})")
    s = run["splits"]
    print(f"splits:  train (n={s['n_train']})  val (n={s['n_val']})  {s['n_folds']}-fold OOF for isotonic")
    print(f"saved:   {run['model_path']}")
    print()
    rows = pd.DataFrame(
        [
            run["metrics"]["train"],
            run["metrics"]["val_raw_v1_1"],
            run["metrics"]["val"],
            run["metrics"]["val_market"],
        ]
    )
    rows.insert(0, "model", ["v1_2 train", "v1.1 raw", "v1.2 calibrated", "market"])
    print(
        rows[["model", "label", "n", "log_loss", "brier", "ece", "accuracy_argmax"]].to_string(
            index=False, float_format=lambda x: f"{x:.4f}"
        )
    )


if __name__ == "__main__":
    _print_run(train_and_eval())

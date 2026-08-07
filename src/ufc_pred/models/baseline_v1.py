"""v1 baseline: logistic regression on static features.

See PLAN.md §2.3 v1. Trains on `train`, evaluates on `val`, saves the model
and metrics. Test set is NOT touched here — it is reserved for the final
version comparison at the end of the project.
"""

from __future__ import annotations

import json
from datetime import datetime

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ufc_pred.backtest.metrics import evaluate, market_no_vig_prob_red
from ufc_pred.features.static_v1 import prepare
from ufc_pred.ingest.kaggle_mdabbert import HISTORY_PARQUET
from ufc_pred.paths import METRICS, MODELS
from ufc_pred.utils.time_splits import TEST_END, TRAIN_END, VAL_END, recency_weights, split

VERSION = "v1_logreg"


def build_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            # max_iter is a budget; lbfgs typically converges in <100 iters here, this is safety headroom.
            ("lr", LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=2000)),
        ]
    )


def train_and_eval() -> dict:
    fights = pd.read_parquet(HISTORY_PARQUET)
    fights = fights[fights["Winner"].isin(["Red", "Blue"])].copy()
    splits = split(fights)

    X_train, y_train, d_train, _ = prepare(splits.train, augment_symmetry=True)
    X_val, y_val, _, _ = prepare(splits.val, augment_symmetry=False)
    X_val = X_val.reindex(columns=X_train.columns, fill_value=0)

    model = build_model()
    model.fit(X_train, y_train, lr__sample_weight=recency_weights(d_train))

    train_metrics = evaluate(y_train, model.predict_proba(X_train)[:, 1], label="train")
    val_metrics = evaluate(y_val, model.predict_proba(X_val)[:, 1], label="val")

    # Market baseline on val for direct comparison on the SAME fights.
    sub = splits.val.dropna(subset=["R_odds", "B_odds"])
    market_val = evaluate(
        (sub["Winner"] == "Red").astype(int).to_numpy(),
        market_no_vig_prob_red(sub),
        label="val_market",
    )

    MODELS.mkdir(parents=True, exist_ok=True)
    METRICS.mkdir(parents=True, exist_ok=True)
    model_path = MODELS / f"{VERSION}.joblib"
    joblib.dump({"model": model, "columns": list(X_train.columns), "one_hot": True}, model_path)

    run = {
        "version": VERSION,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "splits": {
            "train_end": str(TRAIN_END.date()),
            "val_end": str(VAL_END.date()),
            "test_end": str(TEST_END.date()),
            "n_train_pre_aug": int(len(splits.train)),
            "n_train_post_aug": int(len(X_train)),
            "n_val": int(len(splits.val)),
        },
        "data_dates": {
            "history_first": str(fights["date"].min().date()),
            "history_last": str(fights["date"].max().date()),
        },
        "metrics": {"train": train_metrics, "val": val_metrics, "val_market": market_val},
        "model_path": str(model_path),
    }

    metrics_path = METRICS / f"{VERSION}.json"
    metrics_path.write_text(json.dumps(run, indent=2))
    return run


def _print_run(run: dict) -> None:
    print(f"version: {run['version']}  (trained {run['trained_at']})")
    print(f"data:    {run['data_dates']['history_first']} → {run['data_dates']['history_last']}")
    s = run["splits"]
    print(
        f"splits:  train≤{s['train_end']} (n={s['n_train_pre_aug']}, aug={s['n_train_post_aug']})"
        f"  val≤{s['val_end']} (n={s['n_val']})  test held out"
    )
    print(f"saved:   {run['model_path']}")
    print()
    rows = pd.DataFrame([run["metrics"]["train"], run["metrics"]["val"], run["metrics"]["val_market"]])
    rows.insert(0, "model", [VERSION, VERSION, "market"])
    print(
        rows[["model", "label", "n", "log_loss", "brier", "ece", "accuracy_argmax"]].to_string(
            index=False, float_format=lambda x: f"{x:.4f}"
        )
    )


if __name__ == "__main__":
    _print_run(train_and_eval())

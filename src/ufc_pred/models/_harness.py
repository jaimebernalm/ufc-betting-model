"""Shared training harness for the CatBoost rungs of the model ladder.

Replaces the `train_and_eval()` that was previously copy-pasted into every
`baseline_v*.py`. The pipeline is fixed — split, prepare, recency-weight, fit,
evaluate, persist — and a `ModelSpec` supplies the only parts that vary.

Guarantees that matter for comparability:
  * the train/val split dates come from `utils.time_splits` and are never
    passed in per version;
  * the test set is never read here;
  * CatBoost parameters come from `BASE_CATBOOST_PARAMS`, so a version can only
    differ where its spec says it does.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from ufc_pred.backtest.bet_eval import evaluate_bets
from ufc_pred.backtest.metrics import evaluate, market_no_vig_prob_red
from ufc_pred.features.joins import flip_signed_columns
from ufc_pred.features.static_v1 import prepare
from ufc_pred.ingest.kaggle_mdabbert import HISTORY_PARQUET
from ufc_pred.models._spec import BETTING_SCENARIOS, ModelSpec
from ufc_pred.paths import METRICS, MODELS
from ufc_pred.utils.time_splits import TEST_END, TRAIN_END, VAL_END, recency_weights, split

BET_EDGE_THRESHOLD = 0.05


def build_matrices(spec: ModelSpec) -> dict[str, Any]:
    """Load fights and build the train/val design matrices for `spec`.

    Split out from `run_training` so tests can assert feature parity without
    paying for a fit.
    """
    fights = pd.read_parquet(HISTORY_PARQUET)

    join = spec.recipe.join
    if join is not None and spec.recipe.join_before_winner_filter:
        fights = join(fights)
    fights = fights[fights["Winner"].isin(["Red", "Blue"])].copy()
    if join is not None and not spec.recipe.join_before_winner_filter:
        fights = join(fights)

    splits = split(fights)

    X_train, y_train, d_train, cat_features = prepare(splits.train, augment_symmetry=True, one_hot=False)
    X_train = flip_signed_columns(X_train, spec.recipe.flip_columns)

    X_val, y_val, _, _ = prepare(splits.val, augment_symmetry=False, one_hot=False)
    X_val = X_val.reindex(columns=X_train.columns, fill_value=None)
    for c in cat_features:
        X_val[c] = X_val[c].fillna("__missing__").astype(str)

    return {
        "fights": fights,
        "splits": splits,
        "X_train": X_train,
        "y_train": y_train,
        "d_train": d_train,
        "X_val": X_val,
        "y_val": y_val,
        "cat_features": cat_features,
    }


def _fit(spec: ModelSpec, train_pool: Pool, val_pool: Pool | None) -> list[CatBoostClassifier]:
    members: list[CatBoostClassifier] = []
    for seed in range(spec.ensemble_size):
        model = CatBoostClassifier(**spec.catboost_params(seed=seed))
        if spec.early_stopping:
            model.fit(train_pool, eval_set=val_pool, use_best_model=True)
        else:
            model.fit(train_pool, verbose=False)
        members.append(model)
        if spec.ensemble_size > 1:
            print(f"  seed {seed}: tree_count={model.tree_count_}")
    return members


def _predict(members: list[CatBoostClassifier], X: pd.DataFrame) -> np.ndarray:
    if len(members) == 1:
        return members[0].predict_proba(X)[:, 1]
    return np.mean(np.vstack([m.predict_proba(X)[:, 1] for m in members]), axis=0)


def run_training(spec: ModelSpec) -> dict:
    """Train, evaluate, persist and return the run record for one rung."""
    data = build_matrices(spec)
    splits = data["splits"]
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    cat_features = data["cat_features"]

    train_pool = Pool(X_train, y_train, cat_features=cat_features, weight=recency_weights(data["d_train"]))
    val_pool = Pool(X_val, y_val, cat_features=cat_features) if spec.early_stopping else None

    if spec.ensemble_size > 1:
        print(f"Training {spec.ensemble_size}-seed ensemble ({spec.recipe.name})...")
    members = _fit(spec, train_pool, val_pool)

    p_train = _predict(members, X_train)
    p_val = _predict(members, X_val)

    sub = splits.val.dropna(subset=["R_odds", "B_odds"])
    run: dict[str, Any] = {
        "version": spec.version,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "feature_recipe": spec.recipe.name,
        "splits": {
            "train_end": str(TRAIN_END.date()),
            "val_end": str(VAL_END.date()),
            "test_end": str(TEST_END.date()),
            "n_train_pre_aug": int(len(splits.train)),
            "n_train_post_aug": int(len(X_train)),
            "n_val": int(len(splits.val)),
        },
        "metrics": {
            "train": evaluate(y_train, p_train, label="train"),
            "val": evaluate(y_val, p_val, label="val"),
            "val_market": evaluate(
                (sub["Winner"] == "Red").astype(int).to_numpy(),
                market_no_vig_prob_red(sub),
                label="val_market",
            ),
        },
    }

    if spec.ensemble_size > 1:
        run["ensemble"] = {
            "size": spec.ensemble_size,
            "seeds": list(range(spec.ensemble_size)),
            "tree_counts": [int(m.tree_count_) for m in members],
        }
    else:
        run["catboost"] = {
            "best_iteration": int(members[0].get_best_iteration() or 0),
            "tree_count": int(members[0].tree_count_),
        }

    if spec.betting_eval:
        y_red_val = np.asarray(y_val)
        val_df = splits.val.reset_index(drop=True)
        run["betting"] = {}
        for label, use_no_vig, fee in BETTING_SCENARIOS:
            r = evaluate_bets(
                p_val,
                y_red_val,
                val_df["R_odds"],
                val_df["B_odds"],
                edge_threshold=BET_EDGE_THRESHOLD,
                fee_rate=fee,
                use_no_vig=use_no_vig,
            )
            run["betting"][label] = {
                "roi_pct": r.roi_pct,
                "ci95_low": r.ci95_roi_pct[0],
                "ci95_high": r.ci95_roi_pct[1],
                "n_bets": r.n_bets,
                "mean_ev_pct": r.mean_ev_pct,
                "sharpe": r.sharpe,
            }

    if spec.diagnostics is not None:
        run.update(spec.diagnostics(X_train=X_train, X_val=X_val, model=members[0], fights=data["fights"]))

    MODELS.mkdir(parents=True, exist_ok=True)
    METRICS.mkdir(parents=True, exist_ok=True)
    model_path = MODELS / f"{spec.version}.joblib"

    payload: dict[str, Any] = {
        "columns": list(X_train.columns),
        "one_hot": False,
        "cat_features": cat_features,
    }
    if spec.ensemble_size > 1:
        payload["models"] = members
        payload["ensemble_size"] = spec.ensemble_size
        payload["seeds"] = list(range(spec.ensemble_size))
    else:
        payload["model"] = members[0]
    joblib.dump(payload, model_path)

    run["model_path"] = str(model_path)
    (METRICS / f"{spec.version}.json").write_text(json.dumps(run, indent=2, default=float))
    return run

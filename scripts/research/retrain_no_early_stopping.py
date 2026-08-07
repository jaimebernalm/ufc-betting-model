"""Retrain v3, v3.1, v3.3 single-models without early stopping for fair comparison
with v7 (which removed early stopping). Saves new artifacts with `_full2000`
suffix so the original (early-stopped) artifacts stay intact for history.

Each model uses seed=0 and runs the full 2000 iterations. CatBoost config
otherwise byte-identical to the original baselines.
"""

from __future__ import annotations

import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from ufc_pred.backtest.bet_eval import evaluate_bets
from ufc_pred.backtest.metrics import evaluate, market_no_vig_prob_red
from ufc_pred.features.joins import flip_signed_columns, join_skill_stacked
from ufc_pred.features.skill_v3_1_pipeline import OUTPUT as SKILL_V3_1_PARQUET
from ufc_pred.features.skill_v3_pipeline import OUTPUT as SKILL_V3_PARQUET
from ufc_pred.features.static_v1 import prepare
from ufc_pred.ingest.kaggle_mdabbert import HISTORY_PARQUET
from ufc_pred.models._spec import BASE_CATBOOST_PARAMS

V3_3_FLIP_COLS = ("skill_diff_mean_v3", "skill_diff_mean_v3_1")


def _join_skill(fights):
    return join_skill_stacked(fights)


def _augmented_skill_columns(X):
    return flip_signed_columns(X, V3_3_FLIP_COLS)


def build_model():
    """The v3.3 CatBoost config (early stopping on), for calibration work."""
    return CatBoostClassifier(**BASE_CATBOOST_PARAMS, early_stopping_rounds=100, random_seed=0)


from ufc_pred.paths import METRICS, MODELS
from ufc_pred.utils.time_splits import recency_weights, split


def _build():
    return CatBoostClassifier(
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


def _join_v3_only(fights: pd.DataFrame) -> pd.DataFrame:
    """v3-style join: scalar skill features as `skill_diff_mean` / `_std`."""
    sk = pd.read_parquet(SKILL_V3_PARQUET)
    sk["date"] = pd.to_datetime(sk["date"])
    return fights.merge(
        sk[["date", "R_fighter", "B_fighter", "skill_diff_mean", "skill_diff_std"]],
        on=["date", "R_fighter", "B_fighter"],
        how="left",
        validate="many_to_one",
    )


def _join_v3_1_only(fights: pd.DataFrame) -> pd.DataFrame:
    """v3.1-style join: time-varying skill features as `skill_diff_mean` / `_std`."""
    sk = pd.read_parquet(SKILL_V3_1_PARQUET)
    sk["date"] = pd.to_datetime(sk["date"])
    return fights.merge(
        sk[["date", "R_fighter", "B_fighter", "skill_diff_mean", "skill_diff_std"]],
        on=["date", "R_fighter", "B_fighter"],
        how="left",
        validate="many_to_one",
    )


def _join_v3_3(fights: pd.DataFrame) -> pd.DataFrame:
    """v3.3-style join: both skill versions, suffixed."""
    sk_v3 = pd.read_parquet(SKILL_V3_PARQUET).rename(
        columns={"skill_diff_mean": "skill_diff_mean_v3", "skill_diff_std": "skill_diff_std_v3"}
    )
    sk_v3["date"] = pd.to_datetime(sk_v3["date"])
    sk_v3_1 = pd.read_parquet(SKILL_V3_1_PARQUET).rename(
        columns={"skill_diff_mean": "skill_diff_mean_v3_1", "skill_diff_std": "skill_diff_std_v3_1"}
    )
    sk_v3_1["date"] = pd.to_datetime(sk_v3_1["date"])
    f = fights.merge(
        sk_v3[["date", "R_fighter", "B_fighter", "skill_diff_mean_v3", "skill_diff_std_v3"]],
        on=["date", "R_fighter", "B_fighter"],
        how="left",
        validate="many_to_one",
    )
    f = f.merge(
        sk_v3_1[["date", "R_fighter", "B_fighter", "skill_diff_mean_v3_1", "skill_diff_std_v3_1"]],
        on=["date", "R_fighter", "B_fighter"],
        how="left",
        validate="many_to_one",
    )
    return f


def _augment_for(version: str, X: pd.DataFrame) -> pd.DataFrame:
    """Sign-flip skill_diff_mean column(s) on the augmented half."""
    n = len(X) // 2
    if len(X) != 2 * n:
        return X
    X = X.copy()
    cols_to_flip = (
        ["skill_diff_mean_v3", "skill_diff_mean_v3_1"] if version == "v3_3" else ["skill_diff_mean"]
    )
    second = X.index[n:]
    for c in cols_to_flip:
        if c in X.columns:
            X.loc[second, c] = -X.loc[second, c]
    return X


def train_one(version: str, joiner) -> dict:
    fights = pd.read_parquet(HISTORY_PARQUET)
    fights = fights[fights["Winner"].isin(["Red", "Blue"])].copy()
    fights["date"] = pd.to_datetime(fights["date"])
    fights = joiner(fights)
    splits = split(fights)

    X_train, y_train, d_train, cat_features = prepare(splits.train, augment_symmetry=True, one_hot=False)
    X_train = _augment_for(version, X_train)
    X_val, y_val, _, _ = prepare(splits.val, augment_symmetry=False, one_hot=False)
    X_val = X_val.reindex(columns=X_train.columns, fill_value=None)
    for c in cat_features:
        X_val[c] = X_val[c].fillna("__missing__").astype(str)

    sample_weight = recency_weights(d_train)
    pool = Pool(X_train, y_train, cat_features=cat_features, weight=sample_weight)
    m = _build()
    m.fit(pool, verbose=False)

    p_val = m.predict_proba(X_val)[:, 1]
    val_metrics = evaluate(np.asarray(y_val), p_val, label="val")

    sub = splits.val.dropna(subset=["R_odds", "B_odds"])
    market_val = evaluate(
        (sub["Winner"] == "Red").astype(int).to_numpy(),
        market_no_vig_prob_red(sub),
        label="val_market",
    )

    val_df = splits.val.reset_index(drop=True)
    y_red_val = np.asarray(y_val)
    bet_results = {}
    for label, use_no_vig, fee in [
        ("sportsbook_with_vig", False, 0.0),
        ("no_vig_no_fee", True, 0.0),
        ("kalshi_like", True, 0.07),
    ]:
        r = evaluate_bets(
            p_val,
            y_red_val,
            val_df["R_odds"],
            val_df["B_odds"],
            edge_threshold=0.05,
            fee_rate=fee,
            use_no_vig=use_no_vig,
        )
        bet_results[label] = {
            "roi_pct": r.roi_pct,
            "n_bets": r.n_bets,
            "ci95": list(r.ci95_roi_pct),
            "sharpe": r.sharpe,
            "mean_ev_pct": r.mean_ev_pct,
        }

    save_name = f"{version}_catboost_full2000"
    model_path = MODELS / f"{save_name}.joblib"
    joblib.dump(
        {
            "model": m,
            "columns": list(X_train.columns),
            "one_hot": False,
            "cat_features": cat_features,
        },
        model_path,
    )

    run = {
        "version": save_name,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "note": "Retrained without early stopping (full 2000 iterations) for "
        "fair comparison with v7 ensemble. seed=0.",
        "metrics": {"val": val_metrics, "val_market": market_val},
        "betting": bet_results,
        "model_path": str(model_path),
    }
    (METRICS / f"{save_name}.json").write_text(json.dumps(run, indent=2, default=float))
    return run


def main():
    rows = []
    for version, joiner in [
        ("v3", _join_v3_only),
        ("v3_1", _join_v3_1_only),
        ("v3_3", _join_v3_3),
    ]:
        print(f"\n=== Retraining {version} (no early stopping, seed=0) ===")
        run = train_one(version, joiner)
        m = run["metrics"]["val"]
        b = run["betting"]
        row = {
            "version": run["version"],
            "log_loss": m["log_loss"],
            "ece": m["ece"],
            "brier": m["brier"],
            "roi_sportsbook": b["sportsbook_with_vig"]["roi_pct"],
            "roi_no_vig": b["no_vig_no_fee"]["roi_pct"],
            "roi_kalshi": b["kalshi_like"]["roi_pct"],
            "n_bets_kalshi": b["kalshi_like"]["n_bets"],
        }
        rows.append(row)
        print(
            f"  log_loss={m['log_loss']:.4f}  ECE={m['ece']:.4f}  "
            f"ROI(kalshi)={b['kalshi_like']['roi_pct']:+.2f}%  "
            f"n_bets={b['kalshi_like']['n_bets']}"
        )
    print("\n=== Summary ===")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:+.4f}"))


if __name__ == "__main__":
    main()

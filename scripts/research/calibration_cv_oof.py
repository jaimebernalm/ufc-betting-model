"""CV OOF + isotonic calibration on v3.3.

Time-respecting k-fold CV on train:
  fold k: train on date < cut[k], predict on cut[k] <= date < cut[k+1]
Concatenate OOF predictions, fit isotonic on (oof_pred, y_train), apply to
val predictions from the FULL-train v3.3.

This is the standard sports-modeling calibration approach. ~5 CatBoost retrains.
"""

from __future__ import annotations

import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from ufc_pred.backtest.bet_eval import evaluate_bets
from ufc_pred.backtest.metrics import evaluate
from ufc_pred.calibration.methods import IsotonicCalibration, PlattScaling, TemperatureScaling
from ufc_pred.features.joins import flip_signed_columns, join_skill_stacked
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

# Time-respecting fold cutoffs on train (2010-03-21 → 2022-12-31).
# Each fold trains on dates < cut[k], predicts on cut[k] <= date < cut[k+1].
FOLD_CUTOFFS = [
    pd.Timestamp("2016-01-01"),
    pd.Timestamp("2018-01-01"),
    pd.Timestamp("2020-01-01"),
    pd.Timestamp("2021-07-01"),
    pd.Timestamp("2022-12-31"),
]


def _train_fold(train_df: pd.DataFrame, cat_features: list[str], reference_date: pd.Timestamp):
    """Train v3.3 on a subset of fights, return the model + the column order."""
    X, y, dates, _ = prepare(train_df, augment_symmetry=True, one_hot=False)
    X = _augmented_skill_columns(X)
    sw = recency_weights(dates, reference_date=reference_date)
    pool = Pool(X, y, cat_features=cat_features, weight=sw)
    m = build_model()
    m.fit(pool, verbose=False)
    return m, list(X.columns)


def _predict_fold(model: CatBoostClassifier, cols: list[str], cat_features: list[str], df: pd.DataFrame):
    X, y, _, _ = prepare(df, augment_symmetry=False, one_hot=False)
    X = X.reindex(columns=cols, fill_value=None)
    for c in cat_features:
        X[c] = X[c].fillna("__missing__").astype(str)
    return model.predict_proba(X)[:, 1], np.asarray(y)


def main():
    fights = pd.read_parquet(HISTORY_PARQUET)
    fights = fights[fights["Winner"].isin(["Red", "Blue"])].copy()
    fights = _join_skill(fights)
    fights["date"] = pd.to_datetime(fights["date"])
    splits = split(fights)
    train = splits.train.copy()
    val = splits.val.reset_index(drop=True)

    # Determine cat_features once.
    _, _, _, cat_features = prepare(train.head(100), augment_symmetry=False, one_hot=False)

    # OOF predictions on train.
    oof_preds, oof_y, oof_dates = [], [], []
    for i in range(len(FOLD_CUTOFFS) - 1):
        cut_lo = FOLD_CUTOFFS[i]
        cut_hi = FOLD_CUTOFFS[i + 1]
        fold_train = train[train["date"] < cut_lo]
        fold_test = train[(train["date"] >= cut_lo) & (train["date"] < cut_hi)]
        if len(fold_test) == 0:
            continue
        print(
            f"fold {i}: train n={len(fold_train)} (≤{cut_lo.date()})  "
            f"test n={len(fold_test)} ({cut_lo.date()}–{cut_hi.date()})"
        )
        model, cols = _train_fold(fold_train, cat_features, reference_date=cut_lo - pd.Timedelta(days=1))
        p, y = _predict_fold(model, cols, cat_features, fold_test)
        oof_preds.append(p)
        oof_y.append(y)
        oof_dates.append(fold_test["date"].to_numpy())

    p_oof = np.concatenate(oof_preds)
    y_oof = np.concatenate(oof_y)
    print(
        f"\nOOF set: n={len(p_oof)}  pred mean={p_oof.mean():.3f}  pred range [{p_oof.min():.3f}, {p_oof.max():.3f}]"
    )

    # Val predictions from FULL-train v3.3 (the actual deployed model).
    payload = joblib.load(MODELS / "v3_3_catboost_stacked_skill.joblib")
    full_model, cols, cat_features = (
        payload["model"],
        payload["columns"],
        payload.get("cat_features", []),
    )
    X_val, y_val, _, _ = prepare(val, augment_symmetry=False, one_hot=False)
    X_val = X_val.reindex(columns=cols, fill_value=None)
    for c in cat_features:
        X_val[c] = X_val[c].fillna("__missing__").astype(str)
    p_val = full_model.predict_proba(X_val)[:, 1]
    y_val_arr = np.asarray(y_val)

    print(
        f"val: n={len(p_val)}  pred mean={p_val.mean():.3f}  pred range [{p_val.min():.3f}, {p_val.max():.3f}]"
    )

    # Fit each calibrator on OOF, apply to val.
    def _eval(name, p):
        base = evaluate(y_val_arr, p, label="val")
        kalshi = evaluate_bets(
            p,
            y_val_arr,
            val["R_odds"],
            val["B_odds"],
            edge_threshold=0.05,
            fee_rate=0.07,
            use_no_vig=True,
        )
        sb = evaluate_bets(
            p,
            y_val_arr,
            val["R_odds"],
            val["B_odds"],
            edge_threshold=0.05,
            fee_rate=0.0,
            use_no_vig=False,
        )
        return {
            "method": name,
            "log_loss": base["log_loss"],
            "ece": base["ece"],
            "brier": base["brier"],
            "roi_kalshi": kalshi.roi_pct,
            "ci95_lo": kalshi.ci95_roi_pct[0],
            "ci95_hi": kalshi.ci95_roi_pct[1],
            "mean_ev_kalshi": kalshi.mean_ev_pct,
            "n_bets_kalshi": kalshi.n_bets,
            "roi_sportsbook": sb.roi_pct,
        }

    rows = [_eval("uncalibrated (full v3.3)", p_val)]
    fitted = {}
    for name, cls in [
        ("temperature", TemperatureScaling),
        ("platt", PlattScaling),
        ("isotonic", IsotonicCalibration),
    ]:
        cal = cls().fit(p_oof, y_oof)
        rows.append(_eval(f"{name} (CV-OOF)", cal.transform(p_val)))
        fitted[name] = cal

    print()
    print(f"Temperature T (CV-OOF): {fitted['temperature'].T:.4f}")
    print(f"Platt a, b   (CV-OOF): {fitted['platt'].a:.4f}, {fitted['platt'].b:+.4f}")
    print()
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    out_path = METRICS / "calibration_cv_oof.json"
    out_path.write_text(
        json.dumps(
            {
                "ran_at": datetime.now().isoformat(timespec="seconds"),
                "fold_cutoffs": [str(t.date()) for t in FOLD_CUTOFFS],
                "n_oof": int(len(p_oof)),
                "n_val": int(len(p_val)),
                "results": rows,
                "fitted_params": {
                    "temperature": {"T": fitted["temperature"].T},
                    "platt": {"a": fitted["platt"].a, "b": fitted["platt"].b},
                },
            },
            indent=2,
            default=float,
        )
    )
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

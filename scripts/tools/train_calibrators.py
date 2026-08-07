"""Phase 7 (calibration) — try 3 calibration methods on v3.3.

Protocol (Option B from the calibration plan):
  1. Carve out a CALIBRATION tail from the end of train (default: last 6 months
     of train = 2022-07-01 to 2022-12-31). This is the "cal_set".
  2. Refit v3.3 on the truncated train (train minus cal_set).
  3. Predict on cal_set → these are the calibrator's fit inputs.
  4. Fit each calibration method on (cal_predictions, cal_labels).
  5. Predict on val using the SAME truncated v3.3 (clean comparison).
  6. Apply each calibrator to val predictions; report log_loss, ECE, ROI@5% in
     each market scenario.

We use a TRUNCATED v3.3 (not the original full-train v3.3) for val predictions
so the calibrators are fit on predictions from the same model architecture they're
being applied to. The original full-train v3.3's val numbers remain in
artifacts/metrics/v3_3_catboost_stacked_skill.json untouched.

The final winner gets wrapped into baseline_v7 separately (uses full-train v3.3
with the calibrator fit on its in-train last-6-months predictions).

Writes: artifacts/metrics/calibration_eval.json
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from ufc_pred.backtest.bet_eval import evaluate_bets
from ufc_pred.backtest.metrics import evaluate
from ufc_pred.calibration.methods import build as build_calibrator
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


from ufc_pred.paths import METRICS
from ufc_pred.utils.time_splits import recency_weights, split

CAL_CUTOFF = pd.Timestamp("2022-07-01")  # cal tail = [2022-07-01, 2022-12-31]
METHODS = ["temperature", "platt", "isotonic"]


def _truncated_train_predict(
    fights: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Train v3.3 on date < CAL_CUTOFF, predict on cal_tail and on val.

    Returns:
        p_cal: predictions on the cal tail (used to fit calibrators)
        y_cal: labels on the cal tail
        df_cal: rows of fights in the cal tail (date, R_fighter, B_fighter, odds, Winner)
        p_val: predictions on val
        y_val: val labels
        df_val: rows of fights in val
    """
    splits_full = split(fights)
    train = splits_full.train.copy()

    # Split train into truncated_train + cal_tail by date.
    truncated_train = train[train["date"] < CAL_CUTOFF].copy()
    cal_tail = train[train["date"] >= CAL_CUTOFF].copy()
    val = splits_full.val.copy()

    print(
        f"truncated train: {len(truncated_train)} fights "
        f"({truncated_train['date'].min().date()} → {truncated_train['date'].max().date()})"
    )
    print(
        f"cal tail:        {len(cal_tail)} fights "
        f"({cal_tail['date'].min().date()} → {cal_tail['date'].max().date()})"
    )
    print(f"val:             {len(val)} fights ({val['date'].min().date()} → {val['date'].max().date()})")

    # Prepare X/y for each.
    X_trunc, y_trunc, d_trunc, cat_features = prepare(truncated_train, augment_symmetry=True, one_hot=False)
    X_trunc = _augmented_skill_columns(X_trunc)

    X_cal, y_cal, _, _ = prepare(cal_tail, augment_symmetry=False, one_hot=False)
    X_cal = X_cal.reindex(columns=X_trunc.columns, fill_value=None)
    for c in cat_features:
        X_cal[c] = X_cal[c].fillna("__missing__").astype(str)

    X_val, y_val, _, _ = prepare(val, augment_symmetry=False, one_hot=False)
    X_val = X_val.reindex(columns=X_trunc.columns, fill_value=None)
    for c in cat_features:
        X_val[c] = X_val[c].fillna("__missing__").astype(str)

    # Train v3.3 architecture on truncated train. NOTE: we use cal_tail as the
    # eval_set for early stopping — this is conventional and doesn't leak labels
    # into the calibrator (early stopping decides #trees, not probabilities).
    sample_weight = recency_weights(d_trunc, reference_date=CAL_CUTOFF - pd.Timedelta(days=1))
    train_pool = Pool(X_trunc, y_trunc, cat_features=cat_features, weight=sample_weight)
    cal_pool = Pool(X_cal, y_cal, cat_features=cat_features)
    model = build_model()
    model.fit(train_pool, eval_set=cal_pool, use_best_model=True)
    print(f"truncated v3.3 best_iteration: {model.get_best_iteration()}")

    p_cal = model.predict_proba(X_cal)[:, 1]
    p_val = model.predict_proba(X_val)[:, 1]
    return (
        p_cal,
        np.asarray(y_cal),
        cal_tail.reset_index(drop=True),
        p_val,
        np.asarray(y_val),
        val.reset_index(drop=True),
    )


def _eval_one(name: str, p_val: np.ndarray, y_val: np.ndarray, df_val: pd.DataFrame) -> dict:
    """Score a probability vector on val: log_loss, ECE, Brier, accuracy, plus
    ROI@5% in the Kalshi-like scenario."""
    base = evaluate(y_val, p_val, label="val")
    R_odds, B_odds = df_val["R_odds"], df_val["B_odds"]
    kalshi = evaluate_bets(
        p_val,
        y_val,
        R_odds,
        B_odds,
        edge_threshold=0.05,
        fee_rate=0.07,
        use_no_vig=True,
    )
    sportsbook = evaluate_bets(
        p_val,
        y_val,
        R_odds,
        B_odds,
        edge_threshold=0.05,
        fee_rate=0.0,
        use_no_vig=False,
    )
    no_vig = evaluate_bets(
        p_val,
        y_val,
        R_odds,
        B_odds,
        edge_threshold=0.05,
        fee_rate=0.0,
        use_no_vig=True,
    )
    return {
        "method": name,
        "log_loss": base["log_loss"],
        "brier": base["brier"],
        "ece": base["ece"],
        "accuracy": base["accuracy_argmax"],
        "roi_pct_sportsbook": sportsbook.roi_pct,
        "roi_pct_no_vig": no_vig.roi_pct,
        "roi_pct_kalshi": kalshi.roi_pct,
        "mean_ev_pct_kalshi": kalshi.mean_ev_pct,
        "n_bets_kalshi": kalshi.n_bets,
        "ci95_low_kalshi": kalshi.ci95_roi_pct[0],
        "ci95_high_kalshi": kalshi.ci95_roi_pct[1],
    }


def main():
    # Load + join skill features for v3.3.
    fights = pd.read_parquet(HISTORY_PARQUET)
    fights = fights[fights["Winner"].isin(["Red", "Blue"])].copy()
    fights = _join_skill(fights)
    fights["date"] = pd.to_datetime(fights["date"])

    p_cal, y_cal, _, p_val, y_val, df_val = _truncated_train_predict(fights)
    print(f"\np_cal range [{p_cal.min():.3f}, {p_cal.max():.3f}]  mean {p_cal.mean():.3f}")
    print(f"p_val range [{p_val.min():.3f}, {p_val.max():.3f}]  mean {p_val.mean():.3f}")

    # Baseline: uncalibrated truncated-v3.3 predictions on val.
    rows = [_eval_one("uncalibrated (truncated v3.3)", p_val, y_val, df_val)]
    fitted = {}

    for name in METHODS:
        cal = build_calibrator(name).fit(p_cal, y_cal)
        p_val_cal = cal.transform(p_val)
        rows.append(_eval_one(name, p_val_cal, y_val, df_val))
        fitted[name] = cal

    # Diagnostic — report calibrator params where applicable.
    print()
    print(f"Temperature T: {fitted['temperature'].T:.3f}  (>1 = squeezed toward 0.5, the expected direction)")
    print(f"Platt a, b:    {fitted['platt'].a:.3f}, {fitted['platt'].b:+.3f}")

    df = pd.DataFrame(rows)
    print("\n=== Calibration eval on val ===")
    show = df[
        [
            "method",
            "log_loss",
            "ece",
            "brier",
            "accuracy",
            "roi_pct_kalshi",
            "ci95_low_kalshi",
            "ci95_high_kalshi",
            "mean_ev_pct_kalshi",
            "n_bets_kalshi",
            "roi_pct_sportsbook",
            "roi_pct_no_vig",
        ]
    ].copy()
    print(show.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    out_path = METRICS / "calibration_eval.json"
    out_path.write_text(
        json.dumps(
            {
                "ran_at": datetime.now().isoformat(timespec="seconds"),
                "cal_cutoff": str(CAL_CUTOFF.date()),
                "n_cal": int(len(y_cal)),
                "n_val": int(len(y_val)),
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

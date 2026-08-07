"""DIAGNOSTIC ONLY — calibration ceiling experiment.

Fits each calibrator on VAL and evaluates on VAL. This is by-design leaky:
the numbers are NOT comparable to any other model and MUST NOT be used for
keep-or-kill. The question we're answering is purely:

  "If we had perfect knowledge of how to calibrate on val, what's the best
   ROI we could get? Is calibration the right lever, or is the failure mode
   somewhere else?"

If even this best-case calibration produces only a tiny ROI improvement,
calibration cannot save this model on this val set and we should look
elsewhere (more data, different model architecture, conformal wrapper, etc.).
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

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


from ufc_pred.paths import MODELS
from ufc_pred.utils.time_splits import split

METHODS = ["temperature", "platt", "isotonic"]


def _eval(name: str, p: np.ndarray, y: np.ndarray, df: pd.DataFrame) -> dict:
    base = evaluate(y, p, label="val")
    kalshi = evaluate_bets(
        p, y, df["R_odds"], df["B_odds"], edge_threshold=0.05, fee_rate=0.07, use_no_vig=True
    )
    sb = evaluate_bets(p, y, df["R_odds"], df["B_odds"], edge_threshold=0.05, fee_rate=0.0, use_no_vig=False)
    return {
        "method": name,
        "log_loss": base["log_loss"],
        "ece": base["ece"],
        "brier": base["brier"],
        "roi_kalshi": kalshi.roi_pct,
        "mean_ev_kalshi": kalshi.mean_ev_pct,
        "n_bets_kalshi": kalshi.n_bets,
        "roi_sportsbook": sb.roi_pct,
    }


def main():
    # Load FULL-train v3.3 (the actual champion model).
    payload = joblib.load(MODELS / "v3_3_catboost_stacked_skill.joblib")
    model = payload["model"]
    cols = payload["columns"]
    cat_features = payload.get("cat_features", [])

    # Build val features.
    fights = pd.read_parquet(HISTORY_PARQUET)
    fights = fights[fights["Winner"].isin(["Red", "Blue"])].copy()
    fights = _join_skill(fights)
    fights["date"] = pd.to_datetime(fights["date"])
    splits = split(fights)
    val = splits.val.reset_index(drop=True)

    X_val, y_val, _, _ = prepare(val, augment_symmetry=False, one_hot=False)
    X_val = X_val.reindex(columns=cols, fill_value=None)
    for c in cat_features:
        X_val[c] = X_val[c].fillna("__missing__").astype(str)

    p_val = model.predict_proba(X_val)[:, 1]
    y_val_arr = np.asarray(y_val)

    rows = [_eval("uncalibrated (full v3.3)", p_val, y_val_arr, val)]
    fitted_params = {}
    for name in METHODS:
        cal = build_calibrator(name).fit(p_val, y_val_arr)  # LEAK — fits on the same val it'll be tested on
        p_cal = cal.transform(p_val)
        rows.append(_eval(f"{name} (LEAK: fit on val)", p_cal, y_val_arr, val))
        if name == "temperature":
            fitted_params["T"] = cal.T
        elif name == "platt":
            fitted_params["a"] = cal.a
            fitted_params["b"] = cal.b

    print("\n*** DIAGNOSTIC — these numbers are leaky, do not use for keep-or-kill ***\n")
    print(f"Temperature T:  {fitted_params['T']:.4f}  (>1 = squeeze toward 0.5)")
    print(f"Platt a, b:     {fitted_params['a']:.4f}, {fitted_params['b']:+.4f}")
    print()
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    # Compare to v3.3 (full train) uncalibrated original numbers from leaderboard:
    print("\n(For reference, full v3.3 uncalibrated on bet_eval leaderboard is ROI Kalshi −1.40%.)")


if __name__ == "__main__":
    main()

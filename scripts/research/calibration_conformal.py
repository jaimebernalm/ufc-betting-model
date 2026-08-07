"""Conformal-style experiments for v3.3 on val.

Two complementary approaches, both addressing overconfidence at the *betting
decision* level rather than rewriting the predictions:

  (A) Seed-ensemble: train K v3.3 with different random_seed, use std across
      predictions as uncertainty. Betting rule uses `mean - z*std` instead of
      `mean` for the EV calculation.

  (B) Doubt threshold: simplest possible heuristic — restrict bets to fights
      with model_prob in [0.3, 0.7]. The extreme high-confidence predictions
      are where overconfidence concentrates; cut them off entirely.
"""

from __future__ import annotations

import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from ufc_pred.backtest.bet_eval import american_to_decimal, evaluate_bets
from ufc_pred.backtest.metrics import evaluate
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

K_SEEDS = 5
DOUBT_RANGE = (0.30, 0.70)
Z_LOWER = 1.0  # 1-sigma lower bound for conformal


def _train_seeded(train_df: pd.DataFrame, cat_features: list[str], seed: int):
    X, y, dates, _ = prepare(train_df, augment_symmetry=True, one_hot=False)
    X = _augmented_skill_columns(X)
    sw = recency_weights(dates)
    pool = Pool(X, y, cat_features=cat_features, weight=sw)
    m = build_model()
    m.set_params(random_seed=seed)
    m.fit(pool, verbose=False)
    return m, list(X.columns)


def _bet_with_lower_bound(
    p_mean: np.ndarray,
    p_lower: np.ndarray,
    y_red: np.ndarray,
    df: pd.DataFrame,
    edge_threshold: float = 0.05,
    fee_rate: float = 0.07,
    use_no_vig: bool = True,
):
    """Like evaluate_bets but uses p_lower (mean - z*std) for the EV decision.

    Realized PnL still uses actual outcomes; the only difference vs evaluate_bets
    is which probability is used in the threshold check.
    """
    dec_R = american_to_decimal(df["R_odds"])
    dec_B = american_to_decimal(df["B_odds"])
    valid = ~(np.isnan(dec_R) | np.isnan(dec_B))

    if use_no_vig:
        from ufc_pred.backtest.metrics import american_to_implied_prob

        p_r_imp = american_to_implied_prob(df["R_odds"])
        p_b_imp = american_to_implied_prob(df["B_odds"])
        total = p_r_imp + p_b_imp
        dec_R = 1.0 / (p_r_imp / total)
        dec_B = 1.0 / (p_b_imp / total)

    eff_R = 1.0 + (1.0 - fee_rate) * (dec_R - 1.0)
    eff_B = 1.0 + (1.0 - fee_rate) * (dec_B - 1.0)

    # EV by side using LOWER BOUND probability — conservative
    p_R_low = p_lower
    (1.0 - p_mean)  # blue side: 1 - mean is its mean prob; for blue lower bound use 1 - upper of red
    # symmetric: if red has mean m and std s, blue mean = 1-m, blue std = s, blue lower = (1-m) - z*s
    # so p_B_lower = 1 - (p_mean + z*sigma) which equals (1 - p_mean) - z*sigma = p_B_low - z*sigma
    # Compute sigma from p_lower
    sigma = p_mean - p_lower  # since p_lower = p_mean - z*sigma → sigma here already has z baked in
    p_B_lower = (1 - p_mean) - sigma

    ev_R = p_R_low * eff_R - 1.0
    ev_B = p_B_lower * eff_B - 1.0

    bet_red = ev_R >= ev_B
    chosen_ev = np.where(bet_red, ev_R, ev_B)
    chosen_dec = np.where(bet_red, eff_R, eff_B)
    bets = valid & (chosen_ev > edge_threshold)

    won = np.where(bet_red, y_red == 1, y_red == 0)
    realized = np.where(won, chosen_dec - 1.0, -1.0)

    n_bets = int(bets.sum())
    if n_bets == 0:
        return {"n_bets": 0, "roi_pct": 0.0, "mean_ev_pct": 0.0, "hit_rate": 0.0}

    bet_realized = realized[bets]
    bet_won = won[bets]
    return {
        "n_bets": n_bets,
        "roi_pct": float(bet_realized.mean() * 100),
        "mean_ev_pct": float(chosen_ev[bets].mean() * 100),
        "hit_rate": float(bet_won.mean()),
        "n_red": int((bets & bet_red).sum()),
        "n_blue": int((bets & ~bet_red).sum()),
    }


def main():
    fights = pd.read_parquet(HISTORY_PARQUET)
    fights = fights[fights["Winner"].isin(["Red", "Blue"])].copy()
    fights = _join_skill(fights)
    fights["date"] = pd.to_datetime(fights["date"])
    splits = split(fights)
    train, val = splits.train, splits.val.reset_index(drop=True)
    _, _, _, cat_features = prepare(train.head(100), augment_symmetry=False, one_hot=False)

    # === (A) Seed-ensemble ===
    print(f"Training {K_SEEDS} v3.3 models with different seeds...")
    val_preds = []
    for s in range(K_SEEDS):
        model, cols = _train_seeded(train, cat_features, seed=s)
        X_val, y_val, _, _ = prepare(val, augment_symmetry=False, one_hot=False)
        X_val = X_val.reindex(columns=cols, fill_value=None)
        for c in cat_features:
            X_val[c] = X_val[c].fillna("__missing__").astype(str)
        val_preds.append(model.predict_proba(X_val)[:, 1])
        print(f"  seed {s}: best_iter={model.get_best_iteration()}")
    y_val_arr = np.asarray(y_val)
    preds_stack = np.vstack(val_preds)
    p_mean = preds_stack.mean(axis=0)
    p_std = preds_stack.std(axis=0)
    p_lower = p_mean - Z_LOWER * p_std

    print(
        f"\nSeed-ensemble stats: pred_mean range [{p_mean.min():.3f}, {p_mean.max():.3f}], "
        f"p_std mean={p_std.mean():.4f} max={p_std.max():.4f}"
    )

    base = evaluate(y_val_arr, p_mean, label="val")
    kalshi_mean = evaluate_bets(
        p_mean,
        y_val_arr,
        val["R_odds"],
        val["B_odds"],
        edge_threshold=0.05,
        fee_rate=0.07,
        use_no_vig=True,
    )
    kalshi_lower = _bet_with_lower_bound(
        p_mean, p_lower, y_val_arr, val, edge_threshold=0.05, fee_rate=0.07, use_no_vig=True
    )

    rows = []
    rows.append(
        {
            "approach": "ensemble mean (5 seeds, no conformal)",
            "log_loss": base["log_loss"],
            "ece": base["ece"],
            "roi_kalshi": kalshi_mean.roi_pct,
            "n_bets": kalshi_mean.n_bets,
            "mean_ev_kalshi": kalshi_mean.mean_ev_pct,
        }
    )
    rows.append(
        {
            "approach": f"ensemble + conformal (z={Z_LOWER} lower bound)",
            "log_loss": base["log_loss"],
            "ece": base["ece"],
            "roi_kalshi": kalshi_lower["roi_pct"],
            "n_bets": kalshi_lower["n_bets"],
            "mean_ev_kalshi": kalshi_lower["mean_ev_pct"],
        }
    )

    # === (B) Doubt threshold using full-train v3.3 predictions ===
    payload = joblib.load(MODELS / "v3_3_catboost_stacked_skill.joblib")
    full_model, cols, cat_features = (
        payload["model"],
        payload["columns"],
        payload.get("cat_features", []),
    )
    X_val, _, _, _ = prepare(val, augment_symmetry=False, one_hot=False)
    X_val = X_val.reindex(columns=cols, fill_value=None)
    for c in cat_features:
        X_val[c] = X_val[c].fillna("__missing__").astype(str)
    p_full = full_model.predict_proba(X_val)[:, 1]

    # Apply doubt threshold: zero out fights where prob is outside DOUBT_RANGE.
    # Mechanically: set p_red to a "no-bet zone" value (we'll just skip them).
    in_doubt = (p_full >= DOUBT_RANGE[0]) & (p_full <= DOUBT_RANGE[1])
    # For evaluate_bets to skip non-doubt fights, replace their prob with 0.5
    # (which produces zero edge and won't clear threshold).
    p_doubt = np.where(in_doubt, p_full, 0.5)
    kalshi_doubt = evaluate_bets(
        p_doubt,
        y_val_arr,
        val["R_odds"],
        val["B_odds"],
        edge_threshold=0.05,
        fee_rate=0.07,
        use_no_vig=True,
    )
    sb_doubt = evaluate_bets(
        p_doubt,
        y_val_arr,
        val["R_odds"],
        val["B_odds"],
        edge_threshold=0.05,
        fee_rate=0.0,
        use_no_vig=False,
    )
    rows.append(
        {
            "approach": f"doubt threshold p∈[{DOUBT_RANGE[0]}, {DOUBT_RANGE[1]}]",
            "log_loss": evaluate(y_val_arr, p_full)["log_loss"],  # log_loss unaffected by doubt
            "ece": evaluate(y_val_arr, p_full)["ece"],
            "roi_kalshi": kalshi_doubt.roi_pct,
            "n_bets": kalshi_doubt.n_bets,
            "mean_ev_kalshi": kalshi_doubt.mean_ev_pct,
        }
    )

    # Reference uncal full v3.3 (kalshi 5% edge): -1.40% (from leaderboard)
    rows.insert(
        0,
        {
            "approach": "uncalibrated full v3.3 (reference)",
            "log_loss": evaluate(y_val_arr, p_full)["log_loss"],
            "ece": evaluate(y_val_arr, p_full)["ece"],
            "roi_kalshi": -1.40,  # from prior bet_eval run
            "n_bets": 320,
            "mean_ev_kalshi": 37.08,
        },
    )

    df = pd.DataFrame(rows)
    print("\n=== Conformal / doubt experiments ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    # Also break out doubt threshold by side counts.
    print(
        f"\nDoubt-threshold detail: kalshi n_bets={kalshi_doubt.n_bets} "
        f"(red {kalshi_doubt.n_bets_red}, blue {kalshi_doubt.n_bets_blue})  "
        f"sportsbook roi={sb_doubt.roi_pct:+.2f}%"
    )
    print(f"In-doubt fraction of val: {in_doubt.mean():.3f}")

    out_path = METRICS / "calibration_conformal.json"
    out_path.write_text(
        json.dumps(
            {
                "ran_at": datetime.now().isoformat(timespec="seconds"),
                "k_seeds": K_SEEDS,
                "z_lower": Z_LOWER,
                "doubt_range": list(DOUBT_RANGE),
                "results": rows,
                "seed_predictions_std_mean": float(p_std.mean()),
            },
            indent=2,
            default=float,
        )
    )
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

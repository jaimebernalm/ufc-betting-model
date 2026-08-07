"""Rank every trained model by betting EV against the BestFightOdds market on val.

For each model in artifacts/models/*.joblib:
  - Build the val feature matrix matching that model's expected columns.
  - Predict P(Red wins) on val.
  - Run bet_eval.evaluate_bets at multiple edge thresholds.
  - Collect: n_bets, ROI%, mean_EV%, Sharpe, hit_rate, 95% bootstrap CI.

Outputs:
  - Console: headline table at threshold=0.05 + threshold sweep per model.
  - artifacts/metrics/bet_eval.json: machine-readable for the notebook.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ufc_pred.backtest.bet_eval import evaluate_bets, evaluate_bets_kelly, evaluate_bets_sweep
from ufc_pred.features.static_v1 import prepare
from ufc_pred.ingest.kaggle_mdabbert import HISTORY_PARQUET
from ufc_pred.paths import METRICS, MODELS
from ufc_pred.utils.time_splits import split

# Per-model recipe: which extra feature parquets to join before calling prepare().
# Skill columns are renamed depending on what the model expects.
SKILL_V3 = Path("data/processed/skill_features_v3.parquet")
SKILL_V3_1 = Path("data/processed/skill_features_v3_1.parquet")

THRESHOLDS = [0.0, 0.02, 0.03, 0.05, 0.075, 0.10]
HEADLINE_THRESHOLD = 0.05

# Kelly simulation knobs (PLAN.md §10.2 defaults).
KELLY_EDGE_THRESHOLD = 0.03  # PLAN.md §10.1: minimum 3% expected return
KELLY_FRACTION = 0.25  # quarter-Kelly
KELLY_MAX_BET = 0.02  # 2% hard cap per bet
KELLY_STARTING_BANKROLL = 1.0  # $1 → final bankroll is the growth factor

# Three market scenarios. Each tuple: (label, use_no_vig, fee_rate).
SCENARIOS = [
    ("sportsbook (with vig)", False, 0.00),  # BestFightOdds reality — hardest bar
    ("no-vig, no fee", True, 0.00),  # theoretical fair market
    ("kalshi-like (no vig, 7% fee)", True, 0.07),  # prediction market approximation
]


def _join_skill(fights: pd.DataFrame, parquet: Path, suffix: str = "") -> pd.DataFrame:
    sk = pd.read_parquet(parquet)
    sk["date"] = pd.to_datetime(sk["date"])
    rename = {}
    if suffix:
        rename = {
            "skill_diff_mean": f"skill_diff_mean{suffix}",
            "skill_diff_std": f"skill_diff_std{suffix}",
        }
        sk = sk.rename(columns=rename)
    cols = ["date", "R_fighter", "B_fighter"] + (
        list(rename.values()) if rename else ["skill_diff_mean", "skill_diff_std"]
    )
    return fights.merge(sk[cols], on=["date", "R_fighter", "B_fighter"], how="left", validate="many_to_one")


def _join_derived(fights: pd.DataFrame) -> pd.DataFrame:
    from ufc_pred.features.derived_v2 import compute as add_derived

    return add_derived(fights)


def build_val_frame_for(model_name: str, fights: pd.DataFrame) -> pd.DataFrame:
    """Return a fights-frame with the right extra columns joined for `model_name`."""
    f = fights.copy()
    f["date"] = pd.to_datetime(f["date"])

    if model_name in (
        "v3_catboost_skill",
        "v3_catboost_full2000",
        "v7_1_catboost_ensemble_v3",
        "v3_full2000_no_skill_corrupted",
    ):
        # corrupted wrapper still needs the v3 skill columns present (it forces
        # them to NaN internally; reindex would otherwise drop them silently).
        f = _join_skill(f, SKILL_V3)
    elif model_name in ("v3_1_catboost_timevarying_skill", "v3_1_catboost_full2000"):
        f = _join_skill(f, SKILL_V3_1)
    elif model_name == "v3_2_catboost_skill_derived":
        f = _join_skill(f, SKILL_V3)
        f = _join_derived(f)
    elif model_name in (
        "v3_3_catboost_stacked_skill",
        "v3_3_catboost_full2000",
        "v7_catboost_ensemble",
    ):
        f = _join_skill(f, SKILL_V3, suffix="_v3")
        f = _join_skill(f, SKILL_V3_1, suffix="_v3_1")
    elif model_name == "v2_catboost_derived":
        f = _join_derived(f)
    # v1_logreg / v1_1_catboost / v1_2_catboost_isotonic: no extras.

    return f


def predict_val(model_name: str, fights: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """Return (model_prob_red_for_val, val_dataframe). Val frame includes Winner/odds."""
    payload = joblib.load(MODELS / f"{model_name}.joblib")
    cols = payload["columns"]
    one_hot = payload.get("one_hot", True)
    cat_features = payload.get("cat_features", [])

    f = build_val_frame_for(model_name, fights)
    splits = split(f)
    X_val, y_val, _, _ = prepare(splits.val, augment_symmetry=False, one_hot=one_hot)
    X_val = X_val.reindex(columns=cols, fill_value=None if not one_hot else 0)
    if not one_hot:
        for c in cat_features:
            if c in X_val.columns:
                X_val[c] = X_val[c].fillna("__missing__").astype(str)

    if "models" in payload:
        # Ensemble — average across members.
        preds = [m.predict_proba(X_val)[:, 1] for m in payload["models"]]
        p = np.mean(np.vstack(preds), axis=0)
    elif "model" in payload:
        p = payload["model"].predict_proba(X_val)[:, 1]
    elif "ranker_path" in payload and "isotonic" in payload:
        ranker = joblib.load(payload["ranker_path"])["model"]
        raw = ranker.predict_proba(X_val)[:, 1]
        p = payload["isotonic"].transform(raw)
    else:
        raise KeyError(f"Unrecognized payload for {model_name}: {list(payload.keys())}")

    return p, splits.val.reset_index(drop=True)


def main():
    fights = pd.read_parquet(HISTORY_PARQUET)
    fights = fights[fights["Winner"].isin(["Red", "Blue"])].copy()
    fights["date"] = pd.to_datetime(fights["date"])

    model_files = sorted(MODELS.glob("*.joblib"))
    model_names = [p.stem for p in model_files]

    # Predict once per model.
    preds: dict[str, tuple[np.ndarray, pd.DataFrame]] = {}
    for name in model_names:
        try:
            p_val, val_df = predict_val(name, fights)
            preds[name] = (p_val, val_df)
        except Exception as e:
            print(f"[skip] {name}: {e}")

    all_results: dict[str, dict] = {}
    for scenario_label, use_no_vig, fee_rate in SCENARIOS:
        headline_rows = []
        sweep_rows = []
        kelly_rows = []
        for name, (p_val, val_df) in preds.items():
            y_red = (val_df["Winner"].to_numpy() == "Red").astype(int)
            R_odds, B_odds = val_df["R_odds"], val_df["B_odds"]
            head = evaluate_bets(
                p_val,
                y_red,
                R_odds,
                B_odds,
                edge_threshold=HEADLINE_THRESHOLD,
                fee_rate=fee_rate,
                use_no_vig=use_no_vig,
            )
            headline_rows.append({"model": name, **head.summary_row()})
            sweep = evaluate_bets_sweep(
                p_val,
                y_red,
                R_odds,
                B_odds,
                thresholds=THRESHOLDS,
                fee_rate=fee_rate,
                use_no_vig=use_no_vig,
            )
            sweep.insert(0, "model", name)
            sweep_rows.append(sweep)

            # Kelly bankroll simulation with PLAN.md §10.2 defaults.
            kelly = evaluate_bets_kelly(
                p_val,
                y_red,
                R_odds,
                B_odds,
                edge_threshold=KELLY_EDGE_THRESHOLD,
                fee_rate=fee_rate,
                use_no_vig=use_no_vig,
                kelly_fraction=KELLY_FRACTION,
                starting_bankroll=KELLY_STARTING_BANKROLL,
                max_bet_fraction=KELLY_MAX_BET,
            )
            kelly_rows.append({"model": name, **kelly})

        headline_df = pd.DataFrame(headline_rows).sort_values("roi_pct", ascending=False)
        sweep_df = pd.concat(sweep_rows, ignore_index=True)

        print(f"\n========== {scenario_label} ==========")
        print(f"=== Headline @ edge threshold = {HEADLINE_THRESHOLD:.0%} ===")
        show = headline_df[
            [
                "model",
                "n_bets",
                "roi_pct",
                "ci95_low",
                "ci95_high",
                "mean_ev_pct",
                "sharpe",
                "hit_rate",
            ]
        ].copy()
        print(show.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

        print("\n=== Threshold sweep — ROI % ===")
        pivot_roi = sweep_df.pivot(index="model", columns="edge_threshold", values="roi_pct")
        pivot_roi = pivot_roi.loc[headline_df["model"].tolist()]
        print(pivot_roi.to_string(float_format=lambda x: f"{x:+.2f}"))

        # Kelly summary (quarter-Kelly, 2% hard cap, 3% edge — PLAN.md §10.2).
        # Strip trajectories out of the display table; keep them in saved JSON.
        kelly_df = pd.DataFrame(
            [
                {
                    "model": r["model"],
                    "n_kelly_bets": r["n_bets"],
                    "final_bankroll": r["final_bankroll"],
                    "total_return_pct": r["total_return_pct"],
                    "max_drawdown_pct": r["max_drawdown_pct"],
                }
                for r in kelly_rows
            ]
        ).sort_values("total_return_pct", ascending=False)
        print("\n=== Kelly (quarter-Kelly, 2% cap, 3% edge), bankroll started at $1 ===")
        print(kelly_df.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

        all_results[scenario_label] = {
            "use_no_vig": use_no_vig,
            "fee_rate": fee_rate,
            "kelly": {
                "edge_threshold": KELLY_EDGE_THRESHOLD,
                "kelly_fraction": KELLY_FRACTION,
                "max_bet_fraction": KELLY_MAX_BET,
                "starting_bankroll": KELLY_STARTING_BANKROLL,
                "rows": kelly_rows,
            },
            "headline": headline_df.to_dict("records"),
            "sweep": sweep_df.to_dict("records"),
        }

    out_path = METRICS / "bet_eval.json"
    out_path.write_text(
        json.dumps(
            {
                "headline_threshold": HEADLINE_THRESHOLD,
                "thresholds": THRESHOLDS,
                "scenarios": all_results,
            },
            indent=2,
            default=float,
        )
    )
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

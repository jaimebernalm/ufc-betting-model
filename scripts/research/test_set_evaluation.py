"""TEST-SET EVALUATION (one-shot).

Pre-committed decision rule, pinned BEFORE any results are inspected.
Do not edit these constants after seeing test output — that would burn the
test set as an evaluation resource.

Models evaluated:
  - v3_catboost_full2000_trainval         (retrained on train+val)
  - v3_full2000_no_skill_corrupted_trainval (diagnostic wrapper)

Outputs:
  artifacts/metrics/test_set_evaluation.json
  console summary tables (flat-stake + 4x5 Kelly grid + verdict)
"""

from __future__ import annotations

# =========================================================================
# PRE-COMMITTED DECISION RULE  —  DO NOT MODIFY AFTER SEEING TEST RESULTS
# =========================================================================
DEPLOY_MODEL = "v3_catboost_full2000_trainval"
DEPLOY_KELLY_FRACTION = 0.25  # ¼-Kelly
DEPLOY_MAX_BET_FRACTION = 1.0  # no cap
DEPLOY_THRESHOLD_BANKROLL = 1.2  # final bankroll on test must be ≥ 1.2× to deploy

# Market / sizing scenario fixed at PLAN.md Kalshi-like defaults.
EDGE_THRESHOLD_KELLY = 0.03  # PLAN.md §10.1
EDGE_THRESHOLD_FLAT = 0.05  # primary flat-stake threshold from STATUS.md
FEE_RATE = 0.07  # Kalshi 7% fee on winnings
USE_NO_VIG = True

# Grid (diagnostic only — does NOT change the deploy decision)
GRID_FRACTIONS = [0.10, 0.25, 0.50, 1.00]
GRID_CAPS = [0.01, 0.02, 0.05, 0.10, 1.00]
# =========================================================================

import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from ufc_pred.backtest.bet_eval import evaluate_bets, evaluate_bets_kelly
from ufc_pred.backtest.metrics import evaluate, market_no_vig_prob_red
from ufc_pred.backtest.universe import add_prior_fight_counts
from ufc_pred.features.skill_v3_pipeline import OUTPUT as SKILL_V3_PARQUET
from ufc_pred.features.static_v1 import prepare
from ufc_pred.ingest.kaggle_mdabbert import HISTORY_PARQUET
from ufc_pred.paths import METRICS, MODELS
from ufc_pred.utils.time_splits import split


def load_test():
    fights = pd.read_parquet(HISTORY_PARQUET)
    fights = fights[fights["Winner"].isin(["Red", "Blue"])].copy()
    fights["date"] = pd.to_datetime(fights["date"])
    sk = pd.read_parquet(SKILL_V3_PARQUET)
    sk["date"] = pd.to_datetime(sk["date"])
    fights = fights.merge(
        sk[["date", "R_fighter", "B_fighter", "skill_diff_mean", "skill_diff_std"]],
        on=["date", "R_fighter", "B_fighter"],
        how="left",
        validate="many_to_one",
    )
    # Prior-fight counts must be computed over the FULL table before splitting,
    # so a test fight sees the fighter's train-window history too.
    fights = add_prior_fight_counts(fights)
    splits = split(fights)
    return splits.test.reset_index(drop=True)


def predict(model_name: str, test_df: pd.DataFrame) -> np.ndarray:
    payload = joblib.load(MODELS / f"{model_name}.joblib")
    cols = payload["columns"]
    cat = payload.get("cat_features", [])
    X, _, _, _ = prepare(test_df, augment_symmetry=False, one_hot=False)
    X = X.reindex(columns=cols, fill_value=None)
    for c in cat:
        X[c] = X[c].fillna("__missing__").astype(str)
    return payload["model"].predict_proba(X)[:, 1]


def kelly_grid(p, y, R, B) -> dict:
    out = {}
    for f in GRID_FRACTIONS:
        for c in GRID_CAPS:
            r = evaluate_bets_kelly(
                p,
                y,
                R,
                B,
                edge_threshold=EDGE_THRESHOLD_KELLY,
                fee_rate=FEE_RATE,
                use_no_vig=USE_NO_VIG,
                kelly_fraction=f,
                max_bet_fraction=c,
                starting_bankroll=1.0,
            )
            out[f"{f:.2f}_{c:.2f}"] = {
                "kelly_fraction": f,
                "max_bet_fraction": c,
                "final_bankroll": r["final_bankroll"],
                "total_return_pct": r["total_return_pct"],
                "max_drawdown_pct": r["max_drawdown_pct"],
                "n_bets": r["n_bets"],
            }
    return out


def grid_to_frame(grid: dict, key: str) -> pd.DataFrame:
    data = {}
    for c in GRID_CAPS:
        col = []
        for f in GRID_FRACTIONS:
            col.append(grid[f"{f:.2f}_{c:.2f}"][key])
        data[("no cap" if c >= 1 else f"{int(c * 100)}%")] = col
    df = pd.DataFrame(data, index=[f"{int(f * 100)}%-K" for f in GRID_FRACTIONS])
    df.index.name = "Kelly fraction"
    df.columns.name = "per-bet cap"
    return df


def evaluate_model(name: str, p: np.ndarray, y: np.ndarray, test_df: pd.DataFrame):
    metrics = evaluate(y, p, label=f"test_{name}")

    # Flat-stake at primary threshold (5%) — Kalshi-like.
    flat = evaluate_bets(
        p,
        y,
        test_df["R_odds"],
        test_df["B_odds"],
        edge_threshold=EDGE_THRESHOLD_FLAT,
        fee_rate=FEE_RATE,
        use_no_vig=USE_NO_VIG,
    )

    # Same thing restricted to the fights the LIVE runner could actually have
    # priced (both fighters already in the history index). The unrestricted
    # number above is not achievable in deployment — see backtest.universe.
    dep = ~test_df["has_debut"].to_numpy()
    flat_dep = evaluate_bets(
        p[dep],
        y[dep],
        test_df.loc[dep, "R_odds"],
        test_df.loc[dep, "B_odds"],
        edge_threshold=EDGE_THRESHOLD_FLAT,
        fee_rate=FEE_RATE,
        use_no_vig=USE_NO_VIG,
    )

    # 4x5 Kelly grid (diagnostic).
    grid = kelly_grid(p, y, test_df["R_odds"], test_df["B_odds"])
    grid_dep = kelly_grid(p[dep], y[dep], test_df.loc[dep, "R_odds"], test_df.loc[dep, "B_odds"])

    # Deploy config cell (the one that matters for the decision).
    deploy_key = f"{DEPLOY_KELLY_FRACTION:.2f}_{DEPLOY_MAX_BET_FRACTION:.2f}"
    deploy_cell = grid[deploy_key]

    def _flat_dict(r):
        return {
            "n_eligible": r.n_eligible,
            "n_bets": r.n_bets,
            "roi_pct": r.roi_pct,
            "ci95_low": r.ci95_roi_pct[0],
            "ci95_high": r.ci95_roi_pct[1],
            "mean_ev_pct": r.mean_ev_pct,
            "sharpe": r.sharpe,
            "hit_rate": r.hit_rate,
            "total_pnl": r.total_pnl,
        }

    return {
        "name": name,
        "predictive_metrics": metrics,
        "flat_stake_kalshi_5pct": _flat_dict(flat),
        "flat_stake_kalshi_5pct_deployable": _flat_dict(flat_dep),
        "n_debut_fights_excluded": int((~dep).sum()),
        "kelly_grid": grid,
        "kelly_grid_deployable": grid_dep,
        "deploy_cell": deploy_cell,
        "deploy_cell_deployable": grid_dep[deploy_key],
    }


def print_predictive(rows):
    print("\n=== Predictive metrics on test ===")
    df = pd.DataFrame(
        [
            {
                "model": r["name"],
                "n": r["predictive_metrics"]["n"],
                "log_loss": r["predictive_metrics"]["log_loss"],
                "brier": r["predictive_metrics"]["brier"],
                "ece": r["predictive_metrics"]["ece"],
                "accuracy": r["predictive_metrics"]["accuracy_argmax"],
            }
            for r in rows
        ]
    )
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def _flat_frame(rows, key):
    return pd.DataFrame(
        [
            {
                "model": r["name"],
                "n_bets": r[key]["n_bets"],
                "roi_pct": r[key]["roi_pct"],
                "ci95_low": r[key]["ci95_low"],
                "ci95_high": r[key]["ci95_high"],
                "mean_ev_pct": r[key]["mean_ev_pct"],
                "hit_rate": r[key]["hit_rate"],
                "sharpe": r[key]["sharpe"],
            }
            for r in rows
        ]
    )


def print_flat(rows):
    print("\n=== Flat-stake ROI@5% (Kalshi-like, no-vig + 7% fee) — ALL fights ===")
    print(
        _flat_frame(rows, "flat_stake_kalshi_5pct").to_string(index=False, float_format=lambda x: f"{x:+.3f}")
    )

    excluded = rows[0]["n_debut_fights_excluded"]
    print(f"\n=== Same, DEPLOYABLE fights only ({excluded} debut fights excluded) ===")
    print("    This is the achievable number. The live runner cannot price a")
    print("    fighter with no history — it raises 'No fighter match' and skips.")
    print(
        _flat_frame(rows, "flat_stake_kalshi_5pct_deployable").to_string(
            index=False, float_format=lambda x: f"{x:+.3f}"
        )
    )


def print_grids(rows):
    for r in rows:
        print(f"\n=== Kelly grid — {r['name']} — final bankroll ($1 → ?) ===")
        bk = grid_to_frame(r["kelly_grid"], "final_bankroll")
        print(bk.to_string(float_format=lambda x: f"${x:.2f}"))
        print(f"\n=== Kelly grid — {r['name']} — max drawdown (%) ===")
        dd = grid_to_frame(r["kelly_grid"], "max_drawdown_pct")
        print(dd.to_string(float_format=lambda x: f"{x:.1f}%"))


def print_verdict(rows):
    deploy_row = next(r for r in rows if r["name"] == DEPLOY_MODEL)
    cell = deploy_row["deploy_cell"]
    bk = cell["final_bankroll"]
    dd = cell["max_drawdown_pct"]
    n = cell["n_bets"]
    passed = bk >= DEPLOY_THRESHOLD_BANKROLL

    print("\n" + "=" * 72)
    print("DEPLOY DECISION  (rule pre-committed before seeing test results)")
    print("=" * 72)
    print(f"Model:              {DEPLOY_MODEL}")
    print(
        f"Config:             {int(DEPLOY_KELLY_FRACTION * 100)}%-Kelly, "
        f"{'no cap' if DEPLOY_MAX_BET_FRACTION >= 1 else f'{int(DEPLOY_MAX_BET_FRACTION * 100)}% cap'}"
    )
    print(f"Threshold:          final bankroll ≥ ${DEPLOY_THRESHOLD_BANKROLL:.2f}")
    print(f"Result:             $1.00 → ${bk:.2f}  (max DD {dd:.1f}%, n_bets={n})")
    print(f"Verdict:            {'✓ PASS — deploy criterion met' if passed else '✗ FAIL — do not deploy'}")

    dep_cell = deploy_row["deploy_cell_deployable"]
    dep_flat = deploy_row["flat_stake_kalshi_5pct_deployable"]
    dep_passed = dep_cell["final_bankroll"] >= DEPLOY_THRESHOLD_BANKROLL
    print("-" * 72)
    print("DEPLOYABLE-ONLY (debut fights removed — what live can actually bet)")
    print(
        f"Result:             $1.00 → ${dep_cell['final_bankroll']:.2f}  "
        f"(max DD {dep_cell['max_drawdown_pct']:.1f}%, n_bets={dep_cell['n_bets']})"
    )
    print(
        f"Flat-stake ROI@5%:  {dep_flat['roi_pct']:+.2f}%  "
        f"CI95 ({dep_flat['ci95_low']:+.2f}%, {dep_flat['ci95_high']:+.2f}%)"
    )
    if dep_flat["ci95_low"] <= 0 <= dep_flat["ci95_high"]:
        print("                    ⚠ CI includes zero — edge is not established")
    print(f"Verdict:            {'✓ PASS' if dep_passed else '✗ FAIL'} on the achievable universe")
    print("=" * 72)
    return passed, bk, dd, n


def main():
    print("Loading test set...")
    test_df = load_test()
    y_test = (test_df["Winner"].to_numpy() == "Red").astype(int)
    print(
        f"  test n = {len(test_df)} fights ({test_df['date'].min().date()} → {test_df['date'].max().date()})"
    )

    # Market reference for context.
    sub = test_df.dropna(subset=["R_odds", "B_odds"])
    market_metrics = evaluate(
        (sub["Winner"] == "Red").astype(int).to_numpy(),
        market_no_vig_prob_red(sub),
        label="test_market",
    )
    print(f"  market no-vig log_loss = {market_metrics['log_loss']:.4f} (reference)")

    models = [DEPLOY_MODEL, "v3_full2000_no_skill_corrupted_trainval"]
    print("\nPredicting...")
    preds = {}
    for name in models:
        preds[name] = predict(name, test_df)
        p = preds[name]
        print(f"  {name}: pred mean {p.mean():.3f}, range [{p.min():.3f}, {p.max():.3f}]")

    print("\nEvaluating...")
    rows = [evaluate_model(name, preds[name], y_test, test_df) for name in models]

    print_predictive(rows)
    print_flat(rows)
    print_grids(rows)
    passed, bk, dd, n = print_verdict(rows)

    # Write JSON.
    out = {
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "decision_rule": {
            "deploy_model": DEPLOY_MODEL,
            "deploy_kelly_fraction": DEPLOY_KELLY_FRACTION,
            "deploy_max_bet_fraction": DEPLOY_MAX_BET_FRACTION,
            "deploy_threshold_bankroll": DEPLOY_THRESHOLD_BANKROLL,
            "edge_threshold_kelly": EDGE_THRESHOLD_KELLY,
            "edge_threshold_flat": EDGE_THRESHOLD_FLAT,
            "fee_rate": FEE_RATE,
            "use_no_vig": USE_NO_VIG,
        },
        "test_n": int(len(test_df)),
        "test_range": [
            str(test_df["date"].min().date()),
            str(test_df["date"].max().date()),
        ],
        "market_reference": market_metrics,
        "models": rows,
        "verdict": {
            "passed": bool(passed),
            "final_bankroll": float(bk),
            "max_drawdown_pct": float(dd),
            "n_bets": int(n),
        },
    }
    out_path = METRICS / "test_set_evaluation.json"
    out_path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

"""Rolling monthly-retrain backtest on Polymarket-matched fights.

For each month in the test window:
  1. Retrain v3 CatBoost on all Kaggle fights through end of previous month.
  2. Predict that month's Polymarket-matched fights with real + corrupted variants.
  3. Run Kelly simulation with bankroll carried from previous month, per account.
  4. Save bankroll for next iteration.

Output: artifacts/metrics/rolling_monthly_backtest.json
        per-account bankroll trajectory + per-month breakdown.

Also runs a FROZEN baseline (v3_full2000_trainval used throughout) for comparison.
"""

from __future__ import annotations

import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from ufc_pred.features.skill_v3_pipeline import OUTPUT as SKILL_V3_PARQUET
from ufc_pred.features.static_v1 import prepare
from ufc_pred.ingest.kaggle_mdabbert import HISTORY_PARQUET
from ufc_pred.models.wrappers import CorruptedSkillModel
from ufc_pred.paths import METRICS, MODELS
from ufc_pred.utils.time_splits import recency_weights

# ============================================================================
# Config
# ============================================================================
START = 300.0
FEE = 0.02  # Polymarket-style
EDGE_THR = 0.03
ACCOUNTS = [
    ("A", "real", 0.10, 0.10),  # 10%-K + 10% cap
    ("B", "real", 0.25, 1.00),  # ¼-K + no cap
    ("C", "corrupted", 0.25, 1.00),  # same, on corrupted model
]


# ============================================================================
# Load matched Polymarket fights (v2 matcher output) + full Kaggle context
# ============================================================================
def load_data():
    matched = pd.read_parquet("data/interim/polymarket_matched_to_kaggle_v2.parquet")
    matched["date"] = pd.to_datetime(matched["date"])

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
    # Realign matched_fights to its Kaggle row by (date, R_fighter, B_fighter)
    key = ["date", "R_fighter", "B_fighter"]
    matched_full = matched[key + ["polymarket_p_red", "polymarket_p_blue"]].merge(
        fights, on=key, how="left", validate="one_to_one"
    )
    return fights, matched_full


# ============================================================================
# Model training (same recipe as scripts/retrain_train_plus_val.py)
# ============================================================================
def train_catboost(train_df, recency_ref):
    """Train v3 architecture on train_df. recency_ref = reference date for weights."""
    X_train, y_train, d_train, cat_features = prepare(train_df, augment_symmetry=True, one_hot=False)
    # sign-flip skill_diff_mean on augmented half
    n = len(X_train) // 2
    if len(X_train) == 2 * n and "skill_diff_mean" in X_train.columns:
        X_train = X_train.copy()
        X_train.loc[X_train.index[n:], "skill_diff_mean"] = -X_train.loc[X_train.index[n:], "skill_diff_mean"]

    w = recency_weights(d_train, reference_date=recency_ref)
    pool = Pool(X_train, y_train, cat_features=cat_features, weight=w)

    m = CatBoostClassifier(
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
    m.fit(pool, verbose=False)
    return m, list(X_train.columns), cat_features


def predict_on(model_obj, cols, cat_features, fights_df):
    X, _, _, _ = prepare(fights_df, augment_symmetry=False, one_hot=False)
    X = X.reindex(columns=cols, fill_value=None)
    for c in cat_features:
        X[c] = X[c].fillna("__missing__").astype(str)
    return model_obj.predict_proba(X)[:, 1]


# ============================================================================
# Kelly per-bet simulation that mutates a passed bankroll list
# ============================================================================
def simulate_kelly_month(p_model, fights_df, bankroll, kelly_frac, cap):
    """Run Kelly on a sequence of fights starting from `bankroll`. Returns
    (final_bankroll, list of per-bet rows, n_bets)."""
    pR_poly = fights_df["polymarket_p_red"].to_numpy()
    pB_poly = fights_df["polymarket_p_blue"].to_numpy()
    dec_R = 1.0 / pR_poly
    dec_B = 1.0 / pB_poly
    eff_R = 1.0 + (1.0 - FEE) * (dec_R - 1.0)
    eff_B = 1.0 + (1.0 - FEE) * (dec_B - 1.0)
    p_R = p_model
    p_B = 1.0 - p_R
    ev_R = p_R * eff_R - 1.0
    ev_B = p_B * eff_B - 1.0
    bet_red = ev_R >= ev_B
    chosen_ev = np.where(bet_red, ev_R, ev_B)
    chosen_dec = np.where(bet_red, eff_R, eff_B)
    chosen_p = np.where(bet_red, p_R, p_B)
    won = np.where(
        bet_red,
        fights_df["Winner"].to_numpy() == "Red",
        fights_df["Winner"].to_numpy() == "Blue",
    ).astype(int)
    mask = chosen_ev > EDGE_THR

    rows = []
    n_bets = 0
    for i in range(len(p_R)):
        if not mask[i]:
            continue
        b = chosen_dec[i] - 1.0
        p = chosen_p[i]
        q = 1.0 - p
        fk = (b * p - q) / b
        if fk <= 0:
            continue
        sf = min(kelly_frac * fk, cap)
        stake = bankroll * sf
        realized = stake * (chosen_dec[i] - 1.0) if won[i] else -stake
        bankroll += realized
        n_bets += 1
        rows.append(
            {
                "stake": stake,
                "won": int(won[i]),
                "realized": realized,
                "bankroll_after": bankroll,
            }
        )
    return bankroll, rows, n_bets


# ============================================================================
# Main rolling loop
# ============================================================================
def main():
    print("Loading data...")
    fights, matched_full = load_data()
    print(f"  Kaggle fights: {len(fights)}")
    print(f"  Polymarket-matched fights: {len(matched_full)}")

    test_start = pd.Timestamp("2024-04-01")
    test_end = pd.Timestamp("2026-04-01")
    month_starts = pd.date_range(test_start, test_end, freq="MS")
    print(f"  Months: {len(month_starts) - 1}")

    # Bankrolls per account, two regimes (rolling vs frozen)
    bank_rolling = {tag: START for tag, *_ in ACCOUNTS}
    bank_frozen = {tag: START for tag, *_ in ACCOUNTS}

    # Frozen models: load the trainval ones
    frozen_real = joblib.load(MODELS / "v3_catboost_full2000_trainval.joblib")
    frozen_corr = joblib.load(MODELS / "v3_full2000_no_skill_corrupted_trainval.joblib")

    monthly_results = []
    skill_cols = ["skill_diff_mean", "skill_diff_std"]

    for i in range(len(month_starts) - 1):
        m_start = month_starts[i]
        m_end = month_starts[i + 1]
        month_label = m_start.strftime("%Y-%m")
        month_fights = matched_full[
            (matched_full["date"] >= m_start) & (matched_full["date"] < m_end)
        ].reset_index(drop=True)
        if len(month_fights) == 0:
            print(f"[{month_label}]  no Polymarket-matched fights — skipping")
            continue

        # -------- Rolling: retrain on data through end of previous month -----
        train_data = fights[fights["date"] < m_start].copy()
        n_train = len(train_data)
        print(
            f"[{month_label}]  n_month={len(month_fights):3d}  "
            f"train cutoff < {m_start.date()}  n_train={n_train}"
        )

        m_real, cols, cat_features = train_catboost(train_data, recency_ref=m_start - pd.Timedelta(days=1))
        m_corr = CorruptedSkillModel(m_real, skill_cols)

        # Predict the month's fights
        p_real_roll = predict_on(m_real, cols, cat_features, month_fights)
        p_corr_roll = predict_on(m_corr, cols, cat_features, month_fights)

        # Predict with frozen models (same fights)
        p_real_frz = predict_on(
            frozen_real["model"],
            frozen_real["columns"],
            frozen_real.get("cat_features", []),
            month_fights,
        )
        p_corr_frz = predict_on(
            frozen_corr["model"],
            frozen_corr["columns"],
            frozen_corr.get("cat_features", []),
            month_fights,
        )

        # Simulate this month for each account, both regimes
        month_summary = {"month": month_label, "n_fights": int(len(month_fights))}
        for tag, src, kf, cap in ACCOUNTS:
            p_roll = p_real_roll if src == "real" else p_corr_roll
            p_frz = p_real_frz if src == "real" else p_corr_frz

            br0 = bank_rolling[tag]
            bf0 = bank_frozen[tag]

            bank_rolling[tag], _, n_bets_r = simulate_kelly_month(p_roll, month_fights, br0, kf, cap)
            bank_frozen[tag], _, n_bets_f = simulate_kelly_month(p_frz, month_fights, bf0, kf, cap)

            month_summary[f"{tag}_rolling_start"] = br0
            month_summary[f"{tag}_rolling_end"] = bank_rolling[tag]
            month_summary[f"{tag}_rolling_pnl"] = bank_rolling[tag] - br0
            month_summary[f"{tag}_frozen_start"] = bf0
            month_summary[f"{tag}_frozen_end"] = bank_frozen[tag]
            month_summary[f"{tag}_frozen_pnl"] = bank_frozen[tag] - bf0
            month_summary[f"{tag}_n_bets"] = n_bets_r

        monthly_results.append(month_summary)
        print(
            f"  ROLLING  A=${bank_rolling['A']:.2f}  B=${bank_rolling['B']:,.2f}  C=${bank_rolling['C']:,.2f}"
        )
        print(f"  FROZEN   A=${bank_frozen['A']:.2f}  B=${bank_frozen['B']:,.2f}  C=${bank_frozen['C']:,.2f}")

    # Save
    out = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "start_bankroll": START,
            "fee": FEE,
            "edge_threshold": EDGE_THR,
            "accounts": [{"tag": t, "src": s, "kelly": k, "cap": c} for t, s, k, c in ACCOUNTS],
        },
        "final_bankrolls": {
            "rolling": bank_rolling,
            "frozen": bank_frozen,
        },
        "monthly": monthly_results,
    }
    out_path = METRICS / "rolling_monthly_backtest.json"
    out_path.write_text(json.dumps(out, indent=2, default=float))

    print()
    print("=" * 72)
    print("FINAL BANKROLLS")
    print("=" * 72)
    for tag, src, kf, cap in ACCOUNTS:
        cap_label = "no cap" if cap >= 1 else f"{int(cap * 100)}% cap"
        print(f"Account {tag} ({src}, {int(kf * 100)}%-K, {cap_label}):")
        print(f"  Rolling: ${START:.2f} → ${bank_rolling[tag]:,.2f}")
        print(f"  Frozen:  ${START:.2f} → ${bank_frozen[tag]:,.2f}")
        ratio = bank_rolling[tag] / bank_frozen[tag]
        print(f"  Ratio (rolling/frozen): {ratio:.2f}×")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

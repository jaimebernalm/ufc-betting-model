"""Verify the most extreme-drawdown Kelly cells on test by:
1. Reproducing each trajectory.
2. Printing the exact bet at which peak / trough occurred.
3. Verifying max_drawdown_pct against an independent recomputation.
4. Saving visualizations to artifacts/figures/.
"""

from __future__ import annotations

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ufc_pred.backtest.bet_eval import (
    american_to_decimal,
    evaluate_bets_kelly,
)
from ufc_pred.backtest.metrics import american_to_implied_prob
from ufc_pred.features.skill_v3_pipeline import OUTPUT as SKILL_V3_PARQUET
from ufc_pred.features.static_v1 import prepare
from ufc_pred.ingest.kaggle_mdabbert import HISTORY_PARQUET
from ufc_pred.paths import MODELS, ROOT
from ufc_pred.utils.time_splits import split

FIG_DIR = ROOT / "artifacts" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_test_and_predict():
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
    splits = split(fights)
    test = splits.test.reset_index(drop=True)
    y_test = (test["Winner"].to_numpy() == "Red").astype(int)

    def predict(name):
        payload = joblib.load(MODELS / f"{name}.joblib")
        X, _, _, _ = prepare(test, augment_symmetry=False, one_hot=False)
        X = X.reindex(columns=payload["columns"], fill_value=None)
        for c in payload.get("cat_features", []):
            X[c] = X[c].fillna("__missing__").astype(str)
        return payload["model"].predict_proba(X)[:, 1]

    return (
        test,
        y_test,
        {
            "v3": predict("v3_catboost_full2000_trainval"),
            "corrupt": predict("v3_full2000_no_skill_corrupted_trainval"),
        },
    )


def simulate_with_logging(
    p,
    y,
    R_odds,
    B_odds,
    kelly_fraction,
    max_bet_fraction,
    edge_threshold=0.03,
    fee_rate=0.07,
    use_no_vig=True,
):
    """Re-implements evaluate_bets_kelly but logs every step.

    Returns a per-bet dataframe with: bet_idx, fight_idx, side, stake_frac,
    stake, decimal_odds, won, bankroll_after, peak_so_far, drawdown_pct.
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=int)

    dec_R = american_to_decimal(R_odds)
    dec_B = american_to_decimal(B_odds)
    valid = ~(np.isnan(dec_R) | np.isnan(dec_B))

    if use_no_vig:
        p_r_imp = american_to_implied_prob(R_odds)
        p_b_imp = american_to_implied_prob(B_odds)
        total = p_r_imp + p_b_imp
        dec_R = 1.0 / (p_r_imp / total)
        dec_B = 1.0 / (p_b_imp / total)

    eff_R = 1.0 + (1.0 - fee_rate) * (dec_R - 1.0)
    eff_B = 1.0 + (1.0 - fee_rate) * (dec_B - 1.0)

    p_R = p
    p_B = 1.0 - p
    ev_R = p_R * eff_R - 1.0
    ev_B = p_B * eff_B - 1.0
    bet_red = ev_R >= ev_B
    chosen_ev = np.where(bet_red, ev_R, ev_B)
    chosen_dec = np.where(bet_red, eff_R, eff_B)
    chosen_p = np.where(bet_red, p_R, p_B)
    bets_mask = valid & (chosen_ev > edge_threshold)
    won_full = np.where(bet_red, y == 1, y == 0)

    bankroll = 1.0
    peak = 1.0
    max_dd = 0.0
    rows = []
    bet_idx = 0
    for i in range(len(p_R)):
        if not bets_mask[i]:
            continue
        b = chosen_dec[i] - 1.0
        pi = chosen_p[i]
        qi = 1.0 - pi
        full_kelly = (b * pi - qi) / b
        if full_kelly <= 0:
            continue
        stake_frac = min(kelly_fraction * full_kelly, max_bet_fraction)
        stake = bankroll * stake_frac
        if won_full[i]:
            bankroll += stake * (chosen_dec[i] - 1.0)
        else:
            bankroll -= stake
        peak = max(peak, bankroll)
        dd = (peak - bankroll) / peak
        max_dd = max(max_dd, dd)
        rows.append(
            {
                "bet_idx": bet_idx,
                "fight_idx": i,
                "side": "R" if bet_red[i] else "B",
                "p_chosen": pi,
                "decimal_odds": chosen_dec[i],
                "full_kelly_frac": full_kelly,
                "stake_frac": stake_frac,
                "stake": stake,
                "won": int(won_full[i]),
                "bankroll_after": bankroll,
                "peak_so_far": peak,
                "drawdown_pct": dd * 100,
            }
        )
        bet_idx += 1
    return pd.DataFrame(rows), max_dd * 100


def verify_one(label, p, y, test, kelly_fraction, max_bet_fraction):
    print(f"\n{'=' * 72}")
    print(f"Case: {label}")
    print(f"  Kelly fraction = {kelly_fraction}, cap = {max_bet_fraction}")

    # Independent simulation.
    log, max_dd_ours = simulate_with_logging(
        p,
        y,
        test["R_odds"],
        test["B_odds"],
        kelly_fraction=kelly_fraction,
        max_bet_fraction=max_bet_fraction,
    )

    # Library result for comparison.
    lib = evaluate_bets_kelly(
        p,
        y,
        test["R_odds"],
        test["B_odds"],
        edge_threshold=0.03,
        fee_rate=0.07,
        use_no_vig=True,
        kelly_fraction=kelly_fraction,
        max_bet_fraction=max_bet_fraction,
        starting_bankroll=1.0,
    )

    print(f"  library:  final ${lib['final_bankroll']:.4g}, max DD {lib['max_drawdown_pct']:.3f}%")
    print(f"  ours:     final ${log['bankroll_after'].iloc[-1]:.4g}, max DD {max_dd_ours:.3f}%")
    print(
        f"  match:    {np.isclose(lib['max_drawdown_pct'], max_dd_ours, atol=1e-6)}"
        f" / {np.isclose(lib['final_bankroll'], log['bankroll_after'].iloc[-1], rtol=1e-9)}"
    )

    # Find peak and trough bets.
    peak_bet = log["peak_so_far"].idxmax()
    peak_value = log["peak_so_far"].iloc[peak_bet]
    max_dd_bet = log["drawdown_pct"].idxmax()
    trough_value = log["bankroll_after"].iloc[max_dd_bet]
    trough_peak = log["peak_so_far"].iloc[max_dd_bet]
    final = log["bankroll_after"].iloc[-1]

    print(
        f"  peak bankroll = ${peak_value:.4g} at bet #{peak_bet}/{len(log)} "
        f"(fight idx {log['fight_idx'].iloc[peak_bet]}, "
        f"date {test['date'].iloc[log['fight_idx'].iloc[peak_bet]].date()})"
    )
    print(
        f"  worst drawdown reached at bet #{max_dd_bet}/{len(log)} "
        f"(fight idx {log['fight_idx'].iloc[max_dd_bet]}, "
        f"date {test['date'].iloc[log['fight_idx'].iloc[max_dd_bet]].date()})"
    )
    print(f"    peak-so-far at that bet = ${trough_peak:.4g}")
    print(f"    bankroll at that bet    = ${trough_value:.4g}")
    print(f"    drawdown                = {(1 - trough_value / trough_peak) * 100:.3f}%")
    print(f"  final bankroll = ${final:.4g}")
    print(f"  ratio peak/trough/final = {peak_value:.2g} / {trough_value:.2g} / {final:.2g}")

    # Show context around the trough: prev bet, trough bet, next 3 bets.
    print("\n  Trough context (bets around the worst-DD point):")
    ctx_lo = max(0, max_dd_bet - 1)
    ctx_hi = min(len(log), max_dd_bet + 4)
    ctx = log.iloc[ctx_lo:ctx_hi][
        [
            "bet_idx",
            "p_chosen",
            "decimal_odds",
            "stake_frac",
            "stake",
            "won",
            "bankroll_after",
            "peak_so_far",
            "drawdown_pct",
        ]
    ].copy()
    print(
        ctx.to_string(
            index=False,
            formatters={
                "p_chosen": "{:.3f}".format,
                "decimal_odds": "{:.3f}".format,
                "stake_frac": "{:.3f}".format,
                "stake": "{:.4g}".format,
                "bankroll_after": "{:.4g}".format,
                "peak_so_far": "{:.4g}".format,
                "drawdown_pct": "{:.2f}".format,
            },
        )
    )

    return log


def plot_case(log, label, slug):
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    # Top: log-scale bankroll + peak.
    ax = axes[0]
    ax.plot(log["bet_idx"], log["bankroll_after"], color="#1a9641", linewidth=1.5, label="bankroll")
    ax.plot(
        log["bet_idx"],
        log["peak_so_far"],
        color="#fdae61",
        linewidth=1.5,
        linestyle="--",
        alpha=0.85,
        label="running peak",
    )
    # Mark peak and worst-DD points.
    peak_idx = log["peak_so_far"].idxmax()
    dd_idx = log["drawdown_pct"].idxmax()
    ax.scatter(
        [log["bet_idx"].iloc[peak_idx]],
        [log["peak_so_far"].iloc[peak_idx]],
        color="black",
        s=60,
        zorder=5,
        label=f"overall peak (${log['peak_so_far'].iloc[peak_idx]:.4g})",
    )
    ax.scatter(
        [log["bet_idx"].iloc[dd_idx]],
        [log["bankroll_after"].iloc[dd_idx]],
        color="red",
        s=60,
        marker="v",
        zorder=5,
        label=f"worst-DD trough (${log['bankroll_after'].iloc[dd_idx]:.4g}, "
        f"{log['drawdown_pct'].iloc[dd_idx]:.1f}%)",
    )
    ax.scatter(
        [log["bet_idx"].iloc[-1]],
        [log["bankroll_after"].iloc[-1]],
        color="blue",
        s=60,
        marker="s",
        zorder=5,
        label=f"final (${log['bankroll_after'].iloc[-1]:.4g})",
    )
    ax.set_yscale("log")
    ax.set_ylabel("Bankroll ($, log scale)")
    ax.set_title(f"{label} — bankroll + running peak (log scale)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3, which="both")

    # Bottom: drawdown over time.
    ax = axes[1]
    ax.fill_between(log["bet_idx"], 0, log["drawdown_pct"], color="#d7191c", alpha=0.4)
    ax.plot(log["bet_idx"], log["drawdown_pct"], color="#d7191c", linewidth=1.2)
    ax.scatter(
        [log["bet_idx"].iloc[dd_idx]],
        [log["drawdown_pct"].iloc[dd_idx]],
        color="red",
        s=60,
        marker="v",
        zorder=5,
    )
    ax.axhline(50, color="gray", linestyle=":", alpha=0.5)
    ax.axhline(90, color="gray", linestyle=":", alpha=0.5)
    ax.set_ylim(0, 102)
    ax.set_ylabel("Drawdown from running peak (%)")
    ax.set_xlabel("Bet # (chronological)")
    ax.set_title("Drawdown trajectory")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = FIG_DIR / f"verify_drawdown_{slug}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out}")
    return out


def main():
    test, y_test, preds = load_test_and_predict()

    cases = [
        # (label, model_key, kelly_fraction, max_bet_fraction, slug)
        ("v3_real ¼-K + no cap (deploy B)", "v3", 0.25, 1.00, "v3_real_q_nocap"),
        ("v3_real ½-K + no cap", "v3", 0.50, 1.00, "v3_real_half_nocap"),
        ("v3_real full-K + no cap (RUIN)", "v3", 1.00, 1.00, "v3_real_full_nocap"),
        ("corrupted ¼-K + no cap (deploy C)", "corrupt", 0.25, 1.00, "corrupt_q_nocap"),
        ("corrupted ½-K + no cap (99.9% DD)", "corrupt", 0.50, 1.00, "corrupt_half_nocap"),
        ("v3_real 10%-K + 10% cap (deploy A)", "v3", 0.10, 0.10, "v3_real_10_10"),
    ]

    for label, key, f, c, slug in cases:
        log = verify_one(label, preds[key], y_test, test, kelly_fraction=f, max_bet_fraction=c)
        plot_case(log, label, slug)


if __name__ == "__main__":
    main()

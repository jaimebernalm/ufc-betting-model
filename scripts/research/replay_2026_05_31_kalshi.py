"""Replay the 2026-05-31 (UTC 2026-05-30) card using Kalshi T-90min closing
prices and the deployed 10-seed ensembles, computing final bankrolls per the
DEPLOY.md three-account strategy.

Kalshi fee: 0.07 * P * (1-P) per contract, paid upfront with the stake.
"""

from __future__ import annotations

import sys

import pandas as pd

from ufc_pred.inference.ensemble_predict import predict_ensemble
from ufc_pred.inference.skill_for_upcoming import attach_skill_for_upcoming
from ufc_pred.inference.upcoming_builder import build_upcoming_row, resolve_fighter_name
from ufc_pred.ingest.rankings_attach import load_rankings
from ufc_pred.paths import PROCESSED

EDGE_THR = 0.03
ACCOUNTS = [
    ("A", 300.0, 0.10, 0.10, "real"),  # 10%-Kelly, 10% bankroll cap
    ("B", 300.0, 0.25, None, "real"),  # 25%-Kelly, no cap
    ("C", 300.0, 0.25, None, "corrupted"),  # 25%-Kelly, no cap
]

NAME_FIX = {
    "Yadong Song": "Song Yadong",
    "Qileng Aori": "Aori Qileng",
    "Harris Carlston": "Carlston Harris",
}


def kalshi_kelly_stake(p_model: float, p_market: float, bankroll: float, kfrac: float, cap: float | None):
    """Return (side_indicator, stake, n_contracts, fee_total, net_win, net_loss).

    side_indicator: True means we bet *this side* (p_model is the prob of winning).
    Stake includes upfront fee. n_contracts = stake / (p_market + fee_per_contract).
    """
    if p_model - p_market < EDGE_THR:
        return None
    # Per-contract: cost = p_market + 0.07*p_market*(1-p_market)
    fee_per = 0.07 * p_market * (1 - p_market)
    cost_per = p_market + fee_per
    # Effective decimal odds b = (1 - cost_per) / cost_per
    b = (1.0 - cost_per) / cost_per
    full_kelly = (b * p_model - (1 - p_model)) / b
    if full_kelly <= 0:
        return None
    stake = bankroll * kfrac * full_kelly
    if cap is not None:
        stake = min(stake, bankroll * cap)
    if stake < 0.50:
        return None
    n = stake / cost_per
    fee_total = n * fee_per
    net_win = n - stake  # payoff $1 per contract, minus what we paid
    net_loss = -stake
    return stake, n, fee_total, net_win, net_loss


def main() -> int:
    snap = pd.read_parquet("data/raw/kalshi/snapshots/historical_T-90min_perfight.parquet")
    card = snap[snap["fight_date"].dt.strftime("%Y-%m-%d") == "2026-05-30"].copy()
    card = card.sort_values("close_time")  # rough chronological order
    print(f"Card: {len(card)} fights\n")

    fights = pd.read_parquet(PROCESSED / "fights.parquet")
    rankings = load_rankings()

    bankrolls = {"A": 300.0, "B": 300.0, "C": 300.0}
    rows = []

    for _, fight in card.iterrows():
        fa, fb = fight["fighter_a"], fight["fighter_b"]
        pa_mkt, pb_mkt = fight["close_yes_price_a"], fight["close_yes_price_b"]
        winner_side = fight["winner"]  # "A" or "B"
        fa_q = NAME_FIX.get(fa, fa)
        fb_q = NAME_FIX.get(fb, fb)
        try:
            ca = resolve_fighter_name(fa_q, fights)
            cb = resolve_fighter_name(fb_q, fights)
        except ValueError as e:
            print(f"SKIP {fa} vs {fb}: {e}")
            continue
        # infer weight class as modal of the two fighters' history
        sub = pd.concat(
            [
                fights[fights["R_fighter"] == ca]["weight_class"],
                fights[fights["B_fighter"] == ca]["weight_class"],
                fights[fights["R_fighter"] == cb]["weight_class"],
                fights[fights["B_fighter"] == cb]["weight_class"],
            ]
        )
        wc = sub.mode().iat[0] if not sub.empty else "Bantamweight"
        gender = "FEMALE" if wc.startswith("Women") else "MALE"

        try:
            row = build_upcoming_row(
                fighter_a=ca,
                fighter_b=cb,
                fight_date=pd.Timestamp("2026-05-31"),
                weight_class=wc,
                fights=fights,
                gender=gender,
                rankings=rankings,
            )
            row = attach_skill_for_upcoming(row, fights)
            pred = predict_ensemble(row)
        except Exception as e:
            print(f"SKIP {fa} vs {fb}: {e}")
            continue

        # p_real, p_corr are probabilities that fighter_a (Red) wins
        for acct_name, _bk0, kfrac, cap, model in ACCOUNTS:
            p_model = pred.real_mean if model == "real" else pred.corrupted_mean
            # decide side: A or B, whichever has positive edge after threshold
            best = None
            for side, p_side, p_mkt in [("A", p_model, pa_mkt), ("B", 1 - p_model, pb_mkt)]:
                r = kalshi_kelly_stake(p_side, p_mkt, bankrolls[acct_name], kfrac, cap)
                if r is not None:
                    # pick highest expected value (Kelly stake correlates with edge)
                    if best is None or r[0] > best[0][0]:
                        best = (r, side, p_side, p_mkt)
            if best is None:
                rows.append(
                    {
                        "fight": f"{fa} vs {fb}",
                        "account": acct_name,
                        "side": "-",
                        "p_model": round(p_model, 3),
                        "p_mkt_a": pa_mkt,
                        "decision": "SKIP",
                        "stake": 0,
                        "pnl": 0,
                        "bankroll_after": round(bankrolls[acct_name], 2),
                    }
                )
                continue
            (stake, n, fee, net_win, net_loss), side, p_side, p_mkt = best
            won = side == winner_side
            pnl = net_win if won else net_loss
            bankrolls[acct_name] += pnl
            rows.append(
                {
                    "fight": f"{fa} vs {fb}",
                    "account": acct_name,
                    "side": (fa if side == "A" else fb),
                    "p_model": round(p_side, 3),
                    "p_mkt": p_mkt,
                    "decision": "BET",
                    "stake": round(stake, 2),
                    "won": won,
                    "pnl": round(pnl, 2),
                    "bankroll_after": round(bankrolls[acct_name], 2),
                }
            )
        print(
            f"{fa} vs {fb}: p_real(A)={pred.real_mean:.3f} p_corr(A)={pred.corrupted_mean:.3f} | "
            f"mkt A/B={pa_mkt}/{pb_mkt} | winner={winner_side}"
        )

    df = pd.DataFrame(rows)
    print("\n=== Per-bet log ===")
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    print("\n=== Final bankrolls ===")
    for k, v in bankrolls.items():
        print(f"  Account {k}: $300.00 → ${v:.2f}  ({(v / 300 - 1) * 100:+.1f}%)")
    total = sum(bankrolls.values())
    print(f"  TOTAL:     $900.00 → ${total:.2f}  ({(total / 900 - 1) * 100:+.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

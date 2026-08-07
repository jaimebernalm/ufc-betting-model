"""Replay the 5 most recent Fight Night cards before 2026-05-30, chaining
bankrolls. Kalshi T-90min closes + 10-seed ensembles + DEPLOY.md sizing.
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
    ("A", 0.10, 0.10, "real"),
    ("B", 0.25, None, "real"),
    ("C", 0.25, None, "corrupted"),
]
CARD_DATES = ["2026-04-04", "2026-04-18", "2026-04-25", "2026-05-02", "2026-05-16", "2026-05-30"]

NAME_FIX = {
    "Yadong Song": "Song Yadong",
    "Qileng Aori": "Aori Qileng",
    "Harris Carlston": "Carlston Harris",
    "Mingyang Zhang": "Zhang Mingyang",
    "Yizha": "Yizha",
    "Cong Wang": "Wang Cong",
}


def kalshi_kelly_stake(p_model, p_market, bankroll, kfrac, cap):
    if p_model - p_market < EDGE_THR:
        return None
    if p_market <= 0.01 or p_market >= 0.99:
        return None
    fee_per = 0.07 * p_market * (1 - p_market)
    cost_per = p_market + fee_per
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
    return stake, n - stake, -stake  # stake, net_win, net_loss


def process_card(card_date, snap, fights, rankings, bankrolls):
    card = snap[snap["fight_date"].dt.strftime("%Y-%m-%d") == card_date].copy()
    card = card.sort_values("close_time")
    event_title = card["event_title"].iloc[0] if len(card) else ""
    print(f"\n{'=' * 78}\n{card_date}  {event_title}  ({len(card)} fights)")
    print(f"  Bankrolls in:  A=${bankrolls['A']:.2f}  B=${bankrolls['B']:.2f}  C=${bankrolls['C']:.2f}")

    bets, skips, wins = 0, 0, 0
    pnl_card = {"A": 0.0, "B": 0.0, "C": 0.0}

    for _, fight in card.iterrows():
        fa, fb = fight["fighter_a"], fight["fighter_b"]
        pa, pb = fight["close_yes_price_a"], fight["close_yes_price_b"]
        winner_side = fight["winner"]
        if pd.isna(pa) or pd.isna(pb) or winner_side not in ("A", "B"):
            print(f"  SKIP {fa} vs {fb}: missing price/winner")
            skips += 1
            continue
        try:
            ca = resolve_fighter_name(NAME_FIX.get(fa, fa), fights)
            cb = resolve_fighter_name(NAME_FIX.get(fb, fb), fights)
        except ValueError:
            print(f"  SKIP {fa} vs {fb}: name unresolved")
            skips += 1
            continue
        sub = pd.concat(
            [
                fights[fights["R_fighter"].isin([ca, cb])]["weight_class"],
                fights[fights["B_fighter"].isin([ca, cb])]["weight_class"],
            ]
        )
        wc = sub.mode().iat[0] if not sub.empty else "Bantamweight"
        gender = "FEMALE" if wc.startswith("Women") else "MALE"
        try:
            row = build_upcoming_row(
                fighter_a=ca,
                fighter_b=cb,
                fight_date=pd.Timestamp(card_date),
                weight_class=wc,
                fights=fights,
                gender=gender,
                rankings=rankings,
            )
            row = attach_skill_for_upcoming(row, fights)
            pred = predict_ensemble(row)
        except Exception as e:
            print(f"  SKIP {fa} vs {fb}: {e}")
            skips += 1
            continue

        for acct_name, kfrac, cap, model in ACCOUNTS:
            p_model = pred.real_mean if model == "real" else pred.corrupted_mean
            best = None
            for side, p_side, p_mkt in [("A", p_model, pa), ("B", 1 - p_model, pb)]:
                r = kalshi_kelly_stake(p_side, p_mkt, bankrolls[acct_name], kfrac, cap)
                if r is not None and (best is None or r[0] > best[0][0]):
                    best = (r, side)
            if best is None:
                continue
            (stake, net_win, net_loss), side = best
            won = side == winner_side
            pnl = net_win if won else net_loss
            bankrolls[acct_name] += pnl
            pnl_card[acct_name] += pnl
            bets += 1
            if won:
                wins += 1
            side_name = fa if side == "A" else fb
            print(
                f"  {acct_name}: BUY {side_name:<28} p_mdl={(p_model if side == 'A' else 1 - p_model):.2f} "
                f"mkt={pa if side == 'A' else pb:.2f} stake=${stake:6.2f} "
                f"{'WIN' if won else 'LOSS':4} pnl=${pnl:+7.2f}  br=${bankrolls[acct_name]:.2f}"
            )

    print(
        f"  Card result:  bets={bets} wins={wins} skips={skips}  "
        f"ΔA=${pnl_card['A']:+.2f} ΔB=${pnl_card['B']:+.2f} ΔC=${pnl_card['C']:+.2f}"
    )


def main():
    snap = pd.read_parquet("data/raw/kalshi/snapshots/historical_T-90min_perfight.parquet")
    fights = pd.read_parquet(PROCESSED / "fights.parquet")
    rankings = load_rankings()
    bankrolls = {"A": 300.0, "B": 300.0, "C": 300.0}
    print("Starting:  A=$300.00  B=$300.00  C=$300.00  TOTAL=$900.00")
    for d in CARD_DATES:
        process_card(d, snap, fights, rankings, bankrolls)
    print(f"\n{'=' * 78}\nFINAL after {len(CARD_DATES)} Fight Night cards:")
    for k, v in bankrolls.items():
        print(f"  Account {k}: $300.00 → ${v:.2f}  ({(v / 300 - 1) * 100:+.1f}%)")
    total = sum(bankrolls.values())
    print(f"  TOTAL:     $900.00 → ${total:.2f}  ({(total / 900 - 1) * 100:+.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

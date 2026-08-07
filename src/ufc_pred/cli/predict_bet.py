"""End-to-end bet sizing for the next UFC card on Polymarket.

Usage:
    PYTHONPATH=src .conda/bin/python scripts/predict_bet.py
    ufc-predict --fighter-a "Song Yadong" --fighter-b "Deiveson Figueiredo"

Pipeline:
  1. (auto) Refresh ufcstats if local data > 5 days stale
  2. Pull next UFC card from Polymarket + live order books
  3. For each fight: build synthetic row, fit skill (cached per month), run
     real + corrupted 10-seed ensembles
  4. Apply Kelly + bankroll cap + liquidity cap per account
  5. Print recommendation table
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime

import pandas as pd

from ufc_pred.inference import predict_one_fight
from ufc_pred.inference.polymarket_card import CardFight, fetch_next_card
from ufc_pred.inference.upcoming_builder import resolve_fighter_name
from ufc_pred.ingest.rankings_attach import load_rankings
from ufc_pred.ingest.ufcstats_state import UFCStatsStateSource
from ufc_pred.paths import CONFIGS, PROCESSED

FIGHTS_PATH = PROCESSED / "fights.parquet"
BANKROLLS_PATH = CONFIGS / "bankrolls.json"
BETS_DIR = PROCESSED / "bets"
UFCSTATS_STALE_DAYS = 5
POLYMARKET_FEE_COEFF = 0.03


def require_fresh_history(fights: pd.DataFrame) -> pd.DataFrame:
    last = pd.to_datetime(fights["date"]).max()
    age_days = (pd.Timestamp.now() - last).days
    if age_days < UFCSTATS_STALE_DAYS:
        print(f"  [ufcstats] last fight: {last.date()} ({age_days}d ago) — cache OK")
        return fights
    raise RuntimeError(
        f"fights.parquet ends {last.date()} ({age_days}d ago); refusing to size bets "
        "with stale skill/history data. Run scripts/update_ufcstats.py and rebuild "
        "the processed data first."
    )


def load_bankrolls() -> dict[str, float]:
    if not BANKROLLS_PATH.exists():
        return {"A": 300.0, "B": 300.0, "C": 300.0}
    d = json.loads(BANKROLLS_PATH.read_text())
    return {"A": float(d["A"]), "B": float(d["B"]), "C": float(d["C"])}


def _guess_weight_class(fight: CardFight, fights: pd.DataFrame) -> str:
    # Title pulled from Polymarket usually looks like:
    # "UFC Fight Night: A vs B (Bantamweight, Main Card)"
    import re

    m = re.search(r"\(([^,]+)\s*,", fight.event_title)
    if m:
        wc = m.group(1).strip()
        if wc in fights["weight_class"].unique():
            return wc
    # Fall back: each fighter's modal weight class
    a = resolve_fighter_name(fight.fighter_a, fights)
    b = resolve_fighter_name(fight.fighter_b, fights)
    cand = pd.concat(
        [
            fights[fights["R_fighter"] == a]["weight_class"],
            fights[fights["B_fighter"] == a]["weight_class"],
            fights[fights["R_fighter"] == b]["weight_class"],
            fights[fights["B_fighter"] == b]["weight_class"],
        ]
    )
    return cand.mode().iat[0] if not cand.empty else "Bantamweight"


def acct_pad(name: str) -> str:
    return f"{name:<10}"


def process_fight(
    fight: CardFight,
    fights: pd.DataFrame,
    rankings: pd.DataFrame,
    bankrolls: dict[str, float],
    bet_log: list[dict],
    state_source: UFCStatsStateSource,
) -> None:
    print(f"\n{'=' * 78}")
    print(f"  {fight.fighter_a}  vs  {fight.fighter_b}")
    print(f"  {fight.event_title}")
    print(
        f"  Fight date: {fight.fight_date.date()}   "
        f"Volume: ${fight.volume or 0:.0f}   Liquidity: ${fight.liquidity or 0:.0f}"
    )

    if not fight.asks_a or not fight.asks_b:
        print("  ⚠ Empty order book — skipping")
        return

    best_a = fight.asks_a[0][0]
    best_b = fight.asks_b[0][0]
    print(
        f"  Market: {fight.fighter_a} ASK {best_a * 100:.0f}¢  |  {fight.fighter_b} ASK {best_b * 100:.0f}¢"
    )

    try:
        weight_class = _guess_weight_class(fight, fights)
        result = predict_one_fight(
            fight,
            fights,
            rankings,
            bankrolls,
            weight_class_hint=weight_class,
            state_source=state_source,
            fee_coeff=POLYMARKET_FEE_COEFF,
        )
    except (OSError, RuntimeError, ValueError, IndexError, KeyError) as exc:
        print(f"  ⚠ Could not predict fight: {exc}")
        return

    if result["model"] is None:
        print(f"  ⚠ {result['error']}")
        return
    model = result["model"]
    print(
        f"  Model REAL:      p({fight.fighter_a})={model['p_a_real']:.3f}  "
        f"(10-seed std {model['real_per_seed_std']:.3f})"
    )
    print(
        f"  Model CORRUPTED: p({fight.fighter_a})={model['p_a_corrupted']:.3f}  "
        f"(10-seed std {model['corrupted_per_seed_std']:.3f})"
    )

    if not result["orders"]:
        print("\n  → NO BET — all 3 accounts SKIP")
        for rec in result["recommendations"]:
            print(f"    {rec['account']}: {rec['reason']}")
        return

    for _side, order in result["orders"].items():
        side_name = order["side_name"]
        total = order["total_stake_usd"]
        limit = order["limit_price"]
        avg_fill = order["avg_fill_price"]
        total_shares = order["total_shares"]
        total_net = order["total_net_profit_if_win"]
        print(
            f"\n  ➤ ORDER: BUY {side_name}  |  ${total:.2f}  |  limit {limit * 100:.1f}¢  |  "
            f"~{total_shares:.1f} shares  |  expected fill {avg_fill * 100:.2f}¢"
        )
        print(f"    Net profit if win: ${total_net:.2f}   Total loss if lose: ${total:.2f}")
        print("    ┌──────────┬────────┬──────────┬──────────┐")
        print("    │ Account  │ Stake  │ Shares   │ Net win  │  Bookkeeping (1 wallet, 3 virtual accounts)")
        print("    ├──────────┼────────┼──────────┼──────────┤")
        for account_order in order["per_account"]:
            print(
                f"    │ {account_order['account']:<8} │ "
                f"${account_order['stake_usd']:>5.2f} │ "
                f"{account_order['shares']:>8.2f} │ "
                f"${account_order['net_profit_if_win']:>6.2f} │"
            )
        print("    └──────────┴────────┴──────────┴──────────┘")

        bet_log.append(
            {
                "fight": f"{fight.fighter_a} vs {fight.fighter_b}",
                "fight_date": str(fight.fight_date.date()),
                "market_id": fight.market_id,
                "side_bet_on": side_name,
                "token_id": order["token"],
                "total_stake_usd": round(total, 2),
                "limit_price": round(limit, 4),
                "expected_avg_fill": round(avg_fill, 4),
                "total_shares": round(total_shares, 2),
                "per_account": [
                    {
                        "account": account_order["account"],
                        "stake_usd": account_order["stake_usd"],
                        "shares": account_order["shares"],
                        "net_profit_if_win": account_order["net_profit_if_win"],
                        "loss_if_lose": account_order["stake_usd"],
                    }
                    for account_order in order["per_account"]
                ],
                "winner": None,  # filled in by update_bankrolls.py
            }
        )
    skip_recs = [r for r in result["recommendations"] if r["decision"] == "SKIP"]
    if skip_recs:
        print("  (skip: " + ", ".join(f"{r['account']} {r['reason']}" for r in skip_recs) + ")")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fighter-a", help="Override: predict a single fight (Red corner)")
    parser.add_argument("--fighter-b", help="Override: predict a single fight (Blue corner)")
    parser.add_argument("--fight-date", help="Override fight date YYYY-MM-DD (with --fighter-a)")
    parser.add_argument(
        "--weight-class", default="Bantamweight", help="Override weight class (with --fighter-a)"
    )
    args = parser.parse_args()

    print(f"═══ UFC Bet Sizing — {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')} ═══\n")
    fights = pd.read_parquet(FIGHTS_PATH)
    try:
        fights = require_fresh_history(fights)
    except RuntimeError as exc:
        print(f"⛔ {exc}")
        return 2
    rankings = load_rankings()
    bankrolls = load_bankrolls()
    print(f"  [bankrolls] A=${bankrolls['A']:.2f}  B=${bankrolls['B']:.2f}  C=${bankrolls['C']:.2f}")

    if args.fighter_a and args.fighter_b:
        import json as _json

        import requests

        from ufc_pred.ingest.polymarket import GAMMA_BASE, fetch_order_book, iter_h2h_markets

        sess = requests.Session()
        r = sess.get(
            f"{GAMMA_BASE}/events",
            params={"closed": "false", "tag_slug": "ufc", "limit": 100},
            timeout=30,
        )
        target_market = None
        target_event = None
        for e in r.json():
            for m in iter_h2h_markets(e, include_closed=False):
                q = (m.get("question") or "").lower()
                if args.fighter_a.lower() in q and args.fighter_b.lower() in q:
                    target_market = m
                    target_event = e
                    break
            if target_market:
                break
        if not target_market:
            print(f"⚠ No live Polymarket market found for {args.fighter_a} vs {args.fighter_b}")
            return 1
        tokens = _json.loads(target_market["clobTokenIds"])
        outcomes = _json.loads(target_market["outcomes"])
        end = pd.Timestamp(target_market.get("endDate") or target_event.get("endDate"))
        if end.tzinfo is None:
            end = end.tz_localize("UTC")
        book_a = fetch_order_book(tokens[0], session=sess)
        book_b = fetch_order_book(tokens[1], session=sess)
        from ufc_pred.inference.polymarket_card import _book_to_levels

        fight = CardFight(
            event_title=target_event.get("title", ""),
            fight_date=end,
            fighter_a=outcomes[0],
            fighter_b=outcomes[1],
            token_a=str(tokens[0]),
            token_b=str(tokens[1]),
            market_id=str(target_market.get("id")),
            asks_a=_book_to_levels(book_a, "asks", False),
            asks_b=_book_to_levels(book_b, "asks", False),
            bids_a=_book_to_levels(book_a, "bids", True),
            bids_b=_book_to_levels(book_b, "bids", True),
            volume=float(target_market["volume"]) if target_market.get("volume") else None,
            liquidity=float(target_market["liquidity"]) if target_market.get("liquidity") else None,
        )
        card = [fight]
    else:
        print("  [polymarket] fetching next card ...")
        card = fetch_next_card()
        if not card:
            print("⚠ No upcoming UFC card found on Polymarket within 14 days.")
            return 1
        print(f"  [polymarket] {len(card)} fights on {card[0].fight_date.date()}")

    bet_log: list[dict] = []
    state_source = UFCStatsStateSource()
    try:
        for fight in card:
            process_fight(fight, fights, rankings, bankrolls, bet_log, state_source)
    finally:
        state_source.close()

    if bet_log:
        BETS_DIR.mkdir(parents=True, exist_ok=True)
        card_date = bet_log[0]["fight_date"]
        out_path = BETS_DIR / f"{card_date}_bets.json"
        out_path.write_text(
            json.dumps(
                {
                    "card_date": card_date,
                    "generated_at": datetime.now(UTC).isoformat(),
                    "bankrolls_at_bet": bankrolls,
                    "bets": bet_log,
                },
                indent=2,
            )
        )
        total_at_risk = sum(b["total_stake_usd"] for b in bet_log)
        total_bankroll = sum(bankrolls.values())
        print(f"\n{'=' * 78}")
        print(
            f"CARD SUMMARY:  {len(bet_log)} bets, total ${total_at_risk:.2f} at risk  "
            f"({total_at_risk / total_bankroll * 100:.1f}% of ${total_bankroll:.0f} bankroll)"
        )
        print(f"Bet plan saved → {out_path.relative_to(PROCESSED.parent.parent)}")
        print(f"After fights resolve: python scripts/update_bankrolls.py {card_date}")
    else:
        print(f"\n{'=' * 78}\nNo bets placed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

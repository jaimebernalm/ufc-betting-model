"""Update configs/bankrolls.json based on a saved bet plan + resolved winners.

Usage:
    ufc-bankrolls 2026-05-31

Reads:
    data/processed/bets/{card_date}_bets.json   (from `ufc-predict`)

For each bet:
  - Asks you (interactive) who won, OR auto-resolves from Polymarket if --auto
  - For each conceptual account: bankroll += shares (if won) - stake (if lost)

Stakes in the saved plan are already fee-adjusted at sizing time (Kalshi
charges its fee per contract upfront — see `inference.sizing`), so settlement
here is a plain shares-minus-stake calculation.

Updates configs/bankrolls.json with new per-account balances.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime

import requests

from ufc_pred.paths import CONFIGS, PROCESSED

GAMMA_BASE = "https://gamma-api.polymarket.com"
BETS_DIR = PROCESSED / "bets"
BANKROLLS_PATH = CONFIGS / "bankrolls.json"


def fetch_winner(market_id: str) -> str | None:
    """Return the winning outcome name from Polymarket, or None if unresolved."""
    r = requests.get(f"{GAMMA_BASE}/markets/{market_id}", timeout=15)
    if r.status_code != 200:
        return None
    m = r.json()
    if not m.get("closed"):
        return None
    outcomes = json.loads(m.get("outcomes") or "[]")
    prices = json.loads(m.get("outcomePrices") or "[]")
    if len(outcomes) != 2 or len(prices) != 2:
        return None
    try:
        return outcomes[0] if float(prices[0]) >= 0.5 else outcomes[1]
    except (TypeError, ValueError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("card_date", help="YYYY-MM-DD of the card")
    parser.add_argument(
        "--auto", action="store_true", help="Auto-resolve winners from Polymarket (default: prompt)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Show updates without writing bankrolls.json")
    args = parser.parse_args()

    bet_path = BETS_DIR / f"{args.card_date}_bets.json"
    if not bet_path.exists():
        print(f"⚠ No bet plan found at {bet_path}")
        return 1
    plan = json.loads(bet_path.read_text())
    bankrolls = json.loads(BANKROLLS_PATH.read_text())
    starting = {k: bankrolls[k] for k in ("A", "B", "C")}

    deltas = {"A": 0.0, "B": 0.0, "C": 0.0}
    fight_results = []

    for bet in plan["bets"]:
        side_bet_on = bet["side_bet_on"]
        winner = None
        if args.auto:
            winner = fetch_winner(bet["market_id"])
            if winner is None:
                print(f"⏳ {bet['fight']}: market still open, skipping")
                continue
        else:
            ans = (
                input(
                    f"\n{bet['fight']}\n"
                    f"  You bet ${bet['total_stake_usd']:.2f} on {side_bet_on}\n"
                    f"  Did {side_bet_on} win? [y/n/skip]: "
                )
                .strip()
                .lower()
            )
            if ans == "skip":
                continue
            won = ans.startswith("y")
            winner = side_bet_on if won else "OTHER"

        won = winner == side_bet_on
        bet["winner"] = winner
        for per_acct in bet["per_account"]:
            acct = per_acct["account"]
            if won:
                deltas[acct] += per_acct["net_profit_if_win"]
            else:
                deltas[acct] -= per_acct["loss_if_lose"]
        fight_results.append(
            {
                "fight": bet["fight"],
                "bet_on": side_bet_on,
                "winner": winner,
                "won": won,
                "stake": bet["total_stake_usd"],
            }
        )

    print(f"\n{'=' * 70}")
    print("RESULTS:")
    for fr in fight_results:
        status = "✓ WIN " if fr["won"] else "✗ LOSS"
        print(f"  {status}  {fr['fight'][:40]:<40}  bet ${fr['stake']:>7.2f} on {fr['bet_on']}")

    print(f"\n{'=' * 70}")
    print("BANKROLL CHANGES:")
    print(f"  {'Account':<10}{'Before':>10}{'Δ':>10}{'After':>10}")
    new_bankrolls = dict(bankrolls)
    for acct in ("A", "B", "C"):
        new_val = starting[acct] + deltas[acct]
        new_bankrolls[acct] = round(new_val, 2)
        sign = "+" if deltas[acct] >= 0 else ""
        print(f"  {acct:<10}${starting[acct]:>8.2f}  {sign}${deltas[acct]:>7.2f}  ${new_val:>8.2f}")
    total_before = sum(starting.values())
    total_after = sum(new_bankrolls[a] for a in ("A", "B", "C"))
    print(f"  {'TOTAL':<10}${total_before:>8.2f}   ${total_after - total_before:+.2f}  ${total_after:>8.2f}")

    new_bankrolls["last_updated"] = datetime.now(UTC).strftime("%Y-%m-%d")

    if args.dry_run:
        print("\n(dry-run — bankrolls.json NOT modified)")
        return 0

    BANKROLLS_PATH.write_text(json.dumps(new_bankrolls, indent=2) + "\n")
    plan["resolved_at"] = datetime.now(UTC).isoformat()
    plan["fight_results"] = fight_results
    bet_path.write_text(json.dumps(plan, indent=2))
    print(f"\n✓ Updated {BANKROLLS_PATH.relative_to(PROCESSED.parent.parent)}")
    print(f"✓ Resolved bet plan saved to {bet_path.relative_to(PROCESSED.parent.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

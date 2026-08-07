"""Preview the next open Kalshi UFC card with the deployed predict+size pipeline.

Read-only. Does NOT write notifications, touch the ledger, or place orders.

Run:  ufc-preview
"""

from __future__ import annotations

import json
import sys

import pandas as pd
from dotenv import load_dotenv

from ufc_pred.inference import predict_one_fight
from ufc_pred.inference.kalshi_card import fetch_next_card
from ufc_pred.ingest.rankings_attach import load_rankings
from ufc_pred.paths import CONFIGS, PROCESSED, ROOT


def main() -> int:
    load_dotenv(ROOT / ".env")
    return _preview()


def _preview() -> int:
    fights = pd.read_parquet(PROCESSED / "fights.parquet")
    last = pd.to_datetime(fights["date"]).max()
    print(f"ufcstats last fight: {last.date()}")

    bk = json.loads((CONFIGS / "bankrolls.json").read_text())
    bankrolls = {"A": float(bk["A"]), "B": float(bk["B"]), "C": float(bk["C"])}
    print(f"bankrolls: A={bankrolls['A']:.2f}  B={bankrolls['B']:.2f}  C={bankrolls['C']:.2f}")

    try:
        rankings = load_rankings()
    except Exception as e:
        print(f"rankings load failed ({e}); continuing without")
        rankings = pd.DataFrame()

    card = fetch_next_card()
    if not card:
        print("No open UFC card found on Kalshi within window.")
        return 0

    print(f"\nNext card: {card[0].fight_date.date()}  ({len(card)} fights)\n")

    for fight in card:
        try:
            res = predict_one_fight(fight, fights, rankings, bankrolls, weight_class_hint=None)
        except Exception as e:
            print("=" * 78)
            print(f"  {fight.fighter_a}  vs  {fight.fighter_b}")
            print(f"  ⚠️ skipped: {type(e).__name__}: {e}")
            continue
        print("=" * 78)
        print(f"  {fight.fighter_a}  vs  {fight.fighter_b}   [{res['fight'].get('weight_class', '?')}]")
        m = res["market"]
        print(
            f"  ask {fight.fighter_a}={m.get('best_ask_a')}  ask {fight.fighter_b}={m.get('best_ask_b')}"
            f"   vol=${m.get('volume') or 0:.0f} liq=${m.get('liquidity') or 0:.0f}"
        )
        if res["status"] != "ok":
            print(f"  status={res['status']}: {res['error']}")
            if res.get("model"):
                md = res["model"]
                print(
                    f"  p({fight.fighter_a}) real={md['p_a_real']:.3f} corr={md['p_a_corrupted']:.3f}"
                    f"  orient_gap_real={md.get('orientation_gap_real', 0):.4f}"
                )
            continue
        md = res["model"]
        print(
            f"  p({fight.fighter_a}) real={md['p_a_real']:.3f} corr={md['p_a_corrupted']:.3f}"
            f"  (raw {md['p_a_real_raw']:.3f})  orient_gap_real={md.get('orientation_gap_real', 0):.4f}"
        )
        for _side, o in res["orders"].items():
            # `side` is the fighter corner (A=fighter_a/Red, B=fighter_b/Blue), NOT an
            # account. total_stake_usd is the COMBINED stake across accounts on that side.
            print(
                f"   → BET {o['side_name']}  ${o['total_stake_usd']:.2f} total @ {o['avg_fill_price']:.3f}"
                f"  (limit {o['limit_price']:.3f}, {o['total_shares']:.0f} sh, "
                f"win +${o['total_net_profit_if_win']:.2f})"
            )
            for pa in o["per_account"]:
                print(f"        acct {pa['account']}: ${pa['stake_usd']:.2f}  ({pa['shares']:.0f} sh)")
        print(f"  {res['summary_line']}")

    print("\n(Preview only — no orders placed, no ledger writes.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Sync per-account bankrolls based on resolved Kalshi fights.

Each bet_notifications/<key>.json records per-account stakes + shares for one
fight. This script applies per-account PnL as soon as a fight is RESOLVED —
either definitively (Kalshi status finalized/determined) or *provisionally*
(market price pinned to ~0.99/0.01 after the fight started and stable for
90s+, per kalshi_resolution.market_resolution). Provisional results are
recorded in `card_state.fights_provisional` and trued-up when Kalshi actually
settles: if the settlement matches, the fight is promoted to fights_excluded
with no PnL change; if it was overturned (rare), the provisional PnL is
reversed and the correct PnL applied.

This kills the settlement lag: Kelly sizing for the NEXT fight sees the
previous fight's PnL within ~2 minutes of the result, instead of waiting
hours for Kalshi's settlement flag.

 1. Reads bankrolls.json (gets current A/B/C, exclusion + provisional state)
 2. Walks bet_notifications/<card_date>_*.json
 3. For each notification NOT in fights_excluded AND not dry_run:
    a. Queries Kalshi for the bet market's resolution state
    b. If settled or provisionally pinned: applies per-account PnL
    c. Tracks provisional fights for settlement true-up
 4. Writes updated bankrolls.json

Per-account PnL math (mirrors Kalshi quadratic fee model):
    payout_per_share = $1 if YES wins, $0 if NO wins
    A_payout = A_shares × payout_per_share
    A_fee    = 0.07 × A_shares × price × (1 - price)
    A_pnl    = A_payout - A_stake - A_fee

Designed to be idempotent — re-running never double-counts.
Call from bet_runner.py at the start of each tick.
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from datetime import UTC, datetime

from dotenv import load_dotenv

from ufc_pred.ingest.kalshi_client import KalshiClient
from ufc_pred.ingest.kalshi_resolution import market_resolution
from ufc_pred.paths import CONFIGS, PROCESSED, ROOT

load_dotenv(ROOT / ".env")


BANKROLLS_PATH = CONFIGS / "bankrolls.json"
NOTIF_DIR = PROCESSED / "bet_notifications"
AUDIT_DIR = PROCESSED.parent / "raw" / "ufc_schedule"
KALSHI_FEE_COEFF = 0.07


# ---------------------------------------------------------------------------
# Name matching for UFC.com results (surname, accent-folded)
# ---------------------------------------------------------------------------


def _fold(s: str) -> str:
    if not s:
        return ""
    return "".join(
        c for c in unicodedata.normalize("NFKD", s.lower()) if not unicodedata.combining(c)
    ).strip()


def _surname(s: str) -> str:
    parts = _fold(s).split()
    return parts[-1] if parts else ""


def _results_lookup(ufc_results) -> dict[frozenset, str]:
    """{frozenset({surname_red, surname_blue}): winner_display_name} for
    fights with a decided winner."""
    out = {}
    for r in ufc_results or []:
        if r.winner:
            out[frozenset({_surname(r.fighter_red), _surname(r.fighter_blue)})] = r.winner
    return out


def _load_ufc_results_for(card_date: str):
    """Self-serve UFC.com results via the audit log's event URL (used when the
    caller didn't pass results, e.g. manual CLI runs)."""
    try:
        audit = json.loads((AUDIT_DIR / f"{card_date}.json").read_text())
        url = audit["event"]["url"]
        from ufc_pred.ingest.ufc_schedule import fetch_card_results

        return fetch_card_results(url)
    except Exception:
        return None


def _load() -> dict:
    return json.loads(BANKROLLS_PATH.read_text())


def _save(data: dict) -> None:
    BANKROLLS_PATH.write_text(json.dumps(data, indent=2))


def _market_settled(client: KalshiClient, ticker: str) -> bool | None:
    """Return True if YES side won, False if NO won, None if not yet settled.
    Definitive settlement only (used for provisional true-up)."""
    try:
        r = client.get_market(ticker)
    except Exception as e:
        print(f"  [sync] kalshi error for {ticker}: {e}", file=sys.stderr)
        return None
    m = r.get("market", r)
    status = m.get("status", "")
    result = m.get("result", "")
    if status in ("finalized", "determined") and result in ("yes", "no"):
        return result == "yes"
    return None


def _market_resolved(client: KalshiClient, ticker: str, fight_start_utc) -> tuple[str, bool] | None:
    """Settled OR provisionally pinned. Returns (kind, yes_won) or None."""
    try:
        return market_resolution(client, ticker, fight_start_utc=fight_start_utc)
    except Exception as e:
        print(f"  [sync] kalshi error for {ticker}: {e}", file=sys.stderr)
        return None


def _per_account_pnl(per_account: list[dict], yes_won: bool, price: float) -> dict[str, float]:
    """Given the per-account bet record + whether YES won, compute per-account PnL.

    Records written after 2026-06-11 carry a `fee_usd` field and their
    `stake_usd` is the full cash outlay (price + upfront fee), so PnL is
    simply payout − stake. Older records' stake excludes the fee, which must
    be charged separately here.
    """
    out = {}
    payout_per_share = 1.0 if yes_won else 0.0
    for entry in per_account:
        acct = entry["account"]
        stake = entry["stake_usd"]
        shares = entry["shares"]
        payout = shares * payout_per_share
        if entry.get("fee_usd") is not None:
            pnl = payout - stake
        else:
            fee = KALSHI_FEE_COEFF * shares * price * (1.0 - price)
            pnl = payout - stake - fee
        out[acct] = round(pnl, 4)
    return out


def sync(*, client: KalshiClient | None = None, ufc_results=None, verbose: bool = False) -> dict:
    """Apply all newly-resolved fights to bankrolls. Returns the updated dict.

    Resolution sources, fastest first:
      1. UFC.com result badges (`ufc_results`, list of FightResult) — winner
         known ~1-2 min after the fight; booked as PROVISIONAL PnL.
      2. Kalshi market resolution (settled -> final; legacy pin -> provisional).
    Provisional entries are trued-up at official Kalshi settlement.

    Idempotent: skips fights already in `card_state.fights_excluded`.
    """
    data = _load()
    state = data.get("card_state", {})
    card_date = state.get("card_date")
    excluded: set[str] = set(state.get("fights_excluded", []))
    provisional: dict = dict(state.get("fights_provisional", {}))

    if not card_date:
        return data  # no card baseline set; nothing to sync

    if ufc_results is None:
        ufc_results = _load_ufc_results_for(card_date)
    ufc_winners = _results_lookup(ufc_results)

    client = client or KalshiClient()
    applied = []

    for notif_path in sorted(NOTIF_DIR.glob(f"{card_date}_*.json")):
        key = notif_path.stem
        if key in excluded:
            continue
        try:
            record = json.loads(notif_path.read_text())
        except json.JSONDecodeError:
            continue
        if record.get("dry_run"):
            continue
        rec = record.get("recommendation", {})
        if rec.get("status") != "ok":
            excluded.add(key)
            continue
        orders = rec.get("orders", {})
        if not orders:
            excluded.add(key)
            continue
        # The watchdog only ever bets one side per fight.
        side_letter, order = next(iter(orders.items()))
        bet_ticker = order["token"]
        bet_price = order["avg_fill_price"]

        # ---- True-up path: PnL already applied provisionally ----
        if key in provisional:
            settled_yes = _market_settled(client, bet_ticker)
            if settled_yes is None:
                continue  # still awaiting official settlement; PnL already in
            prov = provisional.pop(key)
            if bool(settled_yes) != bool(prov["yes_won"]):
                # Overturned result (rare): reverse provisional PnL, apply correct
                for acct, delta in prov["pnl"].items():
                    data[acct] = round(data[acct] - delta, 4)
                pnl = _per_account_pnl(order["per_account"], settled_yes, bet_price)
                for acct, delta in pnl.items():
                    data[acct] = round(data[acct] + delta, 4)
                applied.append((key, pnl, settled_yes))
                print(
                    f"  [sync] {key}: OVERTURNED — provisional reversed, settled PnL applied",
                    file=sys.stderr,
                )
            elif verbose:
                print(f"  [sync] {key}: settlement confirms provisional result")
            excluded.add(key)
            continue

        # ---- New resolution, fastest source first ----
        # (a) UFC.com result badge: winner known ~1-2 min after the fight.
        kind, side_won_yes, source = None, None, None
        fight_meta = record.get("fight", {})
        fkey = frozenset(
            {
                _surname(fight_meta.get("fighter_a_ufc", "")),
                _surname(fight_meta.get("fighter_b_ufc", "")),
            }
        )
        ufc_winner = ufc_winners.get(fkey)
        if ufc_winner is not None:
            kind = "provisional"
            side_won_yes = _surname(ufc_winner) == _surname(order["side_name"])
            source = "ufc"
        else:
            # (b) Kalshi market: settled (final) or legacy pin (provisional).
            resolution = _market_resolved(client, bet_ticker, record.get("fight_date_utc"))
            if resolution is None:
                if verbose:
                    print(f"  [sync] {key}: not resolved yet")
                continue
            kind, side_won_yes = resolution
            source = "kalshi"
            if kind == "closed":
                # Fight over but result not official — nothing to book yet.
                if verbose:
                    print(f"  [sync] {key}: fight ended, awaiting result")
                continue

        pnl = _per_account_pnl(order["per_account"], side_won_yes, bet_price)
        # Apply
        for acct, delta in pnl.items():
            data[acct] = round(data[acct] + delta, 4)
        if kind == "settled":
            excluded.add(key)
        else:
            provisional[key] = {"yes_won": bool(side_won_yes), "pnl": pnl, "source": source}
        applied.append((key, pnl, side_won_yes))
        if verbose:
            side_name = order["side_name"]
            outcome = "WON" if side_won_yes else "LOST"
            tag = "" if kind == "settled" else f" (provisional, via {source})"
            print(
                f"  [sync] {key}: {side_name} {outcome}{tag} | "
                + " ".join(f"{a}=${v:+.2f}" for a, v in pnl.items())
            )

    # Persist updated state
    state["fights_excluded"] = sorted(excluded)
    state["fights_provisional"] = provisional
    data["card_state"] = state
    data["last_synced"] = datetime.now(UTC).isoformat()
    _save(data)

    if verbose and applied:
        print(
            f"  [sync] applied {len(applied)} fight(s); "
            + f"bankrolls now A=${data['A']:.2f} B=${data['B']:.2f} C=${data['C']:.2f}"
        )
    elif verbose:
        print("  [sync] no new resolved fights")
    if verbose:
        # Reconciliation hint only: cash excludes open-position value, so
        # mid-card it will read LOW vs the virtual ledger. Big divergence
        # post-card = drift worth investigating.
        try:
            cash = client.get_balance().get("balance", 0) / 100.0
            virtual = data["A"] + data["B"] + data["C"]
            print(
                f"  [sync] Kalshi cash ${cash:.2f} (excl. open positions) | "
                f"virtual ledger total ${virtual:.2f}"
            )
        except Exception:
            pass

    return data


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verbose", "-v", action="store_true", default=True)
    args = ap.parse_args()
    data = sync(verbose=args.verbose)
    print(
        f"\nA=${data['A']:.2f}  B=${data['B']:.2f}  C=${data['C']:.2f}  "
        f"(total ${data['A'] + data['B'] + data['C']:.2f})"
    )


if __name__ == "__main__":
    main()

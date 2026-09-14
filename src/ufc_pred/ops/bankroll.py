"""Fill-based reconciliation. Legacy helper functions remain for audit compatibility.

Historical design (superseded by fill_ledger below):
Sync per-account bankrolls based on resolved Kalshi fights.

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
    """Reconcile actual fills across all cards; recommendations only allocate fills."""
    import fcntl

    from ufc_pred.ops.fill_ledger import migration_anchor, reconcile
    from ufc_pred.ops.fills import FILLS_PATH

    lock_path = BANKROLLS_PATH.with_suffix(".lock")
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = _load()
        anchor = data.get("fill_ledger")
        if anchor is None:
            anchor = migration_anchor(data)
        fills = json.loads(FILLS_PATH.read_text()) if FILLS_PATH.exists() else {}
        records = []
        for path in sorted(NOTIF_DIR.glob("*.json")) + sorted((NOTIF_DIR / "revisions").glob("*.json")):
            records.append(json.loads(path.read_text()))
        provisional = reconcile(anchor, fills, records, anchor.get("resolutions", {}))
        resolutions = dict(anchor.get("resolutions", {}))
        client = client or KalshiClient()
        for ticker in {e["ticker"] for e in provisional["entries"].values()}:
            if resolutions.get(ticker, {}).get("final"):
                continue
            related = [
                (r, o)
                for r in records
                if not r.get("dry_run")
                for o in r.get("recommendation", {}).get("orders", {}).values()
                if o.get("token") == ticker
            ]
            start = related[-1][0].get("fight_date_utc") if related else None
            if start:
                resolution = _market_resolved(client, ticker, start)
            else:
                settled = _market_settled(client, ticker)
                resolution = ("settled", settled) if settled is not None else None
            if not resolution or resolution[0] != "settled":
                from ufc_pred.ingest.identity import canonical, normalise

                for record, order in related:
                    fight = record.get("fight", {})
                    wc = fight.get("weight_class")
                    try:
                        pair = {
                            normalise(canonical(fight.get("fighter_a_ufc", ""), wc)),
                            normalise(canonical(fight.get("fighter_b_ufc", ""), wc)),
                        }
                        for result in ufc_results or []:
                            if result.winner and pair == {
                                normalise(canonical(result.fighter_red, wc)),
                                normalise(canonical(result.fighter_blue, wc)),
                            }:
                                won = normalise(canonical(result.winner, wc)) == normalise(
                                    canonical(order["side_name"], wc)
                                )
                                resolution = ("provisional", won)
                    except ValueError:
                        pass  # Ambiguous identities may only settle via the venue.

            if resolution and resolution[0] in ("settled", "provisional"):
                resolutions[ticker] = {"yes_won": bool(resolution[1]), "final": resolution[0] == "settled"}
        ledger = reconcile(anchor, fills, records, resolutions)
        data["fill_ledger"] = ledger
        for account in ("A", "B", "C"):
            data[account] = round(ledger["equity"][account], 6)
        data["last_synced"] = datetime.now(UTC).isoformat()
        temp = BANKROLLS_PATH.with_suffix(".tmp")
        temp.write_text(json.dumps(data, indent=2))
        temp.replace(BANKROLLS_PATH)
        if verbose:
            print(f"[fills ledger] {len(ledger['entries'])} fills, {len(ledger['quarantine'])} quarantined")
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

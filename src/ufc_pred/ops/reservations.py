"""Unfilled recommendations reserve cash without ever becoming realised PnL."""

import json
from datetime import datetime
from pathlib import Path

from ufc_pred.ops.fill_ledger import normalise_fill


def pending_cash(directory: Path, client, *, fills=None):
    reserved = 0.0
    records = []
    for path in list(directory.glob("*.json")) + list((directory / "revisions").glob("*.json")):
        record = json.loads(path.read_text())
        if record.get("recommendation", {}).get("status") != "ok":
            continue
        records.append(record)
    # Retries don't create additional reservations for the same market.
    latest = {}
    for record in sorted(records, key=lambda r: r.get("captured_at_utc", "")):
        for order in record.get("recommendation", {}).get("orders", {}).values():
            latest[order["token"]] = (record, order)
    for ticker, (record, order) in latest.items():
        market = client.get_market(ticker)
        market = market.get("market", market)
        if market.get("status") in ("closed", "finalized", "determined", "settled"):
            continue
        filled = 0.0
        if record.get("mode") != "shadow":
            capture = datetime.fromisoformat(record["captured_at_utc"].replace("Z", "+00:00"))
            for raw in (fills or {}).values():
                fill = normalise_fill(raw)
                if (
                    fill["ticker"] == ticker
                    and fill["side"] == "yes"
                    and fill["action"] == "buy"
                    and datetime.fromisoformat(fill["created_time"].replace("Z", "+00:00")) >= capture
                ):
                    filled += fill["quantity"] * fill["price"] + fill["fee"]
        reserved += max(0.0, float(order["total_stake_usd"]) - filled)
    return reserved

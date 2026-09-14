"""Deterministic fill-based cash, positions and PnL, independent of card rollover.

The migration anchor preserves the operator's reconciled balances. Fills before
that timestamp belong to the old ledger and are never applied a second time.
Every later fill is included once by venue ID, including fills without alerts.
"""

import math
from collections import defaultdict
from datetime import UTC, datetime

ACCOUNTS = ("A", "B", "C")


def normalise_fill(fill):
    side = fill.get("side") or fill.get("outcome_side")
    action = fill.get("action")
    if side not in ("yes", "no") or action not in ("buy", "sell"):
        raise ValueError("unsupported fill side/action")
    qty = float(fill.get("count_fp", fill.get("count", 0)))
    price = (
        float(fill[f"{side}_price_dollars"])
        if f"{side}_price_dollars" in fill
        else float(fill[f"{side}_price"]) / 100
    )
    if "fee_cost" not in fill:
        raise ValueError("actual fee_cost missing; refusing estimated accounting")
    fee = float(fill["fee_cost"])
    if not all(math.isfinite(x) for x in (qty, price, fee)) or qty <= 0 or not 0 <= price <= 1 or fee < 0:
        raise ValueError("invalid fill quantity/price/fee")
    return {
        "ticker": fill.get("ticker") or fill["market_ticker"],
        "side": side,
        "action": action,
        "quantity": qty,
        "price": price,
        "fee": fee,
        "created_time": fill["created_time"],
    }


def migration_anchor(data, now=None):
    return {
        "version": 1,
        "started_at": now or datetime.now(UTC).isoformat(),
        "baseline": {a: float(data[a]) for a in ACCOUNTS},
        "baseline_source": "operator ledger preserved; historical PnL is not re-applied",
        "entries": {},
        "quarantine": {},
    }


def allocation_for(fill, records):
    candidates = []
    for record in records:
        if record.get("dry_run") or record.get("mode") == "shadow":
            continue
        captured = record.get("captured_at_utc", "")
        if not captured or datetime.fromisoformat(captured.replace("Z", "+00:00")) > datetime.fromisoformat(
            fill["created_time"].replace("Z", "+00:00")
        ):
            continue
        for order in record.get("recommendation", {}).get("orders", {}).values():
            if order.get("token") == fill["ticker"] and fill["side"] == "yes":
                weights = {a: 0.0 for a in ACCOUNTS}
                for e in order.get("per_account", []):
                    if e["account"] in weights:
                        weights[e["account"]] += float(e.get("shares", 0))
                if sum(weights.values()) > 0:
                    candidates.append((captured, weights))
    if candidates:
        weights = max(candidates, key=lambda x: x[0])[1]
        source = "preceding recommendation share ratios"
    else:
        weights = dict(zip(ACCOUNTS, (1.0, 2.0, 6.0), strict=True))
        source = "documented off-alert 1:2:6 allocation"
    total = sum(weights.values())
    return {a: w / total for a, w in weights.items()}, source


def reconcile(anchor, fills, records, resolutions):
    """Pure replay; resolutions map ticker to {yes_won, final}.

    Provisional marks affect equity, but do not release spendable cash. Sells
    consume the existing position at average cost; unexpected shorts quarantine.
    """
    start = datetime.fromisoformat(anchor["started_at"].replace("Z", "+00:00"))
    entries = dict(anchor.get("entries", {}))
    quarantine = {}
    for fid, raw in fills.items():
        try:
            timestamp = datetime.fromisoformat(raw["created_time"].replace("Z", "+00:00"))
            if timestamp < start:
                continue
            if fid not in entries:
                fill = normalise_fill(raw)
                weights, source = allocation_for(fill, records)
                entries[fid] = {**fill, "weights": weights, "allocation_source": source}
        except (KeyError, ValueError, TypeError) as exc:
            quarantine[fid] = str(exc)
    cash = {a: float(anchor["baseline"][a]) for a in ACCOUNTS}
    positions = defaultdict(lambda: {a: {"quantity": 0.0, "cost": 0.0} for a in ACCOUNTS})
    for fid, e in sorted(entries.items(), key=lambda item: (item[1]["created_time"], item[0])):
        pos = positions[(e["ticker"], e["side"])]
        if e["action"] == "buy":
            for a, w in e["weights"].items():
                cost = (e["quantity"] * e["price"] + e["fee"]) * w
                cash[a] -= cost
                pos[a]["quantity"] += e["quantity"] * w
                pos[a]["cost"] += cost
        else:
            held = sum(v["quantity"] for v in pos.values())
            if held + 1e-6 < e["quantity"]:
                quarantine[fid] = "sell exceeds tracked holdings"
                continue
            for a in ACCOUNTS:
                w = pos[a]["quantity"] / held
                fraction = e["quantity"] / held
                cash[a] += (e["quantity"] * e["price"] - e["fee"]) * w
                pos[a]["quantity"] *= 1 - fraction
                pos[a]["cost"] *= 1 - fraction
    equity = dict(cash)
    locked = dict.fromkeys(ACCOUNTS, 0.0)
    for (ticker, side), pos in positions.items():
        resolution = resolutions.get(ticker)
        for a, p in pos.items():
            if resolution:
                won = bool(resolution["yes_won"]) == (side == "yes")
                payout = p["quantity"] * won
                equity[a] += payout
                if resolution["final"]:
                    cash[a] += payout
                else:
                    locked[a] += p["cost"]
            else:
                equity[a] += p["cost"]
                locked[a] += p["cost"]
    return {
        **anchor,
        "entries": entries,
        "quarantine": quarantine,
        "equity": equity,
        "cash": cash,
        "locked_cost": locked,
        "resolutions": resolutions,
    }

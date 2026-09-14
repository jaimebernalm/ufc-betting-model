"""Prospective scoring and an explicitly hypothetical paper execution ledger.

Every paper fill derives from an archived pre-fight book recommendation.
Nothing here submits an order or modifies real balances. Actual realised ROI
requires real fills and is deliberately not inferred from these records.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ufc_pred.ops.fill_ledger import reconcile


def project(directory: Path, protocol: dict, client=None):
    captures = directory / "captures"
    records = []
    for path in [*captures.glob("*.json"), *(captures / "revisions").glob("*.json")]:
        r = json.loads(path.read_text())
        if r.get("mode") == "shadow" and r.get("recommendation", {}).get("status") in ("ok", "skip"):
            records.append(r)
    # First usable pre-fight prediction is frozen, even if a retry later differs.
    records.sort(key=lambda r: r.get("captured_at_utc", ""))
    first = {}
    for r in records:
        key = r.get("fight", {}).get("kalshi_event")
        if (
            key
            and r.get("model") is not None
            and r.get("capture_trigger") in ("first_fight", "prev_resolved", "fallback_time")
        ):
            first.setdefault(key, r)
    state_path = directory / "paper_ledger.json"
    old = json.loads(state_path.read_text()) if state_path.exists() else {}
    resolutions = dict(old.get("resolutions", {}))
    entries = {}
    account_bets = {a: [] for a in "ABC"}
    for key, r in first.items():
        ticker_a = r.get("fight", {}).get("kalshi_ticker_a")
        if ticker_a and client is not None and not resolutions.get(ticker_a, {}).get("final"):
            market = client.get_market(ticker_a)
            market = market.get("market", market)
            if market.get("status") in ("determined", "finalized", "settled") and market.get("result") in (
                "yes",
                "no",
            ):
                resolutions[ticker_a] = {"yes_won": market["result"] == "yes", "final": True}
        for side, order in r["recommendation"].get("orders", {}).items():
            ticker = order["token"]
            if client is not None and not resolutions.get(ticker, {}).get("final"):
                market = client.get_market(ticker)
                market = market.get("market", market)
                if market.get("status") in ("determined", "finalized", "settled") and market.get(
                    "result"
                ) in ("yes", "no"):
                    resolutions[ticker] = {"yes_won": market["result"] == "yes", "final": True}
            quantity = sum(float(a["shares"]) for a in order["per_account"])
            fees = sum(float(a.get("fee_usd", 0)) for a in order["per_account"])
            weights = {a: 0.0 for a in "ABC"}
            for a in order["per_account"]:
                weights[a["account"]] += float(a["shares"]) / quantity
            entries[f"{key}:{side}"] = {
                "ticker": ticker,
                "side": "yes",
                "action": "buy",
                "quantity": quantity,
                "price": float(order["avg_fill_price"]),
                "fee": fees,
                "created_time": r["captured_at_utc"],
                "weights": weights,
                "allocation_source": "hypothetical paper execution of archived recommendation",
            }
            resolved = resolutions.get(ticker)
            if resolved:
                for a in order["per_account"]:
                    name = a["account"]
                    p = float(r["model"]["p_a_corrupted" if name == "C" else "p_a_real"])
                    if side == "B":
                        p = 1 - p
                    cost = float(a["stake_usd"])
                    payout = float(a["shares"]) * resolved["yes_won"]
                    account_bets[name].append(
                        {
                            "date": pd.Timestamp(r["fight_date_utc"])
                            .tz_convert("US/Eastern")
                            .strftime("%Y-%m-%d"),
                            "p": p,
                            "won": int(resolved["yes_won"]),
                            "flat_pnl": payout / cost - 1,
                            "paper_pnl": payout - cost,
                        }
                    )
    anchor = {
        "version": 1,
        "started_at": protocol["frozen_at_utc"],
        "baseline": protocol["reference_bankrolls"],
        "entries": entries,
    }
    ledger = reconcile(anchor, {}, [], resolutions)
    ledger["execution_assumption"] = "immediate paper fill at captured displayed depth; not an actual fill"
    state_path.write_text(json.dumps(ledger, indent=2))
    bank = {**ledger["equity"], "paper_cash": ledger["cash"], "mode": "shadow"}
    (directory / "bankroll_reference.json").write_text(json.dumps(bank, indent=2))
    metrics = {}
    for name, bets in account_bets.items():
        metric = {
            "settled_bets": len(bets),
            "paper_equity": ledger["equity"][name],
            "paper_growth_multiple": ledger["equity"][name] / protocol["reference_bankrolls"][name],
        }
        if bets:
            p = np.array([r["p"] for r in bets])
            y = np.array([r["won"] for r in bets])
            pnl = np.array([r["flat_pnl"] for r in bets])
            metric.update(
                flat_roi=float(pnl.mean()),
                selected_brier=float(np.mean((p - y) ** 2)),
                mean_probability=float(p.mean()),
                win_rate=float(y.mean()),
                selected_log_loss=float(
                    -np.mean(
                        y * np.log(np.clip(p, 1e-6, 1 - 1e-6))
                        + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1 - 1e-6))
                    )
                ),
            )
            curve = np.r_[
                protocol["reference_bankrolls"][name],
                protocol["reference_bankrolls"][name] + np.cumsum([r["paper_pnl"] for r in bets]),
            ]
            peaks = np.maximum.accumulate(curve)
            metric["paper_resolved_order_max_drawdown"] = float(np.max((peaks - curve) / peaks))
            groups = pd.DataFrame(bets).groupby("date").flat_pnl.agg(["sum", "count"])
            metric["resolved_cards"] = len(groups)
            metric["card_cluster_ci95"] = None
            if len(groups) >= 5:
                rng = np.random.default_rng(914)
                idx = rng.integers(len(groups), size=(10000, len(groups)))
                boot = groups["sum"].to_numpy()[idx].sum(1) / groups["count"].to_numpy()[idx].sum(1)
                metric["card_cluster_ci95"] = np.quantile(boot, [0.025, 0.975]).tolist()
        metrics[name] = metric
    output = {
        "status": "awaiting future settled observations"
        if not any(account_bets.values())
        else "prospective observations accumulating",
        "captured_fights": len(first),
        "raw_capture_records": len(records),
        "realised_trading_roi": None,
        "execution_assumption": ledger["execution_assumption"],
        "accounts": metrics,
    }
    (directory / "prospective_metrics.json").write_text(json.dumps(output, indent=2))
    return ledger

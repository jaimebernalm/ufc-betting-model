"""Paired frozen-artifact comparison and bounded-liquidity cash replay.

No search over strategy parameters. Execution scenarios are sensitivity tests,
not reconstructed historical books when depth was never captured.
"""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from ufc_pred.backtest.strategy_grid import predict
from ufc_pred.backtest.universe import add_prior_fight_counts
from ufc_pred.features.static_v1 import _swap_red_blue
from ufc_pred.inference.ensemble_predict import _load_bundle
from ufc_pred.inference.sizing import default_accounts, size_bets_combined
from ufc_pred.ingest.identity import canonical, normalise
from ufc_pred.paths import ROOT

OUT = ROOT / "artifacts/corrected_2026_09_14"
BASE = ROOT / "artifacts/baselines/2026_09_14_before_fixes"
CUTOFF = pd.Timestamp("2025-11-30")


def sharpen(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return 1 / (1 + np.exp(-1.25 * np.log(p / (1 - p))))


def predictions(frame, models):
    rev = _swap_red_blue(frame)
    out = {}
    for kind in ["real", "corrupted"]:
        ps = []
        for seed in range(10):
            b = _load_bundle(models / f"v3_{kind}_2025_11_30_seed{seed}.joblib")
            ps.append((predict(b, frame) + 1 - predict(b, rev)) / 2)
        out[kind] = sharpen(np.mean(ps, axis=0))
    return out


def key_names(a, b, wc):
    return frozenset((normalise(canonical(a, wc)), normalise(canonical(b, wc))))


def match_row(rows, a, b, date):
    matches = []
    for idx, r in rows.get(pd.Timestamp(date).normalize(), []):
        try:
            if key_names(a, b, r.weight_class) == key_names(r.R_fighter, r.B_fighter, r.weight_class):
                matches.append((idx, normalise(canonical(a, r.weight_class)) == normalise(r.R_fighter)))
        except ValueError:
            pass
    return matches[0] if len(matches) == 1 else None


def flat_metrics(frame, p):
    pr, pb = frame.price_red.to_numpy(float), frame.price_blue.to_numpy(float)
    red = p - pr >= (1 - p) - pb
    prob = np.where(red, p, 1 - p)
    price = np.where(red, pr, pb)
    cost = price + frame.fee_rate.to_numpy(float) * np.power(
        price * (1 - price), frame.fee_exponent.to_numpy(float)
    )
    take = (prob - price >= 0.03 - 1e-12) & (prob > cost) & (~frame.has_debut.to_numpy(bool))
    won = np.where(red, frame.Winner.eq("Red"), frame.Winner.eq("Blue")).astype(int)
    pnl = won[take] / cost[take] - 1
    if len(pnl) == 0:
        return {"opportunities": len(frame), "bets": 0}
    selected = pd.DataFrame(
        {"date": frame.loc[take, "date"].to_numpy(), "p": prob[take], "won": won[take], "pnl": pnl}
    )
    grouped = selected.groupby("date").pnl.agg(["sum", "count"])
    rng = np.random.default_rng(914)
    idx = rng.integers(len(grouped), size=(10000, len(grouped)))
    boot = grouped["sum"].to_numpy()[idx].sum(1) / grouped["count"].to_numpy()[idx].sum(1)
    bins = []
    for lo in np.arange(0, 1, 0.1):
        g = selected[(selected.p >= lo) & (selected.p < lo + 0.1)]
        if len(g):
            bins.append(
                {"lo": float(lo), "n": len(g), "mean_p": float(g.p.mean()), "win_rate": float(g.won.mean())}
            )
    return {
        "opportunities": len(frame),
        "bets": len(pnl),
        "wins": int(won[take].sum()),
        "flat_roi": float(pnl.mean()),
        "card_cluster_ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
        "selected_brier": float(brier_score_loss(won[take], prob[take])),
        "selected_log_loss": float(log_loss(won[take], prob[take], labels=[0, 1])),
        "calibration_bins": bins,
        "top_5_winning_bets_pnl": float(np.sort(pnl)[-5:].sum()),
        "total_flat_pnl": float(pnl.sum()),
    }


def replay(frame, variant, depth, slippage, settle_lag):
    """Cards release cash together; intra-card ordering unavailable => explicit scenario.

    Liquid depth is cash notional within 3c, consumed by all three accounts.
    Cash remains locked until card end or next card, as the scenario specifies.
    """
    banks = json.loads((BASE / "configs/bankrolls.json").read_text())
    banks = {a: float(banks[a]) for a in "ABC"}
    cash = dict(banks)
    initial = dict(banks)
    curves = {a: [banks[a]] for a in "ABC"}
    logs = []
    pending = []
    dates = sorted(frame.date.unique())
    for card_i, date in enumerate(dates):
        if settle_lag == "next_card":
            for a, stake, payout in pending:
                cash[a] += payout
            pending = []
        card = frame[frame.date == date].sort_values(
            ["sequence_proxy", "R_fighter", "B_fighter"], na_position="last"
        )
        for idx, r in card.iterrows():
            if r.has_debut:
                logs.append(
                    {
                        "variant": variant,
                        "date": str(date),
                        "fight": f"{r.R_fighter} vs {r.B_fighter}",
                        "account": "all",
                        "decision": "SKIP",
                        "reason": "no prior UFC history",
                    }
                )
                continue

            # 1% of stated depth at best; the rest at adverse levels.
            def book(price):
                return [
                    (price, depth * 0.01 / price),
                    (min(0.99, price + slippage), depth * 0.99 / min(0.99, price + slippage)),
                ]

            accts = default_accounts(banks)
            probs = {a: float(r[f"{variant}_real" if a != "C" else f"{variant}_corrupted"]) for a in "ABC"}
            recs = size_bets_combined(
                accts,
                probs,
                book(r.price_red),
                book(r.price_blue),
                fee_coeff=float(r.fee_rate),
                available_cash=max(0, sum(cash.values())),
            )
            for rec in recs:
                if rec.decision != "BET":
                    logs.append(
                        {
                            "variant": variant,
                            "date": str(date),
                            "fight": f"{r.R_fighter} vs {r.B_fighter}",
                            "account": rec.account,
                            "decision": "SKIP",
                            "reason": rec.reason,
                        }
                    )
                    continue
                a = rec.account
                scale = min(1, max(0, cash[a]) / rec.stake_usd)
                stake = rec.stake_usd * scale
                shares = rec.shares * scale
                if stake < 0.50:
                    continue
                won = (r.Winner == "Red") == (rec.side == "A")
                payout = shares * won
                pnl = payout - stake
                cash[a] -= stake
                banks[a] += pnl
                curves[a].append(banks[a])
                pending.append((a, stake, payout))
                logs.append(
                    {
                        "variant": variant,
                        "date": str(date),
                        "fight": f"{r.R_fighter} vs {r.B_fighter}",
                        "account": a,
                        "decision": "BET",
                        "stake": stake,
                        "shares": shares,
                        "price": rec.avg_fill_price,
                        "pnl": pnl,
                        "bank": banks[a],
                        "recommendation_stake": rec.stake_usd,
                        "fill_shortfall": rec.stake_usd - stake,
                    }
                )
        if settle_lag == "card_end":
            for a, stake, payout in pending:
                cash[a] += payout
            pending = []
    result = {}
    for a in "ABC":
        bets = [r for r in logs if r["account"] == a and r["decision"] == "BET"]
        curve = np.array(curves[a])
        peak = np.maximum.accumulate(curve)
        result[a] = {
            "bets": len(bets),
            "initial": initial[a],
            "final_equity": banks[a],
            "growth_multiple": banks[a] / initial[a],
            "log_growth": float(np.log(max(1e-12, banks[a]) / initial[a])),
            "max_drawdown": float(((peak - curve) / peak).max()),
            "staked": sum(x["stake"] for x in bets),
            "pnl": banks[a] - initial[a],
            "recommendation_fill_gap": sum(x["fill_shortfall"] for x in bets),
        }
    return result, logs


def run():
    corrected = pd.read_parquet(OUT / "training_features.parquet")
    corrected["date"] = pd.to_datetime(corrected.date)
    old = pd.read_parquet(BASE / "data/processed/fights.parquet")
    old["date"] = pd.to_datetime(old.date)
    oldskill = pd.read_parquet(BASE / "data/processed/skill_features_v3.parquet")
    oldskill["date"] = pd.to_datetime(oldskill.date)
    old = old.merge(oldskill, on=["date", "R_fighter", "B_fighter"], how="left", validate="one_to_one")
    old = old[old.Winner.isin(["Red", "Blue"])].reset_index(drop=True)
    mask = corrected.date >= CUTOFF
    # Use identical original row positions to avoid losing renamed fighters in a join.
    old_eval = old.loc[old.date >= CUTOFF].reset_index(drop=True)
    new_eval = corrected.loc[mask].reset_index(drop=True)
    assert len(old_eval) == len(new_eval)
    assert old_eval.date.equals(new_eval.date)
    universes = add_prior_fight_counts(corrected)
    new_eval["has_debut"] = universes.loc[mask, "has_debut"].to_numpy()
    results = {}
    arrays = {}
    for name, frame, models in [
        ("old_frozen", old_eval, BASE / "artifacts/models"),
        ("old_models_corrected_features", new_eval, BASE / "artifacts/models"),
        ("corrected", new_eval, OUT / "models"),
    ]:
        print("predicting", name, flush=True)
        ps = predictions(frame, models)
        arrays[name] = ps
        results[name] = {}
        for kind, p in ps.items():
            new_eval[f"{name}_{kind}"] = p
            y = new_eval.Winner.eq("Red").astype(int)
            results[name][kind] = {
                "outcome_log_loss": float(log_loss(y, p)),
                "outcome_brier": float(brier_score_loss(y, p)),
            }
    rows = {}
    for idx, r in new_eval.iterrows():
        rows.setdefault(r.date.normalize(), []).append((idx, r))
    prices = pd.read_parquet(BASE / "artifacts/metrics/strict_multivenue_postcutoff_fights.parquet")
    prices = prices[prices.universe.isin(["polymarket", "kalshi"])]
    matched = []
    unmatched = []
    for _, p in prices.iterrows():
        match = match_row(rows, p.R_fighter, p.B_fighter, p.date)
        if match is None:
            unmatched.append(p.to_dict())
            continue
        idx, forward = match
        r = new_eval.iloc[idx].to_dict()
        for field in ["venue", "fee_rate", "fee_exponent", "priced_at_ask", "market_id"]:
            r[field] = p[field]
        r["price_red"] = p.price_red if forward else p.price_blue
        r["price_blue"] = p.price_blue if forward else p.price_red
        matched.append(r)
    priced = pd.DataFrame(matched)
    for name in arrays:
        for kind in ["real", "corrupted"]:
            results[name][kind]["venues"] = {
                venue: flat_metrics(frame, frame[f"{name}_{kind}"].to_numpy())
                for venue, frame in priced.groupby("venue")
            }
            k = priced[priced.venue.str.startswith("kalshi")]
            results[name][kind]["kalshi_true_asks"] = flat_metrics(
                k[k.priced_at_ask.astype(bool)],
                k.loc[k.priced_at_ask.astype(bool), f"{name}_{kind}"].to_numpy(),
            )
    new_eval.to_parquet(OUT / "all_postcutoff_predictions.parquet", index=False)
    priced.to_parquet(OUT / "paired_price_comparison.parquet", index=False)
    snapshot = pd.read_parquet(
        BASE / "data/raw/kalshi/snapshots/historical_T-90min_perfight_combined.parquet"
    )
    close_lookup = snapshot.drop_duplicates("event_ticker").set_index("event_ticker")["close_time"]
    priced["sequence_proxy"] = pd.to_datetime(priced.market_id.map(close_lookup), utc=True)
    kalshi = priced[priced.venue.str.startswith("kalshi") & priced.priced_at_ask.astype(bool)].copy()
    simulations = {}
    logs = []
    for name in ["old_frozen", "corrected"]:
        for depth in [100.0, 1000.0, 10000.0]:
            for slippage in [0.0, 0.01, 0.03]:
                # Settlement lag is modelled by locking cash for the full card.
                scenario = f"{name}_depth{depth:g}_slip{slippage:.2f}_card_locked"
                summary, records = replay(kalshi, name, depth, slippage, "card_end")
                simulations[scenario] = summary
                for r in records:
                    r["scenario"] = scenario
                logs.extend(records)
    coverage = {
        "postcutoff_fights": len(new_eval),
        "paired_prices": len(priced),
        "kalshi_true_ask_fights": len(kalshi),
        "unmatched_prices": len(unmatched),
        "postcutoff_fights_without_any_price": len(new_eval)
        - priced[["date", "R_fighter", "B_fighter"]].drop_duplicates().shape[0],
        "full_historical_depth_available": False,
        "ordering_assumption": "archived market close time as bout-order proxy; cash locked until card end",
        "book_assumption": "1% of scenario depth at best ask, 99% at 0/1/3 cents adverse",
        "replay_label": "conditional scenarios; missing-price opportunities remain unpriced, not wins or losses; no unlimited liquidity",
        "old_baseline_limit": "frozen historical feature rows with actual deployed model files; this is not a claim to reconstruct every past live feature snapshot",
        "prospective_status": "not yet observed",
    }
    (OUT / "comparison_results.json").write_text(
        json.dumps(
            {"coverage": coverage, "paired": results, "execution_scenarios": simulations},
            indent=2,
            default=str,
        )
    )
    pd.DataFrame(logs).to_csv(OUT / "execution_replay.csv", index=False)
    print(json.dumps(coverage, indent=2))


if __name__ == "__main__":
    run()

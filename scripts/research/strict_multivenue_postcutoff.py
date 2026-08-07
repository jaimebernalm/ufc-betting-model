"""Fully strict post-cutoff evaluation on outcomes, Polymarket and Kalshi.

Unlike the legacy venue scripts, this evaluator rebuilds all fighter state
strictly before each bout, retrains the actual 10-seed recipes, reports debut
and non-debut slices separately, and applies each Polymarket market's archived
Gamma fee schedule rather than one global fee guess.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

try:
    from .strict_raw_retrain_audit import (
        CUTOFF,
        _sharpen,
        attach_strict_features,
        build_strict_states,
    )
except ImportError:  # Support direct execution from the scripts directory.
    from strict_raw_retrain_audit import (
        CUTOFF,
        _sharpen,
        attach_strict_features,
        build_strict_states,
    )
from ufc_pred.backtest.strategy_grid import predict, train_corrupted, train_real
from ufc_pred.backtest.universe import add_prior_fight_counts
from ufc_pred.features.skill_v3_pipeline import OUTPUT as SKILL_PATH
from ufc_pred.features.static_v1 import _swap_red_blue
from ufc_pred.ingest.kaggle_mdabbert import HISTORY_PARQUET
from ufc_pred.paths import METRICS

ROOT = Path(__file__).resolve().parents[1]
POLY_PATH = ROOT / "data/interim/polymarket_matched_to_kaggle_v2.parquet"
KALSHI_PATH = ROOT / "data/raw/kalshi/snapshots/historical_T-90min_perfight_combined.parquet"
FEE_CACHE = METRICS / "polymarket_fee_schedule.json"
OUTPUT = METRICS / "strict_multivenue_postcutoff.json"
DETAIL_OUTPUT = METRICS / "strict_multivenue_postcutoff_fights.parquet"


def _fetch_fee_schedule(market_id: str) -> tuple[str, dict]:
    response = requests.get(f"https://gamma-api.polymarket.com/markets/{market_id}", timeout=20)
    response.raise_for_status()
    market = response.json()
    schedule = market.get("feeSchedule") or {}
    return market_id, {
        "fees_enabled": bool(market.get("feesEnabled", False)),
        "fee_type": market.get("feeType"),
        "rate": float(schedule.get("rate", 0.0) or 0.0),
        "exponent": float(schedule.get("exponent", 1.0) or 1.0),
        "taker_only": bool(schedule.get("takerOnly", True)),
    }


def load_fee_schedules(market_ids: list[str]) -> dict[str, dict]:
    cached: dict[str, dict] = {}
    if FEE_CACHE.exists():
        cached = json.loads(FEE_CACHE.read_text())
    missing = sorted(set(market_ids) - set(cached))
    if missing:
        print(f"fetching {len(missing)} Polymarket fee schedules", flush=True)
        with ThreadPoolExecutor(max_workers=12) as pool:
            futures = {pool.submit(_fetch_fee_schedule, mid): mid for mid in missing}
            for future in as_completed(futures):
                market_id, schedule = future.result()
                cached[market_id] = schedule
        METRICS.mkdir(parents=True, exist_ok=True)
        FEE_CACHE.write_text(json.dumps(cached, indent=2, sort_keys=True))
    return cached


def load_strict_fights(source_dir: Path) -> pd.DataFrame:
    states, dobs, profiles = build_strict_states(source_dir)
    fights = pd.read_parquet(HISTORY_PARQUET)
    fights = fights[fights["Winner"].isin(["Red", "Blue"])].copy()
    fights["date"] = pd.to_datetime(fights["date"])
    fights, _ = attach_strict_features(fights, states, dobs, profiles)
    skill = pd.read_parquet(SKILL_PATH)
    skill["date"] = pd.to_datetime(skill["date"])
    fights = fights.merge(
        skill[["date", "R_fighter", "B_fighter", "skill_diff_mean", "skill_diff_std"]],
        on=["date", "R_fighter", "B_fighter"],
        how="left",
        validate="many_to_one",
    )
    return add_prior_fight_counts(fights)


def build_polymarket(fights: pd.DataFrame) -> pd.DataFrame:
    prices = pd.read_parquet(POLY_PATH)
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices[prices["date"] >= CUTOFF].copy()
    columns = [
        "date",
        "R_fighter",
        "B_fighter",
        "polymarket_p_red",
        "polymarket_p_blue",
        "market_id",
        "market_slug",
    ]
    out = prices[columns].merge(
        fights,
        on=["date", "R_fighter", "B_fighter"],
        how="inner",
        validate="one_to_one",
    )
    schedules = load_fee_schedules(out["market_id"].astype(str).tolist())
    out["venue"] = "polymarket_close"
    out["price_red"] = out["polymarket_p_red"].astype(float)
    out["price_blue"] = out["polymarket_p_blue"].astype(float)
    out["fee_rate"] = [schedules[str(mid)]["rate"] for mid in out["market_id"]]
    out["fee_exponent"] = [schedules[str(mid)]["exponent"] for mid in out["market_id"]]
    out["fees_enabled"] = [schedules[str(mid)]["fees_enabled"] for mid in out["market_id"]]
    out.loc[~out["fees_enabled"], "fee_rate"] = 0.0
    out["priced_at_ask"] = False
    return out


def build_kalshi(fights: pd.DataFrame) -> pd.DataFrame:
    snap = pd.read_parquet(KALSHI_PATH)
    snap["date"] = pd.to_datetime(snap["fight_date"]).dt.tz_localize(None).dt.normalize()
    snap = snap[
        snap["canon_a"].notna()
        & snap["canon_b"].notna()
        & snap["settle_result_a"].isin(["yes", "no"])
        & (snap["date"] >= CUTOFF)
    ].copy()

    rows: list[dict] = []
    lookup = {
        (row["R_fighter"], row["B_fighter"], row["date"].normalize()): row for _, row in fights.iterrows()
    }
    for snap_row in snap.itertuples(index=False):
        key_fwd = (snap_row.canon_a, snap_row.canon_b, snap_row.date)
        key_rev = (snap_row.canon_b, snap_row.canon_a, snap_row.date)
        fight = lookup.get(key_fwd)
        if fight is None:
            fight = lookup.get(key_rev)
        if fight is None:
            continue
        ask_ok = all(
            pd.notna(value) and 0.02 <= float(value) <= 0.98
            for value in (snap_row.ask_yes_price_a, snap_row.ask_yes_price_b)
        )
        if ask_ok:
            price_a = float(snap_row.ask_yes_price_a)
            price_b = float(snap_row.ask_yes_price_b)
        else:
            price_a = float(snap_row.close_yes_price_a)
            price_b = float(snap_row.close_yes_price_b)
        if not (0.02 <= price_a <= 0.98 and 0.02 <= price_b <= 0.98):
            continue
        payload = fight.to_dict()
        a_is_red = fight["R_fighter"] == snap_row.canon_a
        payload.update(
            {
                "venue": "kalshi_T90_ask_fallback",
                "price_red": price_a if a_is_red else price_b,
                "price_blue": price_b if a_is_red else price_a,
                "fee_rate": 0.07,
                "fee_exponent": 1.0,
                "fees_enabled": True,
                "priced_at_ask": ask_ok,
                "market_id": snap_row.event_ticker,
            }
        )
        rows.append(payload)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return (
        out.sort_values("priced_at_ask", ascending=False)
        .drop_duplicates(["date", "R_fighter", "B_fighter"], keep="first")
        .reset_index(drop=True)
    )


def _effective_decimal(price: np.ndarray, rate: np.ndarray, exponent: np.ndarray) -> np.ndarray:
    fee_per_share = rate * np.power(price * (1.0 - price), exponent)
    return 1.0 / (price + fee_per_share)


def gate(frame: pd.DataFrame, p_red: np.ndarray, n_boot: int = 20_000) -> dict:
    price_r = frame["price_red"].to_numpy(float)
    price_b = frame["price_blue"].to_numpy(float)
    rate = frame["fee_rate"].to_numpy(float)
    exponent = frame["fee_exponent"].to_numpy(float)
    edge_r, edge_b = p_red - price_r, (1 - p_red) - price_b
    bet_red = edge_r >= edge_b
    take = np.maximum(edge_r, edge_b) >= 0.03
    won = np.where(bet_red, frame["Winner"].eq("Red"), frame["Winner"].eq("Blue"))
    price = np.where(bet_red, price_r, price_b)
    eff = np.where(
        bet_red,
        _effective_decimal(price_r, rate, exponent),
        _effective_decimal(price_b, rate, exponent),
    )
    pnl = np.where(won, eff - 1.0, -1.0)[take]
    if not len(pnl):
        return {"fights": int(len(frame)), "bets": 0}
    rng = np.random.default_rng(7)
    boot = pnl[rng.integers(0, len(pnl), size=(n_boot, len(pnl)))].mean(axis=1)
    bet_ci = np.percentile(boot, [2.5, 97.5])
    dates = pd.to_datetime(frame.loc[take, "date"]).dt.normalize().to_numpy()
    clusters = pd.DataFrame({"date": dates, "pnl": pnl}).groupby("date")["pnl"].agg(["sum", "count"])
    idx = rng.integers(0, len(clusters), size=(n_boot, len(clusters)))
    cluster_roi = clusters["sum"].to_numpy()[idx].sum(axis=1) / clusters["count"].to_numpy()[idx].sum(axis=1)
    cluster_ci = np.percentile(cluster_roi, [2.5, 97.5])
    chosen_price = price[take]
    return {
        "fights": int(len(frame)),
        "bets": int(take.sum()),
        "wins": int(won[take].sum()),
        "hit_rate": float(won[take].mean()),
        "roi": float(pnl.mean()),
        "bet_ci95": [float(x) for x in bet_ci],
        "card_cluster_ci95": [float(x) for x in cluster_ci],
        "underdog_bets": int((chosen_price < 0.5).sum()),
        "underdog_wins": int((won[take] & (chosen_price < 0.5)).sum()),
    }


def outcome_metrics(frame: pd.DataFrame, p_red: np.ndarray) -> dict:
    y = frame["Winner"].eq("Red").to_numpy(int)
    return {
        "fights": int(len(frame)),
        "accuracy": float(accuracy_score(y, p_red >= 0.5)),
        "log_loss": float(log_loss(y, p_red, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p_red)),
    }


def slices(frame: pd.DataFrame, probabilities: np.ndarray, evaluator) -> dict:
    debut = frame["has_debut"].to_numpy(bool)
    return {
        "all": evaluator(frame.reset_index(drop=True), probabilities),
        "no_debut": evaluator(frame.loc[~debut].reset_index(drop=True), probabilities[~debut]),
        "debut": evaluator(frame.loc[debut].reset_index(drop=True), probabilities[debut]),
    }


def venue_slices(frame: pd.DataFrame, probabilities: np.ndarray) -> dict:
    out = slices(frame, probabilities, gate)
    dates = pd.to_datetime(frame["date"])
    debut = frame["has_debut"].to_numpy(bool)
    through = (dates <= pd.Timestamp("2026-05-16")).to_numpy() & ~debut
    after = (dates > pd.Timestamp("2026-05-16")).to_numpy() & ~debut
    if through.any():
        out["no_debut_through_2026_05_16"] = gate(
            frame.loc[through].reset_index(drop=True), probabilities[through]
        )
    if after.any():
        out["no_debut_after_2026_05_16"] = gate(frame.loc[after].reset_index(drop=True), probabilities[after])
    return out


def detail_rows(
    universe: str,
    frame: pd.DataFrame,
    model_probabilities: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Return the compact per-fight evidence behind the aggregate report."""
    columns = [
        "date",
        "R_fighter",
        "B_fighter",
        "Winner",
        "has_debut",
        "venue",
        "price_red",
        "price_blue",
        "fee_rate",
        "fee_exponent",
        "fees_enabled",
        "priced_at_ask",
        "market_id",
    ]
    out = frame.reindex(columns=columns).copy()
    out.insert(0, "universe", universe)
    for kind, probabilities in model_probabilities.items():
        out[f"p_red_{kind}"] = probabilities
    return out


def run(source_dir: Path, seeds: int) -> dict:
    fights = load_strict_fights(source_dir)
    outcomes = fights[fights["date"] >= CUTOFF].reset_index(drop=True)
    polymarket = build_polymarket(fights)
    kalshi = build_kalshi(fights)
    common_keys = (
        polymarket[["date", "R_fighter", "B_fighter"]]
        .merge(
            kalshi[["date", "R_fighter", "B_fighter"]],
            on=["date", "R_fighter", "B_fighter"],
            how="inner",
        )
        .drop_duplicates()
    )
    polymarket_common = common_keys.merge(polymarket, on=["date", "R_fighter", "B_fighter"], how="left")
    kalshi_common = common_keys.merge(kalshi, on=["date", "R_fighter", "B_fighter"], how="left")
    frames = {
        "outcomes": outcomes,
        "polymarket": polymarket,
        "kalshi": kalshi,
        "polymarket_common": polymarket_common,
        "kalshi_common": kalshi_common,
    }
    reverses = {name: _swap_red_blue(frame) for name, frame in frames.items()}
    predictions = {kind: {name: [] for name in frames} for kind in ("real", "corrupted")}
    for seed in range(seeds):
        print(f"training seed {seed + 1}/{seeds}", flush=True)
        models = {
            "real": train_real(fights, CUTOFF, seed=seed),
            "corrupted": train_corrupted(fights, CUTOFF, seed=seed),
        }
        for kind, model in models.items():
            for name, frame in frames.items():
                sym = 0.5 * (predict(model, frame) + 1.0 - predict(model, reverses[name]))
                predictions[kind][name].append(sym)

    averaged = {
        kind: {name: _sharpen(np.stack(per_seed).mean(axis=0)) for name, per_seed in by_universe.items()}
        for kind, by_universe in predictions.items()
    }

    result = {
        "source_dir": str(source_dir),
        "cutoff": str(CUTOFF.date()),
        "seeds": seeds,
        "universes": {
            name: {
                "fights": int(len(frame)),
                "date_min": str(frame["date"].min().date()),
                "date_max": str(frame["date"].max().date()),
                "debut_fights": int(frame["has_debut"].sum()),
                "fee_enabled_fights": int(
                    frame.get("fees_enabled", pd.Series(False, index=frame.index)).sum()
                ),
                "true_ask_fights": int(frame.get("priced_at_ask", pd.Series(False, index=frame.index)).sum()),
            }
            for name, frame in frames.items()
        },
        "models": {},
    }
    for kind in predictions:
        result["models"][kind] = {}
        for name, p in averaged[kind].items():
            if name == "outcomes":
                result["models"][kind][name] = slices(frames[name], p, outcome_metrics)
            else:
                result["models"][kind][name] = venue_slices(frames[name], p)
    METRICS.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2))
    details = pd.concat(
        [
            detail_rows(
                name,
                frames[name],
                {kind: averaged[kind][name] for kind in averaged},
            )
            for name in ("outcomes", "polymarket", "kalshi")
        ],
        ignore_index=True,
    )
    details.to_parquet(DETAIL_OUTPUT, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ufcstats-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=10)
    args = parser.parse_args()
    result = run(args.ufcstats_dir, args.seeds)
    print(json.dumps(result, indent=2))
    print(f"wrote {OUTPUT}")
    print(f"wrote {DETAIL_OUTPUT}")


if __name__ == "__main__":
    main()

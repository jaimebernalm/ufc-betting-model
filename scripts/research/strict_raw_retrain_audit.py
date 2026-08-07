"""Retrain/evaluate with strictly pre-fight UFCStats career rates.

This is the end-to-end counterpart to ``feature_parity_test.py``.  It consumes
the public CSV export produced by Greco1899/scrape_ufc_stats (or an equivalent
export with the same four files), replaces the five snapshot career-rate
features and age on every historical row, retrains the deployment recipe, and
evaluates only fights after the training cutoff.

Usage:
  PYTHONPATH=src .conda/bin/python scripts/strict_raw_retrain_audit.py \
      --ufcstats-dir /path/to/scrape_ufc_stats --seeds 10
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from ufc_pred.backtest.bet_eval import _effective_decimal
from ufc_pred.backtest.strategy_grid import predict, train_corrupted, train_real
from ufc_pred.features.skill_v3_pipeline import OUTPUT as SKILL_PATH
from ufc_pred.features.static_v1 import _swap_red_blue
from ufc_pred.inference.skill_for_upcoming import attach_skill_for_upcoming
from ufc_pred.inference.upcoming_builder import _DIFF_PAIRS
from ufc_pred.ingest.kaggle_mdabbert import HISTORY_PARQUET
from ufc_pred.paths import METRICS

ROOT = Path(__file__).resolve().parents[1]
CUTOFF = pd.Timestamp("2025-11-30")
RATE_COLS = [
    "avg_SIG_STR_landed",
    "avg_SIG_STR_pct",
    "avg_TD_landed",
    "avg_TD_pct",
    "avg_SUB_ATT",
]
COUNTER_COLS = [
    "current_lose_streak",
    "current_win_streak",
    "draw",
    "longest_win_streak",
    "losses",
    "total_rounds_fought",
    "total_title_bouts",
    "win_by_Decision_Majority",
    "win_by_Decision_Split",
    "win_by_Decision_Unanimous",
    "win_by_KO/TKO",
    "win_by_Submission",
    "win_by_TKO_Doctor_Stoppage",
    "wins",
]


def _norm(value: object) -> str:
    folded = "".join(
        c for c in unicodedata.normalize("NFKD", str(value).casefold()) if not unicodedata.combining(c)
    )
    return re.sub(r"[^a-z0-9]+", " ", folded).strip()


def _pair(value: object) -> tuple[int, int]:
    match = re.match(r"\s*(\d+)\s+of\s+(\d+)", str(value))
    return (int(match.group(1)), int(match.group(2))) if match else (0, 0)


def _elapsed_seconds(row: pd.Series) -> int:
    try:
        minutes, seconds = (int(x) for x in str(row["TIME"]).split(":"))
        return 300 * (int(row["ROUND"]) - 1) + 60 * minutes + seconds
    except (TypeError, ValueError):
        return 0


def _build_counter_states(results: pd.DataFrame) -> pd.DataFrame:
    """Build strict pre-bout record/streak/method counters from result rows."""
    records: list[dict[str, object]] = []
    for row in results.itertuples(index=False):
        names = re.split(r"\s+vs\.?\s+", str(row.BOUT), maxsplit=1)
        outcomes = str(row.OUTCOME).split("/")
        if len(names) != 2 or len(outcomes) != 2:
            continue
        for fighter, outcome in zip(names, outcomes, strict=False):
            records.append(
                {
                    "fighter_key": _norm(fighter),
                    "date": row.date,
                    "outcome": outcome.strip().upper(),
                    "rounds": int(row.ROUND) if pd.notna(row.ROUND) else 0,
                    "title_bout": "title" in str(row.WEIGHTCLASS).casefold(),
                    "method": str(row.METHOD).strip(),
                }
            )

    long = pd.DataFrame(records).sort_values(["fighter_key", "date"], kind="mergesort")
    states: list[dict[str, object]] = []
    for fighter_key, group in long.groupby("fighter_key", sort=False):
        wins = losses = draws = rounds = titles = 0
        current_win = current_lose = longest_win = running_win = 0
        methods = {
            "win_by_Decision_Majority": 0,
            "win_by_Decision_Split": 0,
            "win_by_Decision_Unanimous": 0,
            "win_by_KO/TKO": 0,
            "win_by_Submission": 0,
            "win_by_TKO_Doctor_Stoppage": 0,
        }
        for row in group.itertuples(index=False):
            states.append(
                {
                    "fighter_key": fighter_key,
                    "date": row.date,
                    "current_lose_streak": current_lose,
                    "current_win_streak": current_win,
                    "draw": draws,
                    "longest_win_streak": longest_win,
                    "losses": losses,
                    "total_rounds_fought": rounds,
                    "total_title_bouts": titles,
                    **methods,
                    "wins": wins,
                }
            )

            outcome = row.outcome
            if outcome == "W":
                wins += 1
                current_win += 1
                current_lose = 0
                running_win += 1
                longest_win = max(longest_win, running_win)
                method_key = {
                    "Decision - Majority": "win_by_Decision_Majority",
                    "Decision - Split": "win_by_Decision_Split",
                    "Decision - Unanimous": "win_by_Decision_Unanimous",
                    "KO/TKO": "win_by_KO/TKO",
                    "Submission": "win_by_Submission",
                    "TKO - Doctor's Stoppage": "win_by_TKO_Doctor_Stoppage",
                }.get(row.method)
                if method_key:
                    methods[method_key] += 1
            elif outcome == "L":
                losses += 1
                current_lose += 1
                current_win = 0
                running_win = 0
            elif outcome == "D":
                draws += 1
                running_win = 0
            # Draws and no-contests do not break the displayed current streak,
            # matching the production UFCStats parser.
            rounds += row.rounds
            titles += int(row.title_bout)

    return pd.DataFrame(states).drop_duplicates(["fighter_key", "date"], keep="last")


def _profile_value(value: object, kind: str) -> float | None:
    if pd.isna(value) or str(value).strip() == "--":
        return None
    text = str(value).strip()
    if kind == "height":
        match = re.match(r"(\d+)'\s*(\d+)\"", text)
        return None if not match else round((int(match[1]) * 12 + int(match[2])) * 2.54, 2)
    number = re.search(r"\d+", text)
    if number is None:
        return None
    return round(int(number.group()) * 2.54, 2) if kind == "reach" else float(number.group())


def build_strict_states(
    source_dir: Path,
) -> tuple[pd.DataFrame, dict[str, pd.Timestamp], dict[str, dict[str, object]]]:
    """Return one strict pre-fight rate row per ``(fighter, fight date)``."""
    events = pd.read_csv(source_dir / "ufc_event_details.csv", usecols=["EVENT", "DATE"])
    events["EVENT"] = events["EVENT"].str.strip()
    events["date"] = pd.to_datetime(events["DATE"])

    results = pd.read_csv(source_dir / "ufc_fight_results.csv")
    results["EVENT"] = results["EVENT"].str.strip()
    results["BOUT"] = results["BOUT"].str.strip()
    results = results.merge(events[["EVENT", "date"]], on="EVENT", how="left")
    results["fight_seconds"] = results.apply(_elapsed_seconds, axis=1)
    result_keys = results[["EVENT", "BOUT", "date", "fight_seconds"]].drop_duplicates(
        ["EVENT", "BOUT"], keep="last"
    )

    stats = pd.read_csv(source_dir / "ufc_fight_stats.csv")
    for col in ("EVENT", "BOUT", "FIGHTER"):
        stats[col] = stats[col].str.strip()
    for source, prefix in (("SIG.STR.", "sig"), ("TD", "td")):
        parsed = stats[source].map(_pair)
        stats[f"{prefix}_landed"] = [x[0] for x in parsed]
        stats[f"{prefix}_attempted"] = [x[1] for x in parsed]
    stats["sub_attempted"] = pd.to_numeric(stats["SUB.ATT"], errors="coerce").fillna(0)
    sums = ["sig_landed", "sig_attempted", "td_landed", "td_attempted", "sub_attempted"]
    bouts = (
        stats.groupby(["EVENT", "BOUT", "FIGHTER"], as_index=False)[sums]
        .sum()
        .merge(result_keys, on=["EVENT", "BOUT"], how="left", validate="many_to_one")
    )
    bouts["fighter_key"] = bouts["FIGHTER"].map(_norm)
    bouts = bouts.sort_values(["fighter_key", "date", "EVENT", "BOUT"], kind="mergesort")

    # Cumulative values shifted one bout: target bout never contributes to its
    # own features.  Same-day double appearances are not present in UFC data.
    cumulative_cols = sums + ["fight_seconds"]
    for col in cumulative_cols:
        bouts[f"prior_{col}"] = bouts.groupby("fighter_key")[col].transform(
            lambda values: values.cumsum().shift(fill_value=0)
        )

    sec = bouts["prior_fight_seconds"].replace(0, np.nan)
    bouts["avg_SIG_STR_landed"] = bouts["prior_sig_landed"] * 60.0 / sec
    bouts["avg_SIG_STR_pct"] = bouts["prior_sig_landed"] / bouts["prior_sig_attempted"].replace(0, np.nan)
    bouts["avg_TD_landed"] = bouts["prior_td_landed"] * 900.0 / sec
    bouts["avg_TD_pct"] = bouts["prior_td_landed"] / bouts["prior_td_attempted"].replace(0, np.nan)
    bouts["avg_SUB_ATT"] = bouts["prior_sub_attempted"] * 900.0 / sec
    bouts[RATE_COLS] = bouts[RATE_COLS].fillna(0.0)
    bouts["avg_SIG_STR_landed"] = bouts["avg_SIG_STR_landed"].round(2)
    bouts["avg_SIG_STR_pct"] = bouts["avg_SIG_STR_pct"].round(2)
    bouts["avg_TD_landed"] = bouts["avg_TD_landed"].round(2)
    bouts["avg_TD_pct"] = bouts["avg_TD_pct"].round(2)
    bouts["avg_SUB_ATT"] = bouts["avg_SUB_ATT"].round(1)

    rate_states = bouts[["fighter_key", "date", *RATE_COLS]].drop_duplicates(
        ["fighter_key", "date"], keep="last"
    )
    counter_states = _build_counter_states(results)
    states = rate_states.merge(
        counter_states,
        on=["fighter_key", "date"],
        how="outer",
        validate="one_to_one",
    )
    tott = pd.read_csv(source_dir / "ufc_fighter_tott.csv")
    tott["fighter_key"] = tott["FIGHTER"].map(_norm)
    dobs = dict(zip(tott["fighter_key"], pd.to_datetime(tott["DOB"], errors="coerce"), strict=False))
    profiles = {
        row.fighter_key: {
            "Stance": None if pd.isna(row.STANCE) else str(row.STANCE),
            "Height_cms": _profile_value(row.HEIGHT, "height"),
            "Reach_cms": _profile_value(row.REACH, "reach"),
            "Weight_lbs": _profile_value(row.WEIGHT, "weight"),
        }
        for row in tott.itertuples(index=False)
    }
    for profile in profiles.values():
        if profile["Reach_cms"] is None:
            profile["Reach_cms"] = profile["Height_cms"]
    return states, dobs, profiles


def attach_strict_features(
    fights: pd.DataFrame,
    states: pd.DataFrame,
    dobs: dict[str, pd.Timestamp],
    profiles: dict[str, dict[str, object]],
) -> tuple[pd.DataFrame, dict]:
    out = fights.copy()
    out["date"] = pd.to_datetime(out["date"])
    diagnostics: dict[str, object] = {}
    for side in ("R", "B"):
        key = out[f"{side}_fighter"].map(_norm)
        lookup = pd.DataFrame({"fighter_key": key, "date": out["date"]}).merge(
            states, on=["fighter_key", "date"], how="left", validate="many_to_one"
        )
        diagnostics[f"{side}_rate_coverage"] = float(lookup[RATE_COLS[0]].notna().mean())
        for col in [*RATE_COLS, *COUNTER_COLS]:
            out[f"{side}_{col}"] = lookup[col].to_numpy()
        for col in ("Stance", "Height_cms", "Reach_cms", "Weight_lbs"):
            out[f"{side}_{col}"] = [profiles.get(k, {}).get(col) for k in key]
        ages = []
        for fighter_key, date in zip(key, out["date"], strict=False):
            dob = dobs.get(fighter_key)
            ages.append(
                np.nan
                if dob is None or pd.isna(dob)
                else date.year - dob.year - ((date.month, date.day) < (dob.month, dob.day))
            )
        out[f"{side}_age"] = ages

    for diff_col, base in _DIFF_PAIRS:
        red, blue = out[f"R_{base}"], out[f"B_{base}"]
        out[diff_col] = blue - red
    return out, diagnostics


def _sharpen(probability: np.ndarray, temperature: float = 1.25) -> np.ndarray:
    p = np.clip(probability, 1e-6, 1 - 1e-6)
    return 1.0 / (1.0 + np.exp(-temperature * np.log(p / (1 - p))))


def _gate(frame: pd.DataFrame, p_red: np.ndarray, *, n_boot: int = 20_000) -> dict:
    price_r = frame["polymarket_p_red"].to_numpy(float)
    price_b = frame["polymarket_p_blue"].to_numpy(float)
    edge_r, edge_b = p_red - price_r, (1 - p_red) - price_b
    bet_red = edge_r >= edge_b
    take = np.maximum(edge_r, edge_b) >= 0.03
    won = np.where(bet_red, frame["Winner"].eq("Red"), frame["Winner"].eq("Blue"))
    price = np.where(bet_red, price_r, price_b)
    effective = _effective_decimal(1 / price, 0.02, "polymarket")
    pnl = np.where(won, effective - 1, -1)[take]
    rng = np.random.default_rng(7)
    boot = pnl[rng.integers(0, len(pnl), size=(n_boot, len(pnl)))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    # Fights on one card share market conditions, judging environment and
    # often correlated model errors.  A bet-level bootstrap understates that
    # dependence, so report a card/date cluster bootstrap as the conservative
    # deployment interval.
    bet_dates = pd.to_datetime(frame.loc[take, "date"]).dt.normalize().to_numpy()
    clusters = pd.DataFrame({"date": bet_dates, "pnl": pnl}).groupby("date")["pnl"].agg(["sum", "count"])
    cluster_idx = rng.integers(0, len(clusters), size=(n_boot, len(clusters)))
    cluster_roi = clusters["sum"].to_numpy()[cluster_idx].sum(axis=1) / clusters["count"].to_numpy()[
        cluster_idx
    ].sum(axis=1)
    cluster_lo, cluster_hi = np.percentile(cluster_roi, [2.5, 97.5])
    return {
        "fights": int(len(frame)),
        "bets": int(take.sum()),
        "wins": int(won[take].sum()),
        "hit_rate": float(won[take].mean()),
        "roi": float(pnl.mean()),
        "ci95": [float(lo), float(hi)],
        "card_cluster_ci95": [float(cluster_lo), float(cluster_hi)],
    }


def _gate_live(frame: pd.DataFrame, p_a: np.ndarray) -> dict:
    """Flat-stake gate on the archived live-notification universe."""
    price_a = frame["askA"].to_numpy(float)
    price_b = frame["askB"].to_numpy(float)
    edge_a, edge_b = p_a - price_a, (1 - p_a) - price_b
    bet_a = edge_a >= edge_b
    take = np.maximum(edge_a, edge_b) >= 0.03
    won = np.where(bet_a, frame["winA"], ~frame["winA"])
    price = np.where(bet_a, price_a, price_b)
    effective = _effective_decimal(1 / price, 0.07, "kalshi")
    pnl = np.where(won, effective - 1, -1)[take]
    return {
        "fights": int(len(frame)),
        "bets": int(take.sum()),
        "wins": int(won[take].sum()),
        "hit_rate": float(won[take].mean()),
        "roi": float(pnl.mean()),
    }


def run(source_dir: Path, seeds: int) -> dict:
    states, dobs, profiles = build_strict_states(source_dir)
    fights = pd.read_parquet(HISTORY_PARQUET)
    fights = fights[fights["Winner"].isin(["Red", "Blue"])].copy()
    fights, diagnostics = attach_strict_features(fights, states, dobs, profiles)

    skill = pd.read_parquet(SKILL_PATH)
    skill["date"] = pd.to_datetime(skill["date"])
    fights = fights.merge(
        skill[["date", "R_fighter", "B_fighter", "skill_diff_mean", "skill_diff_std"]],
        on=["date", "R_fighter", "B_fighter"],
        how="left",
        validate="many_to_one",
    )

    prices = pd.read_parquet(ROOT / "data/interim/polymarket_matched_to_kaggle_v2.parquet")
    prices["date"] = pd.to_datetime(prices["date"])
    eval_frame = prices[["date", "R_fighter", "B_fighter", "polymarket_p_red", "polymarket_p_blue"]].merge(
        fights, on=["date", "R_fighter", "B_fighter"], how="inner"
    )
    eval_frame = eval_frame[eval_frame["date"] >= CUTOFF].reset_index(drop=True)

    # Match deployment's no-debut universe: a fighter is usable once they have
    # any UFC fight strictly before the target fight, not necessarily before
    # the model-training cutoff.
    long_dates = pd.concat(
        [
            fights[["date", "R_fighter"]].rename(columns={"R_fighter": "fighter"}),
            fights[["date", "B_fighter"]].rename(columns={"B_fighter": "fighter"}),
        ],
        ignore_index=True,
    )
    first_date = long_dates.groupby("fighter")["date"].min()
    eval_frame = eval_frame[
        (eval_frame["date"] > eval_frame["R_fighter"].map(first_date))
        & (eval_frame["date"] > eval_frame["B_fighter"].map(first_date))
    ].reset_index(drop=True)
    reverse = _swap_red_blue(eval_frame)

    live_fwd_path = ROOT / "data/interim/feature_parity_live_raw_fwd.parquet"
    live_rev_path = ROOT / "data/interim/feature_parity_live_raw_rev.parquet"
    live_meta_path = ROOT / "data/interim/feature_parity_live_raw_meta.parquet"
    live_ledger_path = ROOT / "artifacts/metrics/feature_parity_live_raw.parquet"
    have_live = all(
        path.exists() for path in (live_fwd_path, live_rev_path, live_meta_path, live_ledger_path)
    )
    if have_live:
        live_fwd = pd.read_parquet(live_fwd_path)
        live_rev = pd.read_parquet(live_rev_path)
        # Re-attach every state field from the same strict source used for
        # training.  The archived frames are only the immutable fight/market/
        # skill envelope; they must not preserve an older live-builder quirk.
        live_fwd, _ = attach_strict_features(live_fwd, states, dobs, profiles)
        live_rev, _ = attach_strict_features(live_rev, states, dobs, profiles)
        # The deployed posterior cache was built against a stale fights table
        # on part of the live period. Recompute as-of-month skill from the
        # complete historical result envelope; this remains walk-forward
        # because attach_skill_for_upcoming fits only dates before month start.
        old_skill_fwd = live_fwd["skill_diff_mean"].copy()
        old_skill_rev = live_rev["skill_diff_mean"].copy()
        live_fwd = attach_skill_for_upcoming(live_fwd, fights)
        live_rev = attach_skill_for_upcoming(live_rev, fights)
        diagnostics["live_skill_mean_abs_shift_fwd"] = float(
            (old_skill_fwd - live_fwd["skill_diff_mean"]).abs().mean()
        )
        diagnostics["live_skill_mean_abs_shift_rev"] = float(
            (old_skill_rev - live_rev["skill_diff_mean"]).abs().mean()
        )
        live_meta = pd.read_parquet(live_meta_path)
        order = np.argsort(live_meta["file"].to_numpy())
        live_fwd = live_fwd.iloc[order].reset_index(drop=True)
        live_rev = live_rev.iloc[order].reset_index(drop=True)
        live_meta = live_meta.iloc[order].reset_index(drop=True)
        live_ledger = pd.read_parquet(live_ledger_path).sort_values("file").reset_index(drop=True)
        if not live_meta["file"].equals(live_ledger["file"]):
            raise ValueError("Archived live raw rows and outcome ledger do not align")

    real_predictions = []
    corrupted_predictions = []
    live_real_predictions = []
    live_corrupted_predictions = []
    for seed in range(seeds):
        print(f"training seed {seed + 1}/{seeds}", flush=True)
        real = train_real(fights, CUTOFF, seed=seed)
        corrupted = train_corrupted(fights, CUTOFF, seed=seed)
        real_predictions.append(0.5 * (predict(real, eval_frame) + 1 - predict(real, reverse)))
        corrupted_predictions.append(0.5 * (predict(corrupted, eval_frame) + 1 - predict(corrupted, reverse)))
        if have_live:
            live_real_predictions.append(0.5 * (predict(real, live_fwd) + 1 - predict(real, live_rev)))
            live_corrupted_predictions.append(
                0.5 * (predict(corrupted, live_fwd) + 1 - predict(corrupted, live_rev))
            )

    real_mean = _sharpen(np.stack(real_predictions).mean(axis=0))
    corrupted_mean = _sharpen(np.stack(corrupted_predictions).mean(axis=0))
    result = {
        "source_dir": str(source_dir),
        "cutoff": str(CUTOFF.date()),
        "seeds": seeds,
        "diagnostics": diagnostics,
        "strict_raw_real": _gate(eval_frame, real_mean),
        "strict_raw_corrupted": _gate(eval_frame, corrupted_mean),
    }
    if have_live:
        result["strict_raw_retrained_on_archived_live_universe"] = {
            "real": _gate_live(live_ledger, _sharpen(np.stack(live_real_predictions).mean(axis=0))),
            "corrupted": _gate_live(live_ledger, _sharpen(np.stack(live_corrupted_predictions).mean(axis=0))),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ufcstats-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=10)
    args = parser.parse_args()
    result = run(args.ufcstats_dir, args.seeds)
    METRICS.mkdir(parents=True, exist_ok=True)
    output = METRICS / "strict_raw_retrain_audit.json"
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()

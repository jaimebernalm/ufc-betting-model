"""Test whether UFC prediction-market quality and model ROI change over time.

The post-cutoff model analysis is restricted to non-debut fights and consumes
the per-fight output from ``strict_multivenue_postcutoff.py``.  The longer
Polymarket-only analysis evaluates market closing-price quality and volume on
all matched fights; it does not treat pre-cutoff model predictions as OOS.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

try:
    from .strict_multivenue_postcutoff import _effective_decimal
except ImportError:  # Support direct execution from the scripts directory.
    from strict_multivenue_postcutoff import _effective_decimal
from ufc_pred.inference.sizing import _full_kelly

ROOT = Path(__file__).resolve().parents[1]
DETAILS = ROOT / "artifacts/metrics/strict_multivenue_postcutoff_fights.parquet"
MATCHED = ROOT / "data/interim/polymarket_matched_to_kaggle_v2.parquet"
POLY_RAW = ROOT / "data/raw/polymarket/historical_2024-04-13_to_2026-05-24.parquet"
KALSHI_RAW = ROOT / "data/raw/kalshi/snapshots/historical_T-90min_perfight_combined.parquet"
OUTPUT = ROOT / "artifacts/metrics/temporal_market_edge_audit.json"


def _finite(values: pd.Series | np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float)


def clustered_linear(y, x, clusters) -> dict:
    """OLS slope with a CR1 cluster-robust standard error.

    ``x`` is used on its supplied scale. Calendar regressions pass a 0..1
    coordinate, so their slope is the estimated full-period change.
    """
    frame = pd.DataFrame({"y": y, "x": x, "cluster": clusters}).dropna()
    n = len(frame)
    groups = frame["cluster"].nunique()
    if n < 4 or groups < 3 or frame["x"].nunique() < 2:
        return {"n": n, "clusters": groups, "slope": None, "se": None, "p": None}
    X = np.column_stack([np.ones(n), _finite(frame["x"])])
    target = _finite(frame["y"])
    bread = np.linalg.inv(X.T @ X)
    beta = bread @ X.T @ target
    residual = target - X @ beta
    meat = np.zeros((2, 2))
    cluster_values = frame["cluster"].to_numpy()
    for group in pd.unique(cluster_values):
        use = cluster_values == group
        score = X[use].T @ residual[use]
        meat += np.outer(score, score)
    correction = groups / (groups - 1) * (n - 1) / (n - 2)
    covariance = correction * bread @ meat @ bread
    se = float(np.sqrt(max(covariance[1, 1], 0.0)))
    statistic = float(beta[1] / se) if se else np.inf
    p = float(2 * stats.t.sf(abs(statistic), df=groups - 1))
    return {
        "n": n,
        "clusters": groups,
        "slope": float(beta[1]),
        "se": se,
        "p": p,
    }


def calendar_coordinate(dates: pd.Series) -> np.ndarray:
    values = pd.to_datetime(dates)
    span = (values.max() - values.min()).days
    if span == 0:
        return np.zeros(len(values))
    return (values - values.min()).dt.days.to_numpy(float) / span


def add_venue_metrics(frame: pd.DataFrame, model: str) -> pd.DataFrame:
    out = frame.copy()
    y = out["Winner"].eq("Red").to_numpy(int)
    p = out[f"p_red_{model}"].to_numpy(float)
    price_r = out["price_red"].to_numpy(float)
    price_b = out["price_blue"].to_numpy(float)
    market_p = price_r / (price_r + price_b)
    edge_r = p - price_r
    edge_b = (1.0 - p) - price_b
    bet_red = edge_r >= edge_b
    take = np.maximum(edge_r, edge_b) >= 0.03
    won = np.where(bet_red, y == 1, y == 0)
    rate = out["fee_rate"].to_numpy(float)
    exponent = out["fee_exponent"].to_numpy(float)
    decimal = np.where(
        bet_red,
        _effective_decimal(price_r, rate, exponent),
        _effective_decimal(price_b, rate, exponent),
    )
    selected_p = np.where(bet_red, p, 1.0 - p)
    selected_price = np.where(bet_red, price_r, price_b)
    out["market_p_red"] = market_p
    out["model_logloss"] = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    out["market_logloss"] = -(y * np.log(market_p) + (1 - y) * np.log(1 - market_p))
    out["model_brier"] = np.square(p - y)
    out["market_brier"] = np.square(market_p - y)
    out["abs_model_market_gap"] = np.abs(p - market_p)
    out["predicted_edge"] = np.maximum(edge_r, edge_b)
    out["bet"] = take
    out["bet_red"] = bet_red
    out["selected_side_won"] = won.astype(float)
    out["selected_price_all"] = selected_price
    out["selected_pnl_all"] = np.where(won, decimal - 1.0, -1.0)
    out["bet_won"] = np.where(take, won.astype(float), np.nan)
    out["selected_price"] = np.where(take, selected_price, np.nan)
    out["underdog_bet"] = np.where(take, selected_price < 0.5, np.nan)
    out["model_expected_roi"] = np.where(take, selected_p * decimal - 1.0, np.nan)
    out["pnl"] = np.where(take, np.where(won, decimal - 1.0, -1.0), np.nan)
    return out


def policy_grid(frame: pd.DataFrame) -> list[dict]:
    """Predeclared edge/price filters across early and late venue windows."""
    rows = []
    periods = {"all": np.ones(len(frame), dtype=bool)}
    if frame["universe"].iat[0] == "kalshi":
        dates = pd.to_datetime(frame["date"])
        periods["through_2026_05_16"] = (dates <= pd.Timestamp("2026-05-16")).to_numpy()
        periods["after_2026_05_16"] = (dates > pd.Timestamp("2026-05-16")).to_numpy()
    price_rules = {
        "all_prices": np.ones(len(frame), dtype=bool),
        "favorites_only": frame["selected_price_all"].to_numpy(float) >= 0.5,
        "price_at_least_0.25": frame["selected_price_all"].to_numpy(float) >= 0.25,
    }
    for period, period_mask in periods.items():
        for threshold in (0.03, 0.05, 0.10, 0.15, 0.20):
            edge_mask = frame["predicted_edge"].to_numpy(float) >= threshold
            for price_rule, price_mask in price_rules.items():
                use = period_mask & edge_mask & price_mask
                pnl = frame.loc[use, "selected_pnl_all"]
                won = frame.loc[use, "selected_side_won"]
                rows.append(
                    {
                        "period": period,
                        "edge_threshold": threshold,
                        "price_rule": price_rule,
                        "bets": int(use.sum()),
                        "wins": int(won.sum()),
                        "hit_rate": float(won.mean()) if use.any() else None,
                        "roi": float(pnl.mean()) if use.any() else None,
                    }
                )
    return rows


def card_bootstrap_roi(frame: pd.DataFrame, n_boot: int = 20_000) -> tuple[float, float]:
    cards = frame.groupby("date")["selected_pnl_all"].agg(["sum", "count"])
    if cards.empty:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(61)
    indexes = rng.integers(0, len(cards), size=(n_boot, len(cards)))
    returns = cards["sum"].to_numpy()[indexes].sum(axis=1) / cards["count"].to_numpy()[indexes].sum(axis=1)
    low, high = np.percentile(returns, [2.5, 97.5])
    return float(low), float(high)


def price_side_summary(frame: pd.DataFrame) -> dict:
    eligible = frame[frame["predicted_edge"] >= 0.03].copy()
    out = {}
    for label, selected in {
        "favorite": eligible[eligible["selected_price_all"] >= 0.5],
        "underdog": eligible[eligible["selected_price_all"] < 0.5],
    }.items():
        out[label] = {
            "bets": int(len(selected)),
            "wins": int(selected["selected_side_won"].sum()),
            "hit_rate": float(selected["selected_side_won"].mean()),
            "mean_price": float(selected["selected_price_all"].mean()),
            "mean_predicted_probability": float(
                (selected["selected_price_all"] + selected["predicted_edge"]).mean()
            ),
            "roi": float(selected["selected_pnl_all"].mean()),
            "card_ci95": list(card_bootstrap_roi(selected)),
        }
    return out


def favorite_minus_underdog_test(frame: pd.DataFrame, n_boot: int = 100_000) -> dict:
    eligible = frame[frame["predicted_edge"] >= 0.03].copy()
    eligible["favorite"] = eligible["selected_price_all"] >= 0.5
    cards = []
    for _, group in eligible.groupby("date", sort=True):
        favorite = group[group["favorite"]]
        underdog = group[~group["favorite"]]
        cards.append(
            [
                favorite["selected_pnl_all"].sum(),
                len(favorite),
                underdog["selected_pnl_all"].sum(),
                len(underdog),
            ]
        )
    values = np.asarray(cards, dtype=float)
    rng = np.random.default_rng(812)
    indexes = rng.integers(0, len(values), size=(n_boot, len(values)))
    samples = values[indexes].sum(axis=1)
    valid = (samples[:, 1] > 0) & (samples[:, 3] > 0)
    differences = samples[valid, 0] / samples[valid, 1] - samples[valid, 2] / samples[valid, 3]
    observed = (
        eligible.loc[eligible["favorite"], "selected_pnl_all"].mean()
        - eligible.loc[~eligible["favorite"], "selected_pnl_all"].mean()
    )
    return {
        "favorite_minus_underdog_roi": float(observed),
        "card_bootstrap_ci95": [float(x) for x in np.percentile(differences, [2.5, 97.5])],
        "two_sided_p": float(2 * min(np.mean(differences >= 0), np.mean(differences <= 0))),
    }


def polymarket_price_side_validation(
    no_debut: pd.DataFrame,
    all_fights: pd.DataFrame,
) -> dict:
    eligible = no_debut[no_debut["predicted_edge"] >= 0.03].copy()
    eligible["price_side"] = np.where(eligible["selected_price_all"] >= 0.5, "favorite", "underdog")
    eligible["month"] = pd.to_datetime(eligible["date"]).dt.to_period("M").astype(str)
    dates = sorted(pd.to_datetime(eligible["date"]).unique())
    split_date = pd.Timestamp(dates[len(dates) // 2])
    monthly_rows = []
    for (month, side), group in eligible.groupby(["month", "price_side"]):
        monthly_rows.append(
            {
                "month": month,
                "price_side": side,
                "bets": int(len(group)),
                "wins": int(group["selected_side_won"].sum()),
                "hit_rate": float(group["selected_side_won"].mean()),
                "roi": float(group["selected_pnl_all"].mean()),
            }
        )
    trends = {}
    for side in ("favorite", "underdog"):
        selected = eligible[eligible["price_side"] == side]
        coordinate = calendar_coordinate(selected["date"])
        trends[side] = {
            "roi": clustered_linear(selected["selected_pnl_all"], coordinate, selected["date"]),
            "hit_rate": clustered_linear(selected["selected_side_won"], coordinate, selected["date"]),
        }
    return {
        "no_debut": price_side_summary(no_debut),
        "all_including_debut": price_side_summary(all_fights),
        "favorite_minus_underdog": favorite_minus_underdog_test(no_debut),
        "chronological_split_date": str(split_date.date()),
        "first_nine_cards": price_side_summary(no_debut[pd.to_datetime(no_debut["date"]) < split_date]),
        "last_nine_cards": price_side_summary(no_debut[pd.to_datetime(no_debut["date"]) >= split_date]),
        "monthly": monthly_rows,
        "trends": trends,
    }


def decision_slices(frame: pd.DataFrame) -> list[dict]:
    """Auditable 3pp-policy slices used in the operational recommendation."""
    periods = {"all": np.ones(len(frame), dtype=bool)}
    if frame["universe"].iat[0] == "kalshi":
        dates = pd.to_datetime(frame["date"])
        periods["through_2026_05_16_close_fallback"] = (dates <= pd.Timestamp("2026-05-16")).to_numpy()
        periods["after_2026_05_16_asks"] = (dates > pd.Timestamp("2026-05-16")).to_numpy()
    price_sides = {
        "all": np.ones(len(frame), dtype=bool),
        "favorite": frame["selected_price_all"].to_numpy(float) >= 0.5,
        "underdog": frame["selected_price_all"].to_numpy(float) < 0.5,
    }
    rows = []
    eligible = frame["predicted_edge"].to_numpy(float) >= 0.03
    for period, period_mask in periods.items():
        for price_side, side_mask in price_sides.items():
            use = period_mask & side_mask & eligible
            selected = frame.loc[use]
            ci = card_bootstrap_roi(selected)
            rows.append(
                {
                    "period": period,
                    "price_side": price_side,
                    "fights": int(period_mask.sum()),
                    "bets": int(len(selected)),
                    "wins": int(selected["selected_side_won"].sum()),
                    "hit_rate": float(selected["selected_side_won"].mean()),
                    "mean_price": float(selected["selected_price_all"].mean()),
                    "mean_predicted_probability": float(
                        (selected["selected_price_all"] + selected["predicted_edge"]).mean()
                    ),
                    "roi": float(selected["selected_pnl_all"].mean()),
                    "card_ci95": list(ci),
                }
            )
    return rows


def monthly(frame: pd.DataFrame) -> list[dict]:
    out = frame.copy()
    out["month"] = pd.to_datetime(out["date"]).dt.to_period("M").astype(str)
    rows = []
    for month, group in out.groupby("month", sort=True):
        bets = group[group["bet"]]
        rows.append(
            {
                "month": month,
                "fights": int(len(group)),
                "cards": int(group["date"].nunique()),
                "bets": int(len(bets)),
                "wins": int(bets["bet_won"].sum()),
                "hit_rate": float(bets["bet_won"].mean()) if len(bets) else None,
                "roi": float(bets["pnl"].mean()) if len(bets) else None,
                "bet_rate": float(group["bet"].mean()),
                "mean_predicted_edge": float(bets["predicted_edge"].mean()) if len(bets) else None,
                "mean_model_expected_roi": float(bets["model_expected_roi"].mean()) if len(bets) else None,
                "mean_abs_model_market_gap": float(group["abs_model_market_gap"].mean()),
                "model_logloss": float(group["model_logloss"].mean()),
                "market_logloss": float(group["market_logloss"].mean()),
                "median_volume": float(group["volume"].median()) if "volume" in group else None,
            }
        )
    return rows


def breakpoint_scan(frame: pd.DataFrame, n_perm: int = 20_000) -> dict:
    """Card-level, multiple-break-corrected scan of betting return changes."""
    cards = frame[frame["bet"]].groupby("date", sort=True)["pnl"].agg(["mean", "sum", "count"]).reset_index()
    n = len(cards)
    eligible = np.arange(5, n - 4)
    if n < 10 or not len(eligible):
        return {"cards": n, "break_date": None, "permutation_p": None}
    values = cards["mean"].to_numpy(float)

    def contrasts(order: np.ndarray) -> np.ndarray:
        cumulative = np.cumsum(order)
        total = cumulative[-1]
        return np.array([(total - cumulative[k - 1]) / (n - k) - cumulative[k - 1] / k for k in eligible])

    observed = contrasts(values)
    selected = int(np.argmax(np.abs(observed)))
    split = int(eligible[selected])
    statistic = float(np.max(np.abs(observed)))
    rng = np.random.default_rng(29)
    exceed = 0
    for _ in range(n_perm):
        if np.max(np.abs(contrasts(rng.permutation(values)))) >= statistic:
            exceed += 1
    before = cards.iloc[:split]
    after = cards.iloc[split:]
    return {
        "cards": n,
        "break_date": str(pd.Timestamp(cards.iloc[split]["date"]).date()),
        "before_cards": int(len(before)),
        "after_cards": int(len(after)),
        "before_bets": int(before["count"].sum()),
        "after_bets": int(after["count"].sum()),
        "before_roi": float(before["sum"].sum() / before["count"].sum()),
        "after_roi": float(after["sum"].sum() / after["count"].sum()),
        "card_mean_change": float(observed[selected]),
        "permutation_p": float((exceed + 1) / (n_perm + 1)),
    }


def temporal_tests(frame: pd.DataFrame) -> dict:
    coordinate = calendar_coordinate(frame["date"])
    results = {
        "model_logloss": clustered_linear(frame["model_logloss"], coordinate, frame["date"]),
        "market_logloss": clustered_linear(frame["market_logloss"], coordinate, frame["date"]),
        "model_minus_market_logloss": clustered_linear(
            frame["model_logloss"] - frame["market_logloss"], coordinate, frame["date"]
        ),
        "abs_model_market_gap": clustered_linear(frame["abs_model_market_gap"], coordinate, frame["date"]),
        "bet_rate": clustered_linear(frame["bet"].astype(float), coordinate, frame["date"]),
    }
    bets = frame[frame["bet"]].copy()
    bet_coordinate = calendar_coordinate(bets["date"])
    results.update(
        {
            "bet_hit_rate": clustered_linear(bets["bet_won"], bet_coordinate, bets["date"]),
            "bet_roi": clustered_linear(bets["pnl"], bet_coordinate, bets["date"]),
            "predicted_edge": clustered_linear(bets["predicted_edge"], bet_coordinate, bets["date"]),
            "model_expected_roi": clustered_linear(bets["model_expected_roi"], bet_coordinate, bets["date"]),
            "breakpoint": breakpoint_scan(frame),
        }
    )
    if "volume" in frame and frame["volume"].notna().sum() >= 10:
        log_volume = np.log10(frame["volume"])
        results["log10_volume_over_time"] = clustered_linear(log_volume, coordinate, frame["date"])
        bet_log_volume = np.log10(bets["volume"])
        results["bet_roi_vs_log10_volume"] = clustered_linear(bets["pnl"], bet_log_volume, bets["date"])
        results["gap_vs_log10_volume"] = clustered_linear(
            frame["abs_model_market_gap"], log_volume, frame["date"]
        )
    return results


def summarize(frame: pd.DataFrame) -> dict:
    bets = frame[frame["bet"]]
    return {
        "fights": int(len(frame)),
        "cards": int(frame["date"].nunique()),
        "bets": int(len(bets)),
        "wins": int(bets["bet_won"].sum()),
        "hit_rate": float(bets["bet_won"].mean()),
        "roi": float(bets["pnl"].mean()),
        "model_logloss": float(frame["model_logloss"].mean()),
        "market_logloss": float(frame["market_logloss"].mean()),
        "mean_abs_model_market_gap": float(frame["abs_model_market_gap"].mean()),
        "mean_predicted_edge_on_bets": float(bets["predicted_edge"].mean()),
        "mean_model_expected_roi_on_bets": float(bets["model_expected_roi"].mean()),
    }


def load_volumes() -> pd.DataFrame:
    raw = pd.read_parquet(POLY_RAW)
    raw["market_id"] = raw["market_id"].astype(str)
    return raw.sort_values("scraped_at").drop_duplicates("market_id", keep="last")[["market_id", "volume"]]


def full_polymarket_quality(volumes: pd.DataFrame) -> dict:
    matched = pd.read_parquet(MATCHED)
    matched["date"] = pd.to_datetime(matched["date"])
    matched["market_id"] = matched["market_id"].astype(str)
    frame = matched.merge(volumes, on="market_id", how="left", validate="one_to_one")
    y = frame["Winner"].eq("Red").to_numpy(int)
    price_red = frame["polymarket_p_red"].to_numpy(float)
    p = price_red / (price_red + frame["polymarket_p_blue"].to_numpy(float))
    frame["market_logloss"] = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    frame["market_brier"] = np.square(p - y)
    frame["market_correct"] = (p >= 0.5) == y
    frame["time"] = calendar_coordinate(frame["date"])
    frame["log10_volume"] = np.log10(frame["volume"])
    midpoint = frame["date"].sort_values().iloc[len(frame) // 2]

    def period_summary(part: pd.DataFrame) -> dict:
        return {
            "fights": int(len(part)),
            "accuracy": float(part["market_correct"].mean()),
            "logloss": float(part["market_logloss"].mean()),
            "brier": float(part["market_brier"].mean()),
            "median_volume": float(part["volume"].median()),
        }

    return {
        "date_min": str(frame["date"].min().date()),
        "date_max": str(frame["date"].max().date()),
        "fights": int(len(frame)),
        "volume_coverage": int(frame["volume"].notna().sum()),
        "chronological_midpoint": str(midpoint.date()),
        "first_half": period_summary(frame[frame["date"] <= midpoint]),
        "second_half": period_summary(frame[frame["date"] > midpoint]),
        "trends": {
            "accuracy_over_time": clustered_linear(
                frame["market_correct"].astype(float), frame["time"], frame["date"]
            ),
            "logloss_over_time": clustered_linear(frame["market_logloss"], frame["time"], frame["date"]),
            "brier_over_time": clustered_linear(frame["market_brier"], frame["time"], frame["date"]),
            "log10_volume_over_time": clustered_linear(frame["log10_volume"], frame["time"], frame["date"]),
            "logloss_vs_log10_volume": clustered_linear(
                frame["market_logloss"], frame["log10_volume"], frame["date"]
            ),
        },
        "by_year": {str(year): period_summary(group) for year, group in frame.groupby(frame["date"].dt.year)},
    }


def kalshi_ask_close_comparison(base: pd.DataFrame) -> dict:
    """Hold post-May fights/model fixed and change only ask versus last trade."""
    raw = pd.read_parquet(KALSHI_RAW).drop_duplicates("event_ticker", keep="last")
    columns = [
        "event_ticker",
        "canon_a",
        "canon_b",
        "close_yes_price_a",
        "close_yes_price_b",
        "ask_yes_price_a",
        "ask_yes_price_b",
    ]
    frame = base.merge(
        raw[columns],
        left_on="market_id",
        right_on="event_ticker",
        how="left",
        validate="one_to_one",
    )
    a_is_red = frame["R_fighter"].eq(frame["canon_a"])
    frame["close_red"] = np.where(a_is_red, frame["close_yes_price_a"], frame["close_yes_price_b"])
    frame["close_blue"] = np.where(a_is_red, frame["close_yes_price_b"], frame["close_yes_price_a"])
    post = frame[pd.to_datetime(frame["date"]) > pd.Timestamp("2026-05-16")].copy()
    ask = add_venue_metrics(post, "real")
    close_input = post.copy()
    close_input["price_red"] = close_input["close_red"]
    close_input["price_blue"] = close_input["close_blue"]
    close = add_venue_metrics(close_input, "real")
    red_diff = ask["price_red"] - close["price_red"]
    blue_diff = ask["price_blue"] - close["price_blue"]
    return {
        "fights": int(len(post)),
        "pairs_identical": int(((red_diff == 0.0) & (blue_diff == 0.0)).sum()),
        "contracts_one_cent_higher_at_ask": int(
            np.isclose(red_diff, 0.01).sum() + np.isclose(blue_diff, 0.01).sum()
        ),
        "mean_ask_minus_last_trade_per_contract": float(pd.concat([red_diff, blue_diff]).mean()),
        "bet_eligibility_changes": int((ask["bet"] != close["bet"]).sum()),
        "selected_side_changes": int((ask["bet_red"] != close["bet_red"]).sum()),
        "ask": decision_slices(ask)[0:3],
        "last_trade": decision_slices(close)[0:3],
    }


def post_may_sizing_counterfactual(base: pd.DataFrame) -> dict:
    """Compound the later corrected-model bets under candidate sizing rules."""
    post = base[pd.to_datetime(base["date"]) > pd.Timestamp("2026-05-16")].copy()
    frame = add_venue_metrics(post, "real")
    frame = frame[frame["bet"]].copy()
    selected_p = frame["selected_price"] + frame["predicted_edge"]
    frame["full_kelly"] = [
        _full_kelly(p, price, 0.07) for p, price in zip(selected_p, frame["selected_price"], strict=False)
    ]
    configurations = {
        "current_A_0.10_kelly_cap_0.10": (0.10, 0.10),
        "current_B_0.25_kelly_uncapped": (0.25, None),
        "full_kelly_uncapped": (1.0, None),
        "0.10_kelly_cap_0.02": (0.10, 0.02),
        "flat_fraction_0.005": (None, 0.005),
    }
    result = {}
    slices = {
        "all": frame,
        "favorites": frame[frame["selected_price"] >= 0.5],
        "underdogs": frame[frame["selected_price"] < 0.5],
    }
    for label, selected in slices.items():
        outcomes = {}
        for name, (kelly_fraction, cap) in configurations.items():
            if kelly_fraction is None:
                stake_fraction = np.full(len(selected), cap)
            else:
                stake_fraction = kelly_fraction * selected["full_kelly"].to_numpy()
                if cap is not None:
                    stake_fraction = np.minimum(stake_fraction, cap)
            wealth_factors = 1.0 + stake_fraction * selected["pnl"].to_numpy()
            outcomes[name] = {
                "final_bankroll_multiple": float(np.prod(wealth_factors)),
                "mean_stake_fraction": float(stake_fraction.mean()),
                "max_stake_fraction": float(stake_fraction.max()),
            }
        result[label] = {"bets": int(len(selected)), "configurations": outcomes}
    return result


def sizing_configurations(frame: pd.DataFrame, model: str) -> tuple[dict, pd.DataFrame]:
    priced = add_venue_metrics(frame.copy(), model)
    selected = priced[priced["bet"]].copy()
    selected_p = selected["selected_price"] + selected["predicted_edge"]
    selected["full_kelly"] = [
        _full_kelly(p, price, fee)
        for p, price, fee in zip(selected_p, selected["selected_price"], selected["fee_rate"], strict=False)
    ]
    configurations = {
        "A_0.10_kelly_cap_0.10": (0.10, 0.10),
        "B_or_C_0.25_kelly_uncapped": (0.25, None),
        "full_kelly_uncapped": (1.0, None),
        "0.10_kelly_cap_0.02": (0.10, 0.02),
        "flat_fraction_0.005": (None, 0.005),
    }
    outcomes = {}
    for name, (kelly_fraction, cap) in configurations.items():
        if kelly_fraction is None:
            stake_fraction = np.full(len(selected), cap)
        else:
            stake_fraction = kelly_fraction * selected["full_kelly"].to_numpy()
            if cap is not None:
                stake_fraction = np.minimum(stake_fraction, cap)
        wealth_factor = 1.0 + stake_fraction * selected["pnl"].to_numpy()
        selected[f"log_factor_{name}"] = np.log(wealth_factor)
        card_factor = (
            pd.DataFrame({"date": selected["date"].to_numpy(), "factor": wealth_factor})
            .groupby("date")["factor"]
            .prod()
        )
        path = card_factor.cumprod()
        running = np.r_[1.0, path.to_numpy()]
        drawdown = 1.0 - running / np.maximum.accumulate(running)
        outcomes[name] = {
            "final_bankroll_multiple": float(np.prod(wealth_factor)),
            "mean_log_growth_per_bet": float(np.log(wealth_factor).mean()),
            "card_end_max_drawdown": float(drawdown.max()),
            "mean_stake_fraction": float(stake_fraction.mean()),
            "max_stake_fraction": float(stake_fraction.max()),
        }
    return outcomes, selected


def paired_model_growth_bootstrap(frame: pd.DataFrame, n_boot: int = 100_000) -> dict:
    per_model = {}
    for model in ("real", "corrupted"):
        _, selected = sizing_configurations(frame, model)
        column = "log_factor_B_or_C_0.25_kelly_uncapped"
        per_model[model] = selected.groupby("date")[column].sum()
    cards = pd.concat(per_model, axis=1).fillna(0.0)
    values = cards.to_numpy(float)
    rng = np.random.default_rng(9921)
    indexes = rng.integers(0, len(values), size=(n_boot, len(values)))
    samples = values[indexes].mean(axis=1)
    result = {}
    for index, model in enumerate(("real", "corrupted")):
        distribution = samples[:, index]
        result[model] = {
            "mean_card_log_growth": float(values[:, index].mean()),
            "card_bootstrap_ci95": [float(x) for x in np.percentile(distribution, [2.5, 97.5])],
            "bootstrap_fraction_positive": float(np.mean(distribution > 0)),
        }
    difference = samples[:, 0] - samples[:, 1]
    result["real_minus_corrupted"] = {
        "mean_card_log_growth_difference": float((values[:, 0] - values[:, 1]).mean()),
        "card_bootstrap_ci95": [float(x) for x in np.percentile(difference, [2.5, 97.5])],
        "two_sided_p": float(2 * min(np.mean(difference >= 0), np.mean(difference <= 0))),
    }
    return result


def model_selection_overlap(frame: pd.DataFrame) -> dict:
    real = add_venue_metrics(frame.copy(), "real")
    corrupted = add_venue_metrics(frame.copy(), "corrupted")
    both = real["bet"] & corrupted["bet"]
    same_side = real.loc[both, "bet_red"].to_numpy() == corrupted.loc[both, "bet_red"].to_numpy()
    return {
        "fights": int(len(frame)),
        "real_bets": int(real["bet"].sum()),
        "corrupted_bets": int(corrupted["bet"].sum()),
        "both_bet": int(both.sum()),
        "same_side_when_both_bet": int(same_side.sum()),
        "opposite_side_when_both_bet": int((~same_side).sum()),
        "prediction_correlation": float(np.corrcoef(real["p_red_real"], corrupted["p_red_corrupted"])[0, 1]),
        "common_bet_pnl_correlation": float(
            np.corrcoef(real.loc[both, "pnl"], corrupted.loc[both, "pnl"])[0, 1]
        ),
    }


def quarter_minus_tenth_kelly_bootstrap(selected: pd.DataFrame, n_boot: int = 100_000) -> dict:
    grouped = selected.groupby("date")
    tenth = grouped["log_factor_A_0.10_kelly_cap_0.10"].sum()
    quarter = grouped["log_factor_B_or_C_0.25_kelly_uncapped"].sum()
    difference = (quarter - tenth).to_numpy(float)
    rng = np.random.default_rng(771)
    indexes = rng.integers(0, len(difference), size=(n_boot, len(difference)))
    distribution = difference[indexes].mean(axis=1)
    return {
        "mean_card_log_growth_difference": float(difference.mean()),
        "card_bootstrap_ci95": [float(x) for x in np.percentile(distribution, [2.5, 97.5])],
        "bootstrap_fraction_quarter_better": float(np.mean(distribution > 0)),
        "two_sided_p": float(2 * min(np.mean(distribution >= 0), np.mean(distribution <= 0))),
    }


def polymarket_sizing_validation(all_fights: pd.DataFrame) -> dict:
    result = {}
    for scope, frame in {
        "no_debut": all_fights[~all_fights["has_debut"]].copy(),
        "all_including_debut": all_fights.copy(),
        "debut_only": all_fights[all_fights["has_debut"]].copy(),
    }.items():
        models = {}
        for model in ("real", "corrupted"):
            configurations, selected = sizing_configurations(frame, model)
            models[model] = {
                "bets": int(len(selected)),
                "wins": int(selected["bet_won"].sum()),
                "flat_roi": float(selected["pnl"].mean()),
                "configurations": configurations,
                "quarter_minus_tenth_kelly": quarter_minus_tenth_kelly_bootstrap(selected),
            }
        result[scope] = {
            "fights": int(len(frame)),
            "cards": int(frame["date"].nunique()),
            "models": models,
            "quarter_kelly_card_bootstrap": paired_model_growth_bootstrap(frame),
            "model_overlap": model_selection_overlap(frame),
        }
    return result


def two_model_sizing_validation(frame: pd.DataFrame) -> dict:
    models = {}
    for model in ("real", "corrupted"):
        configurations, selected = sizing_configurations(frame, model)
        models[model] = {
            "bets": int(len(selected)),
            "wins": int(selected["bet_won"].sum()),
            "flat_roi": float(selected["pnl"].mean()),
            "configurations": configurations,
            "quarter_minus_tenth_kelly": quarter_minus_tenth_kelly_bootstrap(selected),
        }
    return {
        "fights": int(len(frame)),
        "cards": int(frame["date"].nunique()),
        "models": models,
        "quarter_kelly_card_bootstrap": paired_model_growth_bootstrap(frame),
        "model_overlap": model_selection_overlap(frame),
    }


def run() -> dict:
    details = pd.read_parquet(DETAILS)
    details["date"] = pd.to_datetime(details["date"])
    volumes = load_volumes()
    result = {
        "post_cutoff_non_debut": {},
        "full_polymarket_market_quality": None,
        "kalshi_post_may_ask_vs_last_trade": None,
        "polymarket_favorite_underdog_validation": None,
        "common_fight_price_side_validation": None,
        "kalshi_post_may_sizing_counterfactual": None,
        "polymarket_sizing_validation": None,
        "kalshi_post_may_model_sizing_validation": None,
    }
    for universe in ("polymarket", "kalshi"):
        base = details[(details["universe"] == universe) & ~details["has_debut"]].copy()
        if universe == "polymarket":
            base["market_id"] = base["market_id"].astype(str)
            base = base.merge(volumes, on="market_id", how="left", validate="one_to_one")
        for model in ("real", "corrupted"):
            frame = add_venue_metrics(base, model)
            result["post_cutoff_non_debut"].setdefault(universe, {})[model] = {
                "summary": summarize(frame),
                "monthly": monthly(frame),
                "tests": temporal_tests(frame),
                "policy_grid": policy_grid(frame),
                "decision_slices": decision_slices(frame),
            }
    result["full_polymarket_market_quality"] = full_polymarket_quality(volumes)
    kalshi_base = details[(details["universe"] == "kalshi") & ~details["has_debut"]].copy()
    result["kalshi_post_may_ask_vs_last_trade"] = kalshi_ask_close_comparison(kalshi_base)
    result["kalshi_post_may_sizing_counterfactual"] = post_may_sizing_counterfactual(kalshi_base)
    result["kalshi_post_may_model_sizing_validation"] = two_model_sizing_validation(
        kalshi_base[pd.to_datetime(kalshi_base["date"]) > pd.Timestamp("2026-05-16")].copy()
    )
    polymarket_all = details[details["universe"] == "polymarket"].copy()
    polymarket_no_debut = polymarket_all[~polymarket_all["has_debut"]].copy()
    result["polymarket_favorite_underdog_validation"] = polymarket_price_side_validation(
        add_venue_metrics(polymarket_no_debut, "real"),
        add_venue_metrics(polymarket_all, "real"),
    )
    result["polymarket_sizing_validation"] = polymarket_sizing_validation(polymarket_all)
    keys = ["date", "R_fighter", "B_fighter"]
    common = polymarket_no_debut[keys].merge(kalshi_base[keys], on=keys).drop_duplicates()
    common_results = {}
    for venue, base in (
        ("polymarket", polymarket_no_debut),
        ("kalshi", kalshi_base),
    ):
        selected = common.merge(base, on=keys, how="inner", validate="one_to_one")
        common_results[venue] = price_side_summary(add_venue_metrics(selected, "real"))
    result["common_fight_price_side_validation"] = common_results
    OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False))
    return result


def main() -> None:
    result = run()
    print(json.dumps(result, indent=2))
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()

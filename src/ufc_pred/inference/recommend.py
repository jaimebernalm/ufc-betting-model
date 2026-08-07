"""Turn a fight plus a market snapshot into a sized, structured recommendation.

`predict_one_fight` is the single entry point both callers use:
  - scripts/cli/predict_bet.py  (CLI over a full Polymarket/Kalshi card)
  - scripts/cli/bet_runner.py   (per-fight T-90min watchdog)

Nothing here places an order. The output is a recommendation; execution is
manual.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict

import pandas as pd

from ufc_pred.inference.ensemble_predict import predict_ensemble, predict_ensemble_symmetric
from ufc_pred.inference.polymarket_card import CardFight
from ufc_pred.inference.sizing import (
    KALSHI_FEE_COEFF,
    AccountConfig,
    Recommendation,
    default_accounts,
    size_bets_combined,
)
from ufc_pred.inference.skill_for_upcoming import attach_skill_for_upcoming
from ufc_pred.inference.upcoming_builder import build_upcoming_row, resolve_fighter_name
from ufc_pred.ingest.ufcstats_state import UFCStatsStateSource
from ufc_pred.paths import CONFIGS


def _load_sharpen_t() -> float:
    """Logit temperature applied to ensemble means before sizing.
    T=1.25 validated cross-window 2026-06-11; configs/inference.json overrides,
    missing/unreadable file falls back to 1.0 (no sharpening)."""
    try:
        with open(CONFIGS / "inference.json") as f:
            return float(json.load(f).get("sharpen_T", 1.0))
    except (OSError, ValueError):
        return 1.0


def _sharpen(p: float, t: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    z = math.log(p / (1 - p)) * t
    return 1.0 / (1.0 + math.exp(-z))


def _guess_weight_class(fight: CardFight, fights: pd.DataFrame) -> str:
    """Parse weight class from event title (Polymarket convention) or fall
    back to each fighter's modal historical class."""
    m = re.search(r"\(([^,]+)\s*,", fight.event_title)
    if m:
        wc = m.group(1).strip()
        if wc in fights["weight_class"].unique():
            return wc
    a = resolve_fighter_name(fight.fighter_a, fights)
    b = resolve_fighter_name(fight.fighter_b, fights)
    cand = pd.concat(
        [
            fights[fights["R_fighter"] == a]["weight_class"],
            fights[fights["B_fighter"] == a]["weight_class"],
            fights[fights["R_fighter"] == b]["weight_class"],
            fights[fights["B_fighter"] == b]["weight_class"],
        ]
    )
    return cand.mode().iat[0] if not cand.empty else "Bantamweight"


def predict_one_fight(
    fight: CardFight,
    fights: pd.DataFrame,
    rankings: pd.DataFrame,
    bankrolls: dict[str, float],
    *,
    weight_class_hint: str | None = None,
    title_bout: bool = False,
    no_of_rounds: int = 3,
    accounts: list[AccountConfig] | None = None,
    state_source: UFCStatsStateSource | None = None,
    fee_coeff: float = KALSHI_FEE_COEFF,
) -> dict:
    """Run the full predict-and-size pipeline on a single fight.

    Returns a structured dict; never prints, never writes to disk. Callers
    (CLI, watchdog, notification builder) decide what to do with the result.

    Result shape:
      {
        "status": "ok" | "skip" | "error",
        "error": str | None,
        "fight": { fighter_a, fighter_b, market_id, token_a, token_b,
                   fight_date_iso, event_title, weight_class },
        "market": { best_ask_a, best_ask_b, best_bid_a, best_bid_b,
                    volume, liquidity },
        "model": { p_a_real, p_a_corrupted,
                   real_per_seed_std, corrupted_per_seed_std,
                   n_seeds },
        "recommendations": [Recommendation, ...],   # per-account
        "orders": {                                  # one entry per side that BET
          "A": { side_name, total_stake_usd, limit_price, avg_fill_price,
                 total_shares, total_net_profit_if_win,
                 per_account: [{account, stake_usd, shares, net_profit_if_win}] },
          ...
        },
        "summary_line": str,   # one-line human summary for notification subtitle
      }
    """
    if not fight.asks_a or not fight.asks_b:
        return {
            "status": "skip",
            "error": "empty order book",
            "fight": _fight_meta(fight, weight_class_hint),
            "market": _market_meta(fight),
            "model": None,
            "recommendations": [],
            "orders": {},
            "summary_line": f"⚠️ {fight.fighter_a} vs {fight.fighter_b}: empty order book",
        }

    # The UFC.com scraper sometimes returns non-standard classes (e.g.
    # "Catchweight") that don't appear in fights.parquet. Validate the hint;
    # fall back to the modal lookup if unknown.
    known_classes = set(fights["weight_class"].unique())
    if weight_class_hint and weight_class_hint in known_classes:
        weight_class = weight_class_hint
    else:
        weight_class = _guess_weight_class(fight, fights)
    gender = "FEMALE" if weight_class.startswith("Women") else "MALE"

    fight_date = fight.fight_date.tz_localize(None) if fight.fight_date.tzinfo else fight.fight_date
    owned_state_source = state_source is None
    if state_source is None:
        state_source = UFCStatsStateSource()
    try:
        upcoming = build_upcoming_row(
            fighter_a=fight.fighter_a,
            fighter_b=fight.fighter_b,
            fight_date=fight_date,
            weight_class=weight_class,
            fights=fights,
            gender=gender,
            title_bout=title_bout,
            no_of_rounds=no_of_rounds,
            rankings=rankings,
            state_source=state_source,
        )
        # Mirrored orientation: predictions are averaged over both corner
        # orderings so the result cannot depend on market ticker order.
        upcoming_rev = build_upcoming_row(
            fighter_a=fight.fighter_b,
            fighter_b=fight.fighter_a,
            fight_date=fight_date,
            weight_class=weight_class,
            fights=fights,
            gender=gender,
            title_bout=title_bout,
            no_of_rounds=no_of_rounds,
            rankings=rankings,
            state_source=state_source,
        )
    except (OSError, RuntimeError, ValueError, IndexError, KeyError) as exc:
        return {
            "status": "error",
            "error": f"build_upcoming_row failed: {exc}",
            "fight": _fight_meta(fight, weight_class),
            "market": _market_meta(fight),
            "model": None,
            "recommendations": [],
            "orders": {},
            "summary_line": f"⚠️ {fight.fighter_a} vs {fight.fighter_b}: {exc}",
        }
    finally:
        if owned_state_source:
            state_source.close()

    upcoming = attach_skill_for_upcoming(upcoming, fights)
    upcoming_rev = attach_skill_for_upcoming(upcoming_rev, fights)
    pred = predict_ensemble_symmetric(upcoming, upcoming_rev)
    # Per-orientation outputs, kept for monitoring how asymmetric the deployed
    # (non-sign-flipped-augmentation) models still are on this fight.
    pred_fwd = predict_ensemble(upcoming)

    sharpen_t = _load_sharpen_t()
    p_real = _sharpen(pred.real_mean, sharpen_t)
    p_corr = _sharpen(pred.corrupted_mean, sharpen_t)

    accounts = accounts or default_accounts(bankrolls)
    p_models = {a.name: (p_real if a.model == "real" else p_corr) for a in accounts}
    recs = size_bets_combined(
        accounts,
        p_models,
        fight.asks_a,
        fight.asks_b,
        fee_coeff=fee_coeff,
    )

    orders = _aggregate_orders(recs, fight)

    return {
        "status": "ok" if orders else "skip",
        "error": None if orders else "no account elected to bet",
        "fight": _fight_meta(fight, weight_class),
        "market": _market_meta(fight),
        "model": {
            "p_a_real": float(p_real),
            "p_a_corrupted": float(p_corr),
            "p_a_real_raw": float(pred.real_mean),
            "p_a_corrupted_raw": float(pred.corrupted_mean),
            "sharpen_T": float(sharpen_t),
            "real_per_seed_std": float(pred.real_per_seed.std()),
            "corrupted_per_seed_std": float(pred.corrupted_per_seed.std()),
            "n_seeds": int(len(pred.real_per_seed)),
            "symmetrized": True,
            "feature_source": "ufcstats_raw_pre_fight",
            "title_bout": bool(title_bout),
            "no_of_rounds": int(no_of_rounds),
            "fee_coeff": float(fee_coeff),
            "orientation_gap_real": float(2 * (pred_fwd.real_mean - pred.real_mean)),
            "orientation_gap_corrupted": float(2 * (pred_fwd.corrupted_mean - pred.corrupted_mean)),
        },
        "recommendations": [asdict(r) for r in recs],
        "orders": orders,
        "summary_line": _summary_line(fight, pred, orders),
    }


def _fight_meta(fight: CardFight, weight_class: str | None) -> dict:
    fd = fight.fight_date
    return {
        "fighter_a": fight.fighter_a,
        "fighter_b": fight.fighter_b,
        "market_id": fight.market_id,
        "token_a": fight.token_a,
        "token_b": fight.token_b,
        "fight_date_iso": fd.isoformat() if hasattr(fd, "isoformat") else str(fd),
        "event_title": fight.event_title,
        "weight_class": weight_class,
    }


def _market_meta(fight: CardFight) -> dict:
    return {
        "best_ask_a": fight.asks_a[0][0] if fight.asks_a else None,
        "best_ask_b": fight.asks_b[0][0] if fight.asks_b else None,
        "best_bid_a": fight.bids_a[0][0] if fight.bids_a else None,
        "best_bid_b": fight.bids_b[0][0] if fight.bids_b else None,
        "volume": fight.volume,
        "liquidity": fight.liquidity,
    }


def _aggregate_orders(recs: list[Recommendation], fight: CardFight) -> dict:
    by_side: dict[str, list[Recommendation]] = {}
    for rec in recs:
        if rec.decision == "BET" and rec.side:
            by_side.setdefault(rec.side, []).append(rec)
    out = {}
    for side, side_recs in by_side.items():
        side_name = fight.fighter_a if side == "A" else fight.fighter_b
        out[side] = {
            "side_name": side_name,
            "token": fight.token_a if side == "A" else fight.token_b,
            "total_stake_usd": round(sum(r.stake_usd for r in side_recs), 2),
            "limit_price": side_recs[0].limit_price,
            "avg_fill_price": side_recs[0].avg_fill_price,
            "total_shares": round(sum(r.shares for r in side_recs), 2),
            "total_net_profit_if_win": round(sum(r.net_profit_if_win for r in side_recs), 2),
            "edge_cents": side_recs[0].edge_cents,
            "per_account": [
                {
                    "account": r.account,
                    "stake_usd": r.stake_usd,
                    "shares": r.shares,
                    "net_profit_if_win": r.net_profit_if_win,
                    "fee_usd": r.fee_usd,
                    "reason": r.reason,
                }
                for r in side_recs
            ],
        }
    return out


def _summary_line(fight: CardFight, pred, orders: dict) -> str:
    if not orders:
        return f"NO BET — {fight.fighter_a} vs {fight.fighter_b}"
    o = next(iter(orders.values()))
    return (
        f"BET ${o['total_stake_usd']:.2f} on {o['side_name']} "
        f"@ {o['limit_price'] * 100:.0f}¢ "
        f"(edge {o['edge_cents']:+.1f}¢)"
    )

"""Kelly sizing with bankroll cap + liquidity cap + edge threshold.

Mirrors the deployment configs in DEPLOY.md §3:
  Account A: $300, 10%-Kelly, 10% bankroll cap   (real ensemble)
  Account B: $300, 25%-Kelly, no bankroll cap    (real ensemble)
  Account C: $300, 25%-Kelly, no bankroll cap    (corrupted ensemble)

Plus a NEW universal liquidity cap (5% of ask depth within 3¢ of best ask).
The cap is applied to the COMBINED stake across all accounts betting the same
side — since the three accounts share one order book in reality, they cannot
each independently consume the same liquidity. See `size_bets_combined`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

EDGE_THRESHOLD = 0.03  # matches strategy_grid.EDGE_THR (gross edge, pre-fee)
KALSHI_FEE_COEFF = 0.07  # fee per contract = coeff × P × (1−P), paid upfront
LIQUIDITY_CAP_FRACTION = 0.05
LIQUIDITY_DEPTH_BAND_CENTS = 3


def _fee_per_contract(price: float, fee_coeff: float = KALSHI_FEE_COEFF) -> float:
    """Quadratic taker fee per contract at fill price ``price``."""
    return fee_coeff * price * (1.0 - price)


def _full_kelly(p: float, price: float, fee_coeff: float = KALSHI_FEE_COEFF) -> float:
    """Fee-correct full-Kelly fraction for buying at `price` with win prob `p`.

    Cost per contract = price + fee (fee paid upfront); profit if win =
    1 − price − fee; loss if lose = full cost. b = win/cost, f* = (b·p − q)/b.
    """
    fee = _fee_per_contract(price, fee_coeff)
    cost = price + fee
    win = 1.0 - cost
    if win <= 0:
        return 0.0
    b = win / cost
    return (b * p - (1.0 - p)) / b


@dataclass(frozen=True)
class AccountConfig:
    name: str
    bankroll: float
    kelly_fraction: float
    bankroll_cap: float | None  # max stake as fraction of bankroll, None = no cap
    model: Literal["real", "corrupted"]


def default_accounts(bankrolls: dict[str, float]) -> list[AccountConfig]:
    """Build the three deployment accounts from a bankroll dict."""
    return [
        AccountConfig("A", bankrolls["A"], kelly_fraction=0.10, bankroll_cap=0.10, model="real"),
        AccountConfig("B", bankrolls["B"], kelly_fraction=0.25, bankroll_cap=None, model="real"),
        AccountConfig("C", bankrolls["C"], kelly_fraction=0.25, bankroll_cap=None, model="corrupted"),
    ]


@dataclass(frozen=True)
class Recommendation:
    account: str
    decision: Literal["BET", "SKIP"]
    side: str | None
    reason: str
    stake_usd: float
    limit_price: float | None
    shares: float
    avg_fill_price: float | None
    net_profit_if_win: float
    edge_cents: float
    # Upfront Kalshi fee included in stake_usd (0.07 × P × (1−P) × shares).
    # Presence of this field marks the post-2026-06-11 convention where
    # stake_usd is the full cash outlay; sync_bankrolls keys off it.
    fee_usd: float = 0.0


def _walk_book(asks: list[tuple[float, float]], stake_usd: float) -> tuple[float, float, float]:
    """Walk the ask book to spend `stake_usd`. Returns (shares, avg_price, limit).
    `asks` must be sorted ascending by price."""
    spent, shares, limit = 0.0, 0.0, asks[0][0]
    for price, size in asks:
        level_cost = price * size
        if spent + level_cost >= stake_usd:
            extra_shares = (stake_usd - spent) / price
            shares += extra_shares
            spent = stake_usd
            limit = price
            break
        spent += level_cost
        shares += size
        limit = price
    avg = spent / shares if shares > 0 else asks[0][0]
    return shares, avg, limit


def _decide_side_and_kelly(
    account: AccountConfig,
    p_model: float,
    asks_a: list[tuple[float, float]],
    asks_b: list[tuple[float, float]],
    fee_coeff: float,
) -> tuple[str | None, float, float, float, list[tuple[float, float]], str | None]:
    """Internal: pick side, compute desired Kelly stake (pre-liquidity-cap).
    Returns (side_name, desired_stake, p, p_market, asks, skip_reason)."""
    if not math.isfinite(account.bankroll) or account.bankroll <= 0 or account.kelly_fraction <= 0:
        return None, 0.0, 0.0, 0.0, [], "no spendable bankroll"
    if not math.isfinite(p_model) or not 0 <= p_model <= 1:
        return None, 0.0, 0.0, 0.0, [], "invalid probability"
    if not asks_a or not asks_b:
        return None, 0.0, 0.0, 0.0, [], "empty order book"
    p_market_a = asks_a[0][0]
    p_market_b = asks_b[0][0]
    edge_a = p_model - p_market_a
    edge_b = (1 - p_model) - p_market_b
    if edge_a >= edge_b:
        side, asks, p_market, edge = "A", asks_a, p_market_a, edge_a
    else:
        side, asks, p_market, edge = "B", asks_b, p_market_b, edge_b

    if edge < EDGE_THRESHOLD:
        return (
            side,
            0.0,
            0.0,
            p_market,
            asks,
            (f"edge {edge * 100:.2f}¢ < threshold {EDGE_THRESHOLD * 100:.0f}¢"),
        )

    p = p_model if side == "A" else 1 - p_model
    full_kelly = _full_kelly(p, p_market, fee_coeff)
    if full_kelly <= 0:
        return side, 0.0, p, p_market, asks, "Kelly negative after fees"

    stake = account.bankroll * account.kelly_fraction * full_kelly
    if account.bankroll_cap is not None:
        cap = account.bankroll * account.bankroll_cap
        if stake > cap:
            stake = cap
    return side, stake, p, p_market, asks, None


def _cash_walk(asks, budget, fee_coeff):
    """Walk actual depth with fees inside the cash budget, never invent shares."""
    shares = notional = fees = 0.0
    limit = None
    for price, quantity in asks:
        cost = price + _fee_per_contract(price, fee_coeff)
        take = min(quantity, max(0.0, budget - notional - fees) / cost)
        if take <= 0:
            break
        shares += take
        notional += price * take
        fees += _fee_per_contract(price, fee_coeff) * take
        limit = price
    return shares, notional, fees, limit


def size_bets_combined(
    accounts: list[AccountConfig],
    p_models: dict[str, float],
    asks_side_a: list[tuple[float, float]],
    asks_side_b: list[tuple[float, float]],
    *,
    fee_coeff: float = KALSHI_FEE_COEFF,
    available_cash: float | None = None,
) -> list[Recommendation]:
    """Frozen Kelly policy, bounded by shared executable depth and cash.

    Every consumed level must retain the original 3-cent gross edge AND
    positive EV after fees for every participating account. A recommendation
    is not a fill: the ledger always uses the venue's actual quantity/fees.
    """

    def clean(book):
        return sorted(
            (float(p), float(q))
            for p, q in book
            if math.isfinite(p) and math.isfinite(q) and 0 < p < 1 and q > 0
        )

    asks_a, asks_b = clean(asks_side_a), clean(asks_side_b)
    desired = [_decide_side_and_kelly(a, p_models[a.name], asks_a, asks_b, fee_coeff) for a in accounts]
    allocations = {}
    for side in ("A", "B"):
        idxs = [i for i, d in enumerate(desired) if d[0] == side and d[5] is None and d[1] > 0]
        if not idxs:
            continue
        book = desired[idxs[0]][4]
        min_p = min(desired[i][2] for i in idxs)
        executable = [
            (p, q)
            for p, q in book
            if min_p - p >= EDGE_THRESHOLD - 1e-12 and min_p > p + _fee_per_contract(p, fee_coeff)
        ]
        depth = sum(p * q for p, q in book if p <= book[0][0] + LIQUIDITY_DEPTH_BAND_CENTS / 100 + 1e-12)
        requested = sum(desired[i][1] for i in idxs)
        budget = min(requested, depth * LIQUIDITY_CAP_FRACTION)
        allocations[side] = (idxs, executable, requested, budget)
    total = sum(a[3] for a in allocations.values())
    cash_scale = 1.0
    if available_cash is not None:
        if not math.isfinite(available_cash) or available_cash < 0:
            raise ValueError("available_cash must be finite and nonnegative")
        cash_scale = min(1.0, available_cash / total) if total else 0.0
    outputs = {}
    for side, (idxs, book, requested, budget) in allocations.items():
        # Reserve one cent for venue fee rounding; fills replace this estimate.
        quantity, notional, fees, limit = _cash_walk(book, max(0, budget * cash_scale - 0.01), fee_coeff)
        if quantity == 0:
            continue
        rounded_fee = math.ceil((fees - 1e-12) * 100) / 100
        cash = notional + rounded_fee
        avg = notional / quantity
        for i in idxs:
            ratio = desired[i][1] / requested
            stake, shares = cash * ratio, quantity * ratio
            p = desired[i][2]
            if stake < 0.50 or p * shares <= stake:
                continue
            outputs[i] = Recommendation(
                accounts[i].name,
                "BET",
                side,
                "Kelly with shared depth, executable EV and cash limits",
                round(stake, 4),
                limit,
                round(shares, 6),
                avg,
                round(shares - stake, 4),
                round((p - limit) * 100, 4),
                round(rounded_fee * ratio, 6),
            )
    return [
        outputs.get(
            i,
            Recommendation(
                a.name,
                "SKIP",
                desired[i][0],
                desired[i][5] or "insufficient executable edge, depth or cash after minimum stake",
                0.0,
                None,
                0.0,
                None,
                0.0,
                0.0,
            ),
        )
        for i, a in enumerate(accounts)
    ]


def size_bet(account, p_model, asks_side_a, asks_side_b, *, fee_coeff=KALSHI_FEE_COEFF, available_cash=None):
    """Single-account adapter to the same execution engine."""
    return size_bets_combined(
        [account],
        {account.name: p_model},
        asks_side_a,
        asks_side_b,
        fee_coeff=fee_coeff,
        available_cash=available_cash,
    )[0]

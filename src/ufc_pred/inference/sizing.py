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


def size_bets_combined(
    accounts: list[AccountConfig],
    p_models: dict[str, float],
    asks_side_a: list[tuple[float, float]],
    asks_side_b: list[tuple[float, float]],
    *,
    fee_coeff: float = KALSHI_FEE_COEFF,
) -> list[Recommendation]:
    """Compute per-account stakes treating the order book as shared.

    Pipeline:
      1. Per account: pick side, compute desired Kelly stake (with bankroll cap)
      2. Group accounts by chosen side
      3. For each side: sum desired stakes, apply liquidity cap to the sum,
         scale each account's stake proportionally if the cap binds
      4. Walk the book once per side with the (possibly scaled) total
         to get a SHARED average fill price for all accounts on that side
      5. Build Recommendation per account
    """
    asks_a = sorted(asks_side_a)
    asks_b = sorted(asks_side_b)
    # Step 1: per-account desired
    desired = []  # list of (acct, side, stake, p, p_market, asks, skip_reason)
    for acct in accounts:
        p_model = p_models[acct.name]
        side, stake, p, p_market, asks, skip = _decide_side_and_kelly(
            acct, p_model, asks_a, asks_b, fee_coeff
        )
        desired.append((acct, side, stake, p, p_market, asks, skip))

    # Step 2-4: group by side, apply liquidity cap to sum, get shared fill
    groups: dict[str, list[int]] = {}
    for i, (_, side, stake, _, _, _, skip) in enumerate(desired):
        if skip is None and stake > 0 and side is not None:
            groups.setdefault(side, []).append(i)

    shared_fill: dict[str, tuple[float, float, list[float]]] = {}
    # side -> (avg_fill_price, limit_price, scaled_stakes_by_index)
    scaled_stakes: dict[int, float] = {}
    cap_messages: dict[int, str] = {}

    for side, idxs in groups.items():
        asks = desired[idxs[0]][5]  # all accounts in same group share the asks
        ask_depth = sum(p * s for p, s in asks if p <= asks[0][0] + LIQUIDITY_DEPTH_BAND_CENTS / 100)
        liq_cap = ask_depth * LIQUIDITY_CAP_FRACTION
        total_desired = sum(desired[i][2] for i in idxs)

        if total_desired > liq_cap:
            scale = liq_cap / total_desired
            for i in idxs:
                scaled_stakes[i] = desired[i][2] * scale
                cap_messages[i] = (
                    f"combined liquidity cap: scaled from "
                    f"${desired[i][2]:.2f} → ${scaled_stakes[i]:.2f} "
                    f"(group total ${total_desired:.2f} > cap ${liq_cap:.2f}, depth ${ask_depth:.2f})"
                )
            total_used = liq_cap
        else:
            for i in idxs:
                scaled_stakes[i] = desired[i][2]
            total_used = total_desired

        shares_total, avg_fill, limit = _walk_book(asks, total_used)
        shared_fill[side] = (avg_fill, limit, shares_total)

    # Step 5: build Recommendations
    out: list[Recommendation] = []
    for i, (acct, side, _desired_stake, p, p_market, _asks, skip) in enumerate(desired):
        if skip is not None:
            edge = (
                (p_models[acct.name] - p_market)
                if side == "A"
                else ((1 - p_models[acct.name]) - p_market)
                if side
                else 0.0
            )
            out.append(
                Recommendation(acct.name, "SKIP", side, skip, 0.0, None, 0.0, None, 0.0, round(edge * 100, 2))
            )
            continue
        stake = scaled_stakes[i]
        if stake < 0.50:
            out.append(
                Recommendation(
                    acct.name,
                    "SKIP",
                    side,
                    f"stake ${stake:.2f} < $0.50 minimum after caps",
                    0.0,
                    None,
                    0.0,
                    None,
                    0.0,
                    0.0,
                )
            )
            continue
        avg_fill, limit, _shares_total = shared_fill[side]
        # stake_usd is the full cash outlay: shares × (price + upfront fee).
        fee_per = _fee_per_contract(avg_fill, fee_coeff)
        shares = stake / (avg_fill + fee_per)
        fee_usd = shares * fee_per
        net_profit = shares - stake
        edge_cents = (p - p_market) * 100
        reason = cap_messages.get(
            i, f"Kelly-limited @ {(stake / (acct.bankroll * acct.kelly_fraction)) * 100:.2f}%"
        )
        out.append(
            Recommendation(
                account=acct.name,
                decision="BET",
                side=side,
                reason=reason,
                stake_usd=round(stake, 2),
                limit_price=round(limit, 3),
                shares=round(shares, 2),
                avg_fill_price=round(avg_fill, 4),
                net_profit_if_win=round(net_profit, 2),
                edge_cents=round(edge_cents, 2),
                fee_usd=round(fee_usd, 2),
            )
        )
    return out


def size_bet(
    account: AccountConfig,
    p_model: float,
    asks_side_a: list[tuple[float, float]],
    asks_side_b: list[tuple[float, float]],
    *,
    fee_coeff: float = KALSHI_FEE_COEFF,
) -> Recommendation:
    """Compute the stake for one account on a fight.

    `asks_side_a` / `asks_side_b`: order book asks for buying side A / side B,
    each as a list of (price, size) sorted ascending by price.

    `p_model`: model's probability that SIDE A wins. The function picks
    whichever side has the better edge after fees. The returned `side` is
    "A" or "B" — the caller maps to fighter names.
    """
    asks_a = sorted(asks_side_a)
    asks_b = sorted(asks_side_b)
    if not asks_a or not asks_b:
        return Recommendation(account.name, "SKIP", None, "empty order book", 0.0, None, 0.0, None, 0.0, 0.0)
    p_market_a = asks_a[0][0]
    p_market_b = asks_b[0][0]

    edge_a = p_model - p_market_a
    edge_b = (1 - p_model) - p_market_b

    if edge_a >= edge_b:
        side_name, asks, p_market, edge = "A", asks_a, p_market_a, edge_a
    else:
        side_name, asks, p_market, edge = "B", asks_b, p_market_b, edge_b

    if edge < EDGE_THRESHOLD:
        return Recommendation(
            account.name,
            "SKIP",
            side_name,
            f"edge {edge * 100:.2f}¢ < threshold {EDGE_THRESHOLD * 100:.0f}¢",
            0.0,
            None,
            0.0,
            None,
            0.0,
            edge * 100,
        )

    p = p_model if side_name == "A" else 1 - p_model
    full_kelly = _full_kelly(p, p_market, fee_coeff)
    if full_kelly <= 0:
        return Recommendation(
            account.name,
            "SKIP",
            side_name,
            "Kelly negative after fees",
            0.0,
            None,
            0.0,
            None,
            0.0,
            edge * 100,
        )

    stake_kelly = account.bankroll * account.kelly_fraction * full_kelly
    stake = stake_kelly
    cap_reasons = []
    if account.bankroll_cap is not None:
        cap = account.bankroll * account.bankroll_cap
        if stake > cap:
            cap_reasons.append(f"bankroll cap ${cap:.2f}")
            stake = cap

    ask_depth = sum(p * s for p, s in asks if p <= asks[0][0] + LIQUIDITY_DEPTH_BAND_CENTS / 100)
    liq_cap = ask_depth * LIQUIDITY_CAP_FRACTION
    if stake > liq_cap:
        cap_reasons.append(f"liquidity cap ${liq_cap:.2f} (depth ${ask_depth:.2f})")
        stake = liq_cap

    if stake < 0.50:
        return Recommendation(
            account.name,
            "SKIP",
            side_name,
            f"stake ${stake:.2f} < $0.50 minimum",
            0.0,
            None,
            0.0,
            None,
            0.0,
            edge * 100,
        )

    # Walk with the full stake to find the price levels; conservative (walks
    # slightly deeper than the at-price spend since the fee isn't notional).
    _shares_notional, avg_fill, limit = _walk_book(asks, stake)
    fee_per = _fee_per_contract(avg_fill, fee_coeff)
    shares = stake / (avg_fill + fee_per)
    fee_usd = shares * fee_per
    net_profit = shares - stake
    reason = "; ".join(cap_reasons) or f"Kelly-limited @ {full_kelly * 100:.2f}%"
    return Recommendation(
        account.name,
        "BET",
        side_name,
        reason,
        stake_usd=round(stake, 2),
        limit_price=round(limit, 3),
        shares=round(shares, 2),
        avg_fill_price=round(avg_fill, 4),
        net_profit_if_win=round(net_profit, 2),
        edge_cents=round(edge * 100, 2),
        fee_usd=round(fee_usd, 2),
    )

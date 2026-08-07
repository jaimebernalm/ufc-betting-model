"""Betting simulation for model evaluation.

What you get per model: how many bets would have been placed at a given edge
threshold, the realized ROI on flat $1 stakes, the model's expected EV,
Sharpe, hit rate, and a bootstrap 95% CI for ROI.

Conventions
-----------
- Odds inputs are American moneyline (the format in the Kaggle dataset).
- Decimal odds include vig — they're what you actually get paid.
- The "no-vig" probability is the market's true-consensus estimate, used for
  the probability-edge diagnostic. The betting decision uses the *with-vig*
  decimal odds because that's the price you actually receive.
- Flat $1 staking: realized PnL per bet is `+(decimal_odds - 1)` on a win,
  `-1` on a loss. Sum across bets = total profit on a unit-stake strategy.
- Fees: two `fee_model` options.
    * "winnings" (default, Kalshi-style): f of your winnings is taken.
      Effective decimal = `1 + (1-f)*(dec - 1)`.
    * "polymarket": taker fee = `shares × f × p × (1-p)` paid up-front in USDC.
      Per $1 of total capital deployed, effective decimal becomes
      `dec / (1 + f × (1 - 1/dec))`. Polymarket Sports = 3% taker, makers free.
  Pass `fee_rate=0.07, fee_model="winnings"` for Kalshi-like scenarios and
  `fee_rate=0.03, fee_model="polymarket"` for Polymarket-like.

PLAN.md §10.1 sets the canonical edge threshold at EV > 0.03 (3%), which is
what we default to here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ufc_pred.backtest.metrics import american_to_implied_prob


def american_to_decimal(odds: pd.Series | np.ndarray) -> np.ndarray:
    """American moneyline → decimal odds.

    -150 → 1.6667 (bet $1, win $0.667 + $1 back if win)
    +200 → 3.0
    """
    o = pd.to_numeric(pd.Series(odds), errors="coerce").to_numpy(dtype=float)
    out = np.full_like(o, np.nan)
    pos = o > 0
    neg = o < 0
    out[pos] = o[pos] / 100.0 + 1.0
    out[neg] = 100.0 / (-o[neg]) + 1.0
    return out


def _effective_decimal(dec: np.ndarray, fee_rate: float, fee_model: str) -> np.ndarray:
    """Adjust decimal odds for fees.

    fee_model="winnings": Kalshi-style — fee_rate of winnings is deducted.
        eff = 1 + (1 - fee_rate) * (dec - 1)

    fee_model="polymarket": taker fee = shares × fee_rate × p × (1-p), where
        p = 1/dec, paid up-front. Per $1 of total capital deployed:
        eff = dec / (1 + fee_rate × (1 - 1/dec))

    fee_model="kalshi": Kalshi's actual published formula —
        fee = fee_rate × p × (1-p) per $1 of contract notional, paid up-front.
        Per $1 of TOTAL deployed (cost + fee): cost = p + fee_rate*p*(1-p),
        payout = $1 if win, so eff = 1 / (p + fee_rate*p*(1-p)) = 1 / [p*(1+fee_rate*(1-p))]
        With fee_rate=0.07 this matches Kalshi's quadratic fee.
        (Kalshi additionally rounds fee UP to the cent per contract — we ignore
        the rounding here; it makes <0.5% difference for typical sizes.)
    """
    if fee_rate == 0.0:
        return dec
    if fee_model == "winnings":
        return 1.0 + (1.0 - fee_rate) * (dec - 1.0)
    if fee_model == "polymarket":
        with np.errstate(divide="ignore", invalid="ignore"):
            return dec / (1.0 + fee_rate * (1.0 - 1.0 / dec))
    if fee_model == "kalshi":
        with np.errstate(divide="ignore", invalid="ignore"):
            p = 1.0 / dec
            return 1.0 / (p * (1.0 + fee_rate * (1.0 - p)))
    raise ValueError(f"unknown fee_model: {fee_model!r}")


def _bootstrap_ci_mean(
    x: np.ndarray, n_iter: int = 5000, ci: float = 0.95, seed: int = 0
) -> tuple[float, float]:
    """Bootstrap CI for the mean of x."""
    if len(x) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_iter, len(x)))
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(lo), float(hi)


@dataclass
class BetEvalResult:
    n_eligible: int  # fights with both odds available
    n_bets: int  # fights where chosen-side EV > threshold
    n_bets_red: int
    n_bets_blue: int
    roi_pct: float  # mean realized PnL per $1 bet × 100
    mean_ev_pct: float  # mean model-expected return × 100
    sharpe: float  # mean / std of per-bet realized
    hit_rate: float  # fraction of bets that won
    total_pnl: float  # sum of realized PnL on flat $1 stakes
    ci95_roi_pct: tuple[float, float]  # bootstrap 95% CI for ROI %
    edge_threshold: float
    fee_rate: float
    bets_df: pd.DataFrame  # per-bet detail for inspection

    def summary_row(self) -> dict:
        return {
            "edge_threshold": self.edge_threshold,
            "n_bets": self.n_bets,
            "n_eligible": self.n_eligible,
            "roi_pct": self.roi_pct,
            "ci95_low": self.ci95_roi_pct[0],
            "ci95_high": self.ci95_roi_pct[1],
            "mean_ev_pct": self.mean_ev_pct,
            "sharpe": self.sharpe,
            "hit_rate": self.hit_rate,
            "total_pnl": self.total_pnl,
            "n_red": self.n_bets_red,
            "n_blue": self.n_bets_blue,
        }


def evaluate_bets(
    model_prob_red: np.ndarray,
    y_red: np.ndarray,
    R_odds: pd.Series | np.ndarray,
    B_odds: pd.Series | np.ndarray,
    edge_threshold: float = 0.03,
    fee_rate: float = 0.0,
    use_no_vig: bool = False,
    fee_model: str = "winnings",
    extra_cols: dict[str, np.ndarray] | None = None,
) -> BetEvalResult:
    """Simulate flat $1 betting against the market on a set of fights.

    Parameters
    ----------
    model_prob_red : array, len N
        Predicted P(Red wins) for each fight.
    y_red : array, len N
        1 if Red won, 0 if Blue won.
    R_odds, B_odds : American moneyline for each corner (sportsbook prices,
        with vig).
    edge_threshold : float
        Minimum model-EV-per-dollar required to place a bet. 0.03 = 3%.
    fee_rate : float
        Fee taken from winnings, e.g. 0.07 = Kalshi-style 7% off winnings.
        Default 0 (sportsbook — vig is already in the price).
    use_no_vig : bool
        If True, replace each side's decimal odds with the *fair* (no-vig)
        decimal odds derived from the market's no-vig consensus probability.
        Use this to simulate prediction markets like Kalshi/Polymarket where
        the prices are roughly no-vig and cost is in the fee.
        Default False (use the with-vig sportsbook prices as-is).
    extra_cols : optional dict mapping column name → array of length N. Each
        gets attached to the per-bet detail dataframe for inspection.

    Returns
    -------
    BetEvalResult.
    """
    model_prob_red = np.asarray(model_prob_red, dtype=float)
    y_red = np.asarray(y_red, dtype=int)

    dec_R = american_to_decimal(R_odds)
    dec_B = american_to_decimal(B_odds)
    valid = ~(np.isnan(dec_R) | np.isnan(dec_B))

    if use_no_vig:
        # Replace decimals with their no-vig equivalents: d_nv = 1 / p_nv.
        p_r_imp = american_to_implied_prob(R_odds)
        p_b_imp = american_to_implied_prob(B_odds)
        total = p_r_imp + p_b_imp
        p_r_nv = p_r_imp / total
        p_b_nv = p_b_imp / total
        dec_R = 1.0 / p_r_nv
        dec_B = 1.0 / p_b_nv

    # Effective decimal after fees.
    eff_R = _effective_decimal(dec_R, fee_rate, fee_model)
    eff_B = _effective_decimal(dec_B, fee_rate, fee_model)

    p_R = model_prob_red
    p_B = 1.0 - model_prob_red

    ev_R = p_R * eff_R - 1.0
    ev_B = p_B * eff_B - 1.0

    bet_red = ev_R >= ev_B  # ties → bet red (arbitrary; rare)
    chosen_ev = np.where(bet_red, ev_R, ev_B)
    chosen_dec = np.where(bet_red, eff_R, eff_B)

    bets = valid & (chosen_ev > edge_threshold)

    # Did the chosen side win?
    won = np.where(bet_red, y_red == 1, y_red == 0)
    # Flat $1 stake: realized = (dec_eff - 1) on win, -1 on loss.
    realized = np.where(won, chosen_dec - 1.0, -1.0)

    bet_realized = realized[bets]
    bet_ev = chosen_ev[bets]
    bet_won = won[bets]

    # Per-bet detail.
    base_cols = {
        "side": np.where(bet_red[bets], "R", "B"),
        "ev": chosen_ev[bets],
        "decimal_odds": chosen_dec[bets],
        "model_prob_chosen": np.where(bet_red[bets], p_R[bets], p_B[bets]),
        "won": bet_won.astype(int),
        "realized_pnl": bet_realized,
    }
    if extra_cols:
        for k, v in extra_cols.items():
            v = np.asarray(v)
            base_cols[k] = v[bets]
    bets_df = pd.DataFrame(base_cols)

    if bet_realized.size == 0:
        roi = 0.0
        mean_ev = 0.0
        sharpe = 0.0
        hit = 0.0
        ci = (float("nan"), float("nan"))
    else:
        roi = float(bet_realized.mean())
        mean_ev = float(bet_ev.mean())
        std = float(bet_realized.std(ddof=1)) if len(bet_realized) > 1 else 0.0
        sharpe = roi / std if std > 0 else 0.0
        hit = float(bet_won.mean())
        lo, hi = _bootstrap_ci_mean(bet_realized)
        ci = (lo * 100, hi * 100)

    return BetEvalResult(
        n_eligible=int(valid.sum()),
        n_bets=int(bets.sum()),
        n_bets_red=int((bets & bet_red).sum()),
        n_bets_blue=int((bets & ~bet_red).sum()),
        roi_pct=roi * 100,
        mean_ev_pct=mean_ev * 100,
        sharpe=sharpe,
        hit_rate=hit,
        total_pnl=float(bet_realized.sum()),
        ci95_roi_pct=ci,
        edge_threshold=edge_threshold,
        fee_rate=fee_rate,
        bets_df=bets_df,
    )


def evaluate_bets_sweep(
    model_prob_red: np.ndarray,
    y_red: np.ndarray,
    R_odds: pd.Series | np.ndarray,
    B_odds: pd.Series | np.ndarray,
    thresholds: list[float] = (0.0, 0.02, 0.03, 0.05, 0.075, 0.10),
    fee_rate: float = 0.0,
    use_no_vig: bool = False,
    fee_model: str = "winnings",
) -> pd.DataFrame:
    """Run `evaluate_bets` at multiple edge thresholds; return a comparison frame.

    A trustworthy bettor's curve: as threshold rises, n_bets falls, but ROI
    and Sharpe should rise (high-edge signals are real). If ROI is flat or
    falls with stricter thresholds, the model is overconfident on disagreements.
    """
    rows = []
    for t in thresholds:
        r = evaluate_bets(
            model_prob_red,
            y_red,
            R_odds,
            B_odds,
            edge_threshold=t,
            fee_rate=fee_rate,
            use_no_vig=use_no_vig,
            fee_model=fee_model,
        )
        rows.append(r.summary_row())
    return pd.DataFrame(rows)


def evaluate_bets_kelly(
    model_prob_red: np.ndarray,
    y_red: np.ndarray,
    R_odds: pd.Series | np.ndarray,
    B_odds: pd.Series | np.ndarray,
    edge_threshold: float = 0.03,
    fee_rate: float = 0.0,
    use_no_vig: bool = False,
    kelly_fraction: float = 0.25,
    starting_bankroll: float = 1.0,
    max_bet_fraction: float = 0.02,
    fee_model: str = "winnings",
) -> dict:
    """Fractional-Kelly bankroll simulation.

    Per bet: stake = bankroll × min(kelly_fraction × full_kelly, max_bet_fraction).
    Per PLAN.md §10.2: never more than 2% of bankroll on a single fight.

    `use_no_vig` and `fee_rate` follow the same semantics as evaluate_bets —
    use them to simulate sportsbook (vig in price) vs Kalshi/Polymarket
    (fair prices + fee on winnings).

    Returns dict with bankroll trajectory, final balance, max drawdown.
    """
    model_prob_red = np.asarray(model_prob_red, dtype=float)
    y_red = np.asarray(y_red, dtype=int)

    dec_R = american_to_decimal(R_odds)
    dec_B = american_to_decimal(B_odds)
    valid = ~(np.isnan(dec_R) | np.isnan(dec_B))

    if use_no_vig:
        p_r_imp = american_to_implied_prob(R_odds)
        p_b_imp = american_to_implied_prob(B_odds)
        total = p_r_imp + p_b_imp
        dec_R = 1.0 / (p_r_imp / total)
        dec_B = 1.0 / (p_b_imp / total)

    eff_R = _effective_decimal(dec_R, fee_rate, fee_model)
    eff_B = _effective_decimal(dec_B, fee_rate, fee_model)

    p_R = model_prob_red
    p_B = 1.0 - p_R
    ev_R = p_R * eff_R - 1.0
    ev_B = p_B * eff_B - 1.0
    bet_red = ev_R >= ev_B
    chosen_ev = np.where(bet_red, ev_R, ev_B)
    chosen_dec = np.where(bet_red, eff_R, eff_B)
    chosen_p = np.where(bet_red, p_R, p_B)

    bets_mask = valid & (chosen_ev > edge_threshold)
    won_full = np.where(bet_red, y_red == 1, y_red == 0)

    bankroll = starting_bankroll
    trajectory = [bankroll]
    peak = bankroll
    max_dd = 0.0

    for i in range(len(p_R)):
        if not bets_mask[i]:
            continue
        b = chosen_dec[i] - 1.0  # net decimal odds
        p = chosen_p[i]
        q = 1.0 - p
        full_kelly = (b * p - q) / b
        if full_kelly <= 0:  # shouldn't happen given ev > threshold
            continue
        stake_frac = min(kelly_fraction * full_kelly, max_bet_fraction)
        stake = bankroll * stake_frac
        if won_full[i]:
            bankroll += stake * (chosen_dec[i] - 1.0)
        else:
            bankroll -= stake
        peak = max(peak, bankroll)
        max_dd = max(max_dd, (peak - bankroll) / peak)
        trajectory.append(bankroll)

    return {
        "starting_bankroll": float(starting_bankroll),
        "final_bankroll": float(bankroll),
        "total_return_pct": float((bankroll / starting_bankroll - 1.0) * 100),
        "max_drawdown_pct": float(max_dd * 100),
        "n_bets": int(bets_mask.sum()),
        "trajectory": [float(x) for x in trajectory],
        "edge_threshold": edge_threshold,
        "kelly_fraction": kelly_fraction,
        "max_bet_fraction": max_bet_fraction,
        "fee_rate": fee_rate,
        "fee_model": fee_model,
        "use_no_vig": use_no_vig,
    }


def probability_edge(
    model_prob_red: np.ndarray,
    R_odds: pd.Series | np.ndarray,
    B_odds: pd.Series | np.ndarray,
) -> np.ndarray:
    """Signed probability edge: model − market_no_vig. Positive = Red favored more by us."""
    p_r_imp = american_to_implied_prob(R_odds)
    p_b_imp = american_to_implied_prob(B_odds)
    market_p_r = p_r_imp / (p_r_imp + p_b_imp)
    return np.asarray(model_prob_red, dtype=float) - market_p_r

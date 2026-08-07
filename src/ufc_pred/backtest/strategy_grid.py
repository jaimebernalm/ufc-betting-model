"""Reusable training + simulation primitives for the seed-robust strategy
evaluation experiment.

Consolidates the duplicated patterns from these scripts:
  - scripts/seed_variance_test.py
  - scripts/frozen_cutoff_sweep.py
  - scripts/account_c_strategy_comparison.py
  - scripts/age_sweep_and_significance.py
  - scripts/extended_window_backtest.py

All numerics match the seed=0 reference results in
`artifacts/metrics/seed_variance_test.json` and
`artifacts/metrics/account_c_strategy_comparison.json`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from scipy.stats import binomtest

from ufc_pred.features.static_v1 import prepare
from ufc_pred.models.wrappers import CorruptedSkillModel
from ufc_pred.utils.time_splits import recency_weights

# Deployment/backtest constants (must match existing scripts).
FEE = 0.02
EDGE_THR = 0.03
START = 300.0


@dataclass
class ModelBundle:
    """Trained model + the metadata needed to predict on new fights."""

    model: object  # CatBoostClassifier or CorruptedSkillModel
    columns: list[str]
    cat_features: list[str]


def _fit_catboost(
    X: pd.DataFrame, y, d, cat: list[str], cutoff: pd.Timestamp, seed: int
) -> CatBoostClassifier:
    w = recency_weights(d, reference_date=cutoff - pd.Timedelta(days=1))
    pool = Pool(X, y, cat_features=cat, weight=w)
    cb = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        loss_function="Logloss",
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
    )
    cb.fit(pool, verbose=False)
    return cb


def train_real(fights: pd.DataFrame, cutoff: pd.Timestamp, seed: int = 0) -> ModelBundle:
    """Train the real (non-corrupted) CatBoost on `fights[date < cutoff]`."""
    df = fights[fights["date"] < cutoff].copy()
    X, y, d, cat = prepare(df, augment_symmetry=True, one_hot=False)
    cb = _fit_catboost(X, y, d, cat, cutoff, seed)
    return ModelBundle(model=cb, columns=list(X.columns), cat_features=cat)


def train_corrupted(fights: pd.DataFrame, cutoff: pd.Timestamp, seed: int = 0) -> ModelBundle:
    """Train the corrupted (skill-feature-NaN at inference) CatBoost on
    `fights[date < cutoff]`.

    Historically this function sign-flipped `skill_diff_mean` on the swapped
    half itself because `_swap_red_blue` didn't. Since the 2026-06-11 fix,
    `prepare(augment_symmetry=True)` performs that flip (for all signed dif
    columns) — flipping here again would undo it, so the manual flip is gone."""
    df = fights[fights["date"] < cutoff].copy()
    X, y, d, cat = prepare(df, augment_symmetry=True, one_hot=False)
    cb = _fit_catboost(X, y, d, cat, cutoff, seed)
    wrapper = CorruptedSkillModel(cb, ["skill_diff_mean", "skill_diff_std"])
    return ModelBundle(model=wrapper, columns=list(X.columns), cat_features=cat)


def predict(bundle: ModelBundle, df: pd.DataFrame) -> np.ndarray:
    """Probability of Red winning, vectorised, matching every script's helper."""
    X, _, _, _ = prepare(df, augment_symmetry=False, one_hot=False)
    X = X.reindex(columns=bundle.columns, fill_value=None)
    for c in bundle.cat_features:
        X[c] = X[c].fillna("__missing__").astype(str)
    return bundle.model.predict_proba(X)[:, 1]


def simulate_kelly(
    p_red: np.ndarray,
    eval_df: pd.DataFrame,
    kelly_frac: float,
    cap: float,
    start: float = START,
    fee: float = FEE,
    edge_thr: float = EDGE_THR,
) -> tuple[float, list[dict]]:
    """Fractional-Kelly bankroll simulation with stake-fraction cap.

    Identical math to scripts/account_c_strategy_comparison.py:96-115 and
    seed_variance_test.py simulate(): stake = bank * min(kelly_frac * fk, cap).

    Returns (final_bankroll, per_bet_records). Per-bet record schema:
      eval_month, fight_id, side, won, dec_odds, fk, pnl_flat
    """
    pR = eval_df["polymarket_p_red"].to_numpy()
    pB = eval_df["polymarket_p_blue"].to_numpy()
    eR = 1 + (1 - fee) * (1 / pR - 1)
    eB = 1 + (1 - fee) * (1 / pB - 1)
    Y = (eval_df["Winner"].to_numpy() == "Red").astype(int)

    ev_R = p_red * eR - 1
    ev_B = (1 - p_red) * eB - 1
    br = ev_R >= ev_B
    chosen_dec = np.where(br, eR, eB)
    chosen_p = np.where(br, p_red, 1 - p_red)
    chosen_ev = np.where(br, ev_R, ev_B)
    won = np.where(br, Y == 1, Y == 0).astype(int)
    mask = chosen_ev > edge_thr

    fight_ids = [f"{r['date'].date()}|{r['R_fighter']}|{r['B_fighter']}" for _, r in eval_df.iterrows()]

    bank = start
    bets: list[dict] = []
    for i in range(len(eval_df)):
        if not mask[i]:
            continue
        b = chosen_dec[i] - 1
        p = chosen_p[i]
        q = 1 - p
        fk = (b * p - q) / b
        if fk <= 0:
            continue
        stake = bank * min(kelly_frac * fk, cap)
        if won[i]:
            bank += stake * (chosen_dec[i] - 1)
        else:
            bank -= stake
        bets.append(
            {
                "fight_id": fight_ids[i],
                "side": "R" if br[i] else "B",
                "won": int(won[i]),
                "dec_odds": float(chosen_dec[i]),
                "fk": float(fk),
                "pnl_flat": float(chosen_dec[i] - 1) if won[i] else -1.0,
                "bank_after": float(bank),
            }
        )
    return float(bank), bets


# ---------------------------------------------------------------------------
# Strategy schedule
# ---------------------------------------------------------------------------

STRATEGIES = [
    "frozen_2023_01",
    "frozen_2024_01",
    "frozen_2025_01",
    "frozen_2025_07",
    "rolling_1mo",
    "rolling_3mo",
    "rolling_6mo",
    "rolling_12mo",
]

FROZEN_CUTOFFS = {
    "frozen_2023_01": pd.Timestamp("2023-01-01"),
    "frozen_2024_01": pd.Timestamp("2024-01-01"),
    "frozen_2025_01": pd.Timestamp("2025-01-01"),
    "frozen_2025_07": pd.Timestamp("2025-07-01"),
}


DEFAULT_ANCHOR = pd.Timestamp("2025-07-01")


def cutoff_for(
    strategy: str, eval_month: pd.Timestamp, anchor: pd.Timestamp = DEFAULT_ANCHOR
) -> pd.Timestamp:
    """Cutoff date to use when predicting `eval_month` under `strategy`.

    For rolling strategies, retrain blocks are anchored to `anchor` and
    advance every K months (K=1, 3, 6, 12)."""
    if strategy in FROZEN_CUTOFFS:
        return FROZEN_CUTOFFS[strategy]
    if strategy.startswith("rolling_"):
        k = int(strategy.split("_")[1].replace("mo", ""))
        idx = (eval_month.to_period("M") - anchor.to_period("M")).n
        block = (idx // k) * k
        return (anchor + pd.DateOffset(months=block)).normalize()
    raise ValueError(f"Unknown strategy: {strategy}")


def unique_cutoffs(strategies: list[str], eval_months: list[pd.Timestamp]) -> list[pd.Timestamp]:
    s = set()
    for st in strategies:
        for m in eval_months:
            s.add(cutoff_for(st, m))
    return sorted(s)


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar test under null p=0.5 on the b+c discordant pairs."""
    if b + c == 0:
        return 1.0
    return binomtest(min(b, c), n=b + c, p=0.5, alternative="two-sided").pvalue


def boot_ci(
    diff: np.ndarray, n: int = 5000, seed: int = 0, ci: tuple[float, float] = (2.5, 97.5)
) -> tuple[float, float, float]:
    """Bootstrap mean and percentile CI of `diff`."""
    diff = np.asarray(diff)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n, len(diff)))
    means = diff[idx].mean(axis=1)
    return (
        float(diff.mean()),
        float(np.percentile(means, ci[0])),
        float(np.percentile(means, ci[1])),
    )


# ---------------------------------------------------------------------------
# Account specs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AccountSpec:
    name: str  # "A", "B", "C"
    model_kind: str  # "real" or "corrupt"
    kelly_frac: float
    cap: float


ACCOUNT_SPECS = (
    AccountSpec("A", "real", 0.10, 0.10),
    AccountSpec("B", "real", 0.25, 1.00),
    AccountSpec("C", "corrupt", 0.25, 1.00),
)

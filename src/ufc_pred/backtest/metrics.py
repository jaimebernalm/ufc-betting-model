"""Evaluation metrics. Every model gets scored the same way."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Weighted average gap between predicted probability and empirical rate, per bin."""
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(y_prob, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        ece += (mask.sum() / n) * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def evaluate(y_true: np.ndarray, y_prob: np.ndarray, label: str = "") -> dict:
    """Score a probability vector against binary labels. Returns a flat dict."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)
    return {
        "label": label,
        "n": int(len(y_true)),
        "log_loss": float(log_loss(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": expected_calibration_error(y_true, y_prob),
        "accuracy_argmax": float(((y_prob > 0.5) == y_true).mean()),
    }


def american_to_implied_prob(odds: pd.Series | np.ndarray) -> np.ndarray:
    """American moneyline → implied probability (still contains the vig)."""
    o = pd.to_numeric(pd.Series(odds), errors="coerce").to_numpy()
    out = np.full_like(o, np.nan, dtype=float)
    pos = o > 0
    neg = o < 0
    out[pos] = 100 / (o[pos] + 100)
    out[neg] = -o[neg] / (-o[neg] + 100)
    return out


def market_no_vig_prob_red(df: pd.DataFrame) -> np.ndarray:
    """No-vig implied probability that Red wins, derived from R_odds and B_odds."""
    p_r = american_to_implied_prob(df["R_odds"])
    p_b = american_to_implied_prob(df["B_odds"])
    return p_r / (p_r + p_b)

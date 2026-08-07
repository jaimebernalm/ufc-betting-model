"""Probability calibrators for the binary classifier.

Each class implements `.fit(p_raw, y)` and `.transform(p_raw) -> p_calibrated`.
All take and return P(class=1) probabilities in [0, 1].

Three methods, ordered by capacity (and overfit risk):

  1. TemperatureScaling — 1 parameter. logit-space rescale: p = sigmoid(logit(p_raw) / T).
     Can only shrink toward 0.5 (T>1) or sharpen (T<1). Never overfits the data.
  2. PlattScaling — 2 parameters. p = sigmoid(a · logit(p_raw) + b). Adds a shift.
  3. IsotonicRegression — non-parametric monotonic step function. Most flexible,
     most prone to overfitting on small calibration sets (PLAN.md notes v1.2 isotonic
     overfit with ~5k OOF points).

Reference: Guo et al. 2017 "On Calibration of Modern Neural Networks" (temperature
scaling); Platt 1999 (Platt scaling); Zadrozny & Elkan 2002 (isotonic).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

_EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class TemperatureScaling:
    """1-parameter calibrator. p = sigmoid(logit(p_raw) / T).

    T > 1  → squeezes probabilities toward 0.5 (less confident).
    T < 1  → sharpens (more confident; not what we expect to need).
    T = 1  → identity.

    Fit by minimizing NLL on the calibration set with bounded scalar optimization.
    """

    def __init__(self) -> None:
        self.T: float = 1.0

    def fit(self, p_raw: np.ndarray, y: np.ndarray) -> TemperatureScaling:
        p_raw = np.asarray(p_raw, dtype=float)
        y = np.asarray(y, dtype=int)
        logits = _logit(p_raw)

        def nll(T: float) -> float:
            if T <= 0:
                return float("inf")
            p = _sigmoid(logits / T)
            p = np.clip(p, _EPS, 1 - _EPS)
            return -float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

        result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
        self.T = float(result.x)
        return self

    def transform(self, p_raw: np.ndarray) -> np.ndarray:
        return _sigmoid(_logit(np.asarray(p_raw, dtype=float)) / self.T)


class PlattScaling:
    """2-parameter calibrator. p = sigmoid(a · logit(p_raw) + b).

    Fit by logistic regression on (logit(p_raw), y).
    """

    def __init__(self) -> None:
        self.a: float = 1.0
        self.b: float = 0.0

    def fit(self, p_raw: np.ndarray, y: np.ndarray) -> PlattScaling:
        p_raw = np.asarray(p_raw, dtype=float)
        y = np.asarray(y, dtype=int)
        z = _logit(p_raw).reshape(-1, 1)
        lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        lr.fit(z, y)
        self.a = float(lr.coef_[0, 0])
        self.b = float(lr.intercept_[0])
        return self

    def transform(self, p_raw: np.ndarray) -> np.ndarray:
        z = _logit(np.asarray(p_raw, dtype=float))
        return _sigmoid(self.a * z + self.b)


class IsotonicCalibration:
    """Non-parametric monotonic calibrator (sklearn IsotonicRegression).

    Most flexible but most prone to overfitting on small cal sets — that's
    what killed v1.2 in this project.
    """

    def __init__(self) -> None:
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=_EPS, y_max=1 - _EPS)

    def fit(self, p_raw: np.ndarray, y: np.ndarray) -> IsotonicCalibration:
        self.iso.fit(np.asarray(p_raw, dtype=float), np.asarray(y, dtype=int))
        return self

    def transform(self, p_raw: np.ndarray) -> np.ndarray:
        return self.iso.transform(np.asarray(p_raw, dtype=float))


def build(name: str):
    """Factory."""
    return {
        "temperature": TemperatureScaling,
        "platt": PlattScaling,
        "isotonic": IsotonicCalibration,
    }[name]()

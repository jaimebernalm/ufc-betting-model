"""Run the 10-seed real + corrupted ensembles on an upcoming row."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ufc_pred.backtest.strategy_grid import ModelBundle, predict
from ufc_pred.paths import MODELS as MODELS_DIR

DEFAULT_CUTOFF_TAG = "2025_11_30"
DEFAULT_N_SEEDS = 10


@dataclass(frozen=True)
class EnsemblePrediction:
    real_mean: float
    real_per_seed: np.ndarray
    corrupted_mean: float
    corrupted_per_seed: np.ndarray


def _load_bundle(path: Path) -> ModelBundle:
    d = joblib.load(path)
    return ModelBundle(model=d["model"], columns=d["columns"], cat_features=d["cat_features"])


def predict_ensemble(
    upcoming_row: pd.DataFrame,
    *,
    cutoff_tag: str = DEFAULT_CUTOFF_TAG,
    n_seeds: int = DEFAULT_N_SEEDS,
) -> EnsemblePrediction:
    """Return averaged p(Red wins) across both ensembles."""
    real, corr = [], []
    for s in range(n_seeds):
        br = _load_bundle(MODELS_DIR / f"v3_real_{cutoff_tag}_seed{s}.joblib")
        bc = _load_bundle(MODELS_DIR / f"v3_corrupted_{cutoff_tag}_seed{s}.joblib")
        real.append(float(predict(br, upcoming_row.copy())[0]))
        corr.append(float(predict(bc, upcoming_row.copy())[0]))
    real = np.array(real)
    corr = np.array(corr)
    return EnsemblePrediction(
        real_mean=float(real.mean()),
        real_per_seed=real,
        corrupted_mean=float(corr.mean()),
        corrupted_per_seed=corr,
    )


def predict_ensemble_symmetric(
    row_fwd: pd.DataFrame,
    row_rev: pd.DataFrame,
    *,
    cutoff_tag: str = DEFAULT_CUTOFF_TAG,
    n_seeds: int = DEFAULT_N_SEEDS,
) -> EnsemblePrediction:
    """Orientation-invariant prediction: average each seed over both corner
    orderings, p = (p_fwd + (1 - p_rev)) / 2.

    `row_fwd` has fighter_a in the Red corner; `row_rev` is the same fight
    with the corners swapped. The deployed models are not exactly symmetric
    (training augmentation did not sign-flip dif columns), so a single
    orientation's output depends on arbitrary market ticker ordering;
    averaging removes that dependence. Returns p(fighter_a wins).
    """
    both = pd.concat([row_fwd, row_rev], ignore_index=True)
    real, corr = [], []
    for s in range(n_seeds):
        br = _load_bundle(MODELS_DIR / f"v3_real_{cutoff_tag}_seed{s}.joblib")
        bc = _load_bundle(MODELS_DIR / f"v3_corrupted_{cutoff_tag}_seed{s}.joblib")
        pr = predict(br, both.copy())
        pc = predict(bc, both.copy())
        real.append(0.5 * (float(pr[0]) + 1.0 - float(pr[1])))
        corr.append(0.5 * (float(pc[0]) + 1.0 - float(pc[1])))
    real = np.array(real)
    corr = np.array(corr)
    return EnsemblePrediction(
        real_mean=float(real.mean()),
        real_per_seed=real,
        corrupted_mean=float(corr.mean()),
        corrupted_per_seed=corr,
    )

"""Model wrappers — small adapters around a base estimator that change inference
behavior without retraining."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


class CorruptedSkillModel:
    """Wraps a base estimator and forces specified columns to NaN at inference.

    Background: an earlier diagnostic run for v3_full2000 accidentally fed the
    model NaN skill features (because of a feature-rename mismatch). The model
    happened to produce surprisingly good Kelly numbers on val 2023 under that
    condition. To allow honest test-set evaluation of "v3_full2000 predicting
    without its skill signal," we wrap the trained model here and apply the
    NaN substitution deterministically on every prediction.

    This is NOT a recommended deployment model. It's preserved so that future
    test-set evaluation can reproduce the corrupted behavior we observed.
    """

    def __init__(self, base_model, nan_columns: Sequence[str]):
        self.base_model = base_model
        self.nan_columns = list(nan_columns)

    def _corrupt(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for c in self.nan_columns:
            if c in X.columns:
                X[c] = np.nan
        return X

    def predict_proba(self, X) -> np.ndarray:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return self.base_model.predict_proba(self._corrupt(X))

    def predict(self, X) -> np.ndarray:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return self.base_model.predict(self._corrupt(X))

    def __repr__(self) -> str:
        return f"CorruptedSkillModel(base={type(self.base_model).__name__}, nan_columns={self.nan_columns})"

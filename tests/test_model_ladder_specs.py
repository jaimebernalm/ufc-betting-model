"""Guards on the model ladder's comparability contract.

The ladder is only meaningful if versions differ in exactly the dimension they
claim to. These tests assert that structurally, so a future edit that quietly
retunes one rung fails CI instead of silently invalidating the comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ufc_pred.features.joins import flip_signed_columns
from ufc_pred.models import (
    baseline_v1_1,
    baseline_v2,
    baseline_v3,
    baseline_v3_1,
    baseline_v3_2,
    baseline_v3_3,
    baseline_v7,
    baseline_v7_1,
)
from ufc_pred.models._spec import BASE_CATBOOST_PARAMS

LADDER = [
    baseline_v1_1,
    baseline_v2,
    baseline_v3,
    baseline_v3_1,
    baseline_v3_2,
    baseline_v3_3,
    baseline_v7,
    baseline_v7_1,
]

TUNABLE = ("iterations", "learning_rate", "depth", "l2_leaf_reg", "loss_function")


@pytest.mark.parametrize("module", LADDER, ids=lambda m: m.VERSION)
def test_no_hyperparameter_drift_between_rungs(module):
    """Every rung inherits the same CatBoost hyperparameters.

    Versions may differ in features, seeds, and early stopping — never in the
    tuning knobs. A rung that retuned them would not be comparable to the rest.
    """
    params = module.SPEC.catboost_params()
    for key in TUNABLE:
        assert params[key] == BASE_CATBOOST_PARAMS[key], (
            f"{module.VERSION} changed {key}: {params[key]} != {BASE_CATBOOST_PARAMS[key]}"
        )


@pytest.mark.parametrize("module", LADDER, ids=lambda m: m.VERSION)
def test_spec_version_matches_module_constant(module):
    assert module.SPEC.version == module.VERSION


def test_early_stopping_adds_rounds_only_when_enabled():
    assert "early_stopping_rounds" in baseline_v3.SPEC.catboost_params()
    assert "early_stopping_rounds" not in baseline_v7.SPEC.catboost_params()


def test_ensemble_members_differ_only_by_seed():
    spec = baseline_v7_1.SPEC
    a, b = spec.catboost_params(seed=0), spec.catboost_params(seed=7)
    assert a.pop("random_seed") == 0
    assert b.pop("random_seed") == 7
    assert a == b


class TestSymmetryFlip:
    """`flip_signed_columns` replaces the per-model `_augmented_skill_columns`.

    Signed differences must negate on the swapped half; magnitudes must not.
    """

    def _augmented(self) -> pd.DataFrame:
        # 3 original rows followed by their 3 swapped counterparts.
        return pd.DataFrame(
            {
                "skill_diff_mean": [1.0, -2.0, 0.5, 1.0, -2.0, 0.5],
                "skill_diff_std": [0.3, 0.4, 0.5, 0.3, 0.4, 0.5],
            }
        )

    def test_negates_signed_column_on_swapped_half_only(self):
        out = flip_signed_columns(self._augmented(), ("skill_diff_mean",))
        np.testing.assert_allclose(out["skill_diff_mean"].to_numpy()[:3], [1.0, -2.0, 0.5])
        np.testing.assert_allclose(out["skill_diff_mean"].to_numpy()[3:], [-1.0, 2.0, -0.5])

    def test_leaves_magnitude_column_untouched(self):
        out = flip_signed_columns(self._augmented(), ("skill_diff_mean",))
        np.testing.assert_allclose(out["skill_diff_std"], self._augmented()["skill_diff_std"])

    def test_odd_length_means_unaugmented_and_is_passed_through(self):
        df = pd.DataFrame({"skill_diff_mean": [1.0, 2.0, 3.0]})
        pd.testing.assert_frame_equal(flip_signed_columns(df, ("skill_diff_mean",)), df)

    def test_empty_flip_list_is_a_no_op(self):
        df = self._augmented()
        pd.testing.assert_frame_equal(flip_signed_columns(df, ()), df)

    def test_missing_column_is_ignored(self):
        df = self._augmented()
        out = flip_signed_columns(df, ("not_a_column",))
        pd.testing.assert_frame_equal(out, df)


@pytest.mark.parametrize(
    "module,expected",
    [
        (baseline_v1_1, ()),
        (baseline_v2, ()),
        (baseline_v3, ("skill_diff_mean",)),
        (baseline_v3_1, ("skill_diff_mean",)),
        (baseline_v3_2, ("skill_diff_mean",)),
        (baseline_v3_3, ("skill_diff_mean_v3", "skill_diff_mean_v3_1")),
        (baseline_v7, ("skill_diff_mean_v3", "skill_diff_mean_v3_1")),
        (baseline_v7_1, ("skill_diff_mean",)),
    ],
    ids=lambda x: getattr(x, "VERSION", str(x)),
)
def test_flip_columns_match_the_joined_skill_schema(module, expected):
    """Each recipe flips exactly the signed columns its join produces.

    Getting this wrong is silent: training would see a fighter's skill edge with
    the wrong sign on half the augmented rows.
    """
    assert module.SPEC.recipe.flip_columns == expected


def test_v7_pair_shares_architecture_with_its_single_seed_rung():
    """v7 is v3.3 ensembled; v7.1 is v3 ensembled. Same recipes, more seeds."""
    assert baseline_v7.SPEC.recipe.flip_columns == baseline_v3_3.SPEC.recipe.flip_columns
    assert baseline_v7_1.SPEC.recipe.flip_columns == baseline_v3.SPEC.recipe.flip_columns
    assert baseline_v7.SPEC.ensemble_size == baseline_v7_1.SPEC.ensemble_size == 10
    assert baseline_v3_3.SPEC.ensemble_size == baseline_v3.SPEC.ensemble_size == 1

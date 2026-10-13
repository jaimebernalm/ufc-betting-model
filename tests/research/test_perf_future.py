"""Availability and train/serve parity checks, without scores or outcomes."""

import json
from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from ufc_pred.features.static_v1 import _swap_red_blue
from ufc_pred.research.perf_adjusted import FEATURE_COLUMNS as ADJUSTED_COLUMNS
from ufc_pred.research.perf_adjusted import build_adjusted_features
from ufc_pred.research.perf_features import (
    COUNT_COLUMNS,
    FEATURE_COLUMNS,
    OBSERVATION_COLUMNS,
    build_smoothed_features,
)
from ufc_pred.research.perf_future import build_future_performance


def observations():
    rows = []
    for day, bout, red, blue, a, b in (
        ("2015-11-01", "past", "A", "B", (20, 40, 1, 2), (10, 30, 0, 3)),
        ("2015-12-12", "past2", "C", "B", (30, 60, 1, 2), (15, 30, 2, 3)),
        ("2016-01-14", "current_month", "A", "C", (60, 90, 2, 2), (20, 40, 1, 3)),
        ("2016-01-15", "same_day", "A", "B", (90, 100, 3, 4), (40, 80, 1, 4)),
        ("2016-02-01", "future", "A", "B", (200, 200, 10, 10), (80, 90, 2, 3)),
    ):
        common = dict(bout_id=bout, event_id=bout, date=day, division="Lightweight", minutes=10.0)
        for fighter, opponent, own, other in ((red, blue, a, b), (blue, red, b, a)):
            rows.append(
                dict(
                    **common,
                    fighter=fighter,
                    opponent=opponent,
                    **dict(zip(COUNT_COLUMNS, own + other, strict=True)),
                )
            )
    return pd.DataFrame(rows, columns=OBSERVATION_COLUMNS)


def targets():
    return pd.DataFrame(
        dict(
            date=["2016-01-15", "2016-02-20", "2016-04-01"],
            R_fighter=["A", "A", "C"],
            B_fighter=["B", "C", "New"],
            weight_class=["Lightweight"] * 3,
        ),
        index=[8, 8, 2],
    )


def historical_features(obs, target, include_adjusted):
    output = build_smoothed_features(obs, target)
    if include_adjusted:
        adjusted = build_adjusted_features(obs, target)
        for column in ADJUSTED_COLUMNS:
            output[column] = adjusted[column].to_numpy()
    return output


@pytest.mark.parametrize("include_adjusted", [False, True])
def test_same_day_as_of_matches_frozen_historical_builders_exactly(include_adjusted):
    obs, target = observations(), targets().iloc[:1]
    expected = historical_features(obs, target, include_adjusted)
    actual, diagnostics = build_future_performance(
        obs, target, "2016-01-15", include_adjusted=include_adjusted
    )
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert diagnostics["prior_observation_rows"] == 6
    assert diagnostics["excluded_as_of_rows"] == 4
    assert diagnostics["excluded_same_day_rows"] == 2
    assert diagnostics["excluded_future_rows"] == 2
    assert diagnostics["prior_bouts"] == 3
    assert diagnostics["excluded_as_of_bouts"] == 2
    assert diagnostics["latest_prior_date"] == "2016-01-14"
    assert diagnostics["source_history_lag_days"] == 1
    assert diagnostics["labels_read"] is False
    assert "original statistic publication timestamps" in diagnostics["availability_caveat"]
    json.dumps(diagnostics, allow_nan=False)


@pytest.mark.parametrize("include_adjusted", [False, True])
def test_future_and_same_day_stats_cannot_change_any_future_month(include_adjusted):
    obs, target = observations(), targets()
    expected, _ = build_future_performance(
        obs.iloc[:6], target, "2016-01-15", include_adjusted=include_adjusted
    )
    modified = obs.copy()
    # Future values need not be valid: they must be filtered before validation.
    modified.loc[6:, "minutes"] = -1
    modified.loc[6:, "sig_landed"] = np.nan
    modified.loc[6:, "td_landed"] = 100000
    actual, diagnostics = build_future_performance(
        modified, target, "2016-01-15", include_adjusted=include_adjusted
    )
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    if include_adjusted:
        months = diagnostics["b2_monthly_diagnostics"]
        assert [month["cutoff"] for month in months] == ["2016-01-01", "2016-02-01", "2016-04-01"]
        assert [month["n_observations"] for month in months] == [4, 6, 6]
        assert [month["latest_prior_date"] for month in months] == ["2015-12-12", "2016-01-14", "2016-01-14"]


def test_b2_excludes_current_target_month_while_b1_uses_available_days():
    obs, target = observations(), targets().iloc[:1]
    full, diagnostics = build_future_performance(obs, target, "2016-01-15")
    old, _ = build_future_performance(obs.iloc[:4], target, "2016-01-15")
    pd.testing.assert_frame_equal(full[ADJUSTED_COLUMNS], old[ADJUSTED_COLUMNS], check_exact=True)
    assert not full[list(FEATURE_COLUMNS)].equals(old[list(FEATURE_COLUMNS)])
    assert diagnostics["b2_monthly_diagnostics"][0]["cutoff"] == "2016-01-01"
    assert diagnostics["b2_monthly_diagnostics"][0]["n_observations"] == 4


def test_exact_schema_keys_dtypes_order_duplicate_index_and_permutation():
    obs, target = observations(), targets()
    target["date"] = pd.to_datetime(target.date)
    expected, _ = build_future_performance(obs, target, pd.Timestamp("2016-01-15"))
    pd.testing.assert_frame_equal(expected.iloc[:, :3], target[["date", "R_fighter", "B_fighter"]])
    assert list(expected.columns) == ["date", "R_fighter", "B_fighter", *FEATURE_COLUMNS, *ADJUSTED_COLUMNS]
    actual, _ = build_future_performance(
        obs.sample(frac=1, random_state=4), target.iloc[::-1], date(2016, 1, 15)
    )
    pd.testing.assert_frame_equal(actual, expected.iloc[::-1], check_exact=True)


def test_red_blue_swap_preserves_paired_columns_and_signed_differences():
    original, _ = build_future_performance(observations(), targets(), "2016-01-15")
    swapped, _ = build_future_performance(observations(), _swap_red_blue(targets()), "2016-01-15")
    expected = _swap_red_blue(original)
    pd.testing.assert_frame_equal(swapped[expected.columns], expected, check_exact=True)


def test_labels_and_odds_are_not_selected_or_propagated_and_inputs_unchanged():
    obs, target = observations(), targets()
    obs["OUTCOME"] = [object()] * len(obs)
    target["Winner"] = [object()] * len(target)
    target["r_odds"] = [object()] * len(target)
    before_obs, before_target = obs.copy(deep=True), target.copy(deep=True)
    actual, diagnostics = build_future_performance(obs, target, "2016-01-15")
    expected, _ = build_future_performance(
        obs.drop(columns="OUTCOME"), target.drop(columns=["Winner", "r_odds"]), "2016-01-15"
    )
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    pd.testing.assert_frame_equal(obs, before_obs)
    pd.testing.assert_frame_equal(target, before_target)
    assert diagnostics["labels_read"] is False
    assert "Winner" not in actual and "r_odds" not in actual


@pytest.mark.parametrize(
    "as_of",
    [
        None,
        20160115,
        "01/15/2016",
        "2016-1-15",
        "2016-02-30",
        "2016-01-15T00:00:00",
        "2016-01-15 12:00",
        pd.NaT,
        pd.Timestamp("2016-01-15", tz="UTC"),
        datetime(2016, 1, 15, 1),
    ],
)
def test_as_of_rejects_ambiguous_missing_nonmidnight_or_timezone_dates(as_of):
    with pytest.raises(ValueError, match="as_of"):
        build_future_performance(observations(), targets(), as_of)


@pytest.mark.parametrize("field", ["date", "R_fighter", "B_fighter", "weight_class"])
def test_missing_required_target_fields_fail(field):
    with pytest.raises(ValueError, match="Missing future target"):
        build_future_performance(observations(), targets().drop(columns=field), "2016-01-15")


@pytest.mark.parametrize(
    "bad_date", ["2016-01-14", "2016-01-15T00:00:00Z", pd.Timestamp("2016-01-15 01:00"), pd.NaT]
)
def test_invalid_target_dates_fail(bad_date):
    target = targets().astype({"date": "object"})
    target.iloc[0, 0] = bad_date
    with pytest.raises(ValueError, match="target date"):
        build_future_performance(observations(), target, "2016-01-15")


def test_bad_observation_date_fails_even_when_statistic_would_be_excluded():
    obs = observations()
    obs.iloc[-1, obs.columns.get_loc("date")] = "2016-02-01T12:00:00"
    with pytest.raises(ValueError, match="observation date"):
        build_future_performance(obs, targets(), "2016-01-15")


@pytest.mark.parametrize("reversed_corners", [False, True])
def test_duplicate_canonical_bout_keys_fail_even_with_different_index_or_alias(reversed_corners):
    target = targets().iloc[:1].copy()
    target["R_fighter"] = "Sumudaerji"
    duplicate = target.copy()
    duplicate["R_fighter"] = "Su Mudaerji"
    if reversed_corners:
        duplicate = _swap_red_blue(duplicate)
    with pytest.raises(ValueError, match="Duplicate future bout key"):
        build_future_performance(
            observations(), pd.concat([target, duplicate], ignore_index=True), "2016-01-15"
        )


@pytest.mark.parametrize(
    "red,blue,division",
    [
        ("A", " a ", "Lightweight"),
        ("Su Mudaerji", "Sumudaerji", "Bantamweight"),
        ("Bruno Silva", "Bruno Silva [MW]", "Middleweight"),
    ],
)
def test_self_fights_after_canonical_identity_fail(red, blue, division):
    target = targets().iloc[:1].assign(R_fighter=red, B_fighter=blue, weight_class=division)
    with pytest.raises(ValueError, match="self-fight"):
        build_future_performance(observations(), target, "2016-01-15")


def test_unresolved_homonyms_and_debutants_retain_missing_flags():
    target = (
        targets().iloc[:1].assign(R_fighter="Bruno Silva", B_fighter="Debut", weight_class="Catch Weight")
    )
    output, _ = build_future_performance(observations(), target, "2016-01-15")
    assert output.R_perf_missing.iloc[0] == output.B_perf_missing.iloc[0] == 1
    assert output.R_perf_adj_missing.iloc[0] == output.B_perf_adj_missing.iloc[0] == 1
    assert np.isfinite(output.iloc[:, 3:].to_numpy()).all()


def test_empty_history_and_empty_targets_return_finite_schema_and_plain_diagnostics():
    obs = observations().iloc[:0]
    target = targets()
    output, diagnostics = build_future_performance(obs, target, "2016-01-15")
    assert np.isfinite(output.iloc[:, 3:].to_numpy()).all()
    assert diagnostics["latest_prior_date"] is None
    assert diagnostics["source_history_lag_days"] is None
    assert diagnostics["prior_observation_rows"] == 0
    json.dumps(diagnostics, allow_nan=False)
    empty, empty_diag = build_future_performance(obs, target.iloc[:0], "2016-01-15")
    assert empty.empty
    assert list(empty.columns) == list(output.columns)
    assert empty_diag["b2_monthly_diagnostics"] == []


def test_adjusted_option_requires_boolean():
    with pytest.raises(ValueError, match="boolean"):
        build_future_performance(observations(), targets(), "2016-01-15", include_adjusted="false")

"""Structural prior-only and numerical checks; no model-score evaluation."""

import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from pandas.testing import assert_frame_equal
from sklearn.exceptions import ConvergenceWarning

from ufc_pred.research import perf_adjusted as adjusted


def observations():
    rows = []
    for i, (date, a, b, division, minutes, sig_a, sig_b, td_a, td_b) in enumerate(
        [
            ("2015-01-02", "A", "B", "Lightweight", 10.0, 45, 15, 3, 1),
            ("2015-03-12", "A", "C", "Lightweight", 15.0, 60, 22, 2, 0),
            ("2015-05-03", "B", "C", "Lightweight", 12.0, 28, 31, 1, 2),
            ("2015-08-04", "D", "E", "Heavyweight", 8.0, 12, 9, 1, 1),
            ("2016-01-02", "A", "B", "Lightweight", 10.0, 400, 1, 9, 0),
            ("2016-02-01", "Future", "A", "Futureweight", 10.0, 500, 2, 8, 0),
        ]
    ):
        for fighter, opponent, sig, td in ((a, b, sig_a, td_a), (b, a, sig_b, td_b)):
            rows.append(
                dict(
                    bout_id=f"bout-{i}",
                    event_id=f"event-{i}",
                    date=date,
                    fighter=fighter,
                    opponent=opponent,
                    division=division,
                    minutes=minutes,
                    sig_landed=sig,
                    td_landed=td,
                )
            )
    return pd.DataFrame(rows)


def targets():
    return pd.DataFrame(
        {
            "date": ["2016-01-23", "2016-01-02", "2016-02-12"],
            "R_fighter": ["A", "A", "C"],
            "B_fighter": ["B", "New", "A"],
            "weight_class": ["Lightweight", "Lightweight", "Heavyweight"],
            "Winner": ["Blue", "Red", "Red"],
        },
        index=[7, 7, 3],
    )


def test_future_and_current_month_cannot_change_features():
    obs, target = observations(), targets().iloc[:2]
    full, diag = adjusted.build_adjusted_features(obs, target, return_diagnostics=True)
    past = adjusted.build_adjusted_features(obs[pd.to_datetime(obs.date) < "2016-01-01"], target)
    assert_frame_equal(full, past, check_exact=True)
    assert diag.loc[0, "n_observations"] == 8
    assert diag.loc[0, "n_fighters"] == 5
    assert diag.loc[0, "latest_prior_date"] < diag.loc[0, "cutoff"]
    assert "Future" not in adjusted.fit_adjusted_state(obs.iloc[:8], "2016-01-01").sig.attack


def test_fitter_rejects_same_cutoff_and_future_rows():
    with pytest.raises(AssertionError, match="strictly before"):
        adjusted.fit_adjusted_state(observations(), "2016-01-02")


def test_permutation_order_and_exact_keys():
    obs, target = observations(), targets()
    full = adjusted.build_adjusted_features(obs, target)
    shuffled = adjusted.build_adjusted_features(obs.sample(frac=1, random_state=7), target)
    assert_frame_equal(full, shuffled, check_exact=True)
    assert_frame_equal(full.iloc[:, :3], target.iloc[:, :3], check_exact=True)
    assert list(full.columns) == ["date", "R_fighter", "B_fighter"] + adjusted.FEATURE_COLUMNS
    assert len(adjusted.FEATURE_COLUMNS) == 20
    # Target permutation does not change monthly state or per-row feature values.
    assert_frame_equal(adjusted.build_adjusted_features(obs, target.iloc[::-1]), full.iloc[::-1])


def test_swap_and_blue_minus_red_matchup_rates():
    target = targets()
    swapped = target.copy()
    swapped[["R_fighter", "B_fighter"]] = target[["B_fighter", "R_fighter"]].to_numpy()
    original = adjusted.build_adjusted_features(observations(), target)
    other = adjusted.build_adjusted_features(observations(), swapped)
    for column in adjusted.FEATURE_COLUMNS:
        if column.startswith("R_"):
            assert_allclose(original[column], other[column.replace("R_", "B_", 1)], atol=0)
        elif column.endswith("_dif"):
            assert_allclose(original[column], -other[column], atol=0)
    assert_allclose(
        original.perf_adj_sig_lograte_dif,
        original.perf_adj_sig_attack_dif + original.perf_adj_sig_defense_dif,
    )
    assert_allclose(
        original.perf_adj_td_lograte_dif, original.perf_adj_td_attack_dif + original.perf_adj_td_defense_dif
    )


def test_no_history_unknown_and_unseen_division():
    obs = observations()
    early = targets().iloc[:1].assign(date="2014-01-01")
    result, diag = adjusted.build_adjusted_features(obs, early, return_diagnostics=True)
    assert_allclose(result.R_perf_adj_sig_lograte, np.log(3.0))
    assert_allclose(result.B_perf_adj_td_lograte, np.log(0.1))
    assert result.R_perf_adj_missing.iloc[0] == 1
    assert diag.sig_fallback_reason.iloc[0] == "no_prior_observations"
    target = targets().iloc[:1].assign(R_fighter="Debut", B_fighter="Other", weight_class="Unseen")
    state = adjusted.fit_adjusted_state(obs.iloc[:8], "2016-01-01")
    result = adjusted.build_adjusted_features(obs, target)
    assert result.R_perf_adj_sig_attack.iloc[0] == 0
    assert result.R_perf_adj_sig_defense.iloc[0] == 0
    assert result.R_perf_adj_missing.iloc[0] == 1
    assert_allclose(result.R_perf_adj_sig_lograte, state.sig.intercept)
    assert_allclose(result.B_perf_adj_td_lograte, state.td.intercept)


def test_scheduled_division_is_used_per_target():
    target = pd.concat([targets().iloc[:1]] * 3, ignore_index=True)
    target["weight_class"] = ["Lightweight", "Heavyweight", "UFC Lightweight Title Bout"]
    result = adjusted.build_adjusted_features(observations(), target)
    state = adjusted.fit_adjusted_state(observations().iloc[:8], "2016-01-01")
    assert_allclose(
        result.R_perf_adj_sig_lograte.iloc[0] - result.R_perf_adj_sig_lograte.iloc[1],
        state.sig.division_effects["lightweight"] - state.sig.division_effects["heavyweight"],
    )
    assert result.R_perf_adj_sig_lograte.iloc[0] == result.R_perf_adj_sig_lograte.iloc[2]


def test_weighted_rate_and_unpenalized_intercept_score():
    obs = observations().iloc[:8]
    cutoff = pd.Timestamp("2016-01-01")
    state = adjusted.fit_adjusted_state(obs, cutoff)
    for name in ("sig", "td"):
        channel = getattr(state, name)
        lograte = np.array(
            [
                channel.intercept
                + channel.division_effects[adjusted._division(row.division)]
                + channel.attack[adjusted._fighter_key(row.fighter, row.division)]
                - channel.defense[adjusted._fighter_key(row.opponent, row.division)]
                for row in obs.itertuples()
            ]
        )
        weight = obs.minutes.to_numpy() * 0.5 ** (
            ((cutoff - pd.to_datetime(obs.date)).dt.total_seconds().to_numpy() / 86400 / 365.25) / 4
        )
        # The unpenalized intercept has zero Poisson score: sum(w * mu)=sum(w*y).
        assert_allclose(
            np.average(np.exp(lograte), weights=weight),
            np.average(obs[f"{name}_landed"] / obs.minutes, weights=weight),
            rtol=2e-6,
        )


@pytest.mark.parametrize(
    "column,value",
    [
        ("minutes", 0),
        ("minutes", -1),
        ("minutes", np.inf),
        ("sig_landed", -1),
        ("td_landed", np.nan),
        ("td_landed", 1.2),
    ],
)
def test_invalid_exposure_and_counts_fail(column, value):
    obs = observations().iloc[:8].copy()
    obs[column] = obs[column].astype(float)
    obs.loc[0, column] = value
    with pytest.raises(ValueError):
        adjusted.fit_adjusted_state(obs, "2016-01-01")


def test_all_zero_count_channel_shrinks_with_exposure():
    obs = observations().iloc[:8].copy()
    obs["td_landed"] = 0
    state = adjusted.fit_adjusted_state(obs, "2016-01-01")
    assert state.td.fallback_reason == "all_zero_counts"
    expected = 0.1 * 30 / (30 + state.diagnostics["weighted_minutes"])
    assert_allclose(np.exp(state.td.intercept), expected)
    assert 0 < expected < 0.1
    obs["minutes"] *= 2
    more = adjusted.fit_adjusted_state(obs, "2016-01-01")
    assert np.exp(more.td.intercept) < expected


def test_unconverged_state_cannot_be_served(monkeypatch):
    original_fit = adjusted.PoissonRegressor.fit

    def warn_fit(self, *args, **kwargs):
        result = original_fit(self, *args, **kwargs)
        warnings.warn("Forced convergence failure", ConvergenceWarning, stacklevel=2)
        return result

    monkeypatch.setattr(adjusted.PoissonRegressor, "fit", warn_fit)
    with pytest.raises(adjusted.AdjustedConvergenceError, match="Unconverged sig"):
        adjusted.build_adjusted_features(observations(), targets())


def test_empty_targets_preserve_keys_and_feature_schema():
    result, diag = adjusted.build_adjusted_features(
        observations(), targets().iloc[:0], return_diagnostics=True
    )
    assert result.empty and diag.empty
    assert list(result.columns) == ["date", "R_fighter", "B_fighter"] + adjusted.FEATURE_COLUMNS


def test_parser_lowercase_divisions_match_titlecase_scheduled_division():
    obs = observations().iloc[:8].copy()
    obs["division"] = obs["division"].str.lower()
    target = targets().iloc[:1]
    result = adjusted.build_adjusted_features(obs, target)
    state = adjusted.fit_adjusted_state(obs, "2016-01-01")
    expected = (
        state.sig.intercept
        + state.sig.division_effects["lightweight"]
        + state.sig.attack["a"]
        - state.sig.defense["b"]
    )
    assert_allclose(result.R_perf_adj_sig_lograte, expected)
    normalized = target.assign(weight_class="lightweight")
    assert_frame_equal(result, adjusted.build_adjusted_features(obs, normalized), check_exact=True)


def test_canonical_normalized_identities_preserve_display_keys():
    obs = observations().iloc[:8].copy()
    replacements = {"A": "José Aldo", "B": "Sumudaerji"}
    obs["fighter"] = obs["fighter"].replace(replacements)
    obs["opponent"] = obs["opponent"].replace(replacements)
    target = targets().iloc[:1].assign(R_fighter=" JOSE ALDO ", B_fighter="Su Mudaerji")
    original = targets().iloc[:1].assign(R_fighter="José Aldo", B_fighter="Sumudaerji")
    result = adjusted.build_adjusted_features(obs, target)
    baseline = adjusted.build_adjusted_features(obs, original)
    assert_frame_equal(result.iloc[:, 3:], baseline.iloc[:, 3:], check_exact=True)
    assert_frame_equal(result.iloc[:, :3], target.iloc[:, :3], check_exact=True)
    assert result.R_perf_adj_missing.iloc[0] == 0
    assert result.B_perf_adj_missing.iloc[0] == 0


def test_ambiguous_target_homonym_is_missing_without_invented_identity():
    target = targets().iloc[:1].assign(R_fighter="Bruno Silva", weight_class="Unseen")
    result = adjusted.build_adjusted_features(observations(), target)
    assert result.R_perf_adj_missing.iloc[0] == 1
    assert result.R_perf_adj_sig_attack.iloc[0] == 0

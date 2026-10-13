"""B1 tests exercise leakage, exposure arithmetic, identity, and raw integrity."""

import numpy as np
import pandas as pd
import pytest

from ufc_pred.features.static_v1 import _swap_red_blue
from ufc_pred.research.perf_features import (
    COUNT_COLUMNS,
    FEATURE_COLUMNS,
    OBSERVATION_COLUMNS,
    PerformanceRecipe,
    _division,
    bout_minutes,
    build_smoothed_features,
    features_for_targets,
    fit_smoothed_state,
    load_bouts,
)


def pair(
    day="2020-01-01",
    bout="b1",
    red="A",
    blue="B",
    own=(10, 20, 0, 0),
    other=(20, 40, 2, 4),
    minutes=5,
    division="lightweight",
):
    common = dict(bout_id=bout, event_id="e1", date=pd.Timestamp(day), division=division, minutes=minutes)
    return [
        dict(**common, fighter=red, opponent=blue, **dict(zip(COUNT_COLUMNS, own + other, strict=True))),
        dict(**common, fighter=blue, opponent=red, **dict(zip(COUNT_COLUMNS, other + own, strict=True))),
    ]


def observations(*rows):
    return pd.DataFrame([item for p in rows for item in p], columns=OBSERVATION_COLUMNS)


def targets(*days):
    return pd.DataFrame(
        dict(
            date=pd.to_datetime(days or ("2020-01-02",)),
            R_fighter="A",
            B_fighter="B",
            weight_class="Lightweight Bout",
        )
    )


def test_gamma_poisson_and_beta_binomial_hand_calculation():
    result = build_smoothed_features(observations(pair()), targets()).iloc[0]
    w = 0.5 ** (1 / (365.25 * 4))
    a_rate = (10 * w + 30 * 3) / (5 * w + 30)
    b_rate = (20 * w + 30 * 3) / (5 * w + 30)
    assert result.perf_sig_attack_dif == pytest.approx(np.log(b_rate) - np.log(a_rate))
    assert result.perf_sig_defense_dif == pytest.approx(np.log(b_rate) - np.log(a_rate))
    assert result.perf_sig_accuracy_dif == pytest.approx(0)
    pooled_td = (2 * w + 20 * 0.3) / (4 * w + 20)
    division_td = (2 * w + 20 * pooled_td) / (4 * w + 20)
    a_prevention = 1 - (2 * w + 20 * division_td) / (4 * w + 20)
    b_prevention = 1 - division_td  # no attempts retains the division prior

    def logit(x):
        return np.log(x / (1 - x))

    assert result.perf_td_prevention_dif == pytest.approx(logit(b_prevention) - logit(a_prevention))
    assert result.R_perf_log_minutes == pytest.approx(np.log1p(5 * w))
    assert result.R_perf_log_sig_attempts == pytest.approx(np.log1p(20 * w))
    assert result.R_perf_log_opp_td_attempts == pytest.approx(np.log1p(4 * w))


def test_future_additions_and_same_day_bouts_do_not_leak():
    past = observations(pair())
    future = observations(
        pair(),
        pair("2020-01-02", "today", own=(1000, 1000, 100, 100)),
        pair("2030-01-01", "future", own=(400, 400, 20, 20)),
    )
    pd.testing.assert_frame_equal(
        build_smoothed_features(past, targets()), build_smoothed_features(future, targets())
    )
    same_day = build_smoothed_features(past, targets("2020-01-01"))
    assert same_day.loc[0, "R_perf_missing"] == 1
    assert same_day.loc[0, "B_perf_missing"] == 1
    assert (same_day.filter(regex="_dif$") == 0).all().all()


def test_observation_and_target_permutation_and_frozen_state_parity():
    obs = observations(pair(), pair("2021-01-01", "b2", own=(30, 40, 3, 4)))
    ts = targets("2022-01-01", "2020-01-02", "2021-01-01", "2022-01-01")
    ts.index = [11, 12, 13, 14]
    expected = build_smoothed_features(obs, ts)
    actual = build_smoothed_features(obs.sample(frac=1, random_state=2), ts.iloc[[3, 1, 0, 2]])
    pd.testing.assert_frame_equal(actual.loc[expected.index], expected)
    frozen = fit_smoothed_state(obs, "2022-01-01")
    pd.testing.assert_frame_equal(features_for_targets(frozen, ts.iloc[[0, 3]]), expected.iloc[[0, 3]])
    pd.testing.assert_frame_equal(features_for_targets(frozen, ts.iloc[[3, 0]]), expected.iloc[[3, 0]])


def test_red_blue_swap_has_exact_signed_difference_and_exposure_symmetry():
    obs, ts = observations(pair()), targets()
    expected = _swap_red_blue(build_smoothed_features(obs, ts))
    actual = build_smoothed_features(obs, _swap_red_blue(ts))
    pd.testing.assert_frame_equal(actual.loc[:, expected.columns], expected)


def test_keys_preserved_with_unknown_and_ambiguous_identity():
    ts = pd.DataFrame(
        dict(
            date=["2020-01-02"] * 3,
            R_fighter=["Newcomer", "Bruno Silva", "A"],
            B_fighter=["B"] * 3,
            weight_class=["Lightweight", "Catch Weight", "Lightweight"],
            Winner=["Red", "Blue", None],
            custom=[1, 2, 3],
        ),
        index=[9, 9, 4],
    )
    result = build_smoothed_features(observations(pair()), ts)
    pd.testing.assert_frame_equal(result.iloc[:, :3], ts[["date", "R_fighter", "B_fighter"]])
    assert list(result.columns) == ["date", "R_fighter", "B_fighter", *FEATURE_COLUMNS]
    assert result.R_perf_missing.tolist() == [1, 1, 0]
    assert np.isfinite(result[list(FEATURE_COLUMNS)].to_numpy()).all()
    different_outcomes = ts.assign(Winner=["Blue", "Red", "Red"])
    pd.testing.assert_frame_equal(build_smoothed_features(observations(pair()), different_outcomes), result)


def test_zero_attempts_are_valid_exposure_not_missing_or_bad_accuracy():
    obs = observations(pair(own=(0, 0, 0, 0), other=(0, 0, 0, 0)))
    result = build_smoothed_features(obs, targets()).iloc[0]
    assert result.R_perf_missing == result.B_perf_missing == 0
    assert result.R_perf_log_minutes > 0
    assert result.R_perf_log_sig_attempts == 0
    assert result.perf_sig_accuracy_dif == result.perf_td_prevention_dif == 0
    assert np.isfinite(result[list(FEATURE_COLUMNS)].to_numpy(dtype=float)).all()


def test_fit_rejects_cutoff_leakage_and_invalid_pairs():
    obs = observations(pair())
    with pytest.raises(ValueError, match="strictly before"):
        fit_smoothed_state(obs, "2020-01-01")
    state = fit_smoothed_state(obs, "2020-01-02")
    with pytest.raises(ValueError, match="precedes"):
        features_for_targets(state, targets("2020-01-01"))
    bad = obs.copy()
    bad.loc[0, "opp_sig_landed"] = 19
    with pytest.raises(ValueError, match="conserve"):
        fit_smoothed_state(bad, "2020-01-02")
    with pytest.raises(ValueError, match="exactly two"):
        fit_smoothed_state(obs.iloc[:1], "2020-01-02")
    with pytest.raises(ValueError, match="positive"):
        PerformanceRecipe(half_life_years=0)


@pytest.mark.parametrize(
    "round_,clock,fmt,expected",
    [
        (3, "2:30", "5 Rnd(5-5-5-5-5)", 12.5),
        (3, "1:00", "3 Rnd (10-5-5)", 16),
        (2, "3:00", "2 Rnd (12-3)", 15),
        (1, "12:30", "No Time Limit", 12.5),
        (2, "0:00", "3 Rnd (5-5-5)", 5),
        (2, "3:00", "1 Rnd + OT (12-3)", 15),
        (3, "2:00", "1 Rnd + 2OT (15-3-3)", 20),
        (4, "1:00", "3 Rnd + OT (5-5-5-5)", 16),
    ],
)
def test_valid_round_exposure(round_, clock, fmt, expected):
    assert bout_minutes(round_, clock, fmt) == expected


@pytest.mark.parametrize(
    "round_,clock,fmt",
    [
        (0, "1:00", "3 Rnd (5-5-5)"),
        (1.5, "1:00", "3 Rnd (5-5-5)"),
        (1, "5:01", "3 Rnd (5-5-5)"),
        (2, "3:60", "2 Rnd (12-3)"),
        (2, "1:00", "No Time Limit"),
        (1, "0:00", "No Time Limit"),
        (3, "1:00", "2 Rnd (12-3)"),
        (1, "1:00", "3 Rnd (5-5)"),
        (1, "1:00", "Unknown"),
        (2, "1:00", "1 Rnd + 2OT (12-3)"),
    ],
)
def test_invalid_round_exposure(round_, clock, fmt):
    with pytest.raises(ValueError):
        bout_minutes(round_, clock, fmt)


def raw_files(
    tmp_path, *, round_=2, fmt="3 Rnd (10-5-5)", clock="2:30", division="Lightweight Bout", bout="A vs. B"
):
    pd.DataFrame([dict(EVENT="UFC 1", URL="event-url", DATE="January 1, 2020")]).to_csv(
        tmp_path / "ufc_event_details.csv", index=False
    )
    results = pd.DataFrame(
        [
            dict(
                EVENT="UFC 1 ",
                BOUT=bout,
                WEIGHTCLASS=division,
                ROUND=round_,
                TIME=clock,
                URL="bout-url",
                **{"TIME FORMAT": fmt},
                OUTCOME="W/L",
            )
        ]
    )
    names = bout.split(" vs. ")
    rows = [
        dict(
            EVENT="UFC 1",
            BOUT=bout,
            ROUND=f"Round {r}",
            FIGHTER=name,
            **{"SIG.STR.": "1 of 2", "TD": "0 of 0"},
        )
        for r in range(1, round_ + 1)
        for name in names
    ]
    stats = pd.DataFrame(rows)
    results.to_csv(tmp_path / "ufc_fight_results.csv", index=False)
    stats.to_csv(tmp_path / "ufc_fight_stats.csv", index=False)
    return results, stats


def test_raw_aggregation_exposure_and_opponent_conservation(tmp_path):
    raw_files(tmp_path)
    obs, excluded = load_bouts(tmp_path)
    assert excluded.empty
    assert list(obs.columns) == list(OBSERVATION_COLUMNS)
    assert obs.minutes.tolist() == [12.5, 12.5]
    assert obs.sig_landed.tolist() == [2, 2]
    assert obs.td_attempted.tolist() == [0, 0]
    for own, allowed in zip(COUNT_COLUMNS[:4], COUNT_COLUMNS[4:], strict=True):
        assert obs.loc[0, own] == obs.loc[1, allowed]
        assert obs.loc[1, own] == obs.loc[0, allowed]


@pytest.mark.parametrize(
    "mutation,reason",
    [
        ("missing", "missing_stats"),
        ("invalid", "invalid_stats"),
        ("landed", "landed_exceeds_attempts"),
        ("duplicate", "duplicate_stats_side_round"),
        ("round", "incomplete_or_extra_stats_rounds"),
        ("side", "unknown_stats_fighter"),
    ],
)
def test_raw_bad_stats_fail_closed(tmp_path, mutation, reason):
    _, stats = raw_files(tmp_path)
    if mutation == "missing":
        stats.loc[0, "SIG.STR."] = np.nan
    elif mutation == "invalid":
        stats.loc[0, "SIG.STR."] = "-1 of 2"
    elif mutation == "landed":
        stats.loc[0, "SIG.STR."] = "3 of 2"
    elif mutation == "duplicate":
        stats = pd.concat([stats, stats.iloc[:1]])
    elif mutation == "round":
        stats = stats.iloc[:-1]
    elif mutation == "side":
        stats.loc[0, "FIGHTER"] = "Unknown"
    stats.to_csv(tmp_path / "ufc_fight_stats.csv", index=False)
    obs, excluded = load_bouts(tmp_path)
    assert obs.empty
    assert excluded.reason.tolist() == [reason]


def test_ambiguous_same_event_rematch_not_combined(tmp_path):
    results, _ = raw_files(tmp_path)
    second = results.copy().assign(URL="bout-url-2")
    pd.concat([results, second]).to_csv(tmp_path / "ufc_fight_results.csv", index=False)
    obs, excluded = load_bouts(tmp_path)
    assert obs.empty
    assert excluded.reason.tolist() == ["ambiguous_event_bout_results"]


@pytest.mark.parametrize(
    "division,expected",
    [
        ("Flyweight Bout", "Bruno Silva [FLW]"),
        ("Middleweight Bout", "Bruno Silva [MW]"),
        ("Catch Weight Bout", None),
    ],
)
def test_historical_division_disambiguates_bruno(tmp_path, division, expected):
    raw_files(tmp_path, bout="Bruno Silva vs. B", division=division)
    obs, excluded = load_bouts(tmp_path)
    if expected is None:
        assert obs.empty
        assert excluded.reason.tolist() == ["ambiguous_fighter_identity"]
    else:
        assert excluded.empty
        assert expected in obs.fighter.tolist()


def test_raw_non_ufc_events_explicitly_excluded(tmp_path):
    results, stats = raw_files(tmp_path)
    results.EVENT = "DWCS 1"
    stats.EVENT = "DWCS 1"
    results.to_csv(tmp_path / "ufc_fight_results.csv", index=False)
    stats.to_csv(tmp_path / "ufc_fight_stats.csv", index=False)
    obs, excluded = load_bouts(tmp_path)
    assert obs.empty
    assert excluded.reason.tolist() == ["non_ufc_event"]


def test_empty_history_and_targets_have_stable_schema():
    obs = observations()
    ts = targets()
    result = build_smoothed_features(obs, ts)
    assert (result.filter(regex="missing$") == 1).all().all()
    assert (result.filter(regex="_dif$") == 0).all().all()
    empty = build_smoothed_features(obs, ts.iloc[:0])
    assert empty.empty
    assert list(empty.columns) == ["date", "R_fighter", "B_fighter", *FEATURE_COLUMNS]


@pytest.mark.parametrize(
    "label,expected",
    [
        ("Ultimate Fighter 32 Middleweight Tournament Title Bout", "middleweight"),
        ("Road to UFC 3 Women's Strawweight Tournament TitleBout", "women's strawweight"),
        ("Ultimate Fighter 28 Women's Featherweight Tournament Title Bout", "women's featherweight"),
        ("TUF Nations Canada vs. Australia Welterweight Tournament Title Bout", "welterweight"),
        ("Ultimate Fighter 23 Light Heavyweight Tournament Title Bout", "light heavyweight"),
        ("UFC Interim Women's Bantamweight Title Bout", "women's bantamweight"),
        ("UFC Women’s Flyweight Title Bout", "women's flyweight"),
        ("Super Heavyweight Bout", "super heavyweight"),
        ("Superheavyweight Bout", "super heavyweight"),
        ("UFC Superfight Championship Bout", "superfight"),
        ("Catch Weight Bout", "catch weight"),
        ("Open Weight Bout", "open weight"),
        ("Ultimate Ultimate '96 Tournament Title Bout", "__unknown__"),
        ("UFC 10 Tournament Title Bout", "__unknown__"),
        ("Unknown", "__unknown__"),
        (None, "__unknown__"),
        ("Notheavyweight Bout", "__unknown__"),
        ("Heavyweightish Bout", "__unknown__"),
        ("Women's Heavyweight Bout", "__unknown__"),
    ],
)
def test_division_uses_exact_physical_class_and_explicit_unknown(label, expected):
    assert _division(label) == expected


def test_ornate_divisions_share_prior_state_with_plain_target_class():
    obs = observations(pair(division="Ultimate Fighter 32 Middleweight Tournament Title Bout"))
    plain = targets().assign(weight_class="Middleweight")
    ornate = targets().assign(weight_class="UFC Interim Middleweight Title Bout")
    pd.testing.assert_frame_equal(build_smoothed_features(obs, plain), build_smoothed_features(obs, ornate))

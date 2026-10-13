"""Prior-only, exposure-aware UFC performance features (research B1).

No result/outcome field is read. A bout becomes available the following day.
Division priors shrink past division performance toward pooled past performance;
pooled priors shrink toward fixed neutral values using the same tau/kappa. Fixed
neutral priors are 3 significant strikes/minute, 0.1 takedowns/minute, 50% SIG
accuracy, and 30% TD accuracy. These are constants, not full-history estimates.
Every exposure and count receives the same exponential recency weight.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ufc_pred.ingest.identity import canonical, normalise

METRICS = ("sig_attack", "sig_defense", "td_attack", "td_defense", "sig_accuracy", "td_prevention")
COUNT_COLUMNS = (
    "sig_landed",
    "sig_attempted",
    "td_landed",
    "td_attempted",
    "opp_sig_landed",
    "opp_sig_attempted",
    "opp_td_landed",
    "opp_td_attempted",
)
OBSERVATION_COLUMNS = (
    "bout_id",
    "event_id",
    "date",
    "fighter",
    "opponent",
    "division",
    "minutes",
    *COUNT_COLUMNS,
)
KEY_COLUMNS = ("date", "R_fighter", "B_fighter")
FEATURE_COLUMNS = tuple(f"perf_{m}_dif" for m in METRICS) + tuple(
    f"{s}_perf_{m}"
    for s in ("R", "B")
    for m in ("log_minutes", "log_sig_attempts", "log_opp_td_attempts", "missing")
)
_EPS = 1e-8


@dataclass(frozen=True)
class PerformanceRecipe:
    half_life_years: float = 4.0
    rate_prior_minutes: float = 30.0
    accuracy_prior_attempts: float = 20.0

    def __post_init__(self):
        for name in ("half_life_years", "rate_prior_minutes", "accuracy_prior_attempts"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")


DEFAULT_RECIPE = PerformanceRecipe()


def _division(value):
    """Extract an exact physical division from decorated historical labels.

    Match complete words, longest first: women's categories precede their
    men's substrings and light/super heavyweight precede heavyweight. Labels
    without a recognised physical category receive one explicit unknown key;
    tournament/event decorations never become separate divisions.
    """
    if pd.isna(value):
        return "__unknown__"
    text = re.sub(r"\s+", " ", str(value).strip().lower().replace("’", "'"))
    divisions = (
        "women's strawweight",
        "women's flyweight",
        "women's bantamweight",
        "women's featherweight",
        "super heavyweight",
        "light heavyweight",
        "strawweight",
        "flyweight",
        "bantamweight",
        "featherweight",
        "lightweight",
        "welterweight",
        "middleweight",
        "heavyweight",
        "catch weight",
        "open weight",
        "superfight",
    )
    # Common compact special-class spellings are normalised explicitly.
    text = re.sub(r"\bsuperheavyweight\b", "super heavyweight", text)
    text = re.sub(r"\bcatchweight\b", "catch weight", text)
    text = re.sub(r"\bopenweight\b", "open weight", text)
    for division in sorted(divisions, key=len, reverse=True):
        if re.search(r"(?<![a-z0-9])" + re.escape(division) + r"(?![a-z0-9])", text):
            if "women's" in text and not division.startswith("women's"):
                return "__unknown__"
            return division
    return "__unknown__"


def bout_minutes(final_round, clock, time_format):
    """Parse actual duration against the historical scheduled round lengths.

    No Time Limit is usable only with a single positive round-one clock.
    A final zero clock is valid after at least one completed scheduled round.
    """
    try:
        round_float = float(final_round)
        round_number = int(round_float)
    except (ValueError, TypeError, OverflowError):
        raise ValueError("invalid_final_round") from None
    if round_number < 1 or round_float != round_number:
        raise ValueError("invalid_final_round")
    match = re.fullmatch(r"(\d+):(\d{2})", str(clock).strip())
    if not match or int(match[2]) >= 60:
        raise ValueError("invalid_final_clock")
    seconds = 60 * int(match[1]) + int(match[2])
    fmt = str(time_format).strip()
    if fmt.casefold() == "no time limit":
        if round_number != 1 or seconds <= 0:
            raise ValueError("invalid_no_time_limit_exposure")
        return seconds / 60.0
    schedule = re.fullmatch(r"(\d+)\s*Rnd(?:\s*\+\s*(\d*)OT)?\s*\(([0-9]+(?:-[0-9]+)*)\)", fmt, re.I)
    if not schedule:
        raise ValueError("unsupported_time_format")
    lengths = [int(x) for x in schedule[3].split("-")]
    overtime = 0 if schedule[2] is None else int(schedule[2] or 1)
    if (
        len(lengths) != int(schedule[1]) + overtime
        or any(x <= 0 for x in lengths)
        or round_number > len(lengths)
    ):
        raise ValueError("invalid_round_schedule")
    if seconds > 60 * lengths[round_number - 1]:
        raise ValueError("final_clock_exceeds_round")
    duration = sum(lengths[: round_number - 1]) + seconds / 60.0
    if duration <= 0:
        raise ValueError("nonpositive_exposure")
    return duration


def _ufc_event(name):
    # Same universe as ingest.ufcstats_scraper._is_ufc_history_event; avoid its
    # network/parser dependencies in this pure research module.
    return (
        name.startswith(("UFC", "The Ultimate Fighter", "Noche UFC"))
        or name == "Ortiz vs Shamrock 3: The Final Chapter"
    )


def _count_pair(value):
    if pd.isna(value) or str(value).strip() in ("", "---", "--"):
        raise ValueError("missing_stats")
    match = re.fullmatch(r"(\d+)\s+of\s+(\d+)", str(value).strip())
    if not match:
        raise ValueError("invalid_stats")
    landed, attempted = int(match[1]), int(match[2])
    if landed > attempted:
        raise ValueError("landed_exceeds_attempts")
    return landed, attempted


def load_bouts(source_dir):
    """Read three raw CSVs; return paired observations and explicit exclusions.

    Stats have no URL, so EVENT/BOUT may join only to a unique result and a
    unique event. Duplicate source rows, same-event rematches, incomplete
    rounds, unknown sides, invalid exposure, and missing stats fail closed.
    Historical WEIGHTCLASS disambiguates identities such as the Bruno Silvas.
    """
    source_dir = Path(source_dir)
    events = pd.read_csv(source_dir / "ufc_event_details.csv")
    results = pd.read_csv(source_dir / "ufc_fight_results.csv")
    stats = pd.read_csv(source_dir / "ufc_fight_stats.csv")
    for frame in (events, results, stats):
        for col in ("EVENT", "BOUT", "FIGHTER"):
            if col in frame:
                frame[col] = frame[col].fillna("").astype(str).str.strip()
    event_groups = {key: group for key, group in events.groupby("EVENT", sort=False)}
    result_groups = {key: group for key, group in results.groupby(["EVENT", "BOUT"], sort=False)}
    stats_groups = {key: group for key, group in stats.groupby(["EVENT", "BOUT"], sort=False)}
    observations, excluded = [], []
    bout_id_counts = results.URL.value_counts()

    def exclude(event, bout, reason, bout_id=None, **extra):
        excluded.append(dict(EVENT=event, BOUT=bout, bout_id=bout_id, reason=reason, **extra))

    for (event, bout), group in result_groups.items():
        bout_id = group.iloc[0].get("URL")
        if not _ufc_event(event):
            exclude(event, bout, "non_ufc_event", bout_id)
            continue
        if len(group) != 1:
            exclude(event, bout, "ambiguous_event_bout_results", bout_id, result_rows=len(group))
            continue
        if event not in event_groups or len(event_groups[event]) != 1:
            exclude(event, bout, "missing_or_ambiguous_event", bout_id)
            continue
        result, event_row = group.iloc[0], event_groups[event].iloc[0]
        date = pd.to_datetime(event_row.get("DATE"), errors="coerce")
        event_id = event_row.get("URL")
        if pd.isna(date) or pd.isna(bout_id) or not str(bout_id).strip() or pd.isna(event_id):
            exclude(event, bout, "missing_stable_key_or_date", bout_id)
            continue
        if bout_id_counts.get(bout_id, 0) != 1:
            exclude(event, bout, "duplicate_bout_id", bout_id)
            continue
        names = re.split(r"\s+vs\.?\s+", bout, maxsplit=1, flags=re.I)
        if len(names) != 2:
            exclude(event, bout, "invalid_bout_names", bout_id)
            continue
        try:
            canonical_names = [canonical(n, result.WEIGHTCLASS) for n in names]
            keys = [normalise(n) for n in canonical_names]
            if len(set(keys)) != 2:
                raise ValueError("ambiguous_fighter_identity")
            minutes = bout_minutes(result.ROUND, result.TIME, result["TIME FORMAT"])
        except ValueError as exc:
            reason = "ambiguous_fighter_identity" if "ambiguous" in str(exc) else str(exc)
            exclude(event, bout, reason, bout_id)
            continue
        rows = stats_groups.get((event, bout))
        if rows is None:
            exclude(event, bout, "missing_stats", bout_id)
            continue
        try:
            per_round = {}
            totals = {key: np.zeros(4, dtype=float) for key in keys}
            final_round = int(float(result.ROUND))
            for row in rows.to_dict("records"):
                round_match = re.fullmatch(r"(?:Round\s*)?(\d+)", str(row["ROUND"]).strip(), re.I)
                if not round_match:
                    raise ValueError("invalid_stats_round")
                round_number = int(round_match[1])
                name_key = normalise(canonical(row["FIGHTER"], result.WEIGHTCLASS))
                if name_key not in totals:
                    raise ValueError("unknown_stats_fighter")
                pair_key = (round_number, name_key)
                if pair_key in per_round:
                    raise ValueError("duplicate_stats_side_round")
                per_round[pair_key] = True
                counts = (*_count_pair(row["SIG.STR."]), *_count_pair(row["TD"]))
                totals[name_key] += counts
            expected = {(r, key) for r in range(1, final_round + 1) for key in keys}
            if set(per_round) != expected:
                raise ValueError("incomplete_or_extra_stats_rounds")
        except ValueError as exc:
            reason = "ambiguous_fighter_identity" if "ambiguous" in str(exc) else str(exc)
            exclude(event, bout, reason, bout_id)
            continue
        for index in (0, 1):
            other = 1 - index
            observations.append(
                dict(
                    bout_id=bout_id,
                    event_id=event_id,
                    date=date.normalize(),
                    fighter=canonical_names[index],
                    opponent=canonical_names[other],
                    division=_division(result.WEIGHTCLASS),
                    minutes=minutes,
                    **dict(zip(COUNT_COLUMNS, np.r_[totals[keys[index]], totals[keys[other]]], strict=True)),
                )
            )
    for event, bout in stats_groups.keys() - result_groups.keys():
        exclude(event, bout, "stats_without_result")
    obs = pd.DataFrame(observations, columns=OBSERVATION_COLUMNS)
    if len(obs):
        obs = obs.sort_values(["date", "bout_id", "fighter"], kind="stable").reset_index(drop=True)
    exclusions = pd.DataFrame(excluded)
    if exclusions.empty:
        exclusions = pd.DataFrame(columns=["EVENT", "BOUT", "bout_id", "reason"])
    return obs, exclusions


def _day(value):
    day = pd.Timestamp(value)
    if pd.isna(day) or day.tz is not None:
        raise ValueError("Dates must be finite timezone-naive calendar dates")
    return day.normalize()


def _prepare_observations(observations):
    missing = set(OBSERVATION_COLUMNS) - set(observations.columns)
    if missing:
        raise ValueError(f"Missing observation columns: {sorted(missing)}")
    frame = observations.loc[:, OBSERVATION_COLUMNS].copy()
    frame["date"] = frame.date.map(_day)
    frame["division"] = frame.division.map(_division)
    for field_name in ("fighter", "opponent"):
        frame[field_name] = [
            normalise(canonical(n, d)) for n, d in zip(frame[field_name], frame.division, strict=True)
        ]
    values = frame.loc[:, ["minutes", *COUNT_COLUMNS]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values[:, 0] <= 0).any() or (values[:, 1:] < 0).any():
        raise ValueError("Observation exposures/counts must be finite, nonnegative, and minutes positive")
    for landed, attempted in ((1, 2), (3, 4), (5, 6), (7, 8)):
        if (values[:, landed] > values[:, attempted]).any():
            raise ValueError("Observation landed counts exceed attempts")
    if frame.duplicated(["bout_id", "fighter"]).any():
        raise ValueError("Duplicate observation bout/fighter")
    for _, group in frame.groupby("bout_id", sort=False):
        if len(group) != 2:
            raise ValueError("Each observation bout must have exactly two sides")
        a, b = group.iloc[0], group.iloc[1]
        if (
            a.fighter != b.opponent
            or b.fighter != a.opponent
            or a.fighter == b.fighter
            or a.date != b.date
            or a.event_id != b.event_id
            or a.division != b.division
            or a.minutes != b.minutes
        ):
            raise ValueError("Inconsistent paired observation identity/exposure")
        if not np.array_equal(
            a[list(COUNT_COLUMNS[:4])].to_numpy(), b[list(COUNT_COLUMNS[4:])].to_numpy()
        ) or not np.array_equal(b[list(COUNT_COLUMNS[:4])].to_numpy(), a[list(COUNT_COLUMNS[4:])].to_numpy()):
            raise ValueError("Opponent counts do not conserve paired own counts")
    return frame.sort_values(["date", "bout_id", "fighter"], kind="stable")


@dataclass
class SmoothedState:
    """Inspectable sufficient statistics, each anchored to its last update day."""

    cutoff: pd.Timestamp
    recipe: PerformanceRecipe
    fighters: dict = field(default_factory=dict)
    divisions: dict = field(default_factory=dict)
    pooled: tuple | None = None


def _values(record, day, recipe):
    if record is None:
        return np.zeros(9, dtype=float)
    date, values = record
    age = (day - date).total_seconds() / (365.25 * 86400)
    return values * 0.5 ** (age / recipe.half_life_years)


def _add(state, row):
    day = row.date
    values = np.array([row.minutes, *(getattr(row, c) for c in COUNT_COLUMNS)], dtype=float)
    for mapping, key in ((state.fighters, row.fighter), (state.divisions, row.division)):
        mapping[key] = (day, _values(mapping.get(key), day, state.recipe) + values)
    state.pooled = (day, _values(state.pooled, day, state.recipe) + values)


def fit_smoothed_state(observations, cutoff, recipe=DEFAULT_RECIPE):
    """Fit a frozen live/historical state; refuse same-day or future history."""
    cutoff = _day(cutoff)
    obs = _prepare_observations(observations)
    if (obs.date >= cutoff).any():
        raise ValueError("fit_smoothed_state requires every observation strictly before cutoff")
    state = SmoothedState(cutoff, recipe)
    for row in obs.itertuples(index=False):
        _add(state, row)
    return state


def _priors(state, division, day):
    div = _values(state.divisions.get(division), day, state.recipe)
    pool = _values(state.pooled, day, state.recipe)

    def ratio(numerator, denominator, fixed, strength):
        # Both stages are fixed in advance; zero-count small divisions remain
        # well behaved rather than producing epsilon-valued rate priors.
        pooled = (pool[numerator] + strength * fixed) / (pool[denominator] + strength)
        return (div[numerator] + strength * pooled) / (div[denominator] + strength)

    tau, kappa = state.recipe.rate_prior_minutes, state.recipe.accuracy_prior_attempts
    return (
        ratio(1, 0, 3.0, tau),
        ratio(5, 0, 3.0, tau),
        ratio(3, 0, 0.1, tau),
        ratio(7, 0, 0.1, tau),
        ratio(1, 2, 0.5, kappa),
        ratio(7, 8, 0.3, kappa),
    )


def _side_features(state, name, division, day):
    try:
        key = normalise(canonical(name, division))
    except ValueError:
        key = None  # unresolved homonym receives the explicit missing state
    values = _values(state.fighters.get(key), day, state.recipe)
    mins, sig, sig_att, td, td_att, opp_sig, opp_sig_att, opp_td, opp_td_att = values
    priors = _priors(state, division, day)
    tau, kappa = state.recipe.rate_prior_minutes, state.recipe.accuracy_prior_attempts
    rates = (np.array([sig, opp_sig, td, opp_td]) + tau * np.array(priors[:4])) / (mins + tau)
    accuracy = (sig + kappa * priors[4]) / (sig_att + kappa)
    prevention = 1 - (opp_td + kappa * priors[5]) / (opp_td_att + kappa)

    def logit(p):
        return np.log(np.clip(p, _EPS, 1 - _EPS) / (1 - np.clip(p, _EPS, 1 - _EPS)))

    metrics = np.array(
        [
            np.log(rates[0]),
            -np.log(rates[1]),
            np.log(rates[2]),
            -np.log(rates[3]),
            logit(accuracy),
            logit(prevention),
        ]
    )
    exposure = np.array(
        [np.log1p(mins), np.log1p(sig_att), np.log1p(opp_td_att), float(key not in state.fighters)]
    )
    return metrics, exposure


def _target_rows(targets):
    missing = set(KEY_COLUMNS) - set(targets)
    if missing:
        raise ValueError(f"Missing target keys: {sorted(missing)}")
    divisions = (
        targets["weight_class"]
        if "weight_class" in targets
        else targets.get("division", pd.Series("__unknown__", index=targets.index))
    )
    return list(
        zip(
            targets.date.map(_day),
            targets.R_fighter,
            targets.B_fighter,
            divisions.map(_division),
            strict=True,
        )
    )


def _feature_values(state, row):
    day, red, blue, division = row
    red_metrics, red_exposure = _side_features(state, red, division, day)
    blue_metrics, blue_exposure = _side_features(state, blue, division, day)
    return np.r_[blue_metrics - red_metrics, red_exposure, blue_exposure]


def _output(targets, values):
    out = targets.loc[:, KEY_COLUMNS].copy()
    for index, column in enumerate(FEATURE_COLUMNS):
        out[column] = values[:, index]
    return out


def features_for_targets(state, targets):
    """Query one fitted state without mutating it; target dates may be later."""
    rows = _target_rows(targets)
    if any(row[0] < state.cutoff for row in rows):
        raise ValueError("Target date precedes fitted state cutoff")
    values = np.array([_feature_values(state, row) for row in rows]).reshape((-1, len(FEATURE_COLUMNS)))
    return _output(targets, values)


def build_smoothed_features(observations, targets, recipe=DEFAULT_RECIPE):
    """Daily chronological sweep, O(history + targets), with lazy decay.

    Return exactly the original three key columns/index/order and fourteen
    compact features. Winner and all other target fields are never accessed.
    """
    obs = _prepare_observations(observations)
    rows = _target_rows(targets)
    values = np.empty((len(rows), len(FEATURE_COLUMNS)))
    state = SmoothedState(min((r[0] for r in rows), default=pd.Timestamp("1970-01-01")), recipe)
    history = list(obs.itertuples(index=False))
    next_history = 0
    for index in sorted(range(len(rows)), key=lambda i: rows[i][0]):
        day = rows[index][0]
        while next_history < len(history) and history[next_history].date < day:
            _add(state, history[next_history])
            next_history += 1
        values[index] = _feature_values(state, rows[index])
    return _output(targets, values)

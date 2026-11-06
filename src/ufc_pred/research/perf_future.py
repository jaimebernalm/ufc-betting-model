"""As-of adapter for frozen B1/B2 performance features on scheduled bouts.

This module reads no files or labels. The caller supplies a source snapshot;
strictly earlier bout dates establish statistical availability, but the snapshot
cannot establish when each statistic was originally published. B1 uses daily
past history and B2 retains its original start-of-target-month fit convention.
"""

from __future__ import annotations

import re
from datetime import date, datetime

import numpy as np
import pandas as pd

from ufc_pred.ingest.identity import canonical, normalise
from ufc_pred.research.perf_adjusted import FEATURE_COLUMNS as ADJUSTED_COLUMNS
from ufc_pred.research.perf_adjusted import build_adjusted_features
from ufc_pred.research.perf_features import (
    KEY_COLUMNS,
    OBSERVATION_COLUMNS,
    _division,
    build_smoothed_features,
)

_TARGET_COLUMNS = (*KEY_COLUMNS, "weight_class")
_AVAILABILITY_CAVEAT = (
    "Availability is inferred from bout dates in the supplied retrospective snapshot; "
    "original statistic publication timestamps are unavailable. Later corrections to "
    "earlier bouts may therefore be present."
)


def _calendar_date(value, field):
    """Accept explicit ISO dates or naive midnight objects, never infer dates."""
    if isinstance(value, str):
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            raise ValueError(f"{field} must be an unambiguous YYYY-MM-DD calendar date")
    elif not isinstance(value, (date, datetime, pd.Timestamp, np.datetime64)):
        raise ValueError(f"{field} must be a timezone-naive calendar date")
    try:
        stamp = pd.Timestamp(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{field} must be a valid calendar date") from exc
    if pd.isna(stamp):
        raise ValueError(f"{field} must be a present calendar date")
    if stamp.tzinfo is not None:
        raise ValueError(f"{field} must be timezone-naive")
    if stamp != stamp.normalize():
        raise ValueError(f"{field} must be a calendar date without a time of day")
    return stamp


def _canonical_key(name, division):
    try:
        return normalise(canonical(name, division))
    except ValueError:
        return None  # unresolved target homonyms retain the existing missing flags


def _validate_targets(targets, as_of):
    missing = sorted(set(_TARGET_COLUMNS) - set(targets.columns))
    if missing:
        raise ValueError(f"Missing future target columns: {missing}")
    # A narrow projection ensures outcome/odds fields cannot reach either builder.
    safe = targets.loc[:, _TARGET_COLUMNS].copy()
    days = [_calendar_date(value, "target date") for value in safe.date]
    if any(day < as_of for day in days):
        raise ValueError("Every target date must be on or after as_of")
    seen = set()
    for position, (day, red, blue, division) in enumerate(
        zip(days, safe.R_fighter, safe.B_fighter, safe.weight_class, strict=True)
    ):
        for side, name in (("R", red), ("B", blue)):
            if pd.isna(name) or not str(name).strip():
                raise ValueError(f"Target row {position} {side}_fighter must be nonempty")
        division = _division(division)
        red_key, blue_key = _canonical_key(red, division), _canonical_key(blue, division)
        if red_key is not None and red_key == blue_key:
            raise ValueError(f"Target row {position} is a self-fight after canonical identity")
        # Missing homonym identities are not asserted equal, but duplicate raw
        # scheduled entries still fail closed. Corner orientation is irrelevant.
        keys = (red_key or normalise(red), blue_key or normalise(blue))
        bout_key = (day, tuple(sorted(keys)))
        if bout_key in seen:
            raise ValueError(f"Duplicate future bout key at target row {position}")
        seen.add(bout_key)
    return safe


def _plain(value):
    """Monthly fitting diagnostics are standalone JSON-compatible records."""
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (date, datetime, pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).date().isoformat()
    return value.item() if isinstance(value, np.generic) else value


def build_future_performance(observations, targets, as_of, *, include_adjusted=True):
    """Return frozen performance features and separate availability diagnostics.

    The mandatory as_of date is an exclusive history cutoff. Targets may be on
    that day or in later months. No later statistics enter B1 or B2; B2 further
    excludes the target month according to its existing historical recipe.
    Exact target keys, dtypes, order and index (including duplicate index labels)
    are retained. Duplicate canonical bout keys and resolved self-fights fail.
    """
    as_of = _calendar_date(as_of, "as_of")
    if not isinstance(include_adjusted, (bool, np.bool_)):
        raise ValueError("include_adjusted must be boolean")
    safe_targets = _validate_targets(targets, as_of)
    missing = sorted(set(OBSERVATION_COLUMNS) - set(observations.columns))
    if missing:
        raise ValueError(f"Missing future observation columns: {missing}")
    days = pd.Series(
        [_calendar_date(value, "observation date") for value in observations.date],
        dtype="datetime64[ns]",
    )
    available = (days < as_of).to_numpy()
    # Filter first: future counts/exposures are never validated or fitted.
    prior = observations.loc[available, OBSERVATION_COLUMNS].copy()
    prior["date"] = days.loc[available].to_numpy()
    output = build_smoothed_features(prior, safe_targets)
    monthly = []
    if include_adjusted:
        adjusted, fit_diagnostics = build_adjusted_features(prior, safe_targets, return_diagnostics=True)
        for column in ADJUSTED_COLUMNS:
            output[column] = adjusted[column].to_numpy()
        monthly = _plain(fit_diagnostics.to_dict("records"))
    latest = days.loc[available].max() if available.any() else pd.NaT
    excluded = observations.loc[~available, "bout_id"]
    diagnostics = {
        "as_of": as_of.date().isoformat(),
        "include_adjusted": bool(include_adjusted),
        "target_rows": len(safe_targets),
        "observation_rows": len(observations),
        "prior_observation_rows": int(available.sum()),
        "prior_bouts": int(prior.bout_id.nunique()),
        "excluded_as_of_rows": int((~available).sum()),
        "excluded_as_of_bouts": int(excluded.nunique()),
        "excluded_same_day_rows": int((days == as_of).sum()),
        "excluded_future_rows": int((days > as_of).sum()),
        "latest_prior_date": _plain(latest),
        "source_history_lag_days": None if pd.isna(latest) else int((as_of - latest).days),
        "labels_read": False,
        "availability_caveat": _AVAILABILITY_CAVEAT,
        "b2_monthly_diagnostics": monthly,
    }
    return output, diagnostics

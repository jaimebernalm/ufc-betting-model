"""Prior-only opponent-adjusted SIG/TD performance, independent of outcomes.

Each fighter-side count is a Poisson likelihood term with exposure in minutes.
These two sides are not two independent outcome observations. The fixed model is
log(rate) = intercept + division + fighter attack - opponent defense. Ridge
shrinks all effects (including division) toward zero; the intercept is pooled.
Weights are minutes * 0.5 ** (age_years / 4), with age measured at month start.
There is no dense Hessian, outcome input, score tuning, or global entity index.

No history, or a channel with no positive counts, has no finite Poisson intercept
MLE. No-data channels use neutral rates SIG=3/min, TD=0.1/min. All-zero channels
shrink that rate by 30 / (weighted_minutes + 30). Both use zero effects and
missing flags, with the reason recorded outside the feature table.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import PoissonRegressor

from ufc_pred.ingest.identity import canonical, normalise
from ufc_pred.research.perf_features import _division

ALPHA = 0.01
HALF_LIFE_YEARS = 4.0
MAX_ITER = 500
TOL = 1e-7
PRIOR_MINUTES = 30.0
NEUTRAL_RATES = {"sig": 3.0, "td": 0.1}
_METRICS = ("sig_attack", "sig_defense", "td_attack", "td_defense", "sig_lograte", "td_lograte")
FEATURE_COLUMNS = (
    [f"{side}_perf_adj_{metric}" for metric in _METRICS for side in ("R", "B")]
    + [f"perf_adj_{metric}_dif" for metric in _METRICS]
    + ["R_perf_adj_missing", "B_perf_adj_missing"]
)
_REQUIRED = ("bout_id", "event_id", "date", "fighter", "opponent", "division", "minutes",
             "sig_landed", "td_landed")


class AdjustedConvergenceError(RuntimeError):
    """A monthly fit failed; no feature state may be served from that fit."""


@dataclass(frozen=True)
class AdjustedChannel:
    intercept: float
    division_effects: dict[str, float]
    attack: dict[str, float]
    defense: dict[str, float]
    fallback_reason: str | None


@dataclass(frozen=True)
class AdjustedState:
    cutoff: pd.Timestamp
    sig: AdjustedChannel
    td: AdjustedChannel
    diagnostics: dict


def _fighter_key(name, division) -> str | None:
    """Match B1 identities; unresolved target homonyms receive missing effects."""
    try:
        return normalise(canonical(name, division))
    except ValueError:
        return None


def _dates(values) -> pd.Series:
    parsed = pd.to_datetime(values, errors="raise")
    if parsed.isna().any():
        raise ValueError("Performance observation/target dates must be present")
    if parsed.dt.tz is not None:
        parsed = parsed.dt.tz_convert("UTC").dt.tz_localize(None)
    return parsed


def _cutoff(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("cutoff must be present")
    return stamp.tz_convert("UTC").tz_localize(None) if stamp.tzinfo else stamp


def _validate_prior(observations: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    missing = sorted(set(_REQUIRED) - set(observations.columns))
    if missing:
        raise ValueError(f"Missing performance observation columns: {missing}")
    prior = observations.loc[:, _REQUIRED].copy()
    prior["date"] = _dates(prior["date"])
    if not (prior["date"] < cutoff).all():
        raise AssertionError("fit_adjusted_state requires every observation date strictly before cutoff")
    prior["division"] = prior["division"].map(_division)
    for col in ("fighter", "opponent"):
        if prior[col].isna().any() or prior[col].astype(str).str.strip().eq("").any():
            raise ValueError(f"{col} must be a nonempty canonical fighter identity")
        prior[col] = [
            normalise(canonical(name, division))
            for name, division in zip(prior[col], prior["division"], strict=True)
        ]
    for col in ("minutes", "sig_landed", "td_landed"):
        prior[col] = pd.to_numeric(prior[col], errors="raise")
        values = prior[col].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{col} must be finite")
        invalid = (values <= 0).any() if col == "minutes" else (values < 0).any()
        if invalid:
            raise ValueError(f"Invalid {col}: exposure must be positive and counts nonnegative")
        if col != "minutes" and (values != np.floor(values)).any():
            raise ValueError(f"{col} must contain integer counts")
    # Canonical reduction order makes floating-point optimization row-order invariant.
    return prior.sort_values(list(_REQUIRED), kind="mergesort").reset_index(drop=True)


def _neutral(channel: str, reason: str, weighted_minutes: float = 0.0) -> AdjustedChannel:
    rate = NEUTRAL_RATES[channel] * PRIOR_MINUTES / (PRIOR_MINUTES + weighted_minutes)
    return AdjustedChannel(float(np.log(rate)), {}, {}, {}, reason)


def fit_adjusted_state(observations: pd.DataFrame, cutoff) -> AdjustedState:
    """Fit a deterministic state, rejecting any row on/after the explicit cutoff.

    The caller must pass a prior-only slice; build_adjusted_features handles this
    slicing. A convergence warning, exhausted iteration limit or nonfinite fit
    raises AdjustedConvergenceError instead of silently exporting coefficients.
    """
    started = time.perf_counter()
    cutoff = _cutoff(cutoff)
    prior = _validate_prior(observations, cutoff)
    fighters = sorted(set(prior["fighter"]) | set(prior["opponent"]))
    divisions = sorted(set(prior["division"]) - {"__unknown__"})
    fighter_ids = {name: i for i, name in enumerate(fighters)}
    division_ids = {name: i for i, name in enumerate(divisions)}
    n, d, f = len(prior), len(divisions), len(fighters)
    row_ids, col_ids, entries = [], [], []
    for i, row in enumerate(prior.itertuples(index=False)):
        if row.division != "__unknown__":
            row_ids.append(i)
            col_ids.append(division_ids[row.division])
            entries.append(1.0)
        row_ids.extend((i, i))
        col_ids.extend((d + fighter_ids[row.fighter], d + f + fighter_ids[row.opponent]))
        entries.extend((1.0, -1.0))
    design = csr_matrix((entries, (row_ids, col_ids)), shape=(n, d + 2 * f), dtype=float)
    minutes = prior["minutes"].to_numpy(dtype=float)
    ages = (cutoff - prior["date"]).dt.total_seconds().to_numpy() / (365.25 * 86400)
    weights = minutes * np.power(0.5, ages / HALF_LIFE_YEARS)
    diagnostics = {
        "cutoff": cutoff, "n_observations": n, "n_bouts": prior["bout_id"].nunique(),
        "n_fighters": f, "n_divisions": d, "design_nnz": design.nnz,
        "raw_minutes": float(minutes.sum()), "weighted_minutes": float(weights.sum()),
        "latest_prior_date": prior["date"].max() if n else pd.NaT,
        "alpha": ALPHA, "half_life_years": HALF_LIFE_YEARS, "prior_minutes": PRIOR_MINUTES,
        "solver": "lbfgs", "max_iter": MAX_ITER, "tol": TOL,
    }
    channels = {}
    for channel in ("sig", "td"):
        count = prior[f"{channel}_landed"].to_numpy(dtype=float)
        reason = "no_prior_observations" if not n else "all_zero_counts" if not count.any() else None
        diagnostics[f"{channel}_fallback_reason"] = reason
        if reason:
            channels[channel] = _neutral(channel, reason, float(weights.sum()))
            diagnostics[f"{channel}_n_iter"] = 0
            diagnostics[f"{channel}_pooled_rate"] = float(np.exp(channels[channel].intercept))
            continue
        model = PoissonRegressor(alpha=ALPHA, fit_intercept=True, solver="lbfgs",
                                 max_iter=MAX_ITER, tol=TOL, warm_start=False)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(design, count / minutes, sample_weight=weights)
        if (any(issubclass(w.category, ConvergenceWarning) for w in caught)
                or model.n_iter_ >= MAX_ITER or not np.isfinite(model.coef_).all()
                or not np.isfinite(model.intercept_)):
            raise AdjustedConvergenceError(f"Unconverged {channel} Poisson fit at {cutoff.date()}")
        for warning in caught:
            if not issubclass(warning.category, ConvergenceWarning):
                warnings.warn(warning.message, warning.category, stacklevel=2)
        coef = model.coef_
        channels[channel] = AdjustedChannel(
            float(model.intercept_), {name: float(coef[i]) for name, i in division_ids.items()},
            {name: float(coef[d + i]) for name, i in fighter_ids.items()},
            {name: float(coef[d + f + i]) for name, i in fighter_ids.items()}, None,
        )
        diagnostics[f"{channel}_n_iter"] = int(model.n_iter_)
        diagnostics[f"{channel}_pooled_rate"] = float(np.exp(model.intercept_))
    diagnostics["fit_seconds"] = time.perf_counter() - started
    return AdjustedState(cutoff, channels["sig"], channels["td"], diagnostics)


def build_adjusted_features(
    observations: pd.DataFrame, targets: pd.DataFrame, *, return_diagnostics: bool = False,
):
    """Build each target month's features from dates strictly before month start.

    Output preserves exact date/fighter values, input index and row order. Only
    those three keys plus 20 features are exported; diagnostic counts/timing are
    returned separately when requested. `weight_class` (or `division`) on each
    target supplies its scheduled division; absent/unseen divisions use pooled
    intercepts. Unknown fighters have zero effects, with paired missing flags.
    Every signed *_dif is Blue minus Red, including matchup log rates.
    """
    keys = ["date", "R_fighter", "B_fighter"]
    missing = sorted(set(keys) - set(targets.columns))
    if missing:
        raise ValueError(f"Missing target columns: {missing}")
    out = targets.loc[:, keys].copy()
    target_dates = _dates(targets["date"])
    observation_dates = _dates(observations["date"])
    months = target_dates.dt.to_period("M").dt.to_timestamp()
    values = {col: np.zeros(len(targets), dtype=float) for col in FEATURE_COLUMNS}
    diagnostic_rows = []
    division_col = next((col for col in ("weight_class", "division") if col in targets), None)
    for month in sorted(months.unique()):
        state = fit_adjusted_state(observations.loc[(observation_dates < month).to_numpy()], month)
        diagnostic_rows.append(state.diagnostics)
        positions = np.flatnonzero((months == month).to_numpy())
        for pos in positions:
            row = targets.iloc[pos]
            division = _division(row[division_col]) if division_col else "__unknown__"
            for side, other in (("R", "B"), ("B", "R")):
                fighter = _fighter_key(row[f"{side}_fighter"], division)
                opponent = _fighter_key(row[f"{other}_fighter"], division)
                missing_effect = False
                for channel in ("sig", "td"):
                    fitted = getattr(state, channel)
                    attack = fitted.attack.get(fighter, 0.0)
                    defense = fitted.defense.get(fighter, 0.0)
                    values[f"{side}_perf_adj_{channel}_attack"][pos] = attack
                    values[f"{side}_perf_adj_{channel}_defense"][pos] = defense
                    values[f"{side}_perf_adj_{channel}_lograte"][pos] = (
                        fitted.intercept + fitted.division_effects.get(division, 0.0)
                        + attack - fitted.defense.get(opponent, 0.0)
                    )
                    missing_effect |= fighter not in fitted.attack or fitted.fallback_reason is not None
                values[f"{side}_perf_adj_missing"][pos] = float(missing_effect)
    for metric in _METRICS:
        values[f"perf_adj_{metric}_dif"] = (
            values[f"B_perf_adj_{metric}"] - values[f"R_perf_adj_{metric}"]
        )
    for col in FEATURE_COLUMNS:
        out[col] = values[col]
    diagnostics = pd.DataFrame(diagnostic_rows)
    return (out, diagnostics) if return_diagnostics else out

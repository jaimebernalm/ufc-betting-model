"""Compute skill_diff_mean / skill_diff_std for an upcoming fight.

Replicates the walk-forward-monthly protocol used during training:
- For a fight on date D in month M, fit NUTS on all fights with date < first-of-M
- Use that posterior to compute skill_diff for the upcoming fight

Caches the per-month posterior to `data/processed/skill_posteriors/`
so the slow NUTS fit only runs once per (target_month, prior_data_hash) pair.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ufc_pred.features.skill_v3 import (
    build_index,
    encode_fights,
    fit_nuts,
    recency_weights_for,
    skill_diff_for_fights,
)
from ufc_pred.features.skill_v3_pipeline import month_floor
from ufc_pred.ingest.identity import ALIASES, canonicalize_frame
from ufc_pred.ingest.identity import VERSION as IDENTITY_VERSION
from ufc_pred.paths import PROCESSED

POSTERIOR_CACHE = PROCESSED / "skill_posteriors"


def _prior_hash(prior: pd.DataFrame, **settings) -> str:
    """Stable hash of the prior slice — invalidates cache when data changes."""
    sig = pd.util.hash_pandas_object(
        prior[["date", "R_fighter", "B_fighter", "Winner", "weight_class"]]
        .sort_values(["date", "R_fighter", "B_fighter"])
        .reset_index(drop=True),
        index=False,
    ).values.tobytes()
    import jax
    import numpyro

    from ufc_pred.ingest import identity
    from ufc_pred.utils import time_splits

    recipe = {
        "version": 2,
        "identity": IDENTITY_VERSION,
        "aliases": ALIASES,
        "half_life_years": time_splits.RECENCY_HALF_LIFE_YEARS,
        "jax_version": jax.__version__,
        "numpyro_version": numpyro.__version__,
        "numpy_version": np.__version__,
        "jax_x64": bool(jax.config.x64_enabled),
        **settings,
    }
    import ast

    from ufc_pred.features import skill_v3

    code = "\n".join(
        ast.dump(ast.parse(Path(module.__file__).read_text())) for module in (skill_v3, time_splits, identity)
    ).encode()
    return hashlib.sha256(sig + json.dumps(recipe, sort_keys=True).encode() + code).hexdigest()[:24]


def fit_or_load_month_posterior(
    fights: pd.DataFrame,
    target_month: pd.Timestamp,
    *,
    num_warmup: int = 500,
    num_samples: int = 500,
    num_chains: int = 2,
    force: bool = False,
) -> tuple[dict, object]:
    """Return (samples, index) for the posterior fit on data < first-of-target_month."""
    POSTERIOR_CACHE.mkdir(parents=True, exist_ok=True)
    from ufc_pred.ingest.identity import verified_envelope

    fights = verified_envelope(fights)
    fights = fights[fights["Winner"].isin(["Red", "Blue"])].copy()
    fights["date"] = pd.to_datetime(fights["date"])

    m_start = month_floor(target_month)
    prior = (
        fights[fights["date"] < m_start]
        .sort_values(["date", "R_fighter", "B_fighter"])
        .reset_index(drop=True)
    )
    h = _prior_hash(
        prior,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        month=str(m_start.date()),
    )
    cache_file = POSTERIOR_CACHE / f"{m_start.date()}_{h}.pkl"

    if cache_file.exists() and not force:
        with open(cache_file, "rb") as f:
            payload = pickle.load(f)
        return payload["samples"], payload["index"]

    print(f"  [skill] fitting NUTS for {m_start.date()} (n_prior={len(prior)}) ...")
    index = build_index(prior)
    a, b, y = encode_fights(prior, index)
    w = recency_weights_for(prior["date"], reference_date=m_start - pd.Timedelta(days=1))
    samples = fit_nuts(
        a,
        b,
        y,
        index,
        weights=w,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=False,
        seed=int(m_start.year) * 100 + int(m_start.month),
    )
    temp = cache_file.with_suffix(".tmp")
    with open(temp, "wb") as f:
        pickle.dump({"samples": samples, "index": index, "hash": h}, f)
    temp.replace(cache_file)
    return samples, index


def attach_skill_for_upcoming(
    upcoming: pd.DataFrame,
    fights: pd.DataFrame,
    *,
    force: bool = False,
) -> pd.DataFrame:
    """Fill `skill_diff_mean` / `skill_diff_std` on `upcoming`.

    All rows in `upcoming` must share the same target month (typical case:
    one card). If they span months, this still works — it groups by month
    and fits each independently.
    """
    upcoming = canonicalize_frame(upcoming)
    upcoming["date"] = pd.to_datetime(upcoming["date"])
    months = upcoming["date"].apply(month_floor).unique()

    out_mean = np.full(len(upcoming), np.nan)
    out_std = np.full(len(upcoming), np.nan)

    for m in months:
        samples, index = fit_or_load_month_posterior(fights, m, force=force)
        mask = upcoming["date"].apply(month_floor) == m
        sub = upcoming.loc[mask]
        feats = skill_diff_for_fights(sub, samples, index)
        out_mean[mask.values] = feats["skill_diff_mean"].values
        out_std[mask.values] = feats["skill_diff_std"].values

    upcoming["skill_diff_mean"] = out_mean
    upcoming["skill_diff_std"] = out_std
    return upcoming

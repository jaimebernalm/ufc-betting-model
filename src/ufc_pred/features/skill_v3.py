"""v3 Bayesian hierarchical skill model (Bradley-Terry).

Per fighter `f`:
    skill[f] ~ Normal(wc_mean[wc(f)], wc_sigma[wc(f)])

Per fight `i` between fighters a, b:
    P(a beats b) = sigmoid(skill[a] - skill[b])
    y_i ~ Bernoulli weighted by recency w_i.

Outputs two features per fight:
    skill_diff_mean = posterior mean of (skill[R] - skill[B])
    skill_diff_std  = posterior std  of (skill[R] - skill[B])

PLAN.md §3. Anti-leak strategy: posterior for fight at date X is fit using
ONLY fights with date < X (walk-forward; see skill_v3_pipeline).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from numpyro.infer import MCMC, NUTS

from ufc_pred.utils.time_splits import RECENCY_HALF_LIFE_YEARS


@dataclass(frozen=True)
class SkillIndex:
    """Maps fighter names and weight classes to integer ids, with each
    fighter's modal weight class for the hierarchical prior."""

    fighter_to_id: dict[str, int]
    wc_to_id: dict[str, int]
    fighter_wc: np.ndarray  # shape (n_fighters,), value in [0, n_wc)

    @property
    def n_fighters(self) -> int:
        return len(self.fighter_to_id)

    @property
    def n_wc(self) -> int:
        return len(self.wc_to_id)


def build_index(fights: pd.DataFrame) -> SkillIndex:
    """Build a global fighter / weight-class index from ALL fights.

    Using the full history (train+val+test) for the *index only* is not a
    leak — the index just numbers fighters; no outcomes are read. Posteriors
    are still fit on prior-only slices per call.
    """
    names = pd.unique(pd.concat([fights["R_fighter"], fights["B_fighter"]]))
    names = sorted(str(n) for n in names)
    fighter_to_id = {n: i for i, n in enumerate(names)}

    wc_series = fights["weight_class"].fillna("Unknown").astype(str)
    wc_names = sorted(wc_series.unique())
    wc_to_id = {w: i for i, w in enumerate(wc_names)}

    # Modal weight class per fighter.
    long = pd.concat(
        [
            pd.DataFrame({"name": fights["R_fighter"], "wc": wc_series}),
            pd.DataFrame({"name": fights["B_fighter"], "wc": wc_series}),
        ],
        ignore_index=True,
    )
    long["name"] = long["name"].astype(str)
    modal = long.groupby("name")["wc"].agg(lambda s: s.mode().iat[0])
    fighter_wc = np.zeros(len(fighter_to_id), dtype=np.int32)
    for name, wc in modal.items():
        fighter_wc[fighter_to_id[name]] = wc_to_id[wc]
    return SkillIndex(fighter_to_id=fighter_to_id, wc_to_id=wc_to_id, fighter_wc=fighter_wc)


def encode_fights(fights: pd.DataFrame, index: SkillIndex) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (a_ids, b_ids, y) for fights with a known Red/Blue winner.

    Fights with fighters not in the index are dropped (shouldn't happen if
    the index was built from the full history)."""
    df = fights[fights["Winner"].isin(["Red", "Blue"])].copy()
    a = df["R_fighter"].astype(str).map(index.fighter_to_id).to_numpy()
    b = df["B_fighter"].astype(str).map(index.fighter_to_id).to_numpy()
    y = (df["Winner"].to_numpy() == "Red").astype(np.int32)
    mask = ~(pd.isna(a) | pd.isna(b))
    return a[mask].astype(np.int32), b[mask].astype(np.int32), y[mask]


def recency_weights_for(
    dates: pd.Series,
    reference_date: pd.Timestamp,
    half_life_years: float = RECENCY_HALF_LIFE_YEARS,
) -> np.ndarray:
    age_years = (reference_date - pd.to_datetime(dates)).dt.days / 365.25
    age_years = age_years.clip(lower=0)
    return np.power(0.5, age_years / half_life_years).to_numpy(dtype=np.float32)


def _model(
    a_ids: jnp.ndarray,
    b_ids: jnp.ndarray,
    fighter_wc: jnp.ndarray,
    n_fighters: int,
    n_wc: int,
    y: jnp.ndarray | None = None,
    weights: jnp.ndarray | None = None,
):
    wc_mean = numpyro.sample("wc_mean", dist.Normal(0.0, 1.0).expand([n_wc]))
    wc_sigma = numpyro.sample("wc_sigma", dist.HalfNormal(1.0).expand([n_wc]))

    # Non-centered parameterization for sampling geometry.
    with numpyro.plate("fighters", n_fighters):
        skill_raw = numpyro.sample("skill_raw", dist.Normal(0.0, 1.0))
    skill = numpyro.deterministic("skill", wc_mean[fighter_wc] + wc_sigma[fighter_wc] * skill_raw)

    logits = skill[a_ids] - skill[b_ids]

    if y is None:
        return
    log_prob = dist.Bernoulli(logits=logits).log_prob(y)
    if weights is not None:
        log_prob = log_prob * weights
    numpyro.factor("likelihood", log_prob.sum())


def fit_nuts(
    a_ids: np.ndarray,
    b_ids: np.ndarray,
    y: np.ndarray,
    index: SkillIndex,
    weights: np.ndarray | None = None,
    num_warmup: int = 1000,
    num_samples: int = 1000,
    num_chains: int = 2,
    seed: int = 0,
    progress_bar: bool = False,
) -> dict[str, np.ndarray]:
    """Run NUTS and return posterior samples as numpy arrays.

    Keys include 'skill' (n_samples, n_fighters), 'wc_mean', 'wc_sigma'.
    """
    numpyro.set_host_device_count(num_chains)
    kernel = NUTS(_model, target_accept_prob=0.9)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=progress_bar,
        chain_method="sequential",
    )
    mcmc.run(
        jax.random.PRNGKey(seed),
        a_ids=jnp.asarray(a_ids),
        b_ids=jnp.asarray(b_ids),
        fighter_wc=jnp.asarray(index.fighter_wc),
        n_fighters=index.n_fighters,
        n_wc=index.n_wc,
        y=jnp.asarray(y),
        weights=jnp.asarray(weights) if weights is not None else None,
    )
    samples = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
    samples["_extra_fields"] = mcmc.get_extra_fields() if False else None  # placeholder
    return samples


def skill_diff_for_fights(
    fights: pd.DataFrame, samples: dict[str, np.ndarray], index: SkillIndex
) -> pd.DataFrame:
    """For each fight, return posterior mean + std of (skill_R - skill_B).

    Fighters absent from `samples` (shouldn't happen with a global index) get
    NaN; the CatBoost integrator handles NaN natively.
    """
    skill = samples["skill"]  # (n_samples, n_fighters)
    a = fights["R_fighter"].astype(str).map(index.fighter_to_id).to_numpy()
    b = fights["B_fighter"].astype(str).map(index.fighter_to_id).to_numpy()

    out_mean = np.full(len(fights), np.nan, dtype=np.float64)
    out_std = np.full(len(fights), np.nan, dtype=np.float64)
    valid = ~(pd.isna(a) | pd.isna(b))
    a_v = a[valid].astype(np.int64)
    b_v = b[valid].astype(np.int64)
    diff = skill[:, a_v] - skill[:, b_v]  # (n_samples, n_valid)
    out_mean[valid] = diff.mean(axis=0)
    out_std[valid] = diff.std(axis=0)

    return pd.DataFrame(
        {
            "date": fights["date"].to_numpy(),
            "R_fighter": fights["R_fighter"].to_numpy(),
            "B_fighter": fights["B_fighter"].to_numpy(),
            "skill_diff_mean": out_mean,
            "skill_diff_std": out_std,
        }
    )


def rhat_summary(samples: dict[str, np.ndarray]) -> dict[str, float]:
    """Quick convergence summary for the skill parameters. Returns max R-hat
    across `skill` and the hyperparameters. NumPyro's get_samples flattens
    chains; for a real R-hat we'd need the un-flattened version, so this is
    a placeholder that reports posterior std stability as a proxy.
    """
    # Real R-hat needs chain structure preserved via mcmc.get_samples(group_by_chain=True).
    # Kept as a TODO; see fit_nuts. For smoke-test we rely on mcmc.print_summary().
    return {"placeholder": float("nan")}

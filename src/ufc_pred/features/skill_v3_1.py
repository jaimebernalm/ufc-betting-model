"""v3.1 Bayesian time-varying skill (Bradley-Terry + Gaussian random walk).

Each fighter has a per-career-fight skill trajectory:

    skill[f, 0]    ~ Normal(wc_mean[wc(f)], wc_sigma[wc(f)])
    skill[f, t]    = skill[f, t-1] + sqrt(Δt_years) * drift_sigma * z[f, t]

where Δt_years is the elapsed time between fighter f's (t-1)-th and t-th career
fight in the training data. Layoffs widen the posterior naturally — a 3-year
gap adds sqrt(3) ≈ 1.73× more drift than a 1-year gap. That's the structural
fix for the Jones-vs-Gane failure mode v3 surfaced.

Per fight i between (a at career-index t_a) and (b at career-index t_b):
    P(a beats b) = sigmoid(skill[a, t_a] - skill[b, t_b])
    y_i ~ Bernoulli(...)

No per-observation recency weight (the random walk replaces it).

Outputs same two features as v3 per fight: skill_diff_mean, skill_diff_std.
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

from ufc_pred.features.skill_v3 import SkillIndex


@dataclass(frozen=True)
class CareerEncoding:
    """Compact representation of a training slice with per-fighter career
    trajectories indexed by career fight."""

    obs_a_id: np.ndarray  # (n_obs,)  fighter ids
    obs_b_id: np.ndarray
    obs_a_t: np.ndarray  # (n_obs,)  career-fight index of a in this slice
    obs_b_t: np.ndarray
    y: np.ndarray  # (n_obs,)
    dt_years: (
        np.ndarray
    )  # (n_fighters, max_career_len) elapsed years between consecutive career fights; 0 at t=0
    career_len: np.ndarray  # (n_fighters,) number of career fights in this slice
    last_index: np.ndarray  # (n_fighters,) index to read for "current" skill (= max(career_len - 1, 0))
    max_career_len: int


def encode_careers(
    prior_fights: pd.DataFrame, index: SkillIndex, max_career_len_cap: int = 30
) -> CareerEncoding:
    """Walk prior fights in date order and build per-fighter career trajectories.

    Fighters with more career fights than `max_career_len_cap` are truncated to
    their MOST RECENT `max_career_len_cap` fights — older fights get dropped.

    The returned arrays are ALWAYS padded to shape (n_fighters, max_career_len_cap)
    so the model's JAX trace can be reused across monthly fits without recompilation.
    """
    df = prior_fights[prior_fights["Winner"].isin(["Red", "Blue"])].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    n_fighters = index.n_fighters
    # Per-fighter list of (fight_row_index, fight_date, is_red)
    history: list[list[tuple[int, pd.Timestamp, bool]]] = [[] for _ in range(n_fighters)]
    for i, row in enumerate(df.itertuples(index=False)):
        a = index.fighter_to_id.get(str(row.R_fighter))
        b = index.fighter_to_id.get(str(row.B_fighter))
        if a is None or b is None:
            continue
        history[a].append((i, row.date, True))
        history[b].append((i, row.date, False))

    # Truncate to last max_career_len_cap fights per fighter.
    for f in range(n_fighters):
        if len(history[f]) > max_career_len_cap:
            history[f] = history[f][-max_career_len_cap:]

    # Build (fight_index -> (a, a_t, b, b_t)) by re-walking trimmed histories.
    # For each fighter, position t = order in their trimmed history.
    fighter_position: list[dict[int, int]] = [{h[t][0]: t for t in range(len(h))} for h in history]

    obs_a_id, obs_b_id, obs_a_t, obs_b_t, ys = [], [], [], [], []
    for i, row in enumerate(df.itertuples(index=False)):
        a = index.fighter_to_id.get(str(row.R_fighter))
        b = index.fighter_to_id.get(str(row.B_fighter))
        if a is None or b is None:
            continue
        # Skip the fight if either fighter had it trimmed off.
        if i not in fighter_position[a] or i not in fighter_position[b]:
            continue
        obs_a_id.append(a)
        obs_b_id.append(b)
        obs_a_t.append(fighter_position[a][i])
        obs_b_t.append(fighter_position[b][i])
        ys.append(1 if row.Winner == "Red" else 0)

    career_len = np.array([max(1, len(h)) for h in history], dtype=np.int32)
    # Always pad to max_career_len_cap so JAX trace stays cached across months.
    max_career_len = max_career_len_cap

    # dt_years[f, t] = elapsed years between (t-1)-th and t-th fight of f.
    # dt_years[f, 0] = 0 by convention (no drift before the first fight).
    dt_years = np.zeros((n_fighters, max_career_len), dtype=np.float32)
    for f in range(n_fighters):
        h = history[f]
        for t in range(1, len(h)):
            delta = (h[t][1] - h[t - 1][1]).days / 365.25
            dt_years[f, t] = max(delta, 0.001)  # tiny floor for numerical safety

    # last_index[f] = index to read for "current" skill at target time.
    last_index = np.maximum(career_len - 1, 0).astype(np.int32)

    return CareerEncoding(
        obs_a_id=np.array(obs_a_id, dtype=np.int32),
        obs_b_id=np.array(obs_b_id, dtype=np.int32),
        obs_a_t=np.array(obs_a_t, dtype=np.int32),
        obs_b_t=np.array(obs_b_t, dtype=np.int32),
        y=np.array(ys, dtype=np.int32),
        dt_years=dt_years,
        career_len=career_len,
        last_index=last_index,
        max_career_len=max_career_len,
    )


def _model(
    obs_a_id,
    obs_b_id,
    obs_a_t,
    obs_b_t,
    y,
    dt_sqrt: jnp.ndarray,  # (n_fighters, max_career_len)
    fighter_wc: jnp.ndarray,
    n_fighters: int,
    n_wc: int,
    max_career_len: int,
):
    wc_mean = numpyro.sample("wc_mean", dist.Normal(0.0, 1.0).expand([n_wc]))
    wc_sigma = numpyro.sample("wc_sigma", dist.HalfNormal(1.0).expand([n_wc]))
    drift_sigma = numpyro.sample("drift_sigma", dist.HalfNormal(0.3))

    # Non-centered z; one per fighter per career step. Shape: (n_fighters, max_career_len).
    z = numpyro.sample("z", dist.Normal(0.0, 1.0).expand([n_fighters, max_career_len]))

    # init[f] = wc_mean[wc(f)] + wc_sigma[wc(f)] * z[f, 0]
    init = wc_mean[fighter_wc] + wc_sigma[fighter_wc] * z[:, 0]

    if max_career_len > 1:
        # drift[f, t] = drift_sigma * sqrt(dt_years[f, t]) * z[f, t]  for t >= 1
        drifts = drift_sigma * dt_sqrt[:, 1:] * z[:, 1:]
        cum = jnp.cumsum(drifts, axis=1)
        skill = jnp.concatenate([init[:, None], init[:, None] + cum], axis=1)
    else:
        skill = init[:, None]

    skill = numpyro.deterministic("skill", skill)
    logits = skill[obs_a_id, obs_a_t] - skill[obs_b_id, obs_b_t]
    numpyro.factor("likelihood", dist.Bernoulli(logits=logits).log_prob(y).sum())


def fit_nuts(
    enc: CareerEncoding,
    index: SkillIndex,
    num_warmup: int = 500,
    num_samples: int = 500,
    num_chains: int = 2,
    seed: int = 0,
    progress_bar: bool = False,
) -> dict[str, np.ndarray]:
    dt_sqrt = np.sqrt(enc.dt_years)
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
        obs_a_id=jnp.asarray(enc.obs_a_id),
        obs_b_id=jnp.asarray(enc.obs_b_id),
        obs_a_t=jnp.asarray(enc.obs_a_t),
        obs_b_t=jnp.asarray(enc.obs_b_t),
        y=jnp.asarray(enc.y),
        dt_sqrt=jnp.asarray(dt_sqrt),
        fighter_wc=jnp.asarray(index.fighter_wc),
        n_fighters=index.n_fighters,
        n_wc=index.n_wc,
        max_career_len=enc.max_career_len,
    )
    return {k: np.asarray(v) for k, v in mcmc.get_samples().items()}


def current_skill(samples: dict[str, np.ndarray], enc: CareerEncoding) -> tuple[np.ndarray, np.ndarray]:
    """Posterior mean + std of each fighter's CURRENT skill (last career index).

    Returns (mean, std), each of shape (n_fighters,).
    """
    skill = samples["skill"]  # (n_samples, n_fighters, max_career_len)
    n_fighters = skill.shape[1]
    last = enc.last_index  # (n_fighters,)
    s = skill[:, np.arange(n_fighters), last]  # (n_samples, n_fighters)
    return s.mean(axis=0), s.std(axis=0)


def skill_diff_for_target_fights(
    target_fights: pd.DataFrame,
    samples: dict[str, np.ndarray],
    enc: CareerEncoding,
    index: SkillIndex,
) -> pd.DataFrame:
    """Per target fight: posterior mean/std of (skill[R, t_R_last] - skill[B, t_B_last]).

    Uses each fighter's most-recent training career index (`enc.last_index`).
    For debutants (not seen in training), uses skill[f, 0] which is the prior draw."""
    s = samples["skill"]  # (n_samples, n_fighters, max_career_len)
    n_fighters = s.shape[1]
    last = enc.last_index  # (n_fighters,)

    # Skill at each fighter's last training career index.
    s_current = s[:, np.arange(n_fighters), last]  # (n_samples, n_fighters)

    a = target_fights["R_fighter"].astype(str).map(index.fighter_to_id).to_numpy()
    b = target_fights["B_fighter"].astype(str).map(index.fighter_to_id).to_numpy()

    out_mean = np.full(len(target_fights), np.nan, dtype=np.float64)
    out_std = np.full(len(target_fights), np.nan, dtype=np.float64)
    valid = ~(pd.isna(a) | pd.isna(b))
    a_v = a[valid].astype(np.int64)
    b_v = b[valid].astype(np.int64)
    diff = s_current[:, a_v] - s_current[:, b_v]
    out_mean[valid] = diff.mean(axis=0)
    out_std[valid] = diff.std(axis=0)

    return pd.DataFrame(
        {
            "date": target_fights["date"].to_numpy(),
            "R_fighter": target_fights["R_fighter"].to_numpy(),
            "B_fighter": target_fights["B_fighter"].to_numpy(),
            "skill_diff_mean": out_mean,
            "skill_diff_std": out_std,
        }
    )

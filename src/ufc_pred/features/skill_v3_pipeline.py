"""Walk-forward pipeline that produces v3 skill features without leakage.

For each calendar month M in the history, we fit the Bayesian skill model on
all fights with date < first_day_of_M and use the resulting posterior to
assign `skill_diff_mean` / `skill_diff_std` to every fight in month M.

This is PLAN.md §3.4's "refit posteriors monthly, accept small look-ahead
within the month" compromise. ~190 fits × ~10s each on CPU.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ufc_pred.features.skill_v3 import (
    skill_diff_for_fights,
)
from ufc_pred.ingest.kaggle_mdabbert import HISTORY_PARQUET
from ufc_pred.paths import PROCESSED

OUTPUT = PROCESSED / "skill_features_v3.parquet"


def month_floor(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(year=ts.year, month=ts.month, day=1)


def build(
    output_path: Path = OUTPUT,
    num_warmup: int = 500,
    num_samples: int = 500,
    num_chains: int = 2,
    min_train_fights: int = 50,
    history_path: Path = HISTORY_PARQUET,
) -> dict:
    """Build skill_features_v3.parquet via monthly walk-forward.

    Months with fewer than `min_train_fights` historical fights get NaN
    features (the model would just sample the prior). CatBoost handles NaN
    natively.
    """
    fights = pd.read_parquet(history_path)
    fights = fights[fights["Winner"].isin(["Red", "Blue"])].copy()
    fights["date"] = pd.to_datetime(fights["date"])
    fights = fights.sort_values("date").reset_index(drop=True)

    from ufc_pred.ingest.identity import canonicalize_frame

    fights = canonicalize_frame(fights)

    months = sorted({month_floor(d) for d in fights["date"]})
    out_frames: list[pd.DataFrame] = []
    fit_log: list[dict] = []

    for m_start in tqdm(months, desc="monthly fits"):
        prior_mask = fights["date"] < m_start
        target_mask = (fights["date"] >= m_start) & (fights["date"] < m_start + pd.offsets.MonthBegin(1))
        target = fights.loc[target_mask]
        if len(target) == 0:
            continue

        prior = fights.loc[prior_mask]
        n_prior = len(prior)
        if n_prior < min_train_fights:
            # Too early; emit NaN features.
            nan_df = pd.DataFrame(
                {
                    "date": target["date"].to_numpy(),
                    "R_fighter": target["R_fighter"].to_numpy(),
                    "B_fighter": target["B_fighter"].to_numpy(),
                    "skill_diff_mean": float("nan"),
                    "skill_diff_std": float("nan"),
                }
            )
            out_frames.append(nan_df)
            fit_log.append({"month": str(m_start.date()), "n_prior": n_prior, "fit": False})
            continue

        from ufc_pred.inference.skill_for_upcoming import fit_or_load_month_posterior

        samples, index = fit_or_load_month_posterior(
            fights, m_start, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains
        )
        feats = skill_diff_for_fights(target, samples, index)
        out_frames.append(feats)
        fit_log.append({"month": str(m_start.date()), "n_prior": n_prior, "fit": True})

    result = pd.concat(out_frames, ignore_index=True)
    result.to_parquet(output_path, index=False)

    return {
        "rows": int(len(result)),
        "n_months_fit": int(sum(1 for r in fit_log if r["fit"])),
        "n_months_skipped": int(sum(1 for r in fit_log if not r["fit"])),
        "output": str(output_path),
    }


if __name__ == "__main__":
    stats = build()
    for k, v in stats.items():
        print(f"{k}: {v}")

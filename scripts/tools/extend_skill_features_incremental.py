"""Incrementally extend skill_features_v3.parquet to cover new months.

Identifies months in fights.parquet that aren't yet in skill_features_v3,
fits the Bayesian skill model walk-forward for those months only, and
appends to the existing parquet.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

from ufc_pred.features.skill_v3 import (
    build_index,
    encode_fights,
    fit_nuts,
    recency_weights_for,
    skill_diff_for_fights,
)
from ufc_pred.features.skill_v3_pipeline import OUTPUT, month_floor
from ufc_pred.ingest.kaggle_mdabbert import HISTORY_PARQUET


def main():
    fights = pd.read_parquet(HISTORY_PARQUET)
    fights = fights[fights["Winner"].isin(["Red", "Blue"])].copy()
    fights["date"] = pd.to_datetime(fights["date"])
    fights = fights.sort_values("date").reset_index(drop=True)

    existing = pd.read_parquet(OUTPUT)
    existing["date"] = pd.to_datetime(existing["date"])
    existing_keys = set(zip(existing["date"], existing["R_fighter"], existing["B_fighter"], strict=False))

    # Build full fighter index from ALL fights (forward-looking ids OK; we
    # only train on the prior slice anyway).
    index = build_index(fights)

    all_months = sorted({month_floor(d) for d in fights["date"]})
    existing_months = sorted({month_floor(d) for d in existing["date"]})
    # A month needs (re)fitting if it's absent entirely OR contains fights
    # missing from the existing parquet (e.g. the month was originally fit
    # mid-month and a later event in that month arrived afterwards).
    missing_mask = [
        (d, r, b) not in existing_keys
        for d, r, b in zip(fights["date"], fights["R_fighter"], fights["B_fighter"], strict=False)
    ]
    months_missing = {month_floor(d) for d, miss in zip(fights["date"], missing_mask, strict=False) if miss}
    new_months = sorted(months_missing | (set(all_months) - set(existing_months)))
    print(f"Existing months: {len(existing_months)} (latest: {existing_months[-1].date()})")
    print(f"Months to (re)fit: {len(new_months)} {[str(m.date()) for m in new_months]}")
    if not new_months:
        print("Nothing to do.")
        return
    # Drop rows for months being refit — they are fully regenerated below.
    existing = existing[~existing["date"].apply(month_floor).isin(set(new_months))].copy()

    new_frames = []
    for m_start in new_months:
        prior_mask = fights["date"] < m_start
        target_mask = (fights["date"] >= m_start) & (fights["date"] < m_start + pd.offsets.MonthBegin(1))
        target = fights.loc[target_mask]
        if len(target) == 0:
            continue
        prior = fights.loc[prior_mask]
        print(f"  fitting {m_start.date()}  n_prior={len(prior)}  n_target={len(target)}")
        a, b, y = encode_fights(prior, index)
        w = recency_weights_for(prior["date"], reference_date=m_start - pd.Timedelta(days=1))
        samples = fit_nuts(
            a,
            b,
            y,
            index,
            weights=w,
            num_warmup=500,
            num_samples=500,
            num_chains=2,
            progress_bar=False,
            seed=int(m_start.year) * 100 + int(m_start.month),
        )
        feats = skill_diff_for_fights(target, samples, index)
        new_frames.append(feats)

    new_df = pd.concat(new_frames, ignore_index=True)
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.sort_values("date").reset_index(drop=True)
    combined.to_parquet(OUTPUT, index=False)
    print(f"\nWrote {OUTPUT}")
    print(f"  rows: {len(existing)} -> {len(combined)}  (+{len(new_df)})")
    print(f"  date range: {combined.date.min().date()} -> {combined.date.max().date()}")


if __name__ == "__main__":
    main()

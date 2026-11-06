"""Dev-period (2016-2023) comparison of every candidate OOF that exists.

Merges artifacts/r2026_10/claude/oof_dev_*.parquet (+ A0 = E0 stats_v2 restricted to dev years),
adds stacked market calibrations fitted on earlier years only:
  M0     = sigma(a * logit q)                      (FLB recalibration of the line)
  M1_<m> = sigma(a * logit q + b * logit p_m)      (does the model add to the line?)
and prints gate G1 (delta log-loss vs raw q) and G2 (flat ROI at offered odds with vig).
Never reads holdout years.
"""

from __future__ import annotations

import argparse
import glob

import pandas as pd

from ufc_pred.research.harness import (
    DEV_YEARS,
    KEYS,
    OUT,
    load_base,
    logit,
    report,
    roi_by_bucket,
    sigmoid,
    walk_forward_stack,
)

DST = OUT / "claude"


def load(split: str = "dev") -> pd.DataFrame:
    frames = []
    years = DEV_YEARS if split == "dev" else [2024, 2025, 2026]
    e0 = pd.read_parquet(OUT / "e0_oof.parquet").rename(columns={"p_stats_v2": "p_A0"})
    e0 = e0[e0.date.dt.year.isin(years)][KEYS + ["p_A0"]]
    frames.append(e0)
    for f in sorted(glob.glob(str(DST / f"oof_{split}_*.parquet"))):
        o = pd.read_parquet(f)
        frames.append(o[KEYS + [c for c in o if c.startswith("p_")]])
    base = load_base(fill_bfo_from="2024-01-01") if split == "holdout" else load_base()
    base = base[base.year.isin(years)][KEYS + ["card", "year", "y", "q", "R_odds", "B_odds", "debut"]]
    for fr in frames:
        base = base.merge(fr, on=KEYS, how="left", validate="one_to_one")
    if split == "dev":
        assert base.year.max() <= max(DEV_YEARS), "dev report must not read holdout years"
    base = base[base.q.notna()].reset_index(drop=True)
    if "p_A0" in base:  # deployed transform of the control: sharpening T=1.25
        base["p_A0T"] = sigmoid(1.25 * logit(base.p_A0))
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buckets", default="")
    args = ap.parse_args()
    df = load()
    models = [c for c in df if c.startswith("p_")]
    df["p_M0"] = walk_forward_stack(df, ["q"], min_years=2)
    for m in models:
        if not m.startswith("p_R"):
            df[f"p_M1_{m[2:]}"] = walk_forward_stack(df, ["q", m], min_years=2)
    pcols = [c for c in df if c.startswith("p_")]
    # stacks start after a 2-year warmup; compare everything on the common support
    ev = df.dropna(subset=pcols).reset_index(drop=True)
    print(
        f"dev common support: {len(ev)} fights, {ev.card.nunique()} cards, years {sorted(ev.year.unique())}"
    )
    ll, roi = report(ev, pcols)
    pd.set_option("display.width", 200)
    print(ll.round(4).to_string(index=False))
    print(roi.round(4).to_string(index=False))
    ll.to_csv(DST / "dev_ll.csv", index=False)
    roi.to_csv(DST / "dev_roi.csv", index=False)
    for m in [b for b in args.buckets.split(",") if b]:
        print(f"\nROI by price bucket of the side taken: {m}")
        print(roi_by_bucket(ev, m).round(3).to_string())
    print("\nper-year delta log-loss vs q")
    from ufc_pred.research.harness import logloss

    py = (
        ev.assign(**{c: logloss(ev[c], ev.y) - logloss(ev.q, ev.y) for c in pcols})
        .groupby("year")[pcols]
        .mean()
    )
    print(py.round(4).to_string())


if __name__ == "__main__":
    main()

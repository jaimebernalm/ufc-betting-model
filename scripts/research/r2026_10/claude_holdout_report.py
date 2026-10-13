"""Single holdout look (test-years 2024-2026) for the frozen finalists and fixed controls.

Run only after artifacts/r2026_10/manifest.json lists the frozen finalists.
Reports G1/G2 on three slices: Kaggle-original odds, BFO-extended odds, all.
Pre-declared economic rule: EV > 0.02 at the offered price, non-debut fights.
"""

from __future__ import annotations

import argparse
import glob
import json

import numpy as np
import pandas as pd

from ufc_pred.research.harness import KEYS, OUT, delta_ll, load_base, logit, roi_summary, sigmoid

DST = OUT / "claude"
YEARS = [2024, 2025, 2026]


def load(names: list[str]) -> pd.DataFrame:
    base = load_base(fill_bfo_from="2024-01-01")
    base = base[base.year.isin(YEARS) & base.q.notna()][KEYS + ["card", "year", "y", "q", "R_odds", "B_odds", "debut", "odds_source"]]
    e0 = pd.read_parquet(OUT / "e0_oof.parquet").rename(columns={"p_stats_v2": "p_A0"})
    base = base.merge(e0[KEYS + ["p_A0"]], on=KEYS, how="left")
    base["p_A0T"] = sigmoid(1.25 * logit(base.p_A0))
    for n in names:
        o = pd.read_parquet(DST / f"oof_holdout_{n}.parquet")
        base = base.merge(o[KEYS + ["p_" + n]], on=KEYS, how="left", validate="one_to_one")
    return base.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", required=True, help="comma list of holdout OOF names (finalists + ablations)")
    args = ap.parse_args()
    manifest = json.loads((OUT / "manifest.json").read_text())
    assert "frozen_finalists" in manifest and isinstance(manifest["frozen_finalists"], dict), "freeze first"
    names = args.names.split(",")
    df = load(names)
    pcols = ["p_" + n for n in names] + ["p_A0", "p_A0T"]
    slices = {"all": df, "kaggle": df[df.odds_source.eq("kaggle")], "bfo_ext": df[df.odds_source.ne("kaggle")]}
    rows_ll, rows_roi = [], []
    for sname, s in slices.items():
        for c in pcols:
            r = delta_ll(s, c)
            r["slice"] = sname
            rows_ll.append(r)
            eco = s[~s.debut]
            for thr in [0.0, 0.02, 0.04]:
                rr = roi_summary(eco, c, thr)
                rr["slice"] = sname
                rows_roi.append(rr)
            rr = roi_summary(eco, c, 0.0, rule="raw3")
            rr["slice"] = sname
            rows_roi.append(rr)
    ll, roi = pd.DataFrame(rows_ll), pd.DataFrame(rows_roi)
    pd.set_option("display.width", 240)
    print(ll.round(4).to_string(index=False))
    print(roi.round(4).to_string(index=False))
    ll.to_csv(DST / "holdout_ll.csv", index=False)
    roi.to_csv(DST / "holdout_roi.csv", index=False)
    print("\nper-year (all slice, non-debut, EV>.02):")
    for c in pcols:
        yr = {int(y): round(roi_summary(g[~g.debut], c, 0.02).get("roi", np.nan), 3) for y, g in df.groupby("year")}
        print(c, yr)
    df.to_parquet(DST / "holdout_eval.parquet", index=False)


if __name__ == "__main__":
    main()

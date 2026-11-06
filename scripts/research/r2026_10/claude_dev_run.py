"""Run pre-declared candidates through the common walk-forward harness.

Usage:
  PYTHONPATH=src python scripts/research/r2026_10/claude_dev_run.py --cands A1,R0,R1 --seeds 5
  (--holdout runs test-years 2024-2026 instead of dev 2016-2023; only for the final single look)

Each candidate's OOF predictions go to artifacts/r2026_10/claude/oof_<split>_<name>.parquet.
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from ufc_pred.research import anchored as A
from ufc_pred.research.harness import DEV_YEARS, OUT, WalkForward, load_base

DST = OUT / "claude"


def candidates(threads: int, extra: pd.DataFrame | None):
    perf_cols = [
        c for c in (extra.columns if extra is not None else []) if c not in ("date", "R_fighter", "B_fighter")
    ]
    perf_signed = []  # extra columns follow the R_/B_ and *_dif conventions; swap handles them
    return {
        "A1": (A.make_stats_fn(threads=threads), False),
        "A1s": (
            A.make_stats_fn(
                params=dict(iterations=400, learning_rate=0.03, depth=4, l2_leaf_reg=10, rsm=0.5),
                threads=threads,
            ),
            False,
        ),
        "R0": (A.make_resid_fn(threads=threads), True),
        "R1": (A.make_resid_fn(threads=threads, logit_q_feature=True), True),
        "R2": (A.make_resid_fn(threads=threads, logit_q_feature=True, signed_extra=perf_signed), True),
        "MF": (A.make_flb_fn(), True),
        "R3": (A.make_resid_fn(threads=threads, logit_q_feature=True, flb_baseline=True), True),
        "R3c": (
            A.make_resid_fn(
                threads=threads,
                logit_q_feature=True,
                flb_baseline=True,
                keep=A.GLM_DIFFS + ["weight_class", "gender", "title_bout", "no_of_rounds"],
            ),
            True,
        ),
        "G0": (A.make_glm_resid_fn(), True),
        "G0n": (A.make_glm_resid_fn(flb=False), True),
        "G1": (A.make_glm_resid_fn(cols=A.GLM_DIFFS + [c for c in perf_cols if c.endswith("_dif")]), True),
        "R4": (
            A.make_resid_fn(
                threads=threads, logit_q_feature=True, flb_baseline=True, signed_extra=perf_signed
            ),
            True,
        ),
        # extra columns follow the R_/B_ and *_dif conventions, so swap handling is automatic
        "R6": (A.make_resid_fn(threads=threads, logit_q_feature=True, flb_baseline=True), True),
        "G2": (A.make_glm_resid_fn(cols=A.GLM_DIFFS + [c for c in perf_cols if c.endswith("_dif")]), True),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument(
        "--perf",
        default=None,
        help="comma-separated parquets with extra per-fight features keyed by date,R_fighter,B_fighter",
    )
    ap.add_argument("--tag", default="", help="suffix for output names when extra features are used")
    args = ap.parse_args()
    DST.mkdir(parents=True, exist_ok=True)
    df = load_base(fill_bfo_from="2024-01-01") if args.holdout else load_base()
    extra = None
    if args.perf:
        frames = [pd.read_parquet(p) for p in args.perf.split(",")]
        extra = frames[0]
        for fr in frames[1:]:
            extra = extra.merge(fr, on=["date", "R_fighter", "B_fighter"], how="outer", validate="one_to_one")
        n0 = len(df)
        df = df.merge(extra, on=["date", "R_fighter", "B_fighter"], how="left", validate="one_to_one")
        assert len(df) == n0
    years = [2024, 2025, 2026] if args.holdout else DEV_YEARS
    split = "holdout" if args.holdout else "dev"
    cands = candidates(args.threads, extra)
    uses_extra = {"R2", "R4", "R6", "G1", "G2"}
    for name in args.cands.split(","):
        # a candidate without extra features must never see --perf columns (and vice versa)
        assert (name in uses_extra) == (extra is not None), f"{name}: --perf mismatch"
        fn, need_odds = cands[name]
        t0 = time.monotonic()
        wf = WalkForward(years=years, seeds=list(range(args.seeds)), require_odds=False)
        oof = wf.run(df, fn, name + args.tag)
        path = DST / f"oof_{split}_{name}{args.tag}.parquet"
        oof.to_parquet(path, index=False)
        side = {
            "extra_files": args.perf,
            "extra_columns": [
                c
                for c in (extra.columns if extra is not None else [])
                if c not in ("date", "R_fighter", "B_fighter")
            ],
            "feature_names": getattr(fn, "feature_names", None),
            "glm_fits": getattr(fn, "fits", None),
        }
        import json

        path.with_suffix(".features.json").write_text(json.dumps(side, indent=1, default=str))
        print(f"{name}: {len(oof)} rows -> {path.name} ({time.monotonic() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

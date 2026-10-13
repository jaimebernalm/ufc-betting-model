"""Robustness of R3 on dev only: hyper-parameter neighbourhood, per-year and per-seed economics.

Not a search: the R3 recipe stays as declared. A fragile optimum (big swings for small
changes) would be a red flag for the holdout.
"""

import pandas as pd

from ufc_pred.research import anchored as A
from ufc_pred.research.harness import DEV_YEARS, OUT, WalkForward, delta_ll, load_base, roi_summary

GRID = {
    "it150": dict(A.RESID_V0, iterations=150),
    "it600": dict(A.RESID_V0, iterations=600),
    "d2": dict(A.RESID_V0, depth=2),
    "d4": dict(A.RESID_V0, depth=4),
    "l2_5": dict(A.RESID_V0, l2_leaf_reg=5),
    "l2_60": dict(A.RESID_V0, l2_leaf_reg=60),
}


def main():
    df = load_base()
    wf = WalkForward(years=DEV_YEARS, seeds=list(range(5)))
    rows = []
    for tag, params in GRID.items():
        fn = A.make_resid_fn(params=params, logit_q_feature=True, flb_baseline=True, threads=6)
        oof = wf.run(df, fn, "R3_" + tag, verbose=False)
        oof = oof[oof.year >= 2018]
        oof.to_parquet(OUT / f"claude/sens_R3_{tag}.parquet", index=False)
        r = delta_ll(oof.dropna(subset=["q"]), "p_R3_" + tag)
        eco = oof[~oof.debut & oof.q.notna()]
        r.update({k: v for k, v in roi_summary(eco, "p_R3_" + tag, 0.02).items() if k in ("bets", "roi", "roi_lo", "roi_hi", "pnl_per_opp")})
        rows.append(r)
        print(pd.DataFrame([r]).round(4).to_string(index=False), flush=True)
    pd.DataFrame(rows).to_csv(OUT / "claude/r3_sensitivity.csv", index=False)


if __name__ == "__main__":
    main()

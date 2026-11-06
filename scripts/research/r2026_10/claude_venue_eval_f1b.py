"""Kalshi venue evaluation of the co-candidate F1+B1+B2 (descriptive, post hoc).

Same protocol as claude_venue_eval.py (trained once on fights before 2026-01-01, 10 seeds, scored
on the accredited Kalshi opportunities, both anchors), with the B1/B2 columns added. Fights up to
2026-09-12 take B1/B2 from Codex's frozen walk-forward parquets; later fights are rebuilt as of
the card day from the pinned UFCStats snapshot (anchored_shadow.target_performance, prior bouts
only). F1 is refitted alongside and must reproduce the saved venue predictions exactly.

2024-2026 is already spent, and F1+B1+B2 was not part of the original venue look: these numbers
are descriptive only and are not evidence for promotion.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from claude_venue_eval import BFO, OPP, capture_rows, summarize  # noqa: E402

from ufc_pred.research import anchored as A  # noqa: E402
from ufc_pred.research import anchored_shadow as S  # noqa: E402
from ufc_pred.research.harness import KEYS, OUT, bfo_exact, load_base, logloss  # noqa: E402

SEEDS = 10


def main():
    opp = pd.read_parquet(OPP)
    base = load_base(fill_bfo_from="2024-01-01")
    for extra in [S.MKT_EXTRA, S.PERF_B1, S.PERF_B2]:
        base = base.merge(pd.read_parquet(extra), on=KEYS, how="left", validate="one_to_one")
    perf_cols = [
        c
        for c in pd.read_parquet(S.PERF_B1).columns.union(pd.read_parquet(S.PERF_B2).columns)
        if c not in KEYS
    ]
    caps = capture_rows(opp, base)
    rebuilt = []
    for i in range(len(caps)):
        row = caps.iloc[[i]].reset_index(drop=True)
        perf, diag = S.target_performance(row, row.date.iloc[0])
        assert diag["source_history_lag_days"] <= S.MAX_B_HISTORY_LAG_DAYS, diag
        assert pd.Timestamp(diag["latest_prior_date"]) < row.date.iloc[0], diag
        rebuilt.append(pd.concat([row, perf[perf_cols].reset_index(drop=True)], axis=1))
    caps = pd.concat(rebuilt, ignore_index=True) if rebuilt else caps
    print(f"capture rows rebuilt from snapshot {S.current_snapshot().name}: {len(caps)}")
    feats = pd.concat([base, caps], ignore_index=True)
    d = opp.merge(feats.drop(columns=["y", "card", "year"], errors="ignore"), on=KEYS, how="left")
    d["q_book"] = d.q
    bfo = bfo_exact() if BFO.exists() else None
    if bfo is not None and "q_close_consensus_R" in bfo:
        d = d.merge(bfo[KEYS + ["q_close_consensus_R"]], on=KEYS, how="left")
        d["q_book"] = d.q_close_consensus_R.fillna(d.q)
    print(f"opportunities {len(d)}; with B1/B2 {d[perf_cols].notna().any(axis=1).sum()}")

    cut = pd.Timestamp("2026-01-01")
    train_b = base[base.date < cut].reset_index(drop=True)
    train_f1 = train_b.drop(columns=perf_cols)
    fn = A.make_resid_fn(logit_q_feature=True, flb_baseline=True)
    rows = []
    for anchor in ["q_book", "q_kalshi_mid"]:
        t = d[d[anchor].notna() & d.R_wins.notna()].copy()
        t["q"] = t[anchor]
        t["has_odds"] = True
        t["Winner"] = np.where(t.y == 1, "Red", "Blue")
        t = t.reset_index(drop=True)
        t["p_F1_R6_mkt"] = np.mean([fn(train_f1, t.drop(columns=perf_cols), s) for s in range(SEEDS)], axis=0)
        t["p_F1B"] = np.mean([fn(train_b, t, s) for s in range(SEEDS)], axis=0)
        saved = pd.read_parquet(OUT / f"claude/venue_kalshi_{anchor}.parquet")[KEYS + ["p_F1_R6_mkt"]]
        chk = t[KEYS + ["p_F1_R6_mkt"]].merge(saved, on=KEYS, suffixes=("", "_saved"), validate="one_to_one")
        diff = float((chk.p_F1_R6_mkt - chk.p_F1_R6_mkt_saved).abs().max())
        print(f"{anchor}: n={len(t)}; F1 parity vs saved venue run max|diff|={diff:.2e}")
        assert len(chk) == len(t) and diff < 1e-9
        for c in ["p_F1_R6_mkt", "p_F1B"]:
            r = summarize(t, c, 0.02)
            r.update(
                anchor=anchor,
                n=len(t),
                ll=float(logloss(t[c], t.y).mean()),
                ll_kmid=float(logloss(t.q_kalshi_mid, t.y).mean()),
            )
            rows.append(r)
        t[KEYS + ["q", "p_F1_R6_mkt", "p_F1B"]].to_parquet(
            OUT / f"claude/venue_kalshi_f1b_{anchor}.parquet", index=False
        )
    pd.set_option("display.width", 220)
    print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()

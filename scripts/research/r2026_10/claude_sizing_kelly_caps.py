"""Kelly fraction x per-fight cap x per-card cap sweep (user request 2026-10-13).

Pre-stated before running: the choice is made on 2016-2023 only, maximising mean log-growth per
card subject to a cap on P(drawdown >= 50% within 100 cards) estimated by resampling cards.
Three risk tolerances are reported (5%, 10%, 25%). 2024-2026 and Kalshi live are shown for the
chosen policies only as descriptive checks (both periods were already looked at).
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

import claude_sizing_opt as Z

KS = [0.25, 0.33, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
CAPF = [0.02, 0.05, 0.10, 0.15, 0.25, 0.50, 1.0]
CAPC = [0.15, 0.30, 0.50, 0.75, 1.0]
ENTRY = {"EV>3%, sin techo": (0.03, 1.0), "EV>2%, techo 10c": (0.02, 0.10)}
TOL = [0.05, 0.10, 0.25]


def sweep(ev_dev, thr, ceil):
    rows = []
    for k, cf, cc in itertools.product(KS, CAPF, CAPC):
        lg, n = ev_dev.log_growth(thr, k, cf, cc, ceil)
        g = float(np.mean(lg)) if np.isfinite(lg).all() else -np.inf
        risk = ev_dev.risk(lg) if np.isfinite(g) else dict(p_dd50=1.0, p_ruin90=1.0, p10_logw=-np.inf, med_logw=-np.inf)
        rows.append(dict(k=k, capf=cf, capc=cc, g=g, bets=n, **risk))
    return pd.DataFrame(rows)


def cap_txt(x):
    return "sin" if x >= 1 else f"{x:.0%}"


def main():
    pd.set_option("display.width", 250)
    books, kal = Z.load_books(), Z.load_kalshi()
    out = []
    for name in ["F1B", "F1"]:
        pcol = Z.MODELS[name]
        bb = Z.bets_books(books.dropna(subset=[pcol]), pcol)
        bb["year"] = pd.to_datetime(bb.date).dt.year
        dev, conf = bb[bb.year <= 2023], bb[bb.year >= 2024]
        bk = Z.bets_kalshi(kal.dropna(subset=[pcol]), pcol)
        for ename, (thr, ceil) in ENTRY.items():
            res = sweep(Z.Evaluator(dev), thr, ceil)
            res.to_csv(Z.OUT / f"claude/sizing_kelly_caps_{name}_{thr:.2f}.csv", index=False)
            print(f"\n===== {name} | {ename} | best caps per Kelly fraction (chosen on 2016-2023, P(dd>=50%) <= 10%)")
            for k in KS:
                ok = res[(res.k == k) & (res.p_dd50 <= 0.10)]
                tag = ""
                if len(ok) == 0:
                    ok, tag = res[res.k == k], " (ninguna cumple; la de menor riesgo)"
                    best = ok.loc[ok.p_dd50.idxmin()]
                else:
                    best = ok.loc[ok.g.idxmax()]
                pol = (thr, k, best.capf, best.capc, ceil)
                d, c, kk = Z.score(dev, pol), Z.score(conf, pol), Z.score(bk, pol)
                nocap = Z.score(conf, (thr, k, 1.0, 1.0, ceil))
                print(f"  k={k:<4} topes {cap_txt(best.capf):>4}/{cap_txt(best.capc):>4}{tag} | dev {100*d['final']:7.0f}$ dd{d['max_dd']:4.0%} P50 {best.p_dd50:4.0%}"
                      f" | 24-26 {100*c['final']:5.0f}$ dd{c['max_dd']:4.0%} P50 {c['p_dd50']:4.0%} | Kalshi {100*kk['final']:4.0f}$"
                      f" || sin topes 24-26 {100*nocap['final']:5.0f}$ dd{nocap['max_dd']:4.0%}")
            for tol in TOL:
                ok = res[res.p_dd50 <= tol]
                best = ok.loc[ok.g.idxmax()]
                pol = (thr, best.k, best.capf, best.capc, ceil)
                d, c, kk = Z.score(dev, pol), Z.score(conf, pol), Z.score(bk, pol)
                out.append(dict(model=name, entry=ename, tolerance=tol, k=best.k, capf=best.capf, capc=best.capc,
                                dev_final=100 * d["final"], dev_dd=d["max_dd"], conf_final=100 * c["final"], conf_dd=c["max_dd"],
                                conf_p_dd50=c["p_dd50"], kalshi_final=100 * kk["final"], kalshi_dd=kk["max_dd"]))
    picks = pd.DataFrame(out)
    print("\n===== picks per risk tolerance (chosen on 2016-2023)")
    print(picks.round(3).to_string(index=False))
    picks.to_csv(Z.OUT / "claude/sizing_kelly_caps_picks.csv", index=False)


if __name__ == "__main__":
    main()

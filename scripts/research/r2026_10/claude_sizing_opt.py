"""Bet-sizing search with walk-forward selection (EV threshold, Kelly fraction, caps, ceiling).

Question: would a more aggressive policy than 1/4-Kelly with 2%/15% caps grow faster without a
real risk of ruin? Pre-registered before running:

* Grid: threshold x Kelly fraction x per-fight cap x per-card cap x edge ceiling (2,940 policies).
* Equity fixed per card; offered book odds with vig (Kalshi: captured ask + 0.07 p(1-p) fee).
  Debut fights are excluded (the live system cannot bet them).
* Risk is measured by resampling whole cards (500 paths of 100 cards, common random numbers).
* Two selection rules, both reported:
    growth : max mean log-growth per card, subject to P(drawdown >= 50% in 100 cards) <= 5%
    robust : max 10th percentile of log-wealth after 100 cards
* Walk-forward: for each year 2019-2023 the policy is chosen on 2016..Y-1 and scored on Y.
  The final choice uses 2016-2023 and is scored on 2024-2026 and on the Kalshi live window.
  Both are already-spent periods, so those two scores are descriptive only.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from ufc_pred.backtest.universe import add_prior_fight_counts
from ufc_pred.paths import ROOT
from ufc_pred.research.harness import KEYS, OUT, american_to_decimal, card_bootstrap_mean, load_base

MODELS = {"F1": "p_R6_mkt", "F1B": "p_R6_mkt_B1v2B2"}
GRID = dict(
    thr=[0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10],
    k=[0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
    capf=[0.02, 0.05, 0.10, 0.25, 1.0],
    capc=[0.15, 0.30, 0.50, 1.0],
    ceil=[0.05, 0.10, 1.0],
)
FIXED = {
    "recomendada (1/4K, 2%/15%)": (0.02, 0.25, 0.02, 0.15, 0.10),
    "1/2K sin topes": (0.02, 0.5, 1.0, 1.0, 0.10),
    "Kelly completo sin topes": (0.02, 1.0, 1.0, 1.0, 1.0),
    "2x Kelly sin topes": (0.02, 2.0, 1.0, 1.0, 1.0),
}
B, H, SEED = 500, 100, 0
LIVE_HISTORY = ROOT / "data/processed/shadow_candidate_2026_09_14_v2/history.parquet"


def bets_books(df: pd.DataFrame, pcol: str) -> pd.DataFrame:
    p = df[pcol].to_numpy()
    dr, db = american_to_decimal(df.R_odds), american_to_decimal(df.B_odds)
    red = p * dr - 1 >= (1 - p) * db - 1
    return _pack(df, red, np.where(red, p, 1 - p), np.where(red, dr, db), np.where(red, df.q, 1 - df.q))


def bets_kalshi(df: pd.DataFrame, pcol: str) -> pd.DataFrame:
    p = df[pcol].to_numpy()
    cr = df.R_ask + 0.07 * df.R_ask * (1 - df.R_ask)
    cb = df.B_ask + 0.07 * df.B_ask * (1 - df.B_ask)
    dr, db = 1 / cr.to_numpy(), 1 / cb.to_numpy()
    red = p * dr - 1 >= (1 - p) * db - 1
    q = df.q_kalshi_mid.to_numpy()
    return _pack(df, red, np.where(red, p, 1 - p), np.where(red, dr, db), np.where(red, q, 1 - q))


def _pack(df, red, pp, d, mkt):
    win = np.where(red, df.y == 1, df.y == 0)
    ev = pp * d - 1
    out = pd.DataFrame(
        {
            "date": df.date.to_numpy(),
            "card": df.card.to_numpy(),
            "ev": ev,
            "kf": np.maximum(0, ev / (d - 1)),
            "edge": pp - mkt,
            "r": np.where(win, d - 1, -1.0),
        }
    )
    return out[np.isfinite(out.ev)].sort_values("date").reset_index(drop=True)


class Evaluator:
    """Per-card log-growth of a policy on a fixed set of cards, plus card-bootstrap risk."""

    def __init__(self, b: pd.DataFrame):
        self.b = b
        self.card_ids, self.ci = np.unique(b.card.to_numpy(), return_inverse=True)
        self.C = len(self.card_ids)
        self.idx = np.random.default_rng(SEED).integers(0, self.C, size=(B, H))

    def log_growth(self, thr, k, capf, capc, ceil):
        b = self.b
        sel = (b.ev.to_numpy() > thr) & (b.edge.to_numpy() <= ceil) & (b.kf.to_numpy() > 0)
        s = np.where(sel, np.minimum(k * b.kf.to_numpy(), capf), 0.0)
        tot = np.bincount(self.ci, s, self.C)
        s = s * np.where(tot > capc, capc / np.where(tot > 0, tot, 1), 1.0)[self.ci]
        rc = np.bincount(self.ci, s * b.r.to_numpy(), self.C)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(rc > -1, np.log1p(np.maximum(rc, -1 + 1e-300)), -np.inf), int(sel.sum())

    def risk(self, lg):
        path = np.cumsum(lg[self.idx], axis=1)
        runmax = np.maximum.accumulate(np.concatenate([np.zeros((B, 1)), path], axis=1), axis=1)[:, 1:]
        dd = np.max(runmax - path, axis=1)
        return dict(
            p_dd50=float(np.mean(dd >= np.log(2))),
            p_ruin90=float(np.mean(path.min(1) <= np.log(0.1))),
            p10_logw=float(np.quantile(path[:, -1], 0.10)),
            med_logw=float(np.median(path[:, -1])),
        )


def search(ev: Evaluator) -> pd.DataFrame:
    rows = []
    for pol in itertools.product(*GRID.values()):
        lg, n = ev.log_growth(*pol)
        r = dict(zip(GRID, pol), bets=n, g=float(np.mean(lg)) if np.isfinite(lg).all() else -np.inf)
        r.update(
            ev.risk(lg)
            if np.isfinite(r["g"])
            else dict(p_dd50=1.0, p_ruin90=1.0, p10_logw=-np.inf, med_logw=-np.inf)
        )
        rows.append(r)
    return pd.DataFrame(rows)


def choose(res: pd.DataFrame) -> dict:
    ok = res[res.p_dd50 <= 0.05]
    g = ok.loc[ok.g.idxmax()] if len(ok) else res.loc[res.g.idxmax()]
    rb = res.loc[res.p10_logw.idxmax()]
    return {"growth": tuple(g[list(GRID)]), "robust": tuple(rb[list(GRID)])}


def score(b: pd.DataFrame, pol) -> dict:
    ev = Evaluator(b)
    lg, n = ev.log_growth(*pol)
    w = np.exp(np.cumsum(lg))
    peak = np.maximum.accumulate(np.concatenate([[1.0], w]))[1:]
    return dict(
        bets=n,
        cards=ev.C,
        final=float(w[-1]) if len(w) else 1.0,
        max_dd=float(np.max(1 - w / peak)) if len(w) else 0.0,
        **ev.risk(lg),
    )


def fmt(pol) -> str:
    t, k, cf, cc, ce = pol
    return f"EV>{t:.1%} | {k:g}xKelly | tope pelea {cf:.0%} | tope cartelera {cc:.0%} | techo {'-' if ce >= 1 else f'{ce * 100:.0f}c'}"


def load_books() -> pd.DataFrame:
    base = load_base()
    dev = base[base.year.between(2016, 2023) & base.q.notna()][
        KEYS + ["card", "year", "y", "q", "R_odds", "B_odds", "debut"]
    ]
    for m in MODELS.values():
        dev = dev.merge(
            pd.read_parquet(OUT / f"claude/oof_dev_{m[2:]}.parquet")[KEYS + [m]], on=KEYS, how="left"
        )
    conf = pd.read_parquet(OUT / "claude/holdout_eval.parquet")[dev.columns]
    d = pd.concat([dev, conf], ignore_index=True)
    return d[~d.debut.astype(bool) & d.R_odds.notna() & d.B_odds.notna()]


def load_kalshi() -> pd.DataFrame:
    opp = pd.read_parquet(OUT / "opportunities_kalshi.parquet")
    k = opp[KEYS + ["card", "y", "R_ask", "B_ask", "q_kalshi_mid"]].merge(
        pd.read_parquet(OUT / "claude/venue_kalshi_f1b_q_kalshi_mid.parquet")[
            KEYS + ["p_F1_R6_mkt", "p_F1B"]
        ],
        on=KEYS,
    )
    k = k.rename(columns={"p_F1_R6_mkt": MODELS["F1"], "p_F1B": MODELS["F1B"]})
    hist = pd.concat(
        [
            pd.read_parquet("artifacts/shadow_candidate_2026_09_14_v2/verified_history.parquet")[KEYS],
            pd.read_parquet(LIVE_HISTORY)[KEYS],
            k[KEYS],
        ]
    ).drop_duplicates()
    hist = add_prior_fight_counts(hist).drop_duplicates(subset=KEYS)
    k = k.merge(hist[KEYS + ["has_debut"]], on=KEYS, how="left")
    return k[(k.has_debut == False) & k.R_ask.notna() & k.B_ask.notna()]  # noqa: E712


def realisation(b: pd.DataFrame, thr=0.02, ceil=0.10) -> dict:
    """Realised flat ROI over predicted EV on the bets the policy takes: ~ the Kelly multiplier the data supports."""
    t = b[(b.ev > thr) & (b.edge <= ceil)]
    lo, hi = card_bootstrap_mean(t.r.to_numpy(), t.card.to_numpy())
    return dict(
        bets=len(t),
        ev_pred=float(t.ev.mean()),
        roi=float(t.r.mean()),
        roi_lo=lo,
        roi_hi=hi,
        ratio=float(t.r.mean() / t.ev.mean()),
    )


def main():
    pd.set_option("display.width", 250)
    books, kal = load_books(), load_kalshi()
    out = []
    for name, pcol in MODELS.items():
        bb = bets_books(books.dropna(subset=[pcol]), pcol)
        bb["year"] = pd.to_datetime(bb.date).dt.year
        bk = bets_kalshi(kal.dropna(subset=[pcol]), pcol)
        print(f"\n===== {name} =====")
        for per, sub in [
            ("2016-2023", bb[bb.year <= 2023]),
            ("2024-2026", bb[bb.year >= 2024]),
            ("Kalshi", bk),
        ]:
            print(f"edge realisation {per}: {realisation(sub)}")
        # walk-forward: choose on 2016..Y-1, score on Y
        wf = {rule: [] for rule in ["growth", "robust"]}
        fixed = {f: [] for f in FIXED}
        for y in range(2019, 2024):
            pick = choose(search(Evaluator(bb[bb.year < y])))
            test = bb[bb.year == y]
            for rule, pol in pick.items():
                s = score(test, pol)
                wf[rule].append(np.log(s["final"]))
                out.append(dict(model=name, stage=f"wf {y}", policy=rule, rule=fmt(pol), **s))
            for f, pol in FIXED.items():
                fixed[f].append(np.log(score(test, pol)["final"]))
        print("walk-forward 2019-2023, $100 compounded across the five test years:")
        for lbl, v in list(wf.items()) + list(fixed.items()):
            print(
                f"  {lbl:28s} {100 * np.exp(np.sum(v)):10.2f} $   per year: {[round(100 * np.exp(x), 1) for x in v]}"
            )
        res = search(Evaluator(bb[bb.year <= 2023]))
        res.to_csv(OUT / f"claude/sizing_opt_grid_{name}.csv", index=False)
        pick = choose(res)
        for rule, pol in list(pick.items()) + list(FIXED.items()):
            for per, sub in [
                ("dev 2016-2023", bb[bb.year <= 2023]),
                ("conf 2024-2026", bb[bb.year >= 2024]),
                ("Kalshi jun-oct 2026", bk),
            ]:
                out.append(dict(model=name, stage=per, policy=rule, rule=fmt(pol), **score(sub, pol)))
    res = pd.DataFrame(out)
    res.to_csv(OUT / "claude/sizing_opt.csv", index=False)
    show = res[~res.stage.str.startswith("wf")].copy()
    show["final_$100"] = (100 * show.final).round(2)
    print(
        show[
            [
                "model",
                "stage",
                "policy",
                "rule",
                "bets",
                "final_$100",
                "max_dd",
                "p_dd50",
                "p_ruin90",
                "p10_logw",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()

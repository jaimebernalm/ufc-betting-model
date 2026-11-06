"""Common walk-forward harness for the Oct-2026 research block.

Protocol (frozen in artifacts/r2026_10/manifest.json before comparisons):
  * dev test-years 2016-2023, annual expanding refit (train strictly < Jan 1 of the test year)
  * holdout >= 2024-01-01, looked at once at the end
  * market reference q = closing sportsbook moneyline (Kaggle/BFO), proportionally de-vigged
  * gates: G1 paired delta log-loss vs raw q (card-clustered bootstrap, upper bound < 0);
           G2 flat ROI at the *offered* odds (vig included) for EV > threshold.

Everything here reads data, never writes to production paths.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ufc_pred.paths import ROOT

FEATURES = ROOT / "artifacts/shadow_candidate_2026_09_14_v2/training_features.parquet"
VERIFIED_HISTORY = ROOT / "artifacts/shadow_candidate_2026_09_14_v2/verified_history.parquet"
OUT = ROOT / "artifacts/r2026_10"
KEYS = ["date", "R_fighter", "B_fighter"]
DEV_YEARS = list(range(2016, 2024))
HOLDOUT_START = pd.Timestamp("2024-01-01")
# Live input cannot reproduce it (upcoming_builder forces False; 2022-23 rows are NaN).
DROP_ALWAYS = ["empty_arena"]


def american_to_prob(o) -> np.ndarray:
    o = np.asarray(o, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(o < 0, -o / (-o + 100.0), 100.0 / (o + 100.0))


def american_to_decimal(o) -> np.ndarray:
    o = np.asarray(o, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(o < 0, 1 + 100.0 / -o, 1 + o / 100.0)


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(x) -> np.ndarray:
    return 1 / (1 + np.exp(-np.asarray(x, dtype=float)))


def logloss(p, y) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


BFO_MATCHED = ROOT / "artifacts/r2026_10/claude/bfo/bfo_matched.parquet"
# Fixed priority: the first sportsbook with BOTH sides quoted supplies the offered price.
SPORTSBOOK_PRIORITY = ["DraftKings", "FanDuel", "BetMGM", "Caesars", "BetRivers", "Unibet", "BetWay"]


def bfo_exact() -> pd.DataFrame:
    """BFO rows with an unambiguous exact-name match (initial/loose/multi-candidate quarantined)."""
    b = pd.read_parquet(BFO_MATCHED)
    b = b[b.bfo_matchup_id.notna() & b.match_method.isin(["exact", "exact_date+1"])]
    b = b.drop_duplicates(subset=KEYS, keep=False).drop_duplicates(subset=["bfo_matchup_id"], keep=False)
    return b.reset_index(drop=True)


def _fill_from_bfo(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing Kaggle closing odds with one concrete BFO sportsbook pair (fixed priority).

    Declared before the holdout look (manifest). A real offered pair from a single book is used
    (never a cross-book median of American odds, which is not a price and can be invalid).
    Kalshi/Polymarket are not sportsbooks and are never used here.
    """
    b = pd.read_parquet(BFO_MATCHED)
    # only unambiguous exact-name matches feed the fill; initial/loose matches are quarantined
    b = b[b.bfo_matchup_id.notna() & b.match_method.isin(["exact", "exact_date+1"])]
    b = b.drop_duplicates(subset=KEYS, keep=False).drop_duplicates(subset=["bfo_matchup_id"], keep=False)
    b = b.reset_index(drop=True)
    r_odds = pd.Series(np.nan, index=b.index)
    b_odds = pd.Series(np.nan, index=b.index)
    book = pd.Series(None, index=b.index, dtype=object)
    for name in SPORTSBOOK_PRIORITY:
        rc, bc = f"R_close_{name}", f"B_close_{name}"
        if rc not in b:
            continue
        take = book.isna() & b[rc].notna() & b[bc].notna() & b[rc].abs().ge(100) & b[bc].abs().ge(100)
        r_odds[take], b_odds[take], book[take] = b.loc[take, rc], b.loc[take, bc], "bfo_" + name
    fill = b[KEYS].assign(bfo_R=r_odds.to_numpy(), bfo_B=b_odds.to_numpy(), bfo_book=book.to_numpy())
    out = df.merge(fill, on=KEYS, how="left")
    use = (out.R_odds.isna() | out.B_odds.isna()) & out.bfo_book.notna()
    out["odds_source"] = np.where(use, out.bfo_book, "kaggle")
    out.loc[use, "R_odds"] = out.loc[use, "bfo_R"]
    out.loc[use, "B_odds"] = out.loc[use, "bfo_B"]
    return out.drop(columns=["bfo_R", "bfo_B", "bfo_book"])


def load_base(fill_bfo_from: str | None = None) -> pd.DataFrame:
    """Decisive fights with the v2 point-in-time features plus market columns.

    Adds: y (Red wins), q (de-vigged Red closing prob, NaN without valid odds),
    has_odds, card (date string), year, debut (either fighter has no prior UFC bout in the
    verified history, same-day ties not prior; the deployed system skips these, so economic
    gates are reported on ~debut).
    """
    df = pd.read_parquet(FEATURES)
    df = df[df.Winner.isin(["Red", "Blue"])].sort_values(KEYS).reset_index(drop=True)
    df = df.drop(columns=[c for c in DROP_ALWAYS if c in df])
    if fill_bfo_from is not None and BFO_MATCHED.exists():
        filled = _fill_from_bfo(df)
        late = filled.date >= pd.Timestamp(fill_bfo_from)
        df[["R_odds", "B_odds"]] = np.where(late.to_numpy()[:, None], filled[["R_odds", "B_odds"]], df[["R_odds", "B_odds"]])
        df["odds_source"] = np.where(late, filled.odds_source, "kaggle")
    valid = df.R_odds.abs().ge(100) & df.B_odds.abs().ge(100)
    pr, pb = american_to_prob(df.R_odds), american_to_prob(df.B_odds)
    df["q"] = np.where(valid, pr / (pr + pb), np.nan)
    df["has_odds"] = valid
    df["y"] = df.Winner.eq("Red").astype(int)
    df["card"] = df.date.dt.strftime("%Y-%m-%d")
    df["year"] = df.date.dt.year
    # UFC-only prior appearances (v2 counters include external bouts, so they cannot define a debut).
    from ufc_pred.backtest.universe import add_prior_fight_counts

    hist = pd.read_parquet(VERIFIED_HISTORY)[KEYS]
    hist = add_prior_fight_counts(hist).drop_duplicates(subset=KEYS)
    df = df.merge(hist[KEYS + ["has_debut"]], on=KEYS, how="left", validate="one_to_one")
    df["debut"] = df.pop("has_debut").fillna(True).astype(bool)
    return df


# A model function takes (train_df, test_df, seed) and returns P(Red wins) for test_df.
ModelFn = Callable[[pd.DataFrame, pd.DataFrame, int], np.ndarray]


@dataclass
class WalkForward:
    years: list[int]
    seeds: list[int]
    require_odds: bool = False  # train/test only on fights with closing odds

    def run(self, df: pd.DataFrame, fn: ModelFn, name: str, verbose: bool = True) -> pd.DataFrame:
        rows = []
        for year in self.years:
            start, end = pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year + 1}-01-01")
            train = df[df.date < start]
            test = df[(df.date >= start) & (df.date < end)]
            if self.require_odds:
                train, test = train[train.has_odds], test[test.has_odds]
            if test.empty:
                continue
            preds = np.stack([fn(train.reset_index(drop=True), test.reset_index(drop=True), s) for s in self.seeds])
            meta = [c for c in ["odds_source"] if c in test]
            out = test[KEYS + ["card", "year", "y", "q", "R_odds", "B_odds", "debut"] + meta].copy()
            out["p_" + name] = preds.mean(0)
            out["sd_" + name] = preds.std(0)
            for s, pr in zip(self.seeds, preds):
                out[f"seed{s}_{name}"] = pr
            rows.append(out)
            if verbose:
                print(f"  {name} {year}: train={len(train)} test={len(test)}", flush=True)
        return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------- metrics


def card_bootstrap_mean(values: np.ndarray, cards: np.ndarray, n: int = 4000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    uc, inv = np.unique(cards, return_inverse=True)
    sums = np.bincount(inv, weights=values, minlength=len(uc))
    cnts = np.bincount(inv, minlength=len(uc)).astype(float)
    idx = rng.integers(0, len(uc), size=(n, len(uc)))
    stats = sums[idx].sum(1) / np.maximum(cnts[idx].sum(1), 1)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def delta_ll(df: pd.DataFrame, pcol: str, ref: str = "q") -> dict:
    d = df.dropna(subset=[pcol, ref])
    delta = logloss(d[pcol], d.y) - logloss(d[ref], d.y)
    lo, hi = card_bootstrap_mean(delta, d.card.to_numpy())
    return dict(
        model=pcol,
        n=len(d),
        ll=float(logloss(d[pcol], d.y).mean()),
        ll_ref=float(logloss(d[ref], d.y).mean()),
        d_ll=float(delta.mean()),
        d_lo=lo,
        d_hi=hi,
        g1_pass=bool(hi < 0),
    )


def bets_at_book_odds(df: pd.DataFrame, pcol: str, ev_thr: float = 0.0, rule: str = "ev") -> pd.DataFrame:
    """One row per opportunity (fight with odds); NaN prediction = no bet (PnL 0).

    rule="ev":   bet the side with the larger EV at the offered (vig-included) odds when EV > ev_thr.
    rule="raw3": deployed-style filter, p_side - implied(offered) >= 0.03 and EV > 0.
    """
    d = df.dropna(subset=["q"])
    dr, db = american_to_decimal(d.R_odds), american_to_decimal(d.B_odds)
    p = d[pcol].to_numpy()
    has_p = np.isfinite(p)
    p = np.where(has_p, p, 0.5)
    evr, evb = p * dr - 1, (1 - p) * db - 1
    red = evr >= evb
    ev = np.where(red, evr, evb)
    if rule == "ev":
        take = (ev > ev_thr) & has_p
    elif rule == "raw3":
        gap = np.where(red, p - 1 / dr, (1 - p) - 1 / db)
        take = (gap >= 0.03) & (ev > 0) & has_p
    else:
        raise ValueError(rule)
    y = d.y.to_numpy()
    ret = np.where(red, y * dr - 1, (1 - y) * db - 1)
    price = np.where(red, d.q, 1 - d.q)  # de-vigged market prob of the side taken
    return pd.DataFrame(
        {"card": d.card.to_numpy(), "year": d.year.to_numpy(), "ret": ret, "ev": ev, "price": price, "bet": take}
    )


def roi_summary(df: pd.DataFrame, pcol: str, ev_thr: float = 0.0, rule: str = "ev") -> dict:
    """Point ROI / PnL per opportunity whenever there is >= 1 bet; CIs only with >= 5 bets."""
    b = bets_at_book_odds(df, pcol, ev_thr, rule)
    t = b[b.bet]
    pnl_opp = np.where(b.bet, b.ret, 0.0)
    out = dict(model=pcol, rule=rule, ev_thr=ev_thr, opportunities=len(b), bets=len(t),
               pnl_per_opp=float(pnl_opp.mean()) if len(b) else 0.0,
               roi=float(t.ret.mean()) if len(t) else np.nan)
    if len(t) < 5:
        return out
    lo, hi = card_bootstrap_mean(t.ret.to_numpy(), t.card.to_numpy())
    plo, phi = card_bootstrap_mean(pnl_opp, b.card.to_numpy())
    out.update(
        roi_lo=lo,
        roi_hi=hi,
        pnl_lo=plo,
        pnl_hi=phi,
        fav_share=float((t.price >= 0.5).mean()),
        mean_ev=float(t.ev.mean()),
    )
    return out


def roi_by_bucket(df: pd.DataFrame, pcol: str, ev_thr: float = 0.0) -> pd.DataFrame:
    b = bets_at_book_odds(df, pcol, ev_thr)
    t = b[b.bet].copy()
    t["bucket"] = pd.cut(t.price, [0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0])
    return t.groupby("bucket", observed=True).ret.agg(["size", "mean"]).rename(columns={"size": "bets", "mean": "roi"})


def walk_forward_stack(oof: pd.DataFrame, cols: list[str], min_years: int = 3, C: float = 1.0) -> np.ndarray:
    """Symmetric, intercept-free logistic stacking fitted on earlier OOF years only.

    logit p = sum_k a_k * logit(col_k)  (M0 when cols=['q']; M1 when cols=['q', p_model]).
    """
    from sklearn.linear_model import LogisticRegression

    out = np.full(len(oof), np.nan)
    years = sorted(oof.year.unique())
    for i, yr in enumerate(years):
        if i < min_years:
            continue
        tr = oof[(oof.year < yr)].dropna(subset=cols)
        te = (oof.year == yr).to_numpy() & oof[cols].notna().all(1).to_numpy()
        X = np.column_stack([logit(tr[c]) for c in cols])
        X = np.vstack([X, -X])
        yy = np.concatenate([tr.y, 1 - tr.y])
        lr = LogisticRegression(C=C, fit_intercept=False).fit(X, yy)
        Xt = np.column_stack([logit(oof.loc[te, c]) for c in cols])
        out[te] = lr.predict_proba(Xt)[:, 1]
    return out


def report(oof: pd.DataFrame, pcols: list[str], thresholds=(0.0, 0.02, 0.04)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """G1 on all fights with odds; G2 (economics) on the deployable universe (no debuts)."""
    ll_rows = [delta_ll(oof, c) for c in pcols]
    eco = oof[~oof.debut] if "debut" in oof else oof
    roi_rows = [roi_summary(eco, c, t) for c in pcols for t in thresholds]
    roi_rows += [roi_summary(eco, c, 0.0, rule="raw3") for c in pcols]
    return pd.DataFrame(ll_rows), pd.DataFrame(roi_rows)

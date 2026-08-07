"""Compare deployed-config bankroll simulations with and without capping
large model edges, on two venues:

  * Polymarket  — long out-of-sample window 2025-07-01 .. 2026-05-31
                  (cutoff 2025-01-01 analog ensemble; the live 2025-11-30
                   artifact would leak on the Jul-Nov 2025 portion).
  * Kalshi      — real Kalshi closing prices 2026-01-24 .. 2026-05-30
                  (the actual live-deployed 2025-11-30 ensemble; fully OOS).

Both venues use the SAME current deployed methodology:
  - 10-seed real ensemble (accounts A,B) + 10-seed corrupted ensemble (C)
  - orientation-symmetrized predictions (2026-06-11 fix)
  - sharpen_T = 1.25 (configs/inference.json)
  - per-account Kelly: A=10%-K/10%-cap, B=25%-K/no-cap, C=25%-K/no-cap
  - edge threshold 3%, quadratic venue fee (Kalshi 0.07, Polymarket 0.03)

The cap acts on the per-bet probability edge (model_p_chosen - market_price_chosen,
i.e. the same "+Xc edge" the live bet tool prints). Two cap modes:
  - exclude : skip any bet whose edge > cap
  - clip    : still bet, but size Kelly as if the edge equalled the cap

Outputs artifacts/metrics/cap_large_edges_sim.json and prints a summary.
Read-only w.r.t. live state.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

from ufc_pred.backtest.bet_eval import _effective_decimal
from ufc_pred.backtest.strategy_grid import train_corrupted, train_real
from ufc_pred.features.skill_v3_pipeline import OUTPUT as SKILL_V3_PARQUET
from ufc_pred.features.static_v1 import _swap_red_blue, prepare
from ufc_pred.ingest.kaggle_mdabbert import HISTORY_PARQUET

SHARPEN_T = json.loads((ROOT / "configs/inference.json").read_text())["sharpen_T"]
START = 300.0
EDGE_THR = 0.03
SEEDS = list(range(10))
HEADLINE_CAP = 0.15
CAP_SWEEP = [0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.50]  # 0.50 ~ uncapped

# Account specs: (kelly_fraction, stake_cap, model_kind)
ACCOUNTS = {
    "A": dict(kelly=0.10, stake_cap=0.10, kind="real"),
    "B": dict(kelly=0.25, stake_cap=1.00, kind="real"),
    "C": dict(kelly=0.25, stake_cap=1.00, kind="corrupt"),
}


# --------------------------------------------------------------------------
# Prediction helpers (mirror strategy_grid.predict + live symmetrize+sharpen)
# --------------------------------------------------------------------------
def _attr(bundle, name):
    return bundle[name] if isinstance(bundle, dict) else getattr(bundle, name)


def predict_bundle(bundle, df: pd.DataFrame) -> np.ndarray:
    X, _, _, _ = prepare(df, augment_symmetry=False, one_hot=False)
    X = X.reindex(columns=_attr(bundle, "columns"), fill_value=None)
    for c in _attr(bundle, "cat_features"):
        X[c] = X[c].fillna("__missing__").astype(str)
    return _attr(bundle, "model").predict_proba(X)[:, 1]


def sharpen(p: np.ndarray, T: float) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return 1.0 / (1.0 + np.exp(-T * np.log(p / (1.0 - p))))


def ensemble_sym_sharp(bundles, df: pd.DataFrame, T: float = SHARPEN_T) -> np.ndarray:
    """Orientation-symmetrized, sharpened ensemble-mean P(Red wins)."""
    mir = _swap_red_blue(df)
    fwd = np.mean([predict_bundle(b, df) for b in bundles], axis=0)
    rev = np.mean([predict_bundle(b, mir) for b in bundles], axis=0)
    p_sym = 0.5 * (fwd + (1.0 - rev))
    return sharpen(p_sym, T)


# --------------------------------------------------------------------------
# Simulation engine (deployed Kelly + edge cap)
# --------------------------------------------------------------------------
def simulate(
    p_model,
    mkt_p_red,
    mkt_p_blue,
    y_red,
    *,
    kelly,
    stake_cap,
    fee_rate,
    fee_model,
    cap_mode="none",
    edge_cap=HEADLINE_CAP,
    start=START,
    edge_thr=EDGE_THR,
):
    mkt_p_red = np.asarray(mkt_p_red, float)
    mkt_p_blue = np.asarray(mkt_p_blue, float)
    y_red = np.asarray(y_red, int)
    p_R = np.asarray(p_model, float)
    p_B = 1.0 - p_R

    dec_R = 1.0 / mkt_p_red
    dec_B = 1.0 / mkt_p_blue
    eff_R = _effective_decimal(dec_R, fee_rate, fee_model)
    eff_B = _effective_decimal(dec_B, fee_rate, fee_model)

    ev_R = p_R * eff_R - 1.0
    ev_B = p_B * eff_B - 1.0
    bet_red = ev_R >= ev_B
    chosen_ev = np.where(bet_red, ev_R, ev_B)
    chosen_dec = np.where(bet_red, eff_R, eff_B)
    chosen_p = np.where(bet_red, p_R, p_B)
    chosen_price = np.where(bet_red, mkt_p_red, mkt_p_blue)  # what you pay
    prob_edge = chosen_p - chosen_price  # "+Xc edge" in the tool
    won = np.where(bet_red, y_red == 1, y_red == 0).astype(int)

    placeable = chosen_ev > edge_thr
    is_large = placeable & (prob_edge > edge_cap)

    bank = start
    traj = [bank]
    peak = bank
    max_dd = 0.0
    n_bets = n_capped = 0
    recs = []
    for i in range(len(p_R)):
        if not placeable[i]:
            continue
        if cap_mode == "exclude" and is_large[i]:
            n_capped += 1
            continue
        b = chosen_dec[i] - 1.0
        if cap_mode == "clip" and is_large[i]:
            p_size = min(chosen_price[i] + edge_cap, chosen_p[i])
            capped = True
            n_capped += 1
        else:
            p_size = chosen_p[i]
            capped = False
        fk = (b * p_size - (1.0 - p_size)) / b
        if fk <= 0:
            continue
        stake_frac = min(kelly * fk, stake_cap)
        stake = bank * stake_frac
        pnl = stake * (chosen_dec[i] - 1.0) if won[i] else -stake
        bank += pnl
        peak = max(peak, bank)
        max_dd = max(max_dd, (peak - bank) / peak)
        traj.append(bank)
        n_bets += 1
        recs.append(
            dict(
                i=int(i),
                edge=float(prob_edge[i]),
                p=float(chosen_p[i]),
                price=float(chosen_price[i]),
                won=int(won[i]),
                dec=float(chosen_dec[i]),
                large=bool(is_large[i]),
                capped=bool(capped),
                stake=float(stake),
                pnl=float(pnl),
                bank=float(bank),
            )
        )
    return dict(
        final=float(bank),
        ret_pct=float((bank / start - 1) * 100),
        max_dd=float(max_dd * 100),
        n_bets=int(n_bets),
        n_capped=int(n_capped),
        n_large=int(is_large.sum()),
        trajectory=[float(x) for x in traj],
        recs=recs,
    )


# --------------------------------------------------------------------------
# Data builders
# --------------------------------------------------------------------------
def load_fights():
    fights = pd.read_parquet(HISTORY_PARQUET)
    fights = fights[fights["Winner"].isin(["Red", "Blue"])].copy()
    fights["date"] = pd.to_datetime(fights["date"])
    sk = pd.read_parquet(SKILL_V3_PARQUET)
    sk["date"] = pd.to_datetime(sk["date"])
    fights = fights.merge(
        sk[["date", "R_fighter", "B_fighter", "skill_diff_mean", "skill_diff_std"]],
        on=["date", "R_fighter", "B_fighter"],
        how="left",
        validate="many_to_one",
    )
    return fights


def build_polymarket(fights):
    EVAL_START, EVAL_END = pd.Timestamp("2025-07-01"), pd.Timestamp("2026-05-31")
    matched = pd.read_parquet(ROOT / "data/interim/polymarket_matched_to_kaggle_v2.parquet")
    matched["date"] = pd.to_datetime(matched["date"])
    key = ["date", "R_fighter", "B_fighter"]
    full = matched[key + ["polymarket_p_red", "polymarket_p_blue"]].merge(
        fights, on=key, how="left", validate="one_to_one"
    )
    ev = full[(full["date"] >= EVAL_START) & (full["date"] <= EVAL_END)].reset_index(drop=True)
    return ev, ev["polymarket_p_red"].to_numpy(), ev["polymarket_p_blue"].to_numpy()


def build_kalshi(fights):
    import re
    import unicodedata

    from rapidfuzz import fuzz

    AP = "'’ʼ`‘"

    def deep_norm(s):
        if pd.isna(s) or s is None:
            return ""
        s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
        for ch in AP + "-.,":
            s = s.replace(ch, "")
        return re.sub(r"\s+", " ", s).strip().lower()

    def last_token(s):
        s = deep_norm(s)
        return s.split()[-1] if s else ""

    kal = pd.read_parquet(ROOT / "data/raw/kalshi/historical.parquet")
    kal = kal.dropna(subset=["close_yes_price_a", "close_yes_price_b", "winner"]).copy()
    ab = kal["close_yes_price_a"] + kal["close_yes_price_b"]
    keep = (
        (kal["close_yes_price_a"] >= 0.02)
        & (kal["close_yes_price_b"] >= 0.02)
        & (ab >= 0.80)
        & (ab <= 1.30)
        & ((kal["volume_a"] + kal["volume_b"]) >= 100)
    )
    kal = kal[keep].reset_index(drop=True)
    kal["fd_n"] = pd.to_datetime(kal["fight_date"]).dt.tz_localize(None).dt.normalize()
    kal["fd_m1"] = kal["fd_n"] - pd.Timedelta(days=1)
    kal["a_dn"] = kal["fighter_a"].map(deep_norm)
    kal["b_dn"] = kal["fighter_b"].map(deep_norm)
    kal["a_last"] = kal["fighter_a"].map(last_token)
    kal["b_last"] = kal["fighter_b"].map(last_token)

    f = fights.copy()
    f["R_dn"] = f["R_fighter"].map(deep_norm)
    f["B_dn"] = f["B_fighter"].map(deep_norm)
    f["R_last"] = f["R_fighter"].map(last_token)
    f["B_last"] = f["B_fighter"].map(last_token)
    by_date = {d: g for d, g in f.groupby("date")}

    def candidates(row):
        dates = set()
        for d in (row["fd_n"], row["fd_m1"]):
            for off in (-1, 0, 1):
                dates.add(d + pd.Timedelta(days=off))
        pools = [by_date[d] for d in dates if d in by_date]
        return pd.concat(pools) if pools else pd.DataFrame()

    def match_one(row):
        pool = candidates(row)
        if len(pool) > 0:
            a, b = row["a_dn"], row["b_dn"]
            a_l, b_l = row["a_last"], row["b_last"]
            h = pool[
                ((pool["R_dn"] == a) & (pool["B_dn"] == b)) | ((pool["R_dn"] == b) & (pool["B_dn"] == a))
            ]
            if len(h) == 1:
                k = h.iloc[0]
                return k, k["R_dn"] == a
            h = pool[
                ((pool["R_last"] == a_l) & (pool["B_last"] == b_l))
                | ((pool["R_last"] == b_l) & (pool["B_last"] == a_l))
            ]
            if len(h) == 1:
                k = h.iloc[0]
                return k, k["R_last"] == a_l
            h = pool[
                (
                    (pool["R_dn"].str.contains(a_l, regex=False))
                    & (pool["B_dn"].str.contains(b_l, regex=False))
                )
                | (
                    (pool["R_dn"].str.contains(b_l, regex=False))
                    & (pool["B_dn"].str.contains(a_l, regex=False))
                )
            ]
            if len(h) == 1:
                k = h.iloc[0]
                return k, a_l in k["R_dn"]

            def fp(k):
                return (
                    max(fuzz.ratio(a_l, k["R_last"]), fuzz.ratio(a_l, k["B_last"]))
                    + max(fuzz.ratio(b_l, k["R_last"]), fuzz.ratio(b_l, k["B_last"]))
                ) / 2

            pool = pool.copy()
            pool["fuz"] = pool.apply(fp, axis=1)
            best = pool.sort_values("fuz", ascending=False).head(2)
            if (
                len(best) > 0
                and best.iloc[0]["fuz"] >= 88
                and (len(best) == 1 or best.iloc[0]["fuz"] - best.iloc[1]["fuz"] >= 5)
            ):
                k = best.iloc[0]
                return k, fuzz.ratio(a_l, k["R_last"]) >= fuzz.ratio(a_l, k["B_last"])
        d = row["fd_n"]
        w = f[(f["date"] >= d - pd.Timedelta(days=14)) & (f["date"] <= d + pd.Timedelta(days=14))]
        h = w[
            ((w["R_last"] == row["a_last"]) & (w["B_last"] == row["b_last"]))
            | ((w["R_last"] == row["b_last"]) & (w["B_last"] == row["a_last"]))
        ]
        if len(h) == 1:
            k = h.iloc[0]
            return k, k["R_last"] == row["a_last"]
        return None, None

    _rows, kag_idx, pr, pb = [], [], [], []
    for _, row in kal.iterrows():
        k, a_is_red = match_one(row)
        if k is None:
            continue
        kag_idx.append(k.name)
        pa, pbb = row["close_yes_price_a"], row["close_yes_price_b"]
        pr.append(pa if a_is_red else pbb)  # price of RED corner
        pb.append(pbb if a_is_red else pa)  # price of BLUE corner
    ev = fights.loc[kag_idx].reset_index(drop=True)
    return ev, np.asarray(pr, float), np.asarray(pb, float)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def run_market(name, ev, mkt_pr, mkt_pb, bundles_real, bundles_corr, fee_rate):
    y_red = (ev["Winner"].to_numpy() == "Red").astype(int)
    p_real = ensemble_sym_sharp(bundles_real, ev)
    p_corr = ensemble_sym_sharp(bundles_corr, ev)
    preds = {"real": p_real, "corrupt": p_corr}

    out = {
        "n_fights": int(len(ev)),
        "date_range": [str(ev["date"].min().date()), str(ev["date"].max().date())],
        "headline": {},
        "sweep": {},
    }

    for acct, spec in ACCOUNTS.items():
        p = preds[spec["kind"]]
        base = dict(kelly=spec["kelly"], stake_cap=spec["stake_cap"], fee_rate=fee_rate, fee_model="kalshi")
        res = {}
        res["uncapped"] = simulate(p, mkt_pr, mkt_pb, y_red, cap_mode="none", **base)
        res["exclude"] = simulate(p, mkt_pr, mkt_pb, y_red, cap_mode="exclude", edge_cap=HEADLINE_CAP, **base)
        res["clip"] = simulate(p, mkt_pr, mkt_pb, y_red, cap_mode="clip", edge_cap=HEADLINE_CAP, **base)
        out["headline"][acct] = {m: {k: v for k, v in r.items() if k != "recs"} for m, r in res.items()}
        # sweep
        sweep = {"exclude": [], "clip": []}
        for cap in CAP_SWEEP:
            for mode in ("exclude", "clip"):
                r = simulate(p, mkt_pr, mkt_pb, y_red, cap_mode=mode, edge_cap=cap, **base)
                sweep[mode].append(
                    dict(
                        cap=cap,
                        final=r["final"],
                        max_dd=r["max_dd"],
                        n_bets=r["n_bets"],
                        n_capped=r["n_capped"],
                    )
                )
        out["sweep"][acct] = sweep

    # full per-bet book (uncapped) for B(real) and C(corrupt) — flat-ROI buckets
    rB = simulate(
        preds["real"],
        mkt_pr,
        mkt_pb,
        y_red,
        cap_mode="none",
        kelly=0.25,
        stake_cap=1.0,
        fee_rate=fee_rate,
        fee_model="kalshi",
    )
    rC = simulate(
        preds["corrupt"],
        mkt_pr,
        mkt_pb,
        y_red,
        cap_mode="none",
        kelly=0.25,
        stake_cap=1.0,
        fee_rate=fee_rate,
        fee_model="kalshi",
    )
    keep = ("edge", "p", "price", "won", "dec", "large")
    out["book_real"] = [{k: r[k] for k in keep} for r in rB["recs"]]
    out["book_corrupt"] = [{k: r[k] for k in keep} for r in rC["recs"]]
    large = [r for r in rB["recs"] if r["large"]]
    ledger = []
    for r in large:
        row = ev.iloc[r["i"]]
        ledger.append(
            dict(
                date=str(row["date"].date()),
                fight=f"{row['R_fighter']} vs {row['B_fighter']}",
                edge_pp=round(r["edge"] * 100, 1),
                model_p=round(r["p"], 3),
                price=round(r["price"], 3),
                won=r["won"],
            )
        )
    out["large_edge_ledger"] = ledger
    out["large_edge_summary"] = dict(
        n=len(large),
        n_won=sum(r["won"] for r in large),
        hit_rate=round(np.mean([r["won"] for r in large]), 3) if large else None,
    )
    return out


def main():
    t0 = time.time()
    fights = load_fights()
    print(f"fights loaded: {len(fights)}  ({time.time() - t0:.0f}s)")

    # Polymarket — train 2025-01-01 analog ensemble (live 2025-11-30 would leak)
    poly_ev, poly_pr, poly_pb = build_polymarket(fights)
    print(
        f"Polymarket eval: {len(poly_ev)} fights {poly_ev['date'].min().date()}..{poly_ev['date'].max().date()}"
    )
    CUT_POLY = pd.Timestamp("2025-01-01")
    print(f"training Polymarket ensemble @ cutoff {CUT_POLY.date()} (10 seeds x2)...")
    poly_real, poly_corr = [], []
    for s in SEEDS:
        poly_real.append(train_real(fights, CUT_POLY, seed=s))
        poly_corr.append(train_corrupted(fights, CUT_POLY, seed=s))
        print(f"  seed {s} done ({time.time() - t0:.0f}s)", flush=True)

    # Kalshi — load the live-deployed 2025-11-30 ensemble (fully OOS on Jan+ 2026)
    kal_ev, kal_pr, kal_pb = build_kalshi(fights)
    print(f"Kalshi eval: {len(kal_ev)} fights {kal_ev['date'].min().date()}..{kal_ev['date'].max().date()}")
    kal_real = [joblib.load(ROOT / f"artifacts/models/v3_real_2025_11_30_seed{s}.joblib") for s in SEEDS]
    kal_corr = [joblib.load(ROOT / f"artifacts/models/v3_corrupted_2025_11_30_seed{s}.joblib") for s in SEEDS]

    results = {
        "config": dict(
            sharpen_T=SHARPEN_T,
            headline_cap=HEADLINE_CAP,
            edge_thr=EDGE_THR,
            start=START,
            seeds=len(SEEDS),
            cap_sweep=CAP_SWEEP,
            poly_cutoff="2025-01-01",
            kalshi_cutoff="2025-11-30",
        ),
        "polymarket": run_market(
            "polymarket", poly_ev, poly_pr, poly_pb, poly_real, poly_corr, fee_rate=0.03
        ),
        "kalshi": run_market("kalshi", kal_ev, kal_pr, kal_pb, kal_real, kal_corr, fee_rate=0.07),
    }

    outpath = ROOT / "artifacts/metrics/cap_large_edges_sim.json"
    outpath.write_text(json.dumps(results, indent=1))
    print(f"\nwrote {outpath}  ({time.time() - t0:.0f}s)\n")

    # Summary
    for venue in ("kalshi", "polymarket"):
        v = results[venue]
        print(
            f"\n===== {venue.upper()}  ({v['n_fights']} fights, {v['date_range'][0]}..{v['date_range'][1]}) ====="
        )
        print(
            f"  large-edge (>{int(HEADLINE_CAP * 100)}pp) fights: {v['large_edge_summary']['n']}, "
            f"hit rate {v['large_edge_summary']['hit_rate']}"
        )
        hdr = (
            f"  {'acct':4} {'uncapped':>14} {'exclude@15':>14} {'clip@15':>14}  {'n_bets':>6} {'n_large':>7}"
        )
        print(hdr)
        for acct in "ABC":
            h = v["headline"][acct]
            print(
                f"  {acct:4} {h['uncapped']['final']:>14,.0f} {h['exclude']['final']:>14,.0f} "
                f"{h['clip']['final']:>14,.0f}  {h['uncapped']['n_bets']:>6} {h['uncapped']['n_large']:>7}"
            )


if __name__ == "__main__":
    main()

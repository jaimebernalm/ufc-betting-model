"""Task 1.2 — does the model add anything over the market anchor?

Tennis PLAN_TENNIS §12 step 7g: every blend/shade of the trained model into
the anchor hurt; verdict "deploy the anchor pure". This asks the same question
of the UFC deployment ensemble. Variants pre-registered in
TENNIS_PORTED_IDEAS.md §6 (2026-07-09).

Two evaluation windows:
1. Kalshi window (2026-01 → 2026-05, matched T-90min prices, all post the
   deployment ensemble's 2025-11-30 cutoff): model = deployed 10-seed real
   ensemble, orientation-symmetrized, exactly as served live. Settlement at
   the Kalshi price with the verified quadratic fee.
2. Val-2023 replica (bigger n, walk-forward clean for the val-protocol
   ensemble trained on ≤ 2022-12-31): anchor = BFO no-vig, settlement =
   BFO no-vig + Kalshi quadratic fee. Also caches the 10 per-seed val
   predictions to data/interim/val2023_champion_seed_preds.npz for reuse
   by the Tier-2 ablations (same seeds, same protocol).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd

from ufc_pred.backtest.kalshi_match import match_kalshi_to_fights
from ufc_pred.backtest.metrics import american_to_implied_prob
from ufc_pred.backtest.strategy_grid import ModelBundle, predict, train_real
from ufc_pred.features.static_v1 import _swap_red_blue
from ufc_pred.paths import MODELS, ROOT
from ufc_pred.utils.time_splits import TRAIN_END, VAL_END

LOG_PATH = ROOT / "experiments" / "2026-07-09_model_on_top.jsonl"
VAL_PREDS_CACHE = ROOT / "data/interim/val2023_champion_seed_preds.npz"
KALSHI_FEE = 0.07
EDGE_THR = 0.03
SEEDS = list(range(10))


def eff_dec(price, fee_rate=KALSHI_FEE):
    price = np.asarray(price, float)
    return 1.0 / (price * (1.0 + fee_rate * (1.0 - price)))


def flat_roi(
    p_bet: np.ndarray,
    y_red: np.ndarray,
    price_red: np.ndarray,
    price_blue: np.ndarray,
    thr: float = EDGE_THR,
    eligible: np.ndarray | None = None,
    force_side_red: np.ndarray | None = None,
) -> dict:
    """Flat-$1 betting: bet the side with max EV under probabilities `p_bet`.

    `eligible` masks fights allowed to bet at all. `force_side_red`, if given,
    restricts to fights where the chosen side equals this side (veto variants).
    Returns summary dict with bootstrap CI.
    """
    eR, eB = eff_dec(price_red), eff_dec(price_blue)
    ev_R = p_bet * eR - 1.0
    ev_B = (1.0 - p_bet) * eB - 1.0
    side_r = ev_R >= ev_B
    edge = np.where(side_r, ev_R, ev_B)
    dec = np.where(side_r, eR, eB)
    bets = edge > thr
    if eligible is not None:
        bets &= eligible
    if force_side_red is not None:
        bets &= side_r == force_side_red
    won = np.where(side_r, y_red == 1, y_red == 0)
    pnl = np.where(won, dec - 1.0, -1.0)[bets]
    n = int(bets.sum())
    out = {"n_bets": n}
    if n:
        rng = np.random.default_rng(0)
        idx = rng.integers(0, n, size=(5000, n))
        means = pnl[idx].mean(axis=1)
        out.update(
            roi_pct=round(float(pnl.mean() * 100), 2),
            ci95=[round(float(np.quantile(means, q) * 100), 2) for q in (0.025, 0.975)],
            hit_rate=round(float(won[bets].mean()), 3),
            total_pnl=round(float(pnl.sum()), 2),
        )
    else:
        out.update(roi_pct=None, ci95=None, hit_rate=None, total_pnl=0.0)
    return out


def log_row(window: str, name: str, res: dict, extra: dict | None = None):
    row = {"window": window, "variant": name, **res, "ts": datetime.now(UTC).isoformat()}
    if extra:
        row.update(extra)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(row) + "\n")
    ci = f"CI[{res['ci95'][0]:7.2f},{res['ci95'][1]:7.2f}]" if res.get("ci95") else " " * 20
    roi = f"{res['roi_pct']:7.2f}%" if res.get("roi_pct") is not None else "     — "
    print(f"  {name:34s} n={res['n_bets']:4d} ROI={roi} {ci}" + (f"  {extra}" if extra else ""))


# ---------------------------------------------------------------------------
# Window 1: Kalshi 2026
# ---------------------------------------------------------------------------


def kalshi_window():
    print("=" * 78)
    print("WINDOW 1 — Kalshi T-90min prices, 2026-01 → 2026-05 (post-cutoff OOS)")
    print("=" * 78)
    fights = pd.read_parquet(ROOT / "data/processed/fights.parquet")
    fights = fights[fights["Winner"].isin(["Red", "Blue"])].copy()
    sk = pd.read_parquet(ROOT / "data/processed/skill_features_v3.parquet")
    sk["date"] = pd.to_datetime(sk["date"])
    fights["date"] = pd.to_datetime(fights["date"])
    fights = fights.merge(
        sk[["date", "R_fighter", "B_fighter", "skill_diff_mean", "skill_diff_std"]],
        on=["date", "R_fighter", "B_fighter"],
        how="left",
        validate="many_to_one",
    )

    meta, matched = match_kalshi_to_fights(fights)
    y = (matched["Winner"] == "Red").to_numpy(int)
    pr_k = meta["kal_p_red"].to_numpy(float)
    pb_k = meta["kal_p_blue"].to_numpy(float)
    print(f"{len(meta)} matched fights ({meta['date'].min().date()} → {meta['date'].max().date()})")

    # Deployed ensemble, orientation-symmetrized (as served live).
    mirrored = _swap_red_blue(matched)
    per_seed = []
    for s in SEEDS:
        d = joblib.load(MODELS / f"v3_real_2025_11_30_seed{s}.joblib")
        b = ModelBundle(model=d["model"], columns=d["columns"], cat_features=d["cat_features"])
        p_fwd = predict(b, matched)
        p_rev = predict(b, mirrored)
        per_seed.append(0.5 * (p_fwd + 1.0 - p_rev))
    per_seed = np.stack(per_seed)
    p_model = per_seed.mean(axis=0)

    # Anchors.
    pm = pd.read_parquet(ROOT / "data/interim/polymarket_matched_to_kaggle_v2.parquet")
    pm["date"] = pd.to_datetime(pm["date"])
    key = matched[["date", "R_fighter", "B_fighter"]].merge(
        pm[["date", "R_fighter", "B_fighter", "polymarket_p_red", "polymarket_p_blue"]],
        on=["date", "R_fighter", "B_fighter"],
        how="left",
        validate="one_to_one",
    )
    tot = key["polymarket_p_red"] + key["polymarket_p_blue"]
    p_anchor_poly = (key["polymarket_p_red"] / tot).to_numpy(float)
    has_poly = ~np.isnan(p_anchor_poly)

    p_ri = american_to_implied_prob(matched["R_odds"])
    p_bi = american_to_implied_prob(matched["B_odds"])
    p_anchor_bfo = np.asarray(p_ri / (p_ri + p_bi), float)
    has_bfo = ~np.isnan(p_anchor_bfo)
    print(f"anchor coverage: poly {has_poly.sum()}/{len(meta)}, bfo {has_bfo.sum()}/{len(meta)}\n")

    def side_red_of(p):
        return p * eff_dec(pr_k) - 1.0 >= (1 - p) * eff_dec(pb_k) - 1.0

    # (a) anchor pure
    log_row(
        "kalshi",
        "a_anchor_poly_pure",
        flat_roi(np.where(has_poly, p_anchor_poly, np.nan), y, pr_k, pb_k, eligible=has_poly),
    )
    log_row(
        "kalshi",
        "a_anchor_bfo_pure",
        flat_roi(np.where(has_bfo, p_anchor_bfo, np.nan), y, pr_k, pb_k, eligible=has_bfo),
    )
    # (b) model pure — ensemble + per-seed spread
    res_b = flat_roi(p_model, y, pr_k, pb_k)
    seed_rois = [flat_roi(per_seed[i], y, pr_k, pb_k)["roi_pct"] for i in range(len(SEEDS))]
    seed_rois = [r for r in seed_rois if r is not None]
    log_row(
        "kalshi",
        "b_model_pure(deployed_ens_sym)",
        res_b,
        {
            "seed_roi_min": min(seed_rois),
            "seed_roi_med": float(np.median(seed_rois)),
            "seed_roi_max": max(seed_rois),
        },
    )
    # (c) blends with poly anchor
    for w in (0.25, 0.5, 0.75):
        p_blend = np.where(has_poly, w * p_model + (1 - w) * p_anchor_poly, p_model)
        log_row("kalshi", f"c_blend_w{w}", flat_roi(p_blend, y, pr_k, pb_k))
    # (d) veto: model bets only where anchor picks the same side
    agree = side_red_of(p_model) == side_red_of(np.where(has_poly, p_anchor_poly, p_model))
    log_row("kalshi", "d_veto_poly_agrees", flat_roi(p_model, y, pr_k, pb_k, eligible=has_poly & agree))
    # (e) model only where anchor silent
    log_row("kalshi", "e_model_where_no_anchor", flat_roi(p_model, y, pr_k, pb_k, eligible=~has_poly))


# ---------------------------------------------------------------------------
# Window 2: val 2023 replica
# ---------------------------------------------------------------------------


def val_window():
    print("\n" + "=" * 78)
    print("WINDOW 2 — val 2023, BFO no-vig settlement + Kalshi quadratic fee")
    print("=" * 78)
    fights = pd.read_parquet(ROOT / "data/processed/fights.parquet")
    fights = fights[fights["Winner"].isin(["Red", "Blue"])].copy()
    fights["date"] = pd.to_datetime(fights["date"])
    sk = pd.read_parquet(ROOT / "data/processed/skill_features_v3.parquet")
    sk["date"] = pd.to_datetime(sk["date"])
    fights = fights.merge(
        sk[["date", "R_fighter", "B_fighter", "skill_diff_mean", "skill_diff_std"]],
        on=["date", "R_fighter", "B_fighter"],
        how="left",
        validate="many_to_one",
    )

    val = fights[(fights["date"] > TRAIN_END) & (fights["date"] <= VAL_END)]
    val = val.dropna(subset=["R_odds", "B_odds"]).reset_index(drop=True)
    y = (val["Winner"] == "Red").to_numpy(int)
    p_ri = american_to_implied_prob(val["R_odds"])
    p_bi = american_to_implied_prob(val["B_odds"])
    p_anchor = np.asarray(p_ri / (p_ri + p_bi), float)
    price_red, price_blue = p_anchor, 1.0 - p_anchor  # settle at no-vig
    print(f"{len(val)} val fights with BFO odds")

    if VAL_PREDS_CACHE.exists():
        per_seed = np.load(VAL_PREDS_CACHE)["per_seed"]
        print(f"loaded cached val preds {per_seed.shape}")
    else:
        per_seed = []
        for s in SEEDS:
            b = train_real(fights, TRAIN_END + pd.Timedelta(days=1), seed=s)
            per_seed.append(predict(b, val))
            print(f"  trained seed {s}")
        per_seed = np.stack(per_seed)
        VAL_PREDS_CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            VAL_PREDS_CACHE,
            per_seed=per_seed,
            date=val["date"].astype(str).to_numpy(),
            R_fighter=val["R_fighter"].to_numpy(),
            B_fighter=val["B_fighter"].to_numpy(),
        )
    p_model = per_seed.mean(axis=0)

    def side_red_of(p):
        return p * eff_dec(price_red) - 1.0 >= (1 - p) * eff_dec(price_blue) - 1.0

    log_row("val2023", "a_anchor_pure", flat_roi(p_anchor, y, price_red, price_blue))
    res_b = flat_roi(p_model, y, price_red, price_blue)
    seed_rois = [flat_roi(per_seed[i], y, price_red, price_blue)["roi_pct"] for i in range(len(SEEDS))]
    log_row(
        "val2023",
        "b_model_pure(10seed_val_ens)",
        res_b,
        {
            "seed_roi_min": min(seed_rois),
            "seed_roi_med": float(np.median(seed_rois)),
            "seed_roi_max": max(seed_rois),
        },
    )
    for w in (0.25, 0.5, 0.75):
        log_row(
            "val2023",
            f"c_blend_w{w}",
            flat_roi(w * p_model + (1 - w) * p_anchor, y, price_red, price_blue),
        )
    agree = side_red_of(p_model) == side_red_of(p_anchor)
    log_row(
        "val2023",
        "d_veto_anchor_agrees",
        flat_roi(p_model, y, price_red, price_blue, eligible=agree),
    )


if __name__ == "__main__":
    kalshi_window()
    val_window()

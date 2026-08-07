"""Tier-2 ablation harness (Tasks 2.1 Elo / 2.2 fatigue / 2.3 half-life).

Protocol (pre-registered in TENNIS_PORTED_IDEAS.md §6): champion v3
architecture trained on date ≤ 2022-12-31, seeds 0-9, fixed 2000 iterations,
recency-weighted; scored on frozen val 2023 (416 odds fights) via
`evaluate_bets(fee_rate=0.07, fee_model="kalshi", use_no_vig=True)` at edge
thr 3% (primary) and 5%; log_loss/ECE as non-regression gates.

Adoption rule: ensemble val ROI @3% improves AND ≥7/10 seeds directionally
improve (paired by seed vs baseline) AND log_loss worsens < 0.005, ECE < 0.01.

Run: PYTHONPATH=src python scripts/tier2_ablation.py [config ...]
Default: all configs. Results appended to
experiments/2026-07-09_tier2_ablation.jsonl (one row per config).
"""

from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from ufc_pred.backtest.bet_eval import evaluate_bets
from ufc_pred.backtest.metrics import evaluate
from ufc_pred.features.elo import build_elo
from ufc_pred.features.fatigue import F1_COLS, F2_COLS, F3_COLS, build_fatigue
from ufc_pred.features.static_v1 import prepare
from ufc_pred.paths import ROOT
from ufc_pred.utils.time_splits import TRAIN_END, VAL_END

LOG_PATH = ROOT / "experiments" / "2026-07-09_tier2_ablation.jsonl"
SEEDS = list(range(10))
ELO_COLS = ["R_elo", "B_elo", "elo_dif", "R_wc_elo", "B_wc_elo", "wc_elo_dif", "R_elo_n", "B_elo_n"]

CONFIGS: dict[str, dict] = {
    "baseline": {},
    **{f"elo_k{k}": {"elo_k": k} for k in (4, 8, 16, 32)},
    "fat_f1": {"fatigue": F1_COLS},
    "fat_f2": {"fatigue": F2_COLS},
    "fat_f3": {"fatigue": F3_COLS},
    **{f"hl_{h}": {"half_life": float(h)} for h in (2, 3, 6, 8)},
}


def load_base() -> pd.DataFrame:
    fights = pd.read_parquet(ROOT / "data/processed/fights.parquet")
    fights = fights[fights["Winner"].isin(["Red", "Blue"])].copy()
    fights["date"] = pd.to_datetime(fights["date"])
    sk = pd.read_parquet(ROOT / "data/processed/skill_features_v3.parquet")
    sk["date"] = pd.to_datetime(sk["date"])
    return fights.merge(
        sk[["date", "R_fighter", "B_fighter", "skill_diff_mean", "skill_diff_std"]],
        on=["date", "R_fighter", "B_fighter"],
        how="left",
        validate="many_to_one",
    )


def with_features(fights: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = fights
    key = ["date", "R_fighter", "B_fighter"]
    if "elo_k" in cfg:
        df = df.merge(
            build_elo(df, k=float(cfg["elo_k"]))[key + ELO_COLS],
            on=key,
            how="left",
            validate="many_to_one",
        )
    if "fatigue" in cfg:
        fat = build_fatigue(df)
        df = df.merge(fat[key + list(cfg["fatigue"])], on=key, how="left", validate="many_to_one")
    return df


def train_predict(df: pd.DataFrame, val_df: pd.DataFrame, seed: int, half_life: float = 4.0) -> np.ndarray:
    train = df[df["date"] <= TRAIN_END]
    X, y, d, cat = prepare(train, augment_symmetry=True, one_hot=False)
    w = np.power(0.5, ((TRAIN_END - d).dt.days / 365.25).clip(lower=0) / half_life)
    cb = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        loss_function="Logloss",
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
    )
    cb.fit(Pool(X, y, cat_features=cat, weight=w.to_numpy()), verbose=False)
    Xv, _, _, _ = prepare(val_df, augment_symmetry=False, one_hot=False)
    Xv = Xv.reindex(columns=X.columns, fill_value=None)
    for c in cat:
        Xv[c] = Xv[c].fillna("__missing__").astype(str)
    return cb.predict_proba(Xv)[:, 1]


def run_config(name: str, cfg: dict, fights: pd.DataFrame, baseline_seed_rois: list[float] | None) -> dict:
    df = with_features(fights, cfg)
    val = df[(df["date"] > TRAIN_END) & (df["date"] <= VAL_END)]
    val = val.dropna(subset=["R_odds", "B_odds"]).reset_index(drop=True)
    y_val = (val["Winner"] == "Red").to_numpy(int)
    hl = cfg.get("half_life", 4.0)

    per_seed, t0 = [], time.time()
    for s in SEEDS:
        per_seed.append(train_predict(df, val, s, half_life=hl))
        print(f"    seed {s} done ({time.time() - t0:.0f}s)", flush=True)
    per_seed = np.stack(per_seed)
    p_ens = per_seed.mean(axis=0)

    def roi(p, thr):
        return evaluate_bets(
            p,
            y_val,
            val["R_odds"],
            val["B_odds"],
            edge_threshold=thr,
            fee_rate=0.07,
            fee_model="kalshi",
            use_no_vig=True,
        )

    seed_rois = [roi(per_seed[i], 0.03).roi_pct for i in range(len(SEEDS))]
    r3, r5 = roi(p_ens, 0.03), roi(p_ens, 0.05)
    m = evaluate(y_val, p_ens, label=name)
    row = {
        "config": name,
        "cfg": {k: (v if not isinstance(v, list) else "cols") for k, v in cfg.items()},
        "roi3_ens": round(r3.roi_pct, 2),
        "roi3_ci": [round(x, 2) for x in r3.ci95_roi_pct],
        "n_bets3": r3.n_bets,
        "roi5_ens": round(r5.roi_pct, 2),
        "n_bets5": r5.n_bets,
        "seed_rois3": [round(x, 2) for x in seed_rois],
        "seed_roi3_med": round(float(np.median(seed_rois)), 2),
        "log_loss": round(m["log_loss"], 4),
        "ece": round(m["ece"], 4),
        "ts": datetime.now(UTC).isoformat(),
    }
    if baseline_seed_rois is not None:
        wins = sum(a > b for a, b in zip(seed_rois, baseline_seed_rois, strict=False))
        row["seed_wins_vs_baseline"] = f"{wins}/10"
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(
        f"  {name:10s} ROI@3% ens={row['roi3_ens']:+.2f}% "
        f"(seed med {row['seed_roi3_med']:+.2f}, n={row['n_bets3']}) "
        f"ROI@5%={row['roi5_ens']:+.2f}% ll={row['log_loss']} ece={row['ece']}"
        + (f" wins={row.get('seed_wins_vs_baseline')}" if baseline_seed_rois else ""),
        flush=True,
    )
    return row


def main():
    names = sys.argv[1:] or list(CONFIGS)
    fights = load_base()
    baseline_rois = None
    # baseline always first if requested or needed for pairing
    if "baseline" not in names:
        names = ["baseline"] + names
    results = {}
    for name in names:
        print(f"[{name}]", flush=True)
        results[name] = run_config(name, CONFIGS[name], fights, baseline_rois)
        if name == "baseline":
            baseline_rois = results[name]["seed_rois3"]


if __name__ == "__main__":
    main()

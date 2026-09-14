"""Reproduce the corrected, frozen-cutoff model bundle without overwriting live models.

Run with PYTHONPATH=src .conda/bin/python scripts/research/rebuild_corrected_system.py
Prerequisite: monthly skill features from ufc_pred.features.skill_v3_pipeline.build
written to artifacts/corrected_2026_09_14/skill_features.parquet.
"""

import hashlib
import json
import subprocess
from pathlib import Path

import joblib
import pandas as pd

from ufc_pred.backtest.strategy_grid import train_real
from ufc_pred.inference.upcoming_builder import build_upcoming_row
from ufc_pred.ingest.rankings_attach import load_rankings
from ufc_pred.ingest.strict_history import RawHistoryStateSource, apply_state_features
from ufc_pred.models.wrappers import CorruptedSkillModel
from ufc_pred.paths import ROOT

OUT = ROOT / "artifacts/corrected_2026_09_14"
CUTOFF = pd.Timestamp("2025-11-30")
KEYS = ["date", "R_fighter", "B_fighter"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build():
    OUT.mkdir(exist_ok=True, parents=True)
    source = RawHistoryStateSource(ROOT / "data/raw/ufcstats_export")
    ranks = load_rankings()
    raw = pd.read_parquet(OUT / "verified_history.parquet")
    strict = apply_state_features(raw, source, ranks)
    strict.to_parquet(OUT / "strict_fights.parquet", index=False)
    skill = pd.read_parquet(OUT / "skill_features.parquet")
    skill["date"] = pd.to_datetime(skill.date)
    fights = strict.merge(skill, on=KEYS, validate="one_to_one", how="left")
    fights = fights[fights.Winner.isin(["Red", "Blue"])].reset_index(drop=True)
    fights.to_parquet(OUT / "training_features.parquet", index=False)
    # Live builder and batch builder must yield identical state/rank/difference fields.
    parity = []
    known = set()
    samples = []
    for _, r in strict.sort_values("date").iterrows():
        if r.R_fighter in known and r.B_fighter in known:
            samples.append(r)
        known.update([r.R_fighter, r.B_fighter])
    for r in samples[:: max(1, len(samples) // 100)]:
        row = build_upcoming_row(
            r.R_fighter,
            r.B_fighter,
            r.date,
            r.weight_class,
            strict,
            gender=r.gender,
            title_bout=bool(r.title_bout),
            no_of_rounds=int(r.no_of_rounds),
            rankings=ranks,
            state_source=source,
        )
        fields = [
            c
            for c in strict
            if (c.startswith(("R_", "B_")) and c not in ("R_odds", "B_odds", "R_ev", "B_ev"))
            or c.endswith("_dif")
            or c == "better_rank"
        ]
        for col in fields:
            a, b = row.iloc[0][col], r[col]
            if pd.isna(a) and pd.isna(b):
                continue
            if a != b:
                parity.append({"date": str(r.date), "column": col, "batch": str(b), "live": str(a)})
    (OUT / "feature_parity.json").write_text(
        json.dumps(
            {"sampled_rows": len(samples[:: max(1, len(samples) // 100)]), "mismatches": parity}, indent=2
        )
    )
    if parity:
        raise ValueError(f"{len(parity)} training/inference feature mismatches")
    models = OUT / "models"
    models.mkdir(exist_ok=True)
    training_hash = sha(OUT / "training_features.parquet")
    for seed in range(10):
        rp = models / f"v3_real_2025_11_30_seed{seed}.joblib"
        cp = models / f"v3_corrupted_2025_11_30_seed{seed}.joblib"
        if rp.exists() and cp.exists():
            assert joblib.load(rp)["training_sha256"] == training_hash
            print(f"seed {seed}: verified checkpoint", flush=True)
            continue
        print(f"training seed {seed}/9, fixed 2000 trees", flush=True)
        bundle = train_real(fights, CUTOFF, seed)
        meta = {
            "columns": bundle.columns,
            "cat_features": bundle.cat_features,
            "training_sha256": training_hash,
            "cutoff": str(CUTOFF.date()),
            "seed": seed,
            "feature_version": "strict-2026-09-14",
            "n_train_rows": int((fights.date < CUTOFF).sum()),
        }
        joblib.dump({**meta, "model": bundle.model}, rp)
        # train_corrupted uses the identical fit; only its inference wrapper differs.
        joblib.dump(
            {**meta, "model": CorruptedSkillModel(bundle.model, ["skill_diff_mean", "skill_diff_std"])}, cp
        )
    policy = {
        "cutoff": "2025-11-30",
        "seeds": list(range(10)),
        "iterations": 2000,
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 3,
        "recency_half_life_years": 4,
        "sharpen_T": 1.25,
        "kelly": {"A": 0.10, "B": 0.25, "C": 0.25},
        "bankroll_cap": {"A": 0.10, "B": None, "C": None},
        "account_C": "same fit with skill columns NaN at inference",
        "edge_gross": 0.03,
        "liquidity_cap": 0.05,
        "depth_band": 0.03,
        "minimum_stake": 0.50,
        "mode": "shadow",
        "profile_vintage_limit": "static profile historical publication times unavailable; no claim of fully point-in-time metadata",
        "historical_periods": "diagnostic, reused; not independent evidence of edge",
    }
    source_dir = ROOT / "data/raw/ufcstats_export"
    files = [
        OUT / "training_features.parquet",
        OUT / "strict_fights.parquet",
        OUT / "skill_features.parquet",
        ROOT / "configs/fighter_aliases.json",
        ROOT / "configs/inference.json",
        ROOT / "data/raw/martj42_rankings/rankings_history.csv",
        ROOT / "data/raw/ufcstats_verified_profiles/profiles.json",
        *models.glob("*.joblib"),
        *source_dir.glob("*.csv"),
    ]
    manifest = {
        "policy": policy,
        "source_commit": subprocess.check_output(
            ["git", "-C", str(source_dir), "rev-parse", "HEAD"], text=True
        ).strip(),
        "files": {str(p.relative_to(ROOT)): sha(p) for p in files},
        "training_rows": int((fights.date < CUTOFF).sum()),
        "rankings_last_date": str(ranks.date.max().date()),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("Completed corrected model bundle", flush=True)


if __name__ == "__main__":
    build()

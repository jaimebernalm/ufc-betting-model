"""Paired causal-block diagnostics, fixed recipe/strategy, isolated from shadow.

No variant selection, tuning, or deployment is performed by this script.
"""

import argparse
import hashlib
import json
import time

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from ufc_pred.features.static_v1 import _swap_red_blue, prepare
from ufc_pred.inference.upcoming_builder import _DIFF_PAIRS
from ufc_pred.ingest.identity import canonicalize_frame
from ufc_pred.ingest.strict_history import COUNTERS, RATES
from ufc_pred.paths import ROOT
from ufc_pred.utils.time_splits import recency_weights

BASE = ROOT / "artifacts/baselines/2026_09_14_before_fixes"
FIX = ROOT / "artifacts/corrected_2026_09_14"
OUT = ROOT / "artifacts/ablation_2026_09_14"
CUTOFF = pd.Timestamp("2025-11-30")
KEYS = ["date", "R_fighter", "B_fighter"]


def datasets():
    original = pd.read_parquet(BASE / "data/processed/fights.parquet")
    skill = pd.read_parquet(BASE / "data/processed/skill_features_v3.parquet")
    old = original.merge(skill, on=KEYS, how="left", validate="one_to_one")
    old = canonicalize_frame(old)
    old = old[old.Winner.isin(["Red", "Blue"])].reset_index(drop=True)
    new = pd.read_parquet(FIX / "training_features.parquet")
    aligned = old.set_index(KEYS).loc[new.set_index(KEYS).index].reset_index().reindex(columns=new.columns)
    assert aligned.date.equals(new.date)
    groups = {}
    for name, bases in [
        ("rates", RATES),
        ("counters", COUNTERS),
        ("profiles", ["age", "Stance", "Height_cms", "Reach_cms", "Weight_lbs"]),
    ]:
        groups[name] = [s + "_" + c for s in ["R", "B"] for c in bases]
        groups[name] += [d for d, b in _DIFF_PAIRS if b in bases]
    groups["rankings"] = [c for c in new if c.endswith("_rank") or c == "better_rank"]
    groups["skill"] = ["skill_diff_mean", "skill_diff_std"]
    all_groups = sum(groups.values(), [])
    assert len(all_groups) == len(set(all_groups))
    variants = {"00_old_features_fixed_symmetry": old}
    v = aligned.copy()
    v["Winner"] = new.Winner
    variants["01_verified_results"] = v.copy()
    for i, group in enumerate(["rankings", "skill", "profiles", "counters", "rates"], start=2):
        v[groups[group]] = new[groups[group]]
        variants[f"{i:02d}_add_{group}"] = v.copy()
    pd.testing.assert_frame_equal(v, new, check_dtype=False)
    # Reverse interventions check that the sequential order has not concealed interactions.
    for group in ["rates", "counters", "skill"]:
        v = new.copy()
        v[groups[group]] = aligned[groups[group]]
        variants["full_restore_old_" + group] = v
    # Mechanistic audit only: do not reinstate broken symmetry in production.
    variants["legacy_real_augmentation"] = old
    variants["legacy_corrupted_augmentation"] = old
    profile_path = OUT / "completed_profile_attributes.json"
    if profile_path.exists():
        profiles = json.loads(profile_path.read_text())
        expected = pd.read_csv(OUT / "lost_profile_fields.csv").fighter.nunique()
        if len(profiles) == expected:
            filled = new.copy()
            for side in ["R", "B"]:
                for idx, row in filled.iterrows():
                    profile = profiles.get(row[side + "_fighter"], {})
                    if "error" in profile:
                        continue
                    for attr in ["Height_cms", "Reach_cms", "Stance", "age"]:
                        col = side + "_" + attr
                        if pd.notna(row[col]):
                            continue
                        value = profile.get(attr)
                        if attr == "age" and profile.get("DOB"):
                            dob, date = pd.Timestamp(profile["DOB"]), row.date
                            value = date.year - dob.year - ((date.month, date.day) < (dob.month, dob.day))
                        if value is not None and pd.notna(value):
                            filled.at[idx, col] = value
            for diff, base in _DIFF_PAIRS:
                if base in ["age", "Height_cms", "Reach_cms"]:
                    filled[diff] = filled["B_" + base] - filled["R_" + base]
            variants["full_complete_missing_profiles"] = filled
    extended_path = OUT / "strict_extended_counters.parquet"
    if extended_path.exists() and (OUT / "external_counter_coverage.json").exists():
        extended = pd.read_parquet(extended_path)
        pd.testing.assert_frame_equal(extended[KEYS + ["Winner"]], new[KEYS + ["Winner"]])
        candidate = variants.get("full_complete_missing_profiles", new).copy()
        candidate[groups["counters"]] = extended[groups["counters"]]
        variants["profiles_plus_prior_external_counters"] = candidate
    return variants, groups, new, aligned


def matrix(frame, augmentation="fixed"):
    train = frame[frame.date < CUTOFF]
    X, y, d, cat = prepare(train, augment_symmetry=True, one_hot=False)
    if augmentation.startswith("legacy"):
        # Before June 11: corners swapped, signed differences and better_rank left unchanged.
        n = len(train)
        cols = [c for c in X if c.endswith("_dif") or c.startswith("skill_diff_mean") or c == "better_rank"]
        X.loc[n:, cols] = X.iloc[:n][cols].to_numpy()
        if augmentation == "legacy_corrupted_augmentation":
            # Historical corrupted trainer manually negated skill_mean only.
            X.loc[n:, "skill_diff_mean"] = -X.iloc[:n].skill_diff_mean.to_numpy()
    return X, y, d, cat


def predict_both(model, frame):
    both = pd.concat([frame, _swap_red_blue(frame)], ignore_index=True)
    X, _, _, _ = prepare(both, augment_symmetry=False, one_hot=False)
    # Identical column order to fitting.
    X = X.reindex(columns=model.feature_names_)
    n = len(frame)
    real = model.predict_proba(X)[:, 1]
    X[["skill_diff_mean", "skill_diff_std"]] = np.nan
    corr = model.predict_proba(X)[:, 1]
    return np.stack([(real[:n] + 1 - real[n:]) / 2, (corr[:n] + 1 - corr[n:]) / 2])


def run(variants_to_run, threads):
    variants, groups, new, aligned = datasets()
    OUT.mkdir(exist_ok=True)
    (OUT / "feature_blocks.json").write_text(json.dumps(groups, indent=2))
    new.loc[new.date >= CUTOFF].to_parquet(OUT / "evaluation_features.parquet", index=False)
    manifest = {
        "cutoff": str(CUTOFF.date()),
        "seeds": list(range(10)),
        "policy": "2000 trees, depth 6, lr .05, l2 3, recency 4y; T1.25; no strategy tuning",
        "groups": groups,
        "variants": {},
        "scope": "diagnostic reused windows; not prospective validation",
    }
    for name, frame in variants.items():
        p = OUT / (name + ".parquet")
        if not p.exists():
            frame.to_parquet(p, index=False)
        else:
            # Checkpoints belong to these exact rows; never silently reuse a
            # fitted model after refreshing one of the research inputs.
            pd.testing.assert_frame_equal(pd.read_parquet(p), frame, check_dtype=False)
        manifest["variants"][name] = {
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "training_rows": int((frame.date < CUTOFF).sum()),
        }
    (OUT / "study_manifest.json").write_text(json.dumps(manifest, indent=2))
    for name in variants_to_run or list(variants):
        frame = variants[name]
        folder = OUT / name
        folder.mkdir(exist_ok=True)
        eval_frame = frame.loc[frame.date >= CUTOFF].reset_index(drop=True)
        assert eval_frame[KEYS].equals(new.loc[new.date >= CUTOFF, KEYS].reset_index(drop=True))
        X, y, d, cat = matrix(frame, name)
        pool = Pool(
            X, y, cat_features=cat, weight=recency_weights(d, reference_date=CUTOFF - pd.Timedelta(days=1))
        )
        for seed in range(10):
            path = folder / f"seed{seed}.joblib"
            predpath = folder / f"seed{seed}.npy"
            if predpath.exists():
                continue
            start = time.monotonic()
            if name == "06_add_rates":
                model = joblib.load(FIX / f"models/v3_real_2025_11_30_seed{seed}.joblib")["model"]
            elif path.exists():
                model = joblib.load(path)
            else:
                model = CatBoostClassifier(
                    iterations=2000,
                    learning_rate=0.05,
                    depth=6,
                    l2_leaf_reg=3,
                    loss_function="Logloss",
                    random_seed=seed,
                    verbose=False,
                    allow_writing_files=False,
                    thread_count=threads,
                )
                model.fit(pool)
                joblib.dump(model, path)
            ps = predict_both(model, eval_frame)
            np.save(predpath, ps)
            imp = dict(zip(model.feature_names_, model.get_feature_importance(), strict=True))
            (folder / f"seed{seed}_importance.json").write_text(json.dumps(imp, indent=2))
            y_eval = eval_frame.Winner.eq("Red").to_numpy(int)
            p = 1 / (
                1
                + np.exp(
                    -1.25 * np.log(np.clip(ps[0], 1e-6, 1 - 1e-6) / (1 - np.clip(ps[0], 1e-6, 1 - 1e-6)))
                )
            )
            print(
                name,
                seed,
                f"{time.monotonic() - start:.1f}s",
                f"LL {log_loss(y_eval, p):.4f} Brier {brier_score_loss(y_eval, p):.4f} AUC {roc_auc_score(y_eval, p):.4f}",
                flush=True,
            )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variants", nargs="*")
    p.add_argument("--threads", type=int, default=8)
    args = p.parse_args()
    run(args.variants, args.threads)

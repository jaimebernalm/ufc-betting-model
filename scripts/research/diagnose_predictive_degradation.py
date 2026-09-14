"""Decompose errors and feature-time contamination without fitting a new policy."""

import json

import joblib
import numpy as np
import pandas as pd
from ablate_predictive_degradation import BASE, CUTOFF, FIX, OUT, datasets, predict_both
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from ufc_pred.ingest.strict_history import RATES, RawHistoryStateSource
from ufc_pred.paths import ROOT


def sharpen(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return 1 / (1 + np.exp(-1.25 * np.log(p / (1 - p))))


def row_loss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -y * np.log(p) - (1 - y) * np.log(1 - p)


def metrics(y, p):
    return {
        "n": len(y),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "accuracy": float(((p >= 0.5) == y).mean()),
        "mean_confidence": float(np.maximum(p, 1 - p).mean()),
        "extreme_predictions": int(((p < 0.1) | (p > 0.9)).sum()),
    }


def paired_ci(frame, p, reference):
    y = frame.Winner.eq("Red").to_numpy(int)
    delta = pd.DataFrame(
        {
            "date": frame.date.to_numpy(),
            "loss": row_loss(y, p) - row_loss(y, reference),
            "brier": (p - y) ** 2 - (reference - y) ** 2,
        }
    )
    groups = delta.groupby("date").agg(loss=("loss", "sum"), brier=("brier", "sum"), n=("loss", "count"))
    rng = np.random.default_rng(914)
    idx = rng.integers(len(groups), size=(10000, len(groups)))
    return {
        m: {
            "mean": float(delta[m].mean()),
            "card_paired_ci95": np.quantile(
                groups[m].to_numpy()[idx].sum(1) / groups.n.to_numpy()[idx].sum(1), [0.025, 0.975]
            ).tolist(),
        }
        for m in ["loss", "brier"]
    }


def diagnose():
    _, groups, new, old = datasets()
    missing = []
    for side in ["R", "B"]:
        for attr in ["age", "Stance", "Height_cms", "Reach_cms", "Weight_lbs"]:
            col = side + "_" + attr
            lost = new[col].isna() & old[col].notna()
            for idx in new.index[lost]:
                missing.append(
                    {
                        "date": str(new.loc[idx, "date"].date()),
                        "fighter": new.loc[idx, side + "_fighter"],
                        "field": attr,
                        "old": str(old.loc[idx, col]),
                    }
                )
    pd.DataFrame(missing).to_csv(OUT / "lost_profile_fields.csv", index=False)
    mask = new.date >= CUTOFF
    n = new.loc[mask].reset_index(drop=True)
    o = old.loc[mask].reset_index(drop=True)
    original_predictions = pd.read_parquet(FIX / "all_postcutoff_predictions.parquet")
    assert n[["date", "R_fighter", "B_fighter"]].equals(
        original_predictions[["date", "R_fighter", "B_fighter"]]
    )
    n["has_debut"] = original_predictions.has_debut
    source = RawHistoryStateSource(ROOT / "data/raw/ufcstats_export")
    records = []
    for i, r in n.iterrows():
        for side in ["R", "B"]:
            name = r[side + "_fighter"]
            post = source(name, r.date + pd.Timedelta(days=1))
            before = n.loc[i, [side + "_" + c for c in RATES]].to_numpy(float)
            oldval = o.loc[i, [side + "_" + c for c in RATES]].to_numpy(float)
            after = np.array([post[c] for c in RATES])
            eqpre = np.isclose(oldval, before, atol=0.011, rtol=0, equal_nan=True)
            eqpost = np.isclose(oldval, after, atol=0.011, rtol=0, equal_nan=True)
            records.append(
                {
                    "date": r.date,
                    "fighter": name,
                    "opponent": r["B_fighter" if side == "R" else "R_fighter"],
                    "winner": r.Winner == ("Red" if side == "R" else "Blue"),
                    "pre_all5": bool(eqpre.all()),
                    "post_all5": bool(eqpost.all()),
                    "post_only_all5": bool(eqpost.all() and not eqpre.all()),
                    "pre_fields": int(eqpre.sum()),
                    "post_fields": int(eqpost.sum()),
                    "old": oldval.tolist(),
                    "before": before.tolist(),
                    "after": after.tolist(),
                }
            )
    contamination = pd.DataFrame(records)
    contamination.to_json(OUT / "rate_snapshot_timing.json", orient="records", indent=2, date_format="iso")
    n["postonly_any"] = contamination.post_only_all5.to_numpy().reshape(-1, 2).any(1)
    n["post_all5_any"] = contamination.post_all5.to_numpy().reshape(-1, 2).any(1)
    versions = {
        v: original_predictions[[v + "_real", v + "_corrupted"]].to_numpy().T
        for v in ["old_frozen", "corrected"]
    }
    interventions = {}
    intervention_names = [*groups, "all"]
    candidate_path = OUT / "profiles_plus_prior_external_counters.parquet"
    if candidate_path.exists():
        intervention_names.append("candidate")
    for group in intervention_names:
        frame = o.copy()
        if group == "candidate":
            frame = pd.read_parquet(candidate_path)
            frame = frame[frame.date >= CUTOFF].reset_index(drop=True)
        else:
            cols = sum(groups.values(), []) if group == "all" else groups[group]
            frame[cols] = n[cols]
        ps = []
        for seed in range(10):
            # Real and C historically had separate fits; use each actual artifact.
            real = joblib.load(BASE / f"artifacts/models/v3_real_2025_11_30_seed{seed}.joblib")["model"]
            corr = joblib.load(BASE / f"artifacts/models/v3_corrupted_2025_11_30_seed{seed}.joblib")[
                "model"
            ].base_model
            ps.append(np.stack([predict_both(real, frame)[0], predict_both(corr, frame)[1]]))
        interventions[group] = sharpen(np.mean(ps, axis=0))
    if "candidate" in interventions:
        np.save(OUT / "frozen_on_extended.npy", interventions["candidate"])
    np.savez(OUT / "frozen_model_feature_interventions.npz", **interventions)
    versions.update({"old_with_new_" + k: p for k, p in interventions.items()})
    masks = {
        "all": np.ones(len(n), dtype=bool),
        "eligible_no_debut": ~n.has_debut,
        "debut": n.has_debut,
        "early_to_may16": n.date <= pd.Timestamp("2026-05-16"),
        "later_after_may16": n.date > pd.Timestamp("2026-05-16"),
        "postfight_snapshot_exact_any": n.postonly_any,
        "no_exact_postfight_match": ~n.postonly_any,
    }
    slices = {}
    for name, m in masks.items():
        y = n.loc[m, "Winner"].eq("Red").to_numpy(int)
        slices[name] = {
            v: {kind: metrics(y, p[j, m]) for j, kind in enumerate(["real", "corrupted"])}
            for v, p in versions.items()
        }
    delta = {}
    for j, k in enumerate(["real", "corrupted"]):
        delta[k] = paired_ci(n, versions["corrected"][j], versions["old_frozen"][j])
        y = n.Winner.eq("Red").to_numpy(int)
        table = n[["date", "R_fighter", "B_fighter", "Winner", "has_debut", "postonly_any"]].copy()
        table["old_p"] = versions["old_frozen"][j]
        table["new_p"] = versions["corrected"][j]
        table["delta_loss"] = row_loss(y, table.new_p.to_numpy()) - row_loss(y, table.old_p.to_numpy())
        table.sort_values("delta_loss", ascending=False).to_csv(
            OUT / f"fight_loss_changes_{k}.csv", index=False
        )
    rate_summary = {}
    for period, m in [
        ("all", np.ones(len(contamination), dtype=bool)),
        ("early", contamination.date <= pd.Timestamp("2026-05-16")),
        ("late", contamination.date > pd.Timestamp("2026-05-16")),
    ]:
        g = contamination[m]
        rate_summary[period] = {
            "fighter_rows": len(g),
            **{c: int(g[c].sum()) for c in ["pre_all5", "post_all5", "post_only_all5"]},
        }
    n.to_parquet(OUT / "annotated_evaluation.parquet", index=False)
    (OUT / "diagnosis.json").write_text(
        json.dumps({"rate_timing": rate_summary, "slices": slices, "paired_deterioration": delta}, indent=2)
    )
    print(json.dumps({"rate_timing": rate_summary, "paired_deterioration": delta}, indent=2))


if __name__ == "__main__":
    diagnose()

"""Summarize completed, fixed ten-seed block interventions and paired replays."""

import json

import numpy as np
import pandas as pd
from ablate_predictive_degradation import BASE, FIX, OUT
from diagnose_predictive_degradation import metrics, paired_ci, sharpen
from evaluate_corrected_system import flat_metrics, replay


def summarize():
    manifest = json.loads((OUT / "study_manifest.json").read_text())
    frame = pd.read_parquet(OUT / "annotated_evaluation.parquet")
    frozen = pd.read_parquet(FIX / "all_postcutoff_predictions.parquet")
    prices = pd.read_parquet(FIX / "paired_price_comparison.parquet")
    versions = {v: frozen[[v + "_real", v + "_corrupted"]].to_numpy().T for v in ["old_frozen", "corrected"]}
    interventions = np.load(OUT / "frozen_model_feature_interventions.npz")
    versions["old_on_corrected_features"] = interventions["all"]
    if (OUT / "frozen_on_extended.npy").exists():
        versions["old_on_candidate_features"] = np.load(OUT / "frozen_on_extended.npy")
    per_seed = {}
    raw_versions = {}
    for name in manifest["variants"]:
        files = [OUT / name / f"seed{s}.npy" for s in range(10)]
        if not all(p.exists() for p in files):
            continue
        ps = np.stack([np.load(p) for p in files])
        assert ps.shape == (10, 2, len(frame))
        raw_versions[name] = ps.mean(0)
        versions[name] = sharpen(ps.mean(0))
        per_seed[name] = ps
    if "06_add_rates" in versions:
        assert np.max(np.abs(versions["06_add_rates"] - versions["corrected"])) < 1e-10
    result = {
        "completed_variants": list(per_seed),
        "total_planned": len(manifest["variants"]),
        "models": {},
        "incremental": {},
        "execution": {},
    }
    masks = {
        "all": np.ones(len(frame), dtype=bool),
        "eligible": ~frame.has_debut.to_numpy(bool),
        "debut": frame.has_debut.to_numpy(bool),
    }
    for name, p in versions.items():
        entry = {}
        for subset, m in masks.items():
            y = frame.loc[m, "Winner"].eq("Red").to_numpy(int)
            entry[subset] = {}
            for j, kind in enumerate(["real", "corrupted"]):
                scores = metrics(y, p[j, m])
                scores["paired_vs_frozen"] = paired_ci(frame[m], p[j, m], versions["old_frozen"][j, m])
                if name in per_seed:
                    seed_metrics = [metrics(y, sharpen(ps[j, m]))["log_loss"] for ps in per_seed[name]]
                    scores["individual_seed_logloss_range"] = [
                        float(min(seed_metrics)),
                        float(max(seed_metrics)),
                    ]
                    scores["individual_seed_logloss_sd"] = float(np.std(seed_metrics))
                # Probability-quality sensitivity, not a new betting strategy.
                raw = (
                    raw_versions[name][j, m]
                    if name in raw_versions
                    else 1 / (1 + np.exp(-np.log(p[j, m] / (1 - p[j, m])) / 1.25))
                )
                scores["unsharpened_logloss_diagnostic"] = metrics(y, raw)["log_loss"]
                entry[subset][kind] = scores
        pred = frame[["date", "R_fighter", "B_fighter"]].copy()
        pred["ablation_real"] = p[0]
        pred["ablation_corrupted"] = p[1]
        priced = prices.merge(pred, on=["date", "R_fighter", "B_fighter"], validate="many_to_one")
        entry["price_windows"] = {}
        for window, m in [
            ("polymarket", priced.venue.str.startswith("polymarket")),
            ("kalshi_early", priced.venue.str.startswith("kalshi") & (priced.date <= "2026-05-16")),
            ("kalshi_late", priced.venue.str.startswith("kalshi") & (priced.date > "2026-05-16")),
        ]:
            g = priced[m]
            entry["price_windows"][window] = {
                k: flat_metrics(g, g["ablation_" + k].to_numpy()) for k in ["real", "corrupted"]
            }
        kalshi = priced[priced.venue.str.startswith("kalshi") & priced.priced_at_ask.astype(bool)].copy()
        snapshot = pd.read_parquet(
            BASE / "data/raw/kalshi/snapshots/historical_T-90min_perfight_combined.parquet"
        )
        lookup = snapshot.drop_duplicates("event_ticker").set_index("event_ticker").close_time
        kalshi["sequence_proxy"] = pd.to_datetime(kalshi.market_id.map(lookup), utc=True)
        result["execution"][name], _ = replay(kalshi, "ablation", 1000, 0.01, "card_end")
        result["models"][name] = entry
    sequence = [
        "00_old_features_fixed_symmetry",
        "01_verified_results",
        "02_add_rankings",
        "03_add_skill",
        "04_add_profiles",
        "05_add_counters",
        "06_add_rates",
    ]
    pairs = [(a, b) for a, b in zip(sequence, sequence[1:], strict=False)]
    pairs += [("corrected", "full_restore_old_" + g) for g in ["rates", "counters", "skill"]]
    pairs += [("old_frozen", "00_old_features_fixed_symmetry")]
    pairs += [("corrected", "full_complete_missing_profiles")]
    pairs += [("full_complete_missing_profiles", "profiles_plus_prior_external_counters")]
    pairs += [("corrected", "profiles_plus_prior_external_counters")]
    for a, b in pairs:
        if a not in versions or b not in versions:
            continue
        result["incremental"][a + " -> " + b] = {
            subset: {
                kind: paired_ci(frame[m], versions[b][j, m], versions[a][j, m])
                for j, kind in enumerate(["real", "corrupted"])
            }
            for subset, m in masks.items()
        }
    np.savez(OUT / "all_variant_predictions.npz", **versions)
    (OUT / "ablation_results.json").write_text(json.dumps(result, indent=2))
    rows = []
    for name, r in result["models"].items():
        for k in ["real", "corrupted"]:
            row = {
                "variant": name,
                "model": k,
                "logloss_all": r["all"][k]["log_loss"],
                "brier_all": r["all"][k]["brier"],
                "logloss_eligible": r["eligible"][k]["log_loss"],
                "brier_eligible": r["eligible"][k]["brier"],
                "auc_eligible": r["eligible"][k]["auc"],
            }
            row.update({w + "_roi": d[k]["flat_roi"] for w, d in r["price_windows"].items()})
            rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "ablation_summary.csv", index=False)
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print("Completed", len(per_seed), "of", len(manifest["variants"]))


if __name__ == "__main__":
    summarize()

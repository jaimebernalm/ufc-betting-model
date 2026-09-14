"""Reproduce the saved positive reports on exactly their original eligible populations."""

import json

import pandas as pd
from evaluate_corrected_system import BASE, OUT, canonical, flat_metrics, normalise

old = pd.read_parquet(BASE / "artifacts/metrics/strict_multivenue_postcutoff_fights.parquet")
old = old[old.universe.isin(["polymarket", "kalshi"])].reset_index(drop=True)
f = pd.read_parquet(OUT / "paired_price_comparison.parquet")
assert len(old) == len(f)
for i, r in f.iterrows():
    s = old.iloc[i]
    assert pd.Timestamp(s.date) == r.date and s.market_id == r.market_id
    assert normalise(canonical(s.R_fighter, r.weight_class)) == normalise(r.R_fighter)
    assert s.price_red == r.price_red and s.price_blue == r.price_blue
f["historical_report_real"] = old.p_red_real
f["historical_report_corrupted"] = old.p_red_corrupted
f["has_debut_corrected"] = f.has_debut
f["has_debut"] = old.has_debut
result = {}
for name, mask in [
    ("Polymarket positivo", f.venue.str.startswith("polymarket")),
    ("Kalshi positivo", f.venue.str.startswith("kalshi") & (f.date <= pd.Timestamp("2026-05-16"))),
    ("Kalshi posterior", f.venue.str.startswith("kalshi") & (f.date > pd.Timestamp("2026-05-16"))),
]:
    g = f[mask]
    result[name] = {
        "date_min": str(g.date.min().date()),
        "date_max": str(g.date.max().date()),
        "eligible": int((~g.has_debut).sum()),
        "true_asks": int(g.priced_at_ask.sum()),
        "metrics": {},
    }
    for v in ["historical_report", "old_frozen", "corrected"]:
        result[name]["metrics"][v] = {
            k: flat_metrics(g, g[v + "_" + k].to_numpy()) for k in ["real", "corrupted"]
        }
    print(name, result[name]["eligible"], "asks", result[name]["true_asks"])
    for v, d in result[name]["metrics"].items():
        print(v, [(k, round(x.get("flat_roi", 0) * 100, 2), x["bets"]) for k, x in d.items()])
(OUT / "historical_report_window_comparison.json").write_text(
    json.dumps(
        {
            "note": "Same archived fight population, eligibility, prices and policy; original report probabilities versus frozen deployed artifacts versus corrected retraining. Corrected eligibility comparison is separate in positive_window_comparison.json.",
            "windows": result,
        },
        indent=2,
    )
)

# Also report the deployable corrected identity/no-debut eligibility, held
# identical across old and new predictions within this second comparison.
f["has_debut"] = f.has_debut_corrected
corrected_population = {}
for name, mask in [
    ("Polymarket hasta 2026-05-16", f.venue.str.startswith("polymarket")),
    ("Kalshi hasta 2026-05-16", f.venue.str.startswith("kalshi") & (f.date <= pd.Timestamp("2026-05-16"))),
    ("Kalshi posterior", f.venue.str.startswith("kalshi") & (f.date > pd.Timestamp("2026-05-16"))),
]:
    g = f[mask]
    corrected_population[name] = {
        "fights": len(g),
        "eligible": int((~g.has_debut).sum()),
        "true_asks": int(g.priced_at_ask.sum()),
        "models": {
            v: {k: flat_metrics(g, g[v + "_" + k].to_numpy()) for k in ["real", "corrupted"]}
            for v in ["old_frozen", "old_models_corrected_features", "corrected"]
        },
    }
(OUT / "positive_window_comparison.json").write_text(json.dumps(corrected_population, indent=2))

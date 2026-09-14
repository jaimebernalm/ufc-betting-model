"""Compare archived predictions before scheduled start against the corrected bundle."""

import json

import pandas as pd
from evaluate_corrected_system import BASE, OUT, flat_metrics, match_row


def run():
    predictions = pd.read_parquet(OUT / "all_postcutoff_predictions.parquet")
    by_date = {}
    for i, r in predictions.iterrows():
        by_date.setdefault(pd.Timestamp(r.date).normalize(), []).append((i, r))
    rows = []
    exclusions = []
    for path in sorted((BASE / "data/processed/bet_notifications").glob("*.json")):
        r = json.loads(path.read_text())
        model = r.get("model") or {}
        snap = r.get("kalshi_snapshot", {})
        fight = r.get("fight", {})
        if r.get("dry_run") or not model or model.get("sharpen_T") != 1.25:
            exclusions.append(
                {"file": path.name, "reason": "dry-run, missing model, or different historical sharpening"}
            )
            continue
        if pd.Timestamp(r["captured_at_utc"]) >= pd.Timestamp(r["fight_date_utc"]):
            exclusions.append(
                {"file": path.name, "reason": "capture not demonstrably before scheduled start"}
            )
            continue
        date = path.name[:10]
        a, b = fight["fighter_a_ufc"], fight["fighter_b_ufc"]
        if fight.get("kalshi_swap_vs_ufc"):
            a, b = b, a
        match = match_row(by_date, a, b, date)
        if not match:
            exclusions.append({"file": path.name, "reason": "no unique outcome match"})
            continue
        idx, forward = match
        entry = predictions.iloc[idx].to_dict()
        pa, pb = snap.get("a_ask"), snap.get("b_ask")
        if pa is None or pb is None:
            continue
        entry.update(
            price_red=pa if forward else pb,
            price_blue=pb if forward else pa,
            fee_rate=float(model.get("fee_coeff", 0.07)),
            fee_exponent=1.0,
            capture_file=path.name,
        )
        for kind in ["real", "corrupted"]:
            p = model[f"p_a_{kind}"]
            entry[f"archived_{kind}"] = p if forward else 1 - p
        rows.append(entry)
    frame = pd.DataFrame(rows)
    results = {
        "paired_captures": len(frame),
        "exclusions": exclusions,
        "old_predictions": "as actually archived; requires T=1.25, non-dry-run, before scheduled start",
        "models": {},
    }
    for name in ["archived", "corrected"]:
        results["models"][name] = {
            kind: flat_metrics(frame, frame[f"{name}_{kind}"].to_numpy()) for kind in ["real", "corrupted"]
        }
    frame.to_parquet(OUT / "paired_archived_live.parquet", index=False)
    (OUT / "archived_live_comparison.json").write_text(json.dumps(results, indent=2, default=str))
    print("paired pre-fight captures", len(frame))


if __name__ == "__main__":
    run()

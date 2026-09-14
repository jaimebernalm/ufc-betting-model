"""Reconcile training outcome envelope with pinned UFCStats bout results.

NC revisions are final-result corrections; their publication timestamps are
not in this source. Historical backtests must disclose that vintage limitation.
"""

import json
import re

import pandas as pd

from ufc_pred.ingest.identity import canonical, canonicalize_frame, normalise
from ufc_pred.ingest.kaggle_mdabbert import HISTORY_PARQUET
from ufc_pred.paths import ROOT


def build():
    source = ROOT / "data/raw/ufcstats_export"
    out = ROOT / "artifacts/corrected_2026_09_14"
    f = canonicalize_frame(pd.read_parquet(HISTORY_PARQUET))
    f["date"] = pd.to_datetime(f.date)
    events = pd.read_csv(source / "ufc_event_details.csv")
    events["EVENT"] = events.EVENT.str.strip()
    events["date"] = pd.to_datetime(events.DATE)
    results = pd.read_csv(source / "ufc_fight_results.csv")
    results["EVENT"] = results.EVENT.str.strip()
    results = results.merge(events[["EVENT", "date"]], on="EVENT", validate="many_to_one")
    lookup = {}
    for r in results.itertuples(index=False):
        names = re.split(r"\s+vs\.?\s+", r.BOUT, maxsplit=1)
        if len(names) != 2:
            continue
        names = [normalise(canonical(n, r.WEIGHTCLASS)) for n in names]
        lookup[(r.date, frozenset(names))] = (names, str(r.OUTCOME).split("/"), r.URL)
    records = []
    changes = []
    missing = []
    for _, row in f.iterrows():
        names = [normalise(row.R_fighter), normalise(row.B_fighter)]
        match = lookup.get((row.date, frozenset(names)))
        if match is None:
            missing.append(
                {
                    "date": str(row.date.date()),
                    "red": row.R_fighter,
                    "blue": row.B_fighter,
                    "reason": "no completed raw bout at this date; excluded rather than assigned a result",
                }
            )
            continue
        raw_names, outcomes, url = match
        outcome = outcomes[raw_names.index(names[0])].strip()
        winner = {"W": "Red", "L": "Blue", "D": "Draw", "NC": "NC"}.get(outcome)
        if winner is None:
            raise ValueError(f"Unknown outcome {outcome}")
        if winner != row.Winner:
            changes.append(
                {
                    "date": str(row.date.date()),
                    "red": row.R_fighter,
                    "blue": row.B_fighter,
                    "old": row.Winner,
                    "corrected": winner,
                    "source": url,
                }
            )
        row["Winner"] = winner
        records.append(row)
    corrected = pd.DataFrame(records).reset_index(drop=True)
    corrected.to_parquet(out / "verified_history.parquet", index=False)
    (out / "result_corrections.json").write_text(
        json.dumps(
            {
                "changes": changes,
                "excluded": missing,
                "outcome_vintage_limitation": "final NC revisions lack announcement timestamps; historical as-of publication cannot be proven",
            },
            indent=2,
        )
    )
    print(
        f"verified history: {len(corrected)} rows, {len(changes)} result changes, {len(missing)} unmatched excluded"
    )


if __name__ == "__main__":
    build()

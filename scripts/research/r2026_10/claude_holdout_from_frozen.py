"""Reproduce the 2024-2026 confirmation tables from the frozen, self-contained evaluation file.

Reads only artifacts/r2026_10/claude/holdout_eval.parquet (1,367 fights; contains y, q, offered
odds, debut flag, odds source and every model's probability) and prints to stdout. It never
reloads BFO/Kaggle data, never refits and never overwrites the frozen evidence.
"""

import hashlib

import pandas as pd

from ufc_pred.research.harness import OUT, delta_ll, roi_summary

FROZEN = OUT / "claude/holdout_eval.parquet"
EXPECTED_SHA256 = "3a6bc351cf639875a4580eb743415c5c4d3d07ec2903f3e435ecf793abbafd07"


def main():
    digest = hashlib.sha256(FROZEN.read_bytes()).hexdigest()
    assert digest == EXPECTED_SHA256, f"frozen holdout file changed: {digest}"
    df = pd.read_parquet(FROZEN)
    pcols = [c for c in df if c.startswith("p_")]
    pd.set_option("display.width", 220)
    print(f"{len(df)} fights, {df.card.nunique()} cards, {int((~df.debut).sum())} non-debut")
    print(pd.DataFrame([delta_ll(df, c) for c in pcols]).round(5).to_string(index=False))
    eco = df[~df.debut]
    rows = [roi_summary(eco, c, t) for c in pcols for t in (0.0, 0.02)]
    print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()

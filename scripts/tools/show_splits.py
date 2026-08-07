"""Print the frozen split sizes and the market baseline on each split.

This is the wall every model has to climb over.
"""

from __future__ import annotations

import pandas as pd

from ufc_pred.backtest.metrics import evaluate, market_no_vig_prob_red
from ufc_pred.ingest.kaggle_mdabbert import HISTORY_PARQUET
from ufc_pred.utils.time_splits import recency_weights, split


def main() -> None:
    fights = pd.read_parquet(HISTORY_PARQUET)
    fights = fights[fights["Winner"].isin(["Red", "Blue"])].copy()
    splits = split(fights)

    print("=== Split sizes ===")
    print(splits.summary().to_string(index=False))

    print("\n=== Recency weights on training set ===")
    w = recency_weights(splits.train["date"])
    print(
        f"min={w.min():.3f}  median={pd.Series(w).median():.3f}  "
        f"max={w.max():.3f}  effective_n={w.sum():.0f} / {len(w)}"
    )

    print("\n=== Market baseline (no-vig implied prob) ===")
    print(f"{'split':<6} {'n':>5} {'log_loss':>10} {'brier':>8} {'ece':>7} {'acc':>6}")
    for name, part in [("train", splits.train), ("val", splits.val), ("test", splits.test)]:
        sub = part.dropna(subset=["R_odds", "B_odds"])
        y = (sub["Winner"] == "Red").astype(int).to_numpy()
        p = market_no_vig_prob_red(sub)
        m = evaluate(y, p, label=name)
        print(
            f"{name:<6} {m['n']:>5d} {m['log_loss']:>10.4f} {m['brier']:>8.4f} "
            f"{m['ece']:>7.4f} {m['accuracy_argmax']:>6.3f}"
        )


if __name__ == "__main__":
    main()

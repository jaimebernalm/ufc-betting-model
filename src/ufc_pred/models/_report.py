"""Console rendering for a training run record. One implementation, not ten."""

from __future__ import annotations

import pandas as pd

from ufc_pred.models._spec import ModelSpec

_METRIC_COLS = ["model", "label", "n", "log_loss", "brier", "ece", "accuracy_argmax"]


def print_run(run: dict, spec: ModelSpec | None = None) -> None:
    version = run["version"]
    print(f"version: {version}  (trained {run['trained_at']})")

    s = run["splits"]
    print(
        f"splits:  train≤{s['train_end']} (n={s['n_train_pre_aug']}, aug={s['n_train_post_aug']})"
        f"  val≤{s['val_end']} (n={s['n_val']})  test held out"
    )

    if "ensemble" in run:
        e = run["ensemble"]
        print(f"ensemble: K={e['size']}  feature_recipe={run['feature_recipe']}")
    else:
        cb = run["catboost"]
        print(f"catboost: best_iter={cb['best_iteration']}  tree_count={cb['tree_count']}")

    if spec is not None and spec.extra_report is not None:
        for line in spec.extra_report(run):
            print(line)

    print(f"saved:   {run['model_path']}")
    print()

    rows = pd.DataFrame([run["metrics"]["train"], run["metrics"]["val"], run["metrics"]["val_market"]])
    rows.insert(0, "model", [version, version, "market"])
    print(rows[_METRIC_COLS].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if "betting" in run:
        print()
        print("Betting (val, edge ≥ 5%):")
        for k, v in run["betting"].items():
            print(
                f"  {k:22s}  n={v['n_bets']:3d}  roi={v['roi_pct']:+.2f}%  "
                f"CI({v['ci95_low']:+.1f}, {v['ci95_high']:+.1f})  sharpe={v['sharpe']:+.3f}"
            )


def main(spec: ModelSpec) -> None:
    """Standard `__main__` body for a ladder rung."""
    from ufc_pred.models._harness import run_training

    print_run(run_training(spec), spec)

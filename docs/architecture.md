# Architecture

## Data flow

```
                    ┌─────────────────────────────────────────┐
  Kaggle dump ──┐   │  ingest/                                │
  UFCStats  ────┼──▶│  scrape, normalize, name-match          │
  Rankings  ────┘   │  → data/processed/fights.parquet        │
                    └───────────────────┬─────────────────────┘
                                        ▼
                    ┌─────────────────────────────────────────┐
                    │  features/                              │
                    │  static · derived · Bayesian skill      │
                    │  (walk-forward: monthly posterior refit)│
                    └───────────────────┬─────────────────────┘
                                        ▼
                    ┌─────────────────────────────────────────┐
                    │  models/                                │
                    │  ModelSpec → _harness → CatBoost        │
                    │  10-seed ensemble, no early stopping    │
                    └───────────────────┬─────────────────────┘
                                        ▼
     Kalshi ───────▶┌─────────────────────────────────────────┐
     Polymarket ───▶│  inference/                             │
     (live prices)  │  predict → fee-correct → ¼-Kelly → cap  │
                    └───────────────────┬─────────────────────┘
                                        ▼
                          notification (no order placement)
```

## Package layout

| module | responsibility |
|---|---|
| `ingest/` | Kaggle loader, UFCStats scraper + incremental updater, Kalshi and Polymarket clients, fighter name matching, rankings attachment |
| `features/` | `static_v1` (base columns + symmetry augmentation), `derived_v2` (layoff/activity/career ratios), `skill_v3` / `skill_v3_1` (Bayesian skill), `joins` (per-recipe feature assembly), `elo`, `fatigue` |
| `models/` | `_spec` (declarative rung definition), `_harness` (shared training pipeline), `_report` (console rendering), `baseline_v*` (one file per ladder rung), `wrappers` |
| `calibration/` | isotonic and conformal methods |
| `backtest/` | `bet_eval` (fee-correct ROI + CIs), `strategy_grid`, `metrics`, `universe`, `kalshi_match` |
| `inference/` | `upcoming_builder`, `skill_for_upcoming`, `ensemble_predict`, `sizing`, `recommend`, venue card fetchers |
| `ops/` | bankroll reconciliation, fill capture |
| `cli/` | five console entry points |
| `notify.py` | macOS notification + ntfy relay |

Outside the package: `scripts/research/` (one-shot analyses, preserved as a
record), `scripts/tools/` (data pipeline and maintenance), `notebooks/`,
`tests/`.

## Feature families

**Static (`static_v1`)** — the base fight table: physical attributes, stance,
weight class, record, rankings, betting odds. Categorical columns are passed to
CatBoost natively rather than one-hot encoded.

*Symmetry augmentation*: each training fight is duplicated with the Red and Blue
corners swapped and the label flipped, so the model cannot learn a corner bias.
Post-fight columns and odds are excluded from the feature matrix.

**Derived (`derived_v2`)** — 12 columns computable without any scrape:

```
R/B_days_since_last_fight   R/B_fights_last_365d   R/B_fights_last_730d
R/B_career_fights           R/B_win_rate           R/B_finish_rate
```

**Bayesian skill (`skill_v3`)** — a hierarchical Bradley-Terry model in NumPyro:

```
skill[f]        ~ Normal(wc_mean[weight_class(f)], wc_sigma[weight_class(f)])
P(a beats b)    = sigmoid(skill[a] − skill[b])
y_i             ~ Bernoulli, weighted by recency
```

Emits two columns per fight — the posterior **mean** and **standard deviation**
of `skill[Red] − skill[Blue]`. The std matters: it encodes how much the model
actually knows about this matchup, and CatBoost leans on it more heavily than on
the mean (see the v3.3 feature importances in [results.md](results.md)).

`skill_v3_1` is the same model with skill following a random walk over a career,
which addresses the long-layoff failure mode where a static estimate goes stale.

**Anti-leak construction.** The posterior for a fight on date *X* is fit using
only fights strictly before *X*. In practice the pipeline refits monthly on the
prior-only slice and assigns that month's fights from that fit.

## The symmetry sign-flip

This is the subtlest correctness detail in the codebase. Augmentation swaps the
corners, so any column expressing a *signed difference* must be negated on the
swapped half, while magnitude-only columns must not:

```python
flip_signed_columns(X, ("skill_diff_mean",))  # negated
#                       skill_diff_std          # untouched — it is a magnitude
```

Getting this wrong is silent: training simply sees half its rows with the skill
edge pointing the wrong way. `Recipe.flip_columns` declares it per version and
`tests/test_model_ladder_specs.py` asserts each recipe flips exactly the columns
its join produces.

## The model ladder

Each rung changes exactly one thing relative to its predecessor, with CatBoost
hyperparameters held fixed across all of them:

```python
SPEC = ModelSpec(
    version="v3_catboost_skill",
    recipe=Recipe(
        name="v3 (scalar Bayesian skill)",
        join=join_skill_v3,
        flip_columns=("skill_diff_mean",),
    ),
    early_stopping=True,
    diagnostics=_coverage,
)
```

`_harness.run_training` handles split → prepare → recency-weight → fit →
evaluate → persist for every rung, so the *diff between two versions is the diff
between two specs*. A test asserts no rung drifts on `iterations`,
`learning_rate`, `depth`, `l2_leaf_reg`, or `loss_function` — the comparison
would be meaningless if one did.

## Read-only by construction

`KalshiClient` exposes balance, market, and orderbook reads. There is no
order-placement method anywhere in the package. The watchdog computes a
recommendation and sends a notification; a human places the bet. This is a
deliberate constraint, not an unfinished feature.

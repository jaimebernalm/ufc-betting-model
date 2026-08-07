# Methodology

The modeling choices here are mostly *negative* results — things that looked
right and weren't. This document records what was tried, what the evidence said,
and what the system does as a result.

---

## 1. The split is frozen, and the test set is spent

```python
TRAIN_END = 2022-12-31
VAL_END   = 2023-12-31
TEST_END  = 2026-03-28
```

These dates were fixed before any model comparison and never moved. A version
comparison is only meaningful if the protocol is identical across versions, so
`utils/time_splits.py` carries a "DO NOT CHANGE" warning and every rung imports
from it rather than defining its own.

The test set was touched **once**, as a single committed evaluation
([results.md](results.md)). It is spent. Everything else — feature ablations,
hyperparameter decisions, strategy selection — happened on validation.

## 2. Select on ROI, not log-loss

The two pull in opposite directions, and this is the single most consequential
finding in the project.

Same recipe, same seed, same data, varying only early stopping:

| | trees | val log-loss | val ECE | ROI (Kalshi-like) |
|---|---|---|---|---|
| early stopping **on** | 182 | **0.6189** | **0.0703** | −8.26% |
| early stopping **off** | 2000 | 0.6452 | 0.1054 | **+2.86%** |

Every predictive metric degrades, and betting return flips from −8.3% to +2.9%.
Sharpening calibration (T < 1), which minimizes log-loss by construction,
likewise hurt ROI.

The reason is that betting return does not depend on being right on average. It
depends on being right *where the market is wrong*, and those are the tails. A
loss function that rewards well-behaved central predictions will happily trade
away the extreme predictions that carry all the profit.

Consequence: models are chosen on the betting backtest at the deployment edge
threshold and fee model. Log-loss and ECE are reported as guardrails — a version
may be rejected for badly degrading them — but they are not the objective.

## 3. Early stopping on the validation set is a leak

`use_best_model=True` with `eval_set=val_pool` tunes the tree count against
validation log-loss. That same validation set is then used to judge betting
return. The model has been fit to the selection set through a side channel — and
by §2, the objective it is being fit to actively works against profit. The A/B
table above is this leak measured.

Later rungs (v7, v7.1) train to a fixed 2,000 iterations and are selected on the
backtest instead. `ModelSpec.early_stopping` makes this an explicit per-version
flag rather than an accident of copy-paste, and a test asserts that no rung
silently changes its tuning parameters relative to the shared base config.

## 4. Strict walk-forward — no feature may see the future

Every feature must be computable strictly before the fight date.

The Bayesian skill model refits its posterior monthly on the prior-only slice and
assigns each month's fights from that fit. The entity index (fighter id
numbering) is built from full history, which is safe because it reads no
outcomes — but no outcome, rating, or rolling statistic ever crosses the fight
date.

## 5. Recency weighting

```python
weight = 0.5 ** (age_years / 4.0)
```

A 4-year half-life, tuned as a hyperparameter rather than assumed. Fights at the
reference date get weight 1.0; fights four years earlier get 0.5.

## 6. Fee formulas come from primary sources

Cheap flat-percentage approximations of venue fees were wrong by **5–20× near the
extremes**. The real Kalshi fee is quadratic in price:

```
fee per contract = 0.07 × P × (1 − P)
```

charged upfront per contract. Polymarket and sportsbooks differ *structurally* —
commission on net winnings versus vig baked into the price versus per-contract
upfront — so a formula verified at one venue tells you nothing about another.

All Kelly sizing runs on fee-corrected odds. Getting this wrong silently inflates
every backtest, because the edge threshold is applied to a price that isn't real.

## 7. A single seed is not a model

Identical data, identical hyperparameters, differing only in `random_seed`,
produced final bankrolls from **$243,529 to $8,968,774** — a 37× spread, wider
than the spread across eight different training cutoffs at a fixed seed.

Any single-seed result is therefore a sample, not a measurement. Deployment
averages `predict_proba` across seeds 0–9 so it lands near the median instead of
a lucky or unlucky tail.

## 8. Ensembling is not universally good

Ensembling smooths predictions toward 0.5, which kills high-edge bets. Whether
that helps depends on the architecture:

| architecture | single seed | 10-seed ensemble |
|---|---|---|
| v3.3 (weaker, variance-driven) | +4.1% | **+5.2%** |
| v3 (sharper, signal-dense) | **+8.4%** | +2.4% |

For the noisier model, averaging diluted noise. For the sharper one, the extreme
predictions *were* the signal, and smoothing destroyed them. So ensemble-versus-
single is ablated per architecture rather than applied as a blanket rule.

## 9. Bet sizing: fractional Kelly with hard caps

Full Kelly assumes your probability is the true probability. An overconfident
model then overstakes and can ruin an edge that was genuinely positive.

Deployment uses **¼-Kelly** on fee-corrected odds (1/10-Kelly on the conservative
account), with a hard per-fight cap of 10% of bankroll on that account and a
liquidity cap tied to visible order-book depth. Bets are placed only above a
**3% edge** threshold.

Even so, the test-set backtest shows an **84% maximum drawdown** at ¼-Kelly.
Fractional Kelly is a safety margin against model overconfidence, not against
volatility — the volatility is real and large.

---

## What this methodology does not fix

The edge is not stationary. A frozen model evaluated strictly post-cutoff earned
+39% on Kalshi through 2026-05-16 and −30% after it. None of the discipline above
detects or prevents that; it is a property of the market, not the estimator. See
[results.md](results.md#the-edge-is-not-stable-over-time).

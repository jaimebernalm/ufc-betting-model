# Results

All ladder numbers below were regenerated in the pinned environment
(`requirements.lock`, CatBoost 1.2.10, Python 3.11) and reproduce from a clean
clone via `python -m ufc_pred.models.baseline_<version>`.

> **Reproducibility note.** Metrics recorded during the original 2026-05 runs do
> not reproduce exactly today — v3 was logged at val log-loss 0.6119 / 290 trees
> and now yields 0.6189 / 182 on identical code. The cause is library drift, not
> a code change (verified by running the pre-refactor implementation
> side-by-side in the current environment: bit-identical to the refactored
> harness). Every figure on this page comes from the current environment.

---

## The model ladder (validation)

Validation = 2023 fights, n=504. Market row uses no-vig closing probabilities and
covers the n=416 fights that carry odds.

| version | features | val log-loss ↓ | Brier ↓ | ECE ↓ | acc ↑ | trees |
|---|---|---|---|---|---|---|
| v1 | static, logistic regression | 0.6297 | 0.2197 | 0.0411 | 65.3% | — |
| v1.1 | static, CatBoost | 0.6216 | 0.2164 | 0.0373 | 64.1% | 356 |
| v1.2 | v1.1 + isotonic calibration | 0.6296 | 0.2205 | 0.0516 | 64.9% | — |
| v2 | + derived (layoff, activity) | 0.6207 | 0.2152 | 0.0424 | 65.5% | 215 |
| v3 | + Bayesian skill (scalar) | 0.6189 | 0.2144 | 0.0703 | 66.7% | 182 |
| v3.1 | + skill (time-varying) | **0.6118** | **0.2117** | 0.0490 | 66.5% | 404 |
| v3.2 | skill + derived | 0.6174 | 0.2142 | **0.0385** | 67.1% | 270 |
| v3.3 | stacked skill (both) | 0.6132 | 0.2123 | 0.0636 | **67.7%** | 303 |
| v7 | v3.3 × 10 seeds, no early stop | 0.6293 | 0.2171 | 0.0563 | 66.3% | 2000 |
| v7.1 | v3 × 10 seeds, no early stop | 0.6323 | 0.2181 | 0.0676 | 66.5% | 2000 |
| — | **market (no-vig)** | **0.5864** | **0.2004** | 0.0486 | **69.2%** | — |

**The market wins every predictive column.** No rung of the ladder beats the
closing line on log-loss, Brier, or accuracy. The deployed models (v7, v7.1) are
among the *worst* on log-loss. That is the design, not a defect — see below.

### Betting return on validation (edge ≥ 5%)

| version | sportsbook (with vig) | no-vig, no fee | Kalshi-like (7% fee) |
|---|---|---|---|
| v7 | +5.00% (n=326) | +7.01% (n=371) | **+4.69%** (n=342) |
| v7.1 | +4.57% (n=330) | +7.29% (n=371) | **+2.96%** (n=345) |

All confidence intervals at this sample size span zero (v7 Kalshi-like:
[−6.9%, +16.4%]). One year of validation fights is not enough to establish an
edge; it is enough to reject a model.

---

## Early stopping is a leak — the clean A/B

Same v3 recipe, same seed, same data. The only difference is early stopping:

| | trees | val log-loss | val ECE | ROI (Kalshi-like) |
|---|---|---|---|---|
| early stopping **on** | 182 | **0.6189** | **0.0703** | **−8.26%** |
| early stopping **off** | 2000 | 0.6452 | 0.1054 | **+2.86%** |

Turning early stopping off makes the model **worse on every predictive metric**
and flips betting return from −8.3% to +2.9%.

Early stopping tunes the tree count against validation log-loss, and that same
validation set is then used to judge profit. The model has been fit to the
selection set through a side channel — and the thing it optimizes for is
actively anti-correlated with the thing that makes money.

Reproduce: set `early_stopping=False` on any rung's `ModelSpec`.

## What CatBoost actually uses from the skill model

v3.3 feature importances over the four skill columns:

```
skill_diff_std_v3       2.369    ← uncertainty
skill_diff_mean_v3_1    2.222
skill_diff_std_v3_1     1.975    ← uncertainty
skill_diff_mean_v3      0.887    ← the actual skill estimate, least useful
```

The **posterior standard deviation outranks the posterior mean**. What helps most
is not the skill estimate — it is knowing how *uncertain* that estimate is. The
market prices the matchup; the model's advantage comes from recognizing where
its own information is thin.

---

## Test set — the single committed evaluation

Touched once, at the end. **n = 1,141 fights, 2024-01-13 → 2026-03-28.**

| | log-loss ↓ | Brier ↓ | ECE ↓ | accuracy ↑ |
|---|---|---|---|---|
| Model | 0.6226 | 0.2164 | 0.0629 | 65.6% |
| Market (no-vig) | **0.5816** | **0.1986** | **0.0296** | **69.5%** |

Flat stakes, real Kalshi fees, 5% edge threshold:

| universe | bets | hit rate | ROI | 95% CI |
|---|---|---|---|---|
| all fights | 846 | 54.4% | **+10.88%** | [+2.5%, +19.4%] |
| deployable (no debut) | 700 | 53.4% | **+7.79%** | [−1.0%, +16.8%] |

**The deployable interval includes zero.** Excluding the 201 debut fights — which
deployment must exclude, since a debutant has no skill posterior — the test-set
edge is no longer statistically distinguishable from zero.

At ¼-Kelly the same period compounds $1 → $3,500 on the full universe, with an
**84% maximum drawdown**. On the deployable universe it compounds $1 → $23.69
with an 85% drawdown. The gap between those two figures is the honest measure of
how much of the headline result rested on fights the system cannot actually bet.

---

## The edge is not stable over time

Model frozen at 2025-11-30, evaluated strictly forward on fights it never saw.
**n = 331 outcomes, 2025-12-06 → 2026-08-01.**

| venue | segment | bets | hit | ROI | 95% CI |
|---|---|---|---|---|---|
| Polymarket | all | 197 | 56.3% | +13.0% | [−3.3, +29.7] |
| Polymarket | no debut | 156 | 62.2% | **+22.5%** | **[+4.2, +41.5]** |
| Polymarket | debut only | 41 | 34.1% | −22.9% | [−57.3, +14.5] |
| Kalshi | all | 114 | 45.6% | −9.1% | [−32.6, +18.8] |
| Kalshi | no debut | 102 | 49.0% | −1.8% | [−27.1, +29.4] |
| Kalshi | debut only | 12 | 16.7% | **−71.7%** | **[−100, −32.9]** |

Splitting Kalshi's no-debut segment chronologically:

| window | bets | hit | ROI | 95% CI |
|---|---|---|---|---|
| → 2026-05-16 | 42 | 61.9% | **+39.0%** | [−9.4, +103.4] |
| 2026-05-16 → | 60 | 40.0% | **−30.3%** | **[−52.8, −5.2]** |

The post-May confidence interval excludes zero. **This is a real decline, not
noise.**

### What the audits could and could not establish

**Ruled out — a backtest pricing artifact.** The post-May Kalshi fights were
re-priced using true asks rather than last-trade prices: 57 of 63 identical,
mean difference $0.0005/contract, and **0 changes to bet eligibility or side
selection**. The decline is not an artifact of how the backtest read prices.

**Consistent with market maturation.** Median Polymarket volume roughly **6×**
across the sample (\$59.6k → \$354k per fight, first vs second chronological
half). A thicker market prices out inefficiency.

**Not model degradation in the usual sense.** Real and corrupted model
predictions correlate at 0.959 over the post-May window and pick the same side on
54 of 56 shared bets. Both declined together.

## Seed variance

Ten models, identical data and hyperparameters, differing only in `random_seed`:

```
final bankroll:  $243,529  ────────────────────────  $8,968,774
                            37× spread
```

Wider than the spread across eight different training cutoffs at a fixed seed.
Any single-seed backtest is a draw from this distribution, not a measurement of
it — which is why deployment averages 10 seeds and why every result here that
comes from a single seed should be read with that spread in mind.

## Ensembling is architecture-dependent

| architecture | single seed | 10-seed ensemble |
|---|---|---|
| v3.3 (noisier) | +4.1% | **+5.2%** |
| v3 (sharper) | **+8.4%** | +2.4% |

Ensembling smooths predictions toward 0.5. For the noisy model that diluted
noise; for the sharp one it destroyed the extreme predictions that carried the
signal. There is no blanket rule — it is ablated per architecture.

---

## Honest summary

- The model is **worse than the market at predicting fights**, on every metric,
  on both validation and test.
- It was **profitable anyway** on the test set (+10.9%), because betting return
  rewards being right where the price is wrong, not being right on average.
- Excluding fights it cannot actually bet, **the test-set edge is not
  statistically distinguishable from zero** (+7.8%, CI [−1.0, +16.8]).
- Live, post-cutoff, it earned **+22.5% on Polymarket** and **−1.8% on Kalshi**,
  with a **statistically significant decline after 2026-05-16**.
- Drawdowns at the deployed Kelly fraction reach **84%**.

The methodology (frozen splits, walk-forward features, fee-correct sizing, seed
ensembling) is sound and is what made the decline *measurable* rather than
invisible. It did not make the edge durable. Documenting that is the point of
this page.

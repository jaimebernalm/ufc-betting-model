# Deployment

The system runs live against Kalshi. It is **notify-only**: it computes
recommendations and pushes them to a phone. A human places every bet. There is
no order-placement code in the package.

---

## Three parallel accounts

The deployment runs three independent virtual bankrolls off the same card, as a
live experiment rather than a single strategy:

| account | Kelly fraction | per-fight cap | model |
|---|---|---|---|
| **A** | 10% | 10% of bankroll | real (10-seed ensemble) |
| **B** | 25% | none | real (same ensemble as A) |
| **C** | 25% | none | corrupted (skill features NaN'd) |

Account **C** exists because of an accident worth preserving. An early diagnostic
run fed the model NaN skill features due to a rename mismatch, and it produced
unexpectedly strong Kelly numbers. Rather than dismiss it, `CorruptedSkillModel`
wraps the trained estimator and applies the NaN substitution deterministically,
so the behaviour can be evaluated honestly rather than remembered as folklore.

All three share one wallet; no transfers happen between the conceptual accounts.

## Model freezing

Both ensembles are frozen at a **2025-11-30** cutoff and average 10 seeds.

Freezing was not laziness. A seed-robust grid over 8 strategies × 10 seeds × 3
accounts on two non-overlapping evaluation windows found that the median-bankroll
winner is a frozen model roughly **6 months stale** at the start of the window.
Rolling retrains — monthly, quarterly, semi-annual, annual — never won median
bankroll on any (account, window) pair. An earlier "rolling monthly helps"
conclusion turned out to be a seed-0 artifact that did not survive the 10-seed
re-run.

## Sizing

```
edge threshold        3%           (gross, pre-fee)
Kelly                 ¼ (accounts B, C) or 1/10 (account A)
per-fight cap         10% of bankroll (account A only)
liquidity cap         5% of visible depth within a 3¢ band
Kalshi fee            0.07 × P × (1 − P) per contract, charged upfront
logit sharpening      T = 1.25
```

Stakes are computed by walking the live order book, so `avg_fill_price` reflects
depth rather than assuming the top-of-book price fills the whole order. The fee
is included in `stake_usd` — it is a real cash outlay, not a deduction from
winnings.

## The watchdog

`ufc-runner --watchdog` ticks every 60 seconds via a launchd agent and captures
each fight at the right moment. Capture triggers, per fight:

- **first_fight** — card opener has no predecessor; capture at T−60min
- **prev_resolved** — the previous fight's market has settled; capture
  immediately, so Kelly sizes off a bankroll that includes every resolved fight
- **fallback_time** — previous fight still unresolved by T−25min (long fight or
  schedule drift); capture anyway and tag `[LATE]`

Bankrolls are synced from settled markets at the start of each tick, and captures
are idempotent — a restart mid-card does not double-record.

### Installing the agent

```bash
sed -e "s|{{PROJECT_ROOT}}|$(pwd)|g" -e "s|{{PYTHON}}|$(which python)|g" \
    configs/launchd/com.ufcbet.watchdog.plist.template \
    > ~/Library/LaunchAgents/com.ufcbet.watchdog.plist
launchctl load ~/Library/LaunchAgents/com.ufcbet.watchdog.plist
launchctl list | grep ufcbet     # exit status should be 0
```

The template invokes `python -m ufc_pred.cli.bet_runner`, not the `ufc-runner`
console script. When the project path contains a space, pip generates a `/bin/sh`
wrapper for console scripts and launchd refuses to execute it
(`Operation not permitted`).

## Recommendation ≠ fill

`data/processed/bet_notifications/` records what was **recommended**;
`kalshi_fills.json` records what was **actually filled**. These diverge — manual
bets placed without an alert, partial fills, hand-adjusted sizes.

Reconciliation attributes real fills pro-rata against each recommendation's
per-account share ratios. Kalshi drops portfolio history within days of
settlement, so fills must be captured while a card is live or shortly after;
`ops.fills` snapshots them into an append-only local store on every tick.

Bankroll sync keys off **settled markets**, not notifications. An earlier version
synced from recommendations and booked a phantom win on a bet that was
recommended but never filled.

## Operational limits

- **macOS only** — notifications go through `osascript`, with an optional
  ntfy.sh relay to a phone. The topic name *is* the authentication; treat
  `NTFY_TOPIC` as a secret.
- **Debut fights are excluded.** A fighter with no UFC history has no skill
  posterior and no career features. Live post-cutoff ROI on debut fights ran
  −23% to −72%. `configs/bankrolls.json` tracks per-card exclusions.
- **Drawdowns are severe.** The test-set backtest shows 84% maximum drawdown at
  ¼-Kelly. That is the strategy behaving as designed, not a malfunction.
- **The edge is not stationary.** See
  [results.md](results.md#the-edge-is-not-stable-over-time) before drawing any
  conclusion about forward performance.

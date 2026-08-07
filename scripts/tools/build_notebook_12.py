"""Generate notebooks/12_tennis_ported_techniques.ipynb.

Simulations for the techniques ported from TennisPred (TENNIS_PORTED_IDEAS.md):
Tier-1 anchor/model-on-top results, sequential per-fight Kelly on the Kalshi
window, segmentation, and the Tier-2 ablation summary. Regenerable: rerun the
scripts in TENNIS_PORTED_FINDINGS.md §Reproduce first, then all cells.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


CELLS = []

CELLS.append(
    md("""# 12 — Tennis-ported techniques: simulations

Evaluates the techniques ported from the TennisPred sister project
(`TENNIS_PORTED_IDEAS.md`, pre-registered 2026-07-09). Full narrative in
`TENNIS_PORTED_FINDINGS.md`; raw sweep logs in `experiments/2026-07-09_*.jsonl`.

| Task | Verdict |
|---|---|
| 1.1 anchor strategy (model-free) | **Not deployable on Kalshi** (no live anchor source); favorite-longshot structure confirmed on the big Polymarket sample |
| 1.2 model-on-top | **Model pure stays deployed** — beats every anchor variant consistently; blends/veto inconsistent across windows |
| 2.4 segmentation | Favorites (chosen price ≥ 0.60) the only both-window-positive cell; no filter adopted, monitor live |
| 2.1 Elo | **KILLED** — every K below baseline, ECE degrades; tennis low-K prior inverts |
| 2.2 fatigue | **KILLED** — all 3 families fail the paired-seed rule (best 4/10) |
| 2.3 half-life | **4y kept** — curve monotone toward shorter (2y +6.6%) but no candidate passes the full pre-registered rule; retest {2,3}y at the 2026-09-30 retrain |

**Pricing/fees everywhere**: Kalshi T-90min per-fight capture, verified quadratic
taker fee `0.07 × P × (1−P)`; val 2023 settles at BFO no-vig + the same fee model.
**Bankroll sims are sequential fight-by-fight** (rule 0.4): stakes on later fights
of a card depend on earlier settlements that same night.
""")
)

CELLS.append(
    code("""import sys, json
from pathlib import Path

ROOT = Path.cwd().parent

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ufc_pred.backtest.kalshi_match import match_kalshi_to_fights
from ufc_pred.backtest.metrics import american_to_implied_prob
from ufc_pred.backtest.strategy_grid import ModelBundle, predict
from ufc_pred.features.static_v1 import _swap_red_blue

pd.set_option('display.precision', 3)
COLORS = {'A': '#1a9641', 'B': '#1f78b4', 'C': '#d7191c', 'grey': '#888888'}
KALSHI_FEE = 0.07
EDGE_THR = 0.03

def eff_dec(price, fee=KALSHI_FEE):
    price = np.asarray(price, float)
    return 1.0 / (price * (1.0 + fee * (1.0 - price)))
""")
)

CELLS.append(
    md("""## Load the Kalshi window + deployed ensemble (as served live)

43 matched fights 2026-01-24 → 2026-05-16, all after the deployment ensemble's
2025-11-30 training cutoff — genuinely out-of-sample for the deployed model.
Predictions are the 10-seed real ensemble, orientation-symmetrized (the
2026-06-11 fix), identical to what `bet_runner` serves.""")
)

CELLS.append(
    code("""fights = pd.read_parquet(ROOT / 'data/processed/fights.parquet')
fights = fights[fights['Winner'].isin(['Red','Blue'])].copy()
fights['date'] = pd.to_datetime(fights['date'])
sk = pd.read_parquet(ROOT / 'data/processed/skill_features_v3.parquet')
sk['date'] = pd.to_datetime(sk['date'])
fights = fights.merge(sk[['date','R_fighter','B_fighter','skill_diff_mean','skill_diff_std']],
                      on=['date','R_fighter','B_fighter'], how='left', validate='many_to_one')

meta, matched = match_kalshi_to_fights(fights)
y_red = (matched['Winner'] == 'Red').to_numpy(int)
pr_k = meta['kal_p_red'].to_numpy(float)
pb_k = meta['kal_p_blue'].to_numpy(float)

mirrored = _swap_red_blue(matched)
per_seed = []
for s in range(10):
    d = joblib.load(ROOT / f'artifacts/models/v3_real_2025_11_30_seed{s}.joblib')
    b = ModelBundle(model=d['model'], columns=d['columns'], cat_features=d['cat_features'])
    per_seed.append(0.5 * (predict(b, matched) + 1.0 - predict(b, mirrored)))
per_seed = np.stack(per_seed)
p_model = per_seed.mean(axis=0)

# Polymarket anchor (pre-fight, no-vig-normalized) for the ablation rows
pm = pd.read_parquet(ROOT / 'data/interim/polymarket_matched_to_kaggle_v2.parquet')
pm['date'] = pd.to_datetime(pm['date'])
key = matched[['date','R_fighter','B_fighter']].merge(
    pm[['date','R_fighter','B_fighter','polymarket_p_red','polymarket_p_blue']],
    on=['date','R_fighter','B_fighter'], how='left', validate='one_to_one')
tot = key['polymarket_p_red'] + key['polymarket_p_blue']
p_anchor = (key['polymarket_p_red'] / tot).to_numpy(float)

print(f'{len(meta)} fights, {meta["date"].min().date()} → {meta["date"].max().date()}')
print(f'ensemble pred mean {p_model.mean():.3f}; anchor coverage {np.isfinite(p_anchor).sum()}/{len(meta)}')
""")
)

CELLS.append(
    md("""## Task 1.2 — anchor vs model vs blends vs veto (flat $1, thr 3%)

The tennis project's humbling result was "the model adds nothing over the
anchor". **UFC is the opposite**: the deployed ensemble beats every
anchor-involving variant consistently across both evaluation windows. Blends
look great here (n≈30) but *hurt* on the 10× bigger val window — classic
small-sample mirage; per rule 0.3 no change is adopted.""")
)

CELLS.append(
    code("""def flat_roi(p_bet, eligible=None, force_fav=None):
    eR, eB = eff_dec(pr_k), eff_dec(pb_k)
    ev_R, ev_B = p_bet * eR - 1.0, (1 - p_bet) * eB - 1.0
    side_r = ev_R >= ev_B
    edge = np.where(side_r, ev_R, ev_B)
    dec = np.where(side_r, eR, eB)
    bets = edge > EDGE_THR
    if eligible is not None: bets &= eligible
    if force_fav is not None:
        chosen_price = np.where(side_r, pr_k, pb_k)
        bets &= (chosen_price >= 0.5) == force_fav
    won = np.where(side_r, y_red == 1, y_red == 0)
    pnl = np.where(won, dec - 1.0, -1.0)[bets]
    n = int(bets.sum())
    if not n: return {'n_bets': 0, 'roi_pct': np.nan, 'ci_lo': np.nan, 'ci_hi': np.nan}
    rng = np.random.default_rng(0)
    means = pnl[rng.integers(0, n, size=(5000, n))].mean(axis=1)
    return {'n_bets': n, 'roi_pct': pnl.mean()*100,
            'ci_lo': np.quantile(means, .025)*100, 'ci_hi': np.quantile(means, .975)*100}

has_anchor = np.isfinite(p_anchor)
rows = [
    {'variant': '(a) anchor pure (Polymarket)', **flat_roi(np.where(has_anchor, p_anchor, np.nan), eligible=has_anchor)},
    {'variant': '(b) model pure — DEPLOYED', **flat_roi(p_model)},
]
for w in (0.25, 0.5, 0.75):
    rows.append({'variant': f'(c) blend w={w}',
                 **flat_roi(np.where(has_anchor, w*p_model + (1-w)*p_anchor, p_model))})
side_of = lambda p: p * eff_dec(pr_k) - 1.0 >= (1-p) * eff_dec(pb_k) - 1.0
agree = side_of(p_model) == side_of(np.where(has_anchor, p_anchor, p_model))
rows.append({'variant': '(d) veto (anchor agrees on side)', **flat_roi(p_model, eligible=has_anchor & agree)})
rows.append({'variant': '(b) favorites only (price ≥ 0.5)', **flat_roi(p_model, force_fav=True)})
rows.append({'variant': '(b) underdogs only (price < 0.5)', **flat_roi(p_model, force_fav=False)})

seed_rois = []
for i in range(10):
    seed_rois.append(flat_roi(per_seed[i])['roi_pct'])
tbl = pd.DataFrame(rows)
print(f'model-pure per-seed ROI: min {min(seed_rois):+.1f}%  med {np.median(seed_rois):+.1f}%  max {max(seed_rois):+.1f}%  (all positive: {all(r>0 for r in seed_rois)})')
tbl.style.format({'roi_pct': '{:+.2f}', 'ci_lo': '{:+.2f}', 'ci_hi': '{:+.2f}'})
""")
)

CELLS.append(
    md("""## Sequential per-fight Kelly on the Kalshi window (rule 0.4)

Bankroll updates after **every fight settlement in true chronological order**
(fight order within a card = market close order), so later stakes that night
compound earlier results — matching how the operator actually bets. Accounts
per DEPLOY.md: A = 10%-Kelly + 10% cap, B = 25%-Kelly + no cap, both $300.
Two prediction sets: deployed model pure vs the favorites-side-only variant
(the only both-window-positive segment from Task 2.4 — shown for information,
NOT adopted).""")
)

CELLS.append(
    code("""order = np.lexsort((pd.to_datetime(meta['close_time']).to_numpy(),
                    meta['date'].to_numpy()))

def kelly_seq(p_bet, kf, cap, start=300.0, fav_only=False):
    eR, eB = eff_dec(pr_k), eff_dec(pb_k)
    bank, traj, log = start, [start], []
    for i in order:
        ev_R, ev_B = p_bet[i]*eR[i]-1, (1-p_bet[i])*eB[i]-1
        side_r = ev_R >= ev_B
        edge = ev_R if side_r else ev_B
        dec = eR[i] if side_r else eB[i]
        price = pr_k[i] if side_r else pb_k[i]
        pch = p_bet[i] if side_r else 1-p_bet[i]
        if edge <= EDGE_THR or (fav_only and price < 0.5):
            continue
        b = dec - 1.0
        fk = (b*pch - (1-pch)) / b
        if fk <= 0: continue
        stake = bank * min(kf*fk, cap)
        won = (y_red[i] == 1) == side_r
        bank += stake*(dec-1.0) if won else -stake
        traj.append(bank)
        log.append({'date': meta['date'].iloc[i], 'won': int(won), 'stake': stake, 'bank': bank})
    return bank, traj, pd.DataFrame(log)

sims = {
    'A model pure': kelly_seq(p_model, 0.10, 0.10),
    'B model pure': kelly_seq(p_model, 0.25, 1.00),
    'A favorites only': kelly_seq(p_model, 0.10, 0.10, fav_only=True),
    'B favorites only': kelly_seq(p_model, 0.25, 1.00, fav_only=True),
}
for name, (final, traj, log) in sims.items():
    wins = log['won'].sum() if len(log) else 0
    print(f'{name:20s} $300 → ${final:8,.2f}   bets={len(log):2d}  wins={wins}')
""")
)

CELLS.append(
    code("""fig, ax = plt.subplots(figsize=(12, 5.5))
styles = {'A model pure': (COLORS['A'], '-'), 'B model pure': (COLORS['B'], '-'),
          'A favorites only': (COLORS['A'], '--'), 'B favorites only': (COLORS['B'], '--')}
for name, (final, traj, log) in sims.items():
    c, ls = styles[name]
    ax.plot(traj, color=c, linestyle=ls, linewidth=2,
            label=f'{name}: $300 → ${final:,.0f}')
ax.axhline(300, color='black', alpha=0.4, linewidth=0.8)
ax.set_xlabel('Bet # (sequential through the Kalshi window, within-card order)')
ax.set_ylabel('Bankroll ($)')
ax.set_title('Sequential per-fight Kelly on real Kalshi T-90min prices (quad fee)')
ax.legend(loc='upper left'); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()
""")
)

CELLS.append(
    md("""## Task 2.4 — where the ROI lives (val 2023 vs Kalshi window)

Pre-registered slices only. The only cell positive in both windows with a
Kalshi CI excluding zero is **favorites (chosen price ≥ 0.60)** — consistent
with the favorite-longshot bias the anchor study found. Longshot cells flip
sign between windows → no kill-filter adopted.""")
)

CELLS.append(
    code("""seg = pd.read_json(ROOT / 'experiments/2026-07-09_segment_roi.jsonl', lines=True)
piv = seg.pivot_table(index=['dim','cell'], columns='window',
                      values=['roi_pct','n_bets'], aggfunc='first')
piv.columns = [f'{a}_{b}' for a,b in piv.columns]
piv = piv[['n_bets_val2023','roi_pct_val2023','n_bets_kalshi2026','roi_pct_kalshi2026']]
piv['same_sign'] = np.sign(piv['roi_pct_val2023']) == np.sign(piv['roi_pct_kalshi2026'])
piv.style.format({'roi_pct_val2023': '{:+.1f}', 'roi_pct_kalshi2026': '{:+.1f}',
                  'n_bets_val2023': '{:.0f}', 'n_bets_kalshi2026': '{:.0f}'})
""")
)

CELLS.append(
    code("""pb = seg[seg['dim']=='price_bucket'].copy()
pb['ci_lo'] = pb['ci95'].str[0]; pb['ci_hi'] = pb['ci95'].str[1]
cells = ['p<0.40','0.40-0.60','p>=0.60']
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for ax, win in zip(axes, ['val2023','kalshi2026']):
    d = pb[pb['window']==win].set_index('cell').reindex(cells)
    x = np.arange(len(cells))
    colors = [COLORS['C'] if v < 0 else COLORS['A'] for v in d['roi_pct']]
    ax.bar(x, d['roi_pct'], width=0.55, color=colors)
    ax.errorbar(x, d['roi_pct'], yerr=[d['roi_pct']-d['ci_lo'], d['ci_hi']-d['roi_pct']],
                fmt='none', ecolor='black', alpha=0.5, capsize=4)
    for xi, (v, n) in enumerate(zip(d['roi_pct'], d['n_bets'])):
        ax.annotate(f'{v:+.1f}%\\n(n={n:.0f})', (xi, v), ha='center',
                    va='bottom' if v >= 0 else 'top', fontsize=9)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x, cells); ax.set_title(win); ax.grid(alpha=0.3, axis='y')
axes[0].set_ylabel('flat-stake ROI (%)')
fig.suptitle('Ensemble ROI by chosen-side price bucket — favorites carry the edge')
plt.tight_layout(); plt.show()
""")
)

CELLS.append(
    md("""## Task 1.1c — the anchor's favorite-longshot structure (Polymarket, n=638)

Model-free: BFO no-vig as truth, bet Polymarket pre-fight prices when EV
clears the threshold (3% quadratic fee). Dead overall; the pre-registered
buckets split exactly like tennis. Not deployable (Polymarket is blocked in
CT/MA and the anchor source lags months on Kalshi) — shown as the structural
evidence behind the favorites finding.""")
)

CELLS.append(
    code("""anc = pd.read_json(ROOT / 'experiments/2026-07-09_anchor_strategy.jsonl', lines=True)
curve = anc[anc['name']=='anchor_bfo_poly'][['thr','n_bets','roi_pct','ci95_halfwidth']]
cap = anc[anc['name']=='anchor_bfo_poly_cap33'][['thr','n_bets','roi_pct','ci95_halfwidth']]
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.errorbar(curve['thr']*100, curve['roi_pct'], yerr=curve['ci95_halfwidth'],
            marker='o', color=COLORS['grey'], capsize=4, label='all fights')
ax.errorbar(cap['thr']*100, cap['roi_pct'], yerr=cap['ci95_halfwidth'],
            marker='o', color=COLORS['B'], capsize=4, label='price ≥ 0.33 (tennis cap analog)')
ax.axhline(0, color='black', linewidth=0.8)
ax.set_xlabel('EV threshold (%)'); ax.set_ylabel('flat-stake ROI (%)')
ax.set_title('Model-free anchor on Polymarket 2024-04 → 2026-03')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()
buckets = anc[anc['name'].str.startswith('anchor_bfo_poly_p') | anc['name'].str.startswith('anchor_bfo_poly_0')]
buckets[['name','thr','n_bets','roi_pct','ci95_halfwidth','hit_rate']]
""")
)

CELLS.append(
    md("""## Tier 2 — Elo / fatigue / half-life ablations (val 2023, 10 seeds)

Reads `experiments/2026-07-09_tier2_ablation.jsonl`. Adoption rule
(pre-registered): ensemble val ROI @3% improves AND ≥7/10 seeds improve
(paired) AND log_loss/ECE non-regression. `seed_wins_vs_baseline` counts
paired seed wins.""")
)

CELLS.append(
    code("""t2 = pd.read_json(ROOT / 'experiments/2026-07-09_tier2_ablation.jsonl', lines=True)
cols = ['config','roi3_ens','roi3_ci','n_bets3','seed_roi3_med','roi5_ens',
        'log_loss','ece','seed_wins_vs_baseline']
t2[cols]
""")
)

CELLS.append(
    code("""base = t2[t2['config']=='baseline'].iloc[0]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
elo = t2[t2['config'].str.startswith('elo_k')].copy()
elo['K'] = elo['config'].str.replace('elo_k','').astype(int)
elo = elo.sort_values('K')
axes[0].plot(elo['K'], elo['roi3_ens'], marker='o', color=COLORS['B'], label='champion + Elo')
axes[0].scatter(elo['K'], elo['seed_roi3_med'], marker='x', color=COLORS['B'], alpha=0.6, label='seed median')
axes[0].axhline(base['roi3_ens'], color=COLORS['grey'], linestyle='--',
                label=f'baseline ({base["roi3_ens"]:+.1f}%)')
axes[0].set_xscale('log', base=2); axes[0].set_xticks([4,8,16,32], [4,8,16,32])
axes[0].set_xlabel('Elo K'); axes[0].set_ylabel('val ROI @3% (%)')
axes[0].set_title('Task 2.1 — Elo K sweep'); axes[0].legend(); axes[0].grid(alpha=0.3)

hl = t2[t2['config'].str.startswith('hl_')].copy()
hl['h'] = hl['config'].str.replace('hl_','').astype(float)
hl = pd.concat([hl, pd.DataFrame([{'h': 4.0, 'roi3_ens': base['roi3_ens'],
                                   'seed_roi3_med': base['seed_roi3_med']}])]).sort_values('h')
axes[1].plot(hl['h'], hl['roi3_ens'], marker='o', color=COLORS['A'], label='ensemble')
axes[1].scatter(hl['h'], hl['seed_roi3_med'], marker='x', color=COLORS['A'], alpha=0.6, label='seed median')
axes[1].set_xlabel('recency half-life (years)')
axes[1].set_title('Task 2.3 — half-life sweep (4.0 = current)')
axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout(); plt.show()
""")
)

CELLS.append(
    md("""## Takeaways

1. **The deployed model survives its first clean OOS test**: +9.4% flat ROI on
   real Kalshi prices post-cutoff, and *every one of the 10 seeds is positive*
   (min +6.0%). Wide CI (n=39) — keep expectations modest.
2. **The model > anchor** — the opposite of tennis. Deployment stays
   model-pure; no blend/veto/anchor variant is adopted (inconsistent across
   windows).
3. **The edge concentrates on favorite-side bets** in every independent look
   (anchor buckets, veto mechanics, segmentation). Nothing adopted as a filter
   yet; track the favorites/longshots split in live results
   (`sync_bankrolls` PnL by chosen price would confirm cheaply).
4. **Tier-2: nothing adopted.** Elo and fatigue — tennis's two model wins —
   both fail here (the Bayesian skill posterior already owns that niche).
   The one live thread is the **recency half-life**: val ROI is monotone
   toward shorter half-lives (2y +6.6% vs 4y +4.4%), but the 2y candidate
   fails the calibration gates and 3y falls one seed short of the paired-win
   rule — retest {2, 3}y once at the 2026-09-30 scheduled retrain.
5. The Kalshi API does **not** serve old settled markets: refresh the backfill
   within days of each card or the history is lost.
""")
)


def main():
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = ROOT / "notebooks/12_tennis_ported_techniques.ipynb"
    out.write_text(json.dumps(nb, indent=1))
    print(f"wrote {out} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()

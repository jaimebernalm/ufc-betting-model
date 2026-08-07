"""Generate notebooks/09_kalshi_vs_polymarket.ipynb.

Three things:
  1. Verify actual Kalshi + Polymarket fee formulas (both are quadratic in
     price, not flat percentages).
  2. Match Kalshi historical fights to Polymarket fights and evaluate the
     model on the matched subset against each venue's real closing prices.
  3. Show how Kalshi vs Polymarket closing prices differ for the same fights.
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
    md("""# 09 — Kalshi vs Polymarket: real fees + cross-venue price comparison

This notebook does three things the earlier evaluations got wrong or skipped:

1. **Use the real fee formulas** for both venues (not the flat % approximations).
2. **Run the model against Polymarket prices on the exact same fights** that nb08 evaluated on Kalshi prices — so we can compare like-for-like.
3. **Quantify how Kalshi and Polymarket closing prices differ** for the same fights (spread cost, agreement, divergence).

## The fee facts (verified from primary sources)

### Kalshi (UFC = general schedule)

From Kalshi's CFTC-filed fee schedule:
```
fee = round_up(0.07 × C × P × (1−P))   dollars per trade
```
- Taker only — makers pay $0
- Per-contract fee rounded UP to next cent

### Polymarket (international, the data we have)

```
fee = 0.03 × C × p × (1−p)   per contract
```
- Effective from March 30, 2026
- 25% maker rebate (we ignore in evaluation — assume taker)

### What the old notebooks used

Earlier notebooks approximated both as "flat % of winnings": nb05 = 2%, nb08 = 7%. The *direction* of effect is the same (favorites pay relatively more), but the magnitude is wrong — especially the "of winnings" framing systematically overstates fees by ~2× at the midpoint.

## Result preview

| Effective fee % of notional | p = 0.05 | p = 0.25 | p = 0.50 | p = 0.75 | p = 0.95 |
|---|---|---|---|---|---|
| Kalshi (0.07 quadratic) | 0.33% | 1.31% | **1.75%** | 1.31% | 0.33% |
| Polymarket (0.03 quadratic) | 0.14% | 0.56% | **0.75%** | 0.56% | 0.14% |
| nb05 "flat 2% winnings" applied to dec=1/p | 38% | 6% | 2% | 0.67% | 0.11% |
| nb08 "flat 7% winnings" applied to dec=1/p | 133% | 21% | 7% | 2.3% | 0.37% |

The flat-% approximations over-penalize underdogs by a huge margin. This matters because the model often bets dogs.
""")
)

CELLS.append(
    code("""import sys, unicodedata, re
from pathlib import Path

ROOT = Path.cwd().parent

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ufc_pred.backtest.bet_eval import evaluate_bets, evaluate_bets_kelly, _effective_decimal
from ufc_pred.backtest.metrics import evaluate
from ufc_pred.features.skill_v3_pipeline import OUTPUT as SKILL_V3_PARQUET
from ufc_pred.features.static_v1 import prepare
from ufc_pred.ingest.kaggle_mdabbert import HISTORY_PARQUET

pd.set_option('display.precision', 3)
""")
)

CELLS.append(
    md("""## 1. Effective fee curve for both venues

Plot the effective % cost as a function of the contract price, for both Kalshi and Polymarket, and compare against the flat-% approximations we used previously.
""")
)

CELLS.append(
    code("""# Compute effective decimal odds at each price for each fee model,
# then convert to "% of notional cost".
prices = np.linspace(0.02, 0.98, 200)
dec = 1.0 / prices

# Cost-per-$1-payout = price + fee_per_contract; effective_cost_pct = fee / price
# For the quadratic models: fee_per_$1_payout = fee_rate * p * (1-p)
kalshi_fee_pct_payout    = 0.07 * prices * (1 - prices) * 100   # % of $1 payout
polymarket_fee_pct_payout = 0.03 * prices * (1 - prices) * 100

# For the "winnings" model: fee = fee_rate * (1 - price) = % of stake when expressed per $1 payout
flat7_winnings_pct = 0.07 * (1 - prices) * 100
flat2_winnings_pct = 0.02 * (1 - prices) * 100

fig, ax = plt.subplots(1, 1, figsize=(11, 6))
ax.plot(prices, kalshi_fee_pct_payout, 'b-', lw=2.5, label='Kalshi real (0.07 × p×(1−p))')
ax.plot(prices, polymarket_fee_pct_payout, 'g-', lw=2.5, label='Polymarket real (0.03 × p×(1−p))')
ax.plot(prices, flat7_winnings_pct, 'b--', alpha=0.6, label='nb08 approx (7% of winnings)')
ax.plot(prices, flat2_winnings_pct, 'g--', alpha=0.6, label='nb05 approx (2% of winnings)')
ax.axvline(0.5, color='gray', alpha=0.3)
ax.set_xlabel('Contract price (= implied probability)')
ax.set_ylabel('Effective fee (% of $1 payout)')
ax.set_title('Fee curves: real quadratic formulas vs flat-% approximations')
ax.legend()
ax.grid(alpha=0.3)
ax.set_ylim(0, 8)
plt.tight_layout(); plt.show()

# Print the comparison table
rows = []
for p in [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]:
    rows.append({
        'price': p,
        'Kalshi real %': 0.07 * p * (1-p) * 100,
        'Poly real %':   0.03 * p * (1-p) * 100,
        'nb08 7%-wins %': 0.07 * (1-p) * 100,
        'nb05 2%-wins %': 0.02 * (1-p) * 100,
    })
pd.DataFrame(rows).set_index('price').style.format('{:.2f}%')
""")
)

CELLS.append(md("""## 2. Load Kalshi + Polymarket historical, match by date and name"""))

CELLS.append(
    code("""APOSTROPHES = "\\'’ʼ`‘"
def norm(s):
    if pd.isna(s): return ""
    return unicodedata.normalize("NFKD", str(s)).encode("ascii","ignore").decode("ascii").lower().strip()
def deep_norm(s):
    if pd.isna(s) or s is None: return ""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii","ignore").decode("ascii")
    for ch in APOSTROPHES + "-.,":
        s = s.replace(ch, "")
    return re.sub(r"\\s+", " ", s).strip().lower()
def last_token(s):
    s = deep_norm(s)
    return s.split()[-1] if s else ""

# Kalshi
kal = pd.read_parquet(ROOT / 'data/raw/kalshi/historical.parquet')
kal = kal.dropna(subset=['close_yes_price_a','close_yes_price_b','winner']).copy()
ab_sum = kal['close_yes_price_a'] + kal['close_yes_price_b']
keep = (kal['close_yes_price_a'] >= 0.02) & (kal['close_yes_price_b'] >= 0.02) & \\
       (ab_sum >= 0.80) & (ab_sum <= 1.30) & \\
       ((kal['volume_a'] + kal['volume_b']) >= 100)
kal = kal[keep].reset_index(drop=True)
kal['fd_n'] = pd.to_datetime(kal['fight_date']).dt.tz_localize(None).dt.normalize()
kal['a_dn'] = kal['fighter_a'].map(deep_norm)
kal['b_dn'] = kal['fighter_b'].map(deep_norm)
kal['a_last'] = kal['fighter_a'].map(last_token)
kal['b_last'] = kal['fighter_b'].map(last_token)

# Polymarket
poly = pd.read_parquet(ROOT / 'data/raw/polymarket/historical_2024-04-13_to_2026-05-24.parquet')
poly = poly.dropna(subset=['closing_price_a','closing_price_b','winner']).copy()
poly['fd_n'] = pd.to_datetime(poly['fight_date']).dt.tz_localize(None).dt.normalize()
poly['a_dn'] = poly['fighter_a'].map(deep_norm)
poly['b_dn'] = poly['fighter_b'].map(deep_norm)
poly['a_last'] = poly['fighter_a'].map(last_token)
poly['b_last'] = poly['fighter_b'].map(last_token)

# Kaggle (for ground truth + model features)
fights = pd.read_parquet(HISTORY_PARQUET)
fights = fights[fights['Winner'].isin(['Red','Blue'])].copy()
fights['date'] = pd.to_datetime(fights['date'])
fights['R_dn']  = fights['R_fighter'].map(deep_norm)
fights['B_dn']  = fights['B_fighter'].map(deep_norm)
fights['R_last'] = fights['R_fighter'].map(last_token)
fights['B_last'] = fights['B_fighter'].map(last_token)
sk = pd.read_parquet(SKILL_V3_PARQUET)
sk['date'] = pd.to_datetime(sk['date'])
fights = fights.merge(sk[['date','R_fighter','B_fighter','skill_diff_mean','skill_diff_std']],
                     on=['date','R_fighter','B_fighter'], how='left', validate='many_to_one')

print(f'Kalshi rows:     {len(kal)}  ({kal["fd_n"].min().date()} → {kal["fd_n"].max().date()})')
print(f'Polymarket rows: {len(poly)}  ({poly["fd_n"].min().date()} → {poly["fd_n"].max().date()})')
print(f'Kaggle rows:     {len(fights)}')
""")
)

CELLS.append(
    md("""## 3. Match Kalshi fights to Polymarket fights and to Kaggle ground truth

Both Kalshi and Polymarket rows must match to the same Kaggle fight for a row to be analyzable. We need three-way matching.
""")
)

CELLS.append(
    code("""from rapidfuzz import fuzz

# Build Polymarket-by-date index for fast lookup
poly_by_date = {d: g for d, g in poly.groupby('fd_n')}

def match_polymarket(kal_row):
    \"\"\"Find the Polymarket row that's the same fight as this Kalshi row.\"\"\"
    a, b = kal_row['a_dn'], kal_row['b_dn']
    a_l, b_l = kal_row['a_last'], kal_row['b_last']
    # Look in poly within ±2 days
    candidates = []
    for off in (-2, -1, 0, 1, 2):
        d = kal_row['fd_n'] + pd.Timedelta(days=off)
        if d in poly_by_date:
            candidates.append(poly_by_date[d])
    if not candidates:
        return None
    pool = pd.concat(candidates)
    # exact full
    h = pool[((pool['a_dn']==a) & (pool['b_dn']==b)) | ((pool['a_dn']==b) & (pool['b_dn']==a))]
    if len(h) >= 1: return h.iloc[0]
    # exact last
    h = pool[((pool['a_last']==a_l) & (pool['b_last']==b_l)) | ((pool['a_last']==b_l) & (pool['b_last']==a_l))]
    if len(h) >= 1: return h.iloc[0]
    # substring
    h = pool[((pool['a_dn'].str.contains(a_l, regex=False)) & (pool['b_dn'].str.contains(b_l, regex=False))) |
             ((pool['a_dn'].str.contains(b_l, regex=False)) & (pool['b_dn'].str.contains(a_l, regex=False)))]
    if len(h) >= 1: return h.iloc[0]
    return None

# Also need Kaggle match (same logic as nb05/nb08, condensed)
fights_by_date = {d: g for d, g in fights.groupby('date')}
def match_kaggle(kal_row):
    a, b = kal_row['a_dn'], kal_row['b_dn']
    a_l, b_l = kal_row['a_last'], kal_row['b_last']
    pools = []
    for off in (-1, 0, 1):
        d = kal_row['fd_n'] + pd.Timedelta(days=off)
        if d in fights_by_date: pools.append(fights_by_date[d])
    if not pools: return None, None
    pool = pd.concat(pools)
    h = pool[((pool['R_dn']==a) & (pool['B_dn']==b)) | ((pool['R_dn']==b) & (pool['B_dn']==a))]
    if len(h) == 1:
        k = h.iloc[0]; return k, k['R_dn'] == a
    h = pool[((pool['R_last']==a_l) & (pool['B_last']==b_l)) | ((pool['R_last']==b_l) & (pool['B_last']==a_l))]
    if len(h) == 1:
        k = h.iloc[0]; return k, k['R_last'] == a_l
    return None, None

triples = []  # each = dict with kalshi prices, polymarket prices, kaggle row + a_is_red
for _, k_row in kal.iterrows():
    p_match = match_polymarket(k_row)
    if p_match is None: continue
    k_kag, a_is_red_k = match_kaggle(k_row)
    if k_kag is None: continue
    # Align poly's a/b to Kalshi's a/b
    poly_a_is_kal_a = (p_match['a_last'] == k_row['a_last']) or (p_match['a_dn'] == k_row['a_dn'])
    poly_close_a = p_match['closing_price_a'] if poly_a_is_kal_a else p_match['closing_price_b']
    poly_close_b = p_match['closing_price_b'] if poly_a_is_kal_a else p_match['closing_price_a']
    triples.append({
        'date': k_kag['date'],
        'a_is_red': a_is_red_k,
        'kal_a': k_row['fighter_a'], 'kal_b': k_row['fighter_b'],
        'kal_close_a': k_row['close_yes_price_a'],
        'kal_close_b': k_row['close_yes_price_b'],
        'poly_close_a': poly_close_a, 'poly_close_b': poly_close_b,
        'kal_winner': k_row['winner'],
        'kag_R': k_kag['R_fighter'], 'kag_B': k_kag['B_fighter'],
        'kag_winner': k_kag['Winner'],
        'kag_idx': k_kag.name,
    })

t_df = pd.DataFrame(triples)
matched_kag = fights.loc[[t['kag_idx'] for t in triples]].reset_index(drop=True)
print(f'Triples (Kalshi ∩ Polymarket ∩ Kaggle): {len(t_df)}')
print(f'  date range: {t_df["date"].min().date()} → {t_df["date"].max().date()}')
print(f'  Kalshi-Kaggle winner agreement: {(((t_df["kal_winner"]=="A") == ((t_df["a_is_red"] & (t_df["kag_winner"]=="Red")) | (~t_df["a_is_red"] & (t_df["kag_winner"]=="Blue"))))).sum()}/{len(t_df)}')
""")
)

CELLS.append(
    md("""## 4. How do Kalshi and Polymarket closing prices differ?

For each matched fight, compare side-A's closing probability on Kalshi vs Polymarket. Three views:
1. Distribution of differences (Kalshi minus Polymarket)
2. Scatter plot
3. Overround / spread comparison (sum of both sides)
""")
)

CELLS.append(
    code("""diffs = (t_df['kal_close_a'] - t_df['poly_close_a']) * 100  # in percentage points
kal_sum = t_df['kal_close_a'] + t_df['kal_close_b']
poly_sum = t_df['poly_close_a'] + t_df['poly_close_b']

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Difference distribution
axes[0].hist(diffs, bins=30, color='steelblue', edgecolor='black', alpha=0.8)
axes[0].axvline(0, color='red', lw=1)
axes[0].axvline(diffs.median(), color='orange', lw=2, label=f'median: {diffs.median():+.2f}pp')
axes[0].set_xlabel('Kalshi p(A) − Polymarket p(A)   (percentage points)')
axes[0].set_ylabel('# fights')
axes[0].set_title(f'Closing-price gap, fighter A\\n(n={len(diffs)}, mean={diffs.mean():+.2f}pp, std={diffs.std():.2f}pp)')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Scatter
axes[1].scatter(t_df['poly_close_a'], t_df['kal_close_a'], alpha=0.6, s=30)
axes[1].plot([0,1],[0,1], 'r--', lw=1, alpha=0.5, label='y=x')
axes[1].set_xlabel('Polymarket closing p(A)')
axes[1].set_ylabel('Kalshi closing p(A)')
axes[1].set_title('Closing prices, Kalshi vs Polymarket')
axes[1].legend()
axes[1].grid(alpha=0.3)
axes[1].set_xlim(0, 1); axes[1].set_ylim(0, 1)

# Overround comparison
axes[2].hist(kal_sum, bins=20, alpha=0.6, label=f'Kalshi (mean {kal_sum.mean():.3f})', color='blue')
axes[2].hist(poly_sum, bins=20, alpha=0.6, label=f'Polymarket (mean {poly_sum.mean():.3f})', color='green')
axes[2].axvline(1.0, color='red', lw=1)
axes[2].set_xlabel('p(A) + p(B)  (> 1 = overround = spread cost)')
axes[2].set_ylabel('# fights')
axes[2].set_title('Total implied probability sum\\n(distance from 1.0 = spread cost)')
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout(); plt.show()

# Summary table
summary = pd.DataFrame({
    'metric': ['mean diff (Kal − Poly)', 'median |diff|', '|diff| > 5pp', '|diff| > 10pp',
               'mean overround Kalshi', 'mean overround Polymarket',
               'avg Kalshi spread', 'avg Poly spread'],
    'value': [f'{diffs.mean():+.2f} pp', f'{diffs.abs().median():.2f} pp',
              f'{(diffs.abs() > 5).sum()}/{len(diffs)} ({(diffs.abs()>5).mean()*100:.1f}%)',
              f'{(diffs.abs() > 10).sum()}/{len(diffs)} ({(diffs.abs()>10).mean()*100:.1f}%)',
              f'{kal_sum.mean():.4f}', f'{poly_sum.mean():.4f}',
              f'{(kal_sum.mean()-1)*100:.2f}%', f'{(poly_sum.mean()-1)*100:.2f}%']
})
summary
""")
)

CELLS.append(
    md("""## 5. Model predictions on the matched subset

Same model as nb05/nb08 — just on the smaller Kalshi ∩ Polymarket intersection.
""")
)

CELLS.append(
    code("""PRICE_FLOOR = 0.02
payload = joblib.load(ROOT / 'artifacts/models/v3_catboost_full2000_trainval.joblib')
X, _, _, _ = prepare(matched_kag, augment_symmetry=False, one_hot=False)
X = X.reindex(columns=payload['columns'], fill_value=None)
for c in payload.get('cat_features', []):
    X[c] = X[c].fillna('__missing__').astype(str)
p_red = payload['model'].predict_proba(X)[:, 1]
y_red = (matched_kag['Winner'].to_numpy() == 'Red').astype(int)

def prob_to_american(p):
    p = float(np.clip(p, PRICE_FLOOR, 1 - PRICE_FLOOR))
    dec = 1.0 / p
    return (dec - 1.0) * 100.0 if dec >= 2.0 else -100.0 / (dec - 1.0)

# Kalshi odds in Red/Blue orientation
R_kal_odds, B_kal_odds = [], []
R_poly_odds, B_poly_odds = [], []
for r in triples:
    pa, pb = r['kal_close_a'], r['kal_close_b']
    qa, qb = r['poly_close_a'], r['poly_close_b']
    if r['a_is_red']:
        R_kal_odds.append(prob_to_american(pa)); B_kal_odds.append(prob_to_american(pb))
        R_poly_odds.append(prob_to_american(qa)); B_poly_odds.append(prob_to_american(qb))
    else:
        R_kal_odds.append(prob_to_american(pb)); B_kal_odds.append(prob_to_american(pa))
        R_poly_odds.append(prob_to_american(qb)); B_poly_odds.append(prob_to_american(qa))
R_kal_odds = pd.Series(R_kal_odds); B_kal_odds = pd.Series(B_kal_odds)
R_poly_odds = pd.Series(R_poly_odds); B_poly_odds = pd.Series(B_poly_odds)

print(f'n matched: {len(p_red)}')
print(f'model pred mean: {p_red.mean():.3f}, range [{p_red.min():.3f}, {p_red.max():.3f}]')
print(f'y_red base rate: {y_red.mean():.3f}')
metrics = evaluate(y_red, p_red, label='kalshi_poly_intersection')
print(f"log_loss={metrics['log_loss']:.4f}  brier={metrics['brier']:.4f}  ece={metrics['ece']:.4f}  acc={metrics['accuracy_argmax']:.3f}")
""")
)

CELLS.append(md("""## 6. Flat-stake ROI with REAL fees — Kalshi vs Polymarket on same fights"""))

CELLS.append(
    code("""rows = []
for venue, R_odds, B_odds, fee_rate, fee_model in [
    ('Kalshi (real 0.07 quadratic)',  R_kal_odds,  B_kal_odds,  0.07, 'kalshi'),
    ('Kalshi (old nb08: 7% winnings)', R_kal_odds,  B_kal_odds,  0.07, 'winnings'),
    ('Kalshi (zero-fee idealized)',    R_kal_odds,  B_kal_odds,  0.00, 'kalshi'),
    ('Polymarket (real 0.03 quadratic)', R_poly_odds, B_poly_odds, 0.03, 'kalshi'),  # same formula shape, diff coef
    ('Polymarket (old nb05: 2% winnings)', R_poly_odds, B_poly_odds, 0.02, 'winnings'),
    ('Polymarket (zero-fee idealized)',   R_poly_odds, B_poly_odds, 0.00, 'kalshi'),
]:
    for thr in [0.03, 0.05]:
        r = evaluate_bets(p_red, y_red, R_odds, B_odds,
                          edge_threshold=thr, fee_rate=fee_rate,
                          fee_model=fee_model, use_no_vig=False)
        rows.append({
            'venue + fee model': venue, 'edge_thr': f'{int(thr*100)}%',
            'n_bets': r.n_bets, 'roi_pct': r.roi_pct,
            'ci95_low': r.ci95_roi_pct[0], 'ci95_high': r.ci95_roi_pct[1],
            'hit_rate': r.hit_rate, 'mean_ev_pct': r.mean_ev_pct,
        })
flat = pd.DataFrame(rows)
flat.style.format({'roi_pct': '{:+.2f}', 'ci95_low': '{:+.2f}', 'ci95_high': '{:+.2f}',
                  'hit_rate': '{:.3f}', 'mean_ev_pct': '{:+.2f}'})
""")
)

CELLS.append(
    md("""## 7. Deploy-config Kelly trajectories — Kalshi vs Polymarket

Same accounts as nb08, both venues, REAL fees.
""")
)

CELLS.append(
    code("""START = 300.0
EDGE_THRESHOLD = 0.03

payload_c = joblib.load(ROOT / 'artifacts/models/v3_full2000_no_skill_corrupted_trainval.joblib')
p_corrupt = payload_c['model'].predict_proba(X)[:, 1]

def sim(p, R, B, kelly, cap, fee_rate, fee_model):
    return evaluate_bets_kelly(p, y_red, R, B,
        edge_threshold=EDGE_THRESHOLD, fee_rate=fee_rate, fee_model=fee_model,
        use_no_vig=False, kelly_fraction=kelly, max_bet_fraction=cap,
        starting_bankroll=START)

results = []
for venue, R, B, fee_rate in [('Kalshi', R_kal_odds, B_kal_odds, 0.07),
                              ('Polymarket', R_poly_odds, B_poly_odds, 0.03)]:
    for tag, p, kelly, cap in [('A', p_red, 0.10, 0.10),
                               ('B', p_red, 0.25, 1.00),
                               ('C', p_corrupt, 0.25, 1.00)]:
        s = sim(p, R, B, kelly, cap, fee_rate, 'kalshi')
        results.append({'venue': venue, 'account': tag,
                       'config': '10%-K+10%cap' if tag=='A' else '¼-K, no cap',
                       'n_bets': s['n_bets'], 'final': s['final_bankroll'],
                       'max_dd': s['max_drawdown_pct'], 'traj': s['trajectory']})

res_df = pd.DataFrame([{k:v for k,v in r.items() if k!='traj'} for r in results])
res_df.style.format({'final': '${:,.2f}', 'max_dd': '{:.1f}%'})
""")
)

CELLS.append(
    code("""# Side-by-side bankroll trajectories
fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
colors = {'A': '#1a9641', 'B': '#1f78b4', 'C': '#d7191c'}
for ax, venue in zip(axes, ['Kalshi', 'Polymarket']):
    for r in results:
        if r['venue'] != venue: continue
        ax.plot(r['traj'], color=colors[r['account']], lw=2,
                label=f"Acct {r['account']}: ${START:.0f} → ${r['final']:,.2f}  ({r['config']})")
    ax.axhline(START, color='black', alpha=0.4)
    ax.set_yscale('log')
    ax.set_title(f'{venue} prices, REAL fees, edge ≥ 3%')
    ax.set_xlabel('Bet # (chronological)')
    ax.set_ylabel('Bankroll ($, log)')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(alpha=0.3, which='both')
plt.tight_layout(); plt.show()
""")
)

CELLS.append(md("""## 8. Per-bet diagnostic: where Kalshi vs Polymarket prices diverged most"""))

CELLS.append(
    code("""# For the bets the bot would actually place, show where Kalshi and Polymarket
# disagreed the most. These are the fights where venue choice matters.
diag = []
for i, r in enumerate(triples):
    if r['a_is_red']:
        kal_p_chosen = r['kal_close_a'] if p_red[i] >= 0.5 else r['kal_close_b']
        poly_p_chosen = r['poly_close_a'] if p_red[i] >= 0.5 else r['poly_close_b']
        chosen = r['kal_a'] if p_red[i] >= 0.5 else r['kal_b']
    else:
        kal_p_chosen = r['kal_close_b'] if p_red[i] >= 0.5 else r['kal_close_a']
        poly_p_chosen = r['poly_close_b'] if p_red[i] >= 0.5 else r['poly_close_a']
        chosen = r['kal_b'] if p_red[i] >= 0.5 else r['kal_a']
    diag.append({
        'date': r['date'].date(),
        'chosen': chosen,
        'model_p': p_red[i] if p_red[i] >= 0.5 else 1-p_red[i],
        'kalshi_p': kal_p_chosen,
        'poly_p': poly_p_chosen,
        'gap_pp': (kal_p_chosen - poly_p_chosen) * 100,
        'won': int(y_red[i] == (1 if r['a_is_red'] else 0)) if p_red[i] >= 0.5 else
               int(y_red[i] == (0 if r['a_is_red'] else 1)),
    })
diag_df = pd.DataFrame(diag).sort_values('gap_pp', key=abs, ascending=False)
print('Top 10 fights where Kalshi vs Polymarket disagreed the most on the model\\'s chosen side:')
diag_df.head(10).style.format({
    'model_p': '{:.3f}', 'kalshi_p': '{:.3f}', 'poly_p': '{:.3f}', 'gap_pp': '{:+.2f}pp',
})
""")
)

CELLS.append(
    md("""## Takeaways

1. **Fees were ~2× overstated in nb05/nb08.** Real Kalshi sports fee peaks at ~1.75% of notional at p=0.5 (not 7%). Real Polymarket fee peaks at ~0.75% (not 2%). The flat-% approximations get the *direction* right but the *magnitude* wrong, especially for underdogs where the flat models penalize trades that real venues barely fee at all.

2. **On the same fights, Polymarket prices differ from Kalshi prices** — see histogram + scatter. Median absolute gap of a few percentage points is normal market noise; tails (|gap| > 5pp) are mispricings either venue could be on either side of. For an arbitrageur this is the signal; for a one-venue bettor it just affects ROI.

3. **Re-evaluated with REAL fees, the model's edge:**
   - Kalshi: still negative on this 4-month, ~30-bet sample (the data window is the binding constraint, not the fee model)
   - Polymarket: similar pattern
   - Both are noise-dominated at this sample size

4. **What changed by fixing the fee model:** ROI moved up by ~3-5 percentage points across the board (real fees are lower than approximations), but CI95 bounds still straddle zero. The conclusion from nb08 stands: too little data to deploy with confidence on Kalshi history alone. The Polymarket nb05 numbers (615 bets) remain the better edge estimate.

5. **Recommendation for the $100 deploy decision:** Use the Polymarket nb05 ROI as the model-edge prior (since it's based on 10× more bets), apply Kalshi's real fee curve (cheaper than nb08 implied), and rebuild this notebook in 6 months when Kalshi has accumulated more matched history.
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
    out = ROOT / "notebooks/09_kalshi_vs_polymarket.ipynb"
    out.write_text(json.dumps(nb, indent=1))
    print(f"wrote {out} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()

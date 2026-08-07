"""Generate notebooks/08_kalshi_evaluation.ipynb mirroring nb05."""

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
    md("""# 08 — Kalshi evaluation

Notebook 05 evaluated the model against **actual Polymarket closing prices** with ~2% fee. This notebook re-runs the same evaluation against **actual Kalshi closing prices** (where we'll actually deploy from CT/MA since Polymarket US is blocked).

## Data

- **Kalshi historical**: settled `KXUFCFIGHT` markets pulled via the official v2 REST API. One event per fight, two binary markets per event (one per fighter, mutually exclusive). For each market we have closing price (last trade), settlement result, volume, and open interest. UFC-only events (we filter out "Netflix MMA Special" events that share the series).
- **Matching**: Kalshi fighter names match fights.parquet exactly in most cases (Kalshi uses standard display names). Fuzzy fallback with surname-strict guard to avoid first-name false positives.
- All matched fights are within the test window of `fights.parquet` (2010-03-21 → 2026-05-16).

## Key features of Kalshi pricing

- Closing prices are **last trade per side**, NOT a no-vig consensus. The two sides will not sum to exactly 1.0 — there is always a bid-ask spread.
- Decimal odds: `dec_a = 1 / closing_price_a`.
- **Kalshi fee structure**: ~7% on winnings (formula in their docs). This is the central case here; 0% and 2% included for comparison with Polymarket.
- Per-market volume is in contract units. Total card volume is healthy ($100k+ per main event).
- No historical orderbook depth via REST — same limitation as the Polymarket backfill.

## Model

`v3_catboost_full2000_trainval` — the deployed champion. Same model as notebooks 04/05.
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

from ufc_pred.backtest.bet_eval import evaluate_bets, evaluate_bets_kelly
from ufc_pred.backtest.metrics import evaluate
from ufc_pred.features.skill_v3_pipeline import OUTPUT as SKILL_V3_PARQUET
from ufc_pred.features.static_v1 import prepare
from ufc_pred.ingest.kaggle_mdabbert import HISTORY_PARQUET

pd.set_option('display.precision', 3)
""")
)

CELLS.append(md("""## Load Kalshi historical + Kaggle, match by date and name"""))

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

# Kalshi historical (built by scripts/kalshi_backfill.py). Closing prices come
# from `fetch_last_betting_price` (last trade with price in [0.05, 0.95]),
# avoiding the post-fight arbitrage contamination of `last_price_dollars`.
kal = pd.read_parquet(ROOT / 'data/raw/kalshi/historical.parquet')
print(f'raw rows: {len(kal)}')
# Filter rules:
#  - both closing prices in betting range (drops zero-volume markets that never traded)
#  - a winner labeled
#  - prices sum to a sensible range (drops one-sided-trade-only markets where
#    one side has a betting price and the other is 0)
kal = kal.dropna(subset=['close_yes_price_a','close_yes_price_b','winner']).copy()
ab_sum = kal['close_yes_price_a'] + kal['close_yes_price_b']
keep = (kal['close_yes_price_a'] >= 0.02) & (kal['close_yes_price_b'] >= 0.02) & \
       (ab_sum >= 0.80) & (ab_sum <= 1.30) & \
       ((kal['volume_a'] + kal['volume_b']) >= 100)
print(f'after volume + price-sanity filter: {keep.sum()} (dropped {(~keep).sum()} zero-volume/one-sided markets)')
kal = kal[keep].reset_index(drop=True)
kal['fd_n']  = pd.to_datetime(kal['fight_date']).dt.tz_localize(None).dt.normalize()
kal['fd_m1'] = kal['fd_n'] - pd.Timedelta(days=1)
kal['a_dn']  = kal['fighter_a'].map(deep_norm)
kal['b_dn']  = kal['fighter_b'].map(deep_norm)
kal['a_last'] = kal['fighter_a'].map(last_token)
kal['b_last'] = kal['fighter_b'].map(last_token)

# Kaggle
fights = pd.read_parquet(HISTORY_PARQUET)
fights = fights[fights['Winner'].isin(['Red','Blue'])].copy()
fights['date'] = pd.to_datetime(fights['date'])
fights['R_dn']  = fights['R_fighter'].map(deep_norm)
fights['B_dn']  = fights['B_fighter'].map(deep_norm)
fights['R_last'] = fights['R_fighter'].map(last_token)
fights['B_last'] = fights['B_fighter'].map(last_token)

sk = pd.read_parquet(SKILL_V3_PARQUET)
sk['date'] = pd.to_datetime(sk['date'])
fights = fights.merge(
    sk[['date','R_fighter','B_fighter','skill_diff_mean','skill_diff_std']],
    on=['date','R_fighter','B_fighter'], how='left', validate='many_to_one',
)
print(f'Kalshi rows: {len(kal)}  ({kal["fd_n"].min().date()} → {kal["fd_n"].max().date()})')
print(f'Kaggle rows: {len(fights)}')
""")
)

CELLS.append(
    code("""from rapidfuzz import fuzz

fights_by_date = {d: g for d, g in fights.groupby('date')}

def candidates_local(row):
    dates = set()
    for d in (row['fd_n'], row['fd_m1']):
        for off in (-1, 0, 1):
            dates.add(d + pd.Timedelta(days=off))
    pools = [fights_by_date[d] for d in dates if d in fights_by_date]
    return pd.concat(pools) if pools else pd.DataFrame()

def match_one(row):
    pool = candidates_local(row)
    if len(pool) > 0:
        a, b = row['a_dn'], row['b_dn']
        a_l, b_l = row['a_last'], row['b_last']
        h = pool[((pool['R_dn']==a) & (pool['B_dn']==b)) | ((pool['R_dn']==b) & (pool['B_dn']==a))]
        if len(h) == 1:
            k = h.iloc[0]; return k, k['R_dn']==a, 'exact_full'
        h = pool[((pool['R_last']==a_l) & (pool['B_last']==b_l)) | ((pool['R_last']==b_l) & (pool['B_last']==a_l))]
        if len(h) == 1:
            k = h.iloc[0]; return k, k['R_last']==a_l, 'exact_last'
        h = pool[((pool['R_dn'].str.contains(a_l, regex=False)) & (pool['B_dn'].str.contains(b_l, regex=False))) |
                 ((pool['R_dn'].str.contains(b_l, regex=False)) & (pool['B_dn'].str.contains(a_l, regex=False)))]
        if len(h) == 1:
            k = h.iloc[0]; return k, a_l in k['R_dn'], 'substr'
        def fp(k):
            return (max(fuzz.ratio(a_l, k['R_last']), fuzz.ratio(a_l, k['B_last'])) +
                    max(fuzz.ratio(b_l, k['R_last']), fuzz.ratio(b_l, k['B_last']))) / 2
        pool = pool.copy(); pool['fuz'] = pool.apply(fp, axis=1)
        best = pool.sort_values('fuz', ascending=False).head(2)
        if len(best) > 0 and best.iloc[0]['fuz'] >= 88 and (len(best) == 1 or best.iloc[0]['fuz'] - best.iloc[1]['fuz'] >= 5):
            k = best.iloc[0]
            return k, fuzz.ratio(a_l, k['R_last']) >= fuzz.ratio(a_l, k['B_last']), 'fuzzy'
    d = row['fd_n']
    w = fights[(fights['date'] >= d - pd.Timedelta(days=14)) & (fights['date'] <= d + pd.Timedelta(days=14))]
    h = w[((w['R_last']==row['a_last']) & (w['B_last']==row['b_last'])) | ((w['R_last']==row['b_last']) & (w['B_last']==row['a_last']))]
    if len(h) == 1:
        k = h.iloc[0]; return k, k['R_last']==row['a_last'], 'wide_14d'
    return None, None, 'no'

matches = []
matched_kag_rows = []
for i, row in kal.iterrows():
    k, a_is_red, reason = match_one(row)
    if k is None: continue
    matches.append({
        'date': k['date'], 'a_is_red': a_is_red,
        'kal_a': row['fighter_a'], 'kal_b': row['fighter_b'],
        'closing_price_a': row['close_yes_price_a'],
        'closing_price_b': row['close_yes_price_b'],
        'kal_winner': row['winner'],  # "A" or "B"
        'volume_a': row['volume_a'], 'volume_b': row['volume_b'],
        'oi_a': row['open_interest_a'], 'oi_b': row['open_interest_b'],
        'kag_R': k['R_fighter'], 'kag_B': k['B_fighter'],
        'kag_winner': k['Winner'],
        'match_reason': reason,
    })
    matched_kag_rows.append(k.name)

m_df = pd.DataFrame(matches)
matched_fights = fights.loc[matched_kag_rows].reset_index(drop=True)

print(f"Matched: {len(m_df)}/{len(kal)} ({len(m_df)/len(kal)*100:.1f}%)")
print(f"Date range: {m_df['date'].min().date()} → {m_df['date'].max().date()}")
print("By pass:")
for r, n in m_df['match_reason'].value_counts().items():
    print(f"  {r:12s}: {n}")

# Sanity: do Kalshi and Kaggle agree on winner?
m_df['kag_winner_AB'] = np.where(
    (m_df['a_is_red'] & (m_df['kag_winner']=='Red')) | (~m_df['a_is_red'] & (m_df['kag_winner']=='Blue')),
    'A', 'B')
disagree = (m_df['kag_winner_AB'] != m_df['kal_winner']).sum()
print(f"\\nWinner-disagreement Kalshi vs Kaggle: {disagree}/{len(m_df)} (these are dropped from analysis)")
""")
)

CELLS.append(md("""## Compute model predictions on matched fights"""))

CELLS.append(
    code("""PRICE_FLOOR = 0.02  # Kalshi minimum tradeable price
payload = joblib.load(ROOT / 'artifacts/models/v3_catboost_full2000_trainval.joblib')
X, _, _, _ = prepare(matched_fights, augment_symmetry=False, one_hot=False)
X = X.reindex(columns=payload['columns'], fill_value=None)
for c in payload.get('cat_features', []):
    X[c] = X[c].fillna('__missing__').astype(str)
p_red = payload['model'].predict_proba(X)[:, 1]
y_red = (matched_fights['Winner'].to_numpy() == 'Red').astype(int)

def prob_to_american(p):
    p = float(np.clip(p, PRICE_FLOOR, 1 - PRICE_FLOOR))
    dec = 1.0 / p
    return (dec - 1.0) * 100.0 if dec >= 2.0 else -100.0 / (dec - 1.0)

R_kal_odds, B_kal_odds = [], []
for r in matches:
    pa, pb = r['closing_price_a'], r['closing_price_b']
    if r['a_is_red']:
        R_kal_odds.append(prob_to_american(pa)); B_kal_odds.append(prob_to_american(pb))
    else:
        R_kal_odds.append(prob_to_american(pb)); B_kal_odds.append(prob_to_american(pa))
R_kal_odds = pd.Series(R_kal_odds)
B_kal_odds = pd.Series(B_kal_odds)

print(f'model pred mean: {p_red.mean():.3f}, range [{p_red.min():.3f}, {p_red.max():.3f}]')
print(f'y_red base rate: {y_red.mean():.3f}')

metrics = evaluate(y_red, p_red, label='kalshi_subset')
print(f"log_loss={metrics['log_loss']:.4f}  brier={metrics['brier']:.4f}  ece={metrics['ece']:.4f}  acc={metrics['accuracy_argmax']:.3f}")

# Spread (overround) — Kalshi prices won't sum to 1.0
spread = np.array([r['closing_price_a'] + r['closing_price_b'] for r in matches])
print(f'\\nKalshi close-price sum stats: mean={spread.mean():.3f} median={np.median(spread):.3f} '
      f'(>1 = overround / spread crossing the midpoint)')
""")
)

CELLS.append(
    md("""## Orientation-symmetrized predictions (matches live serving, 2026-06-11 fix)

The deployed models are not corner-symmetric: training augmentation did not sign-flip
the `*_dif` / `skill_diff_mean` columns, so a single orientation's prediction depends on
which fighter happens to sit in the Red corner. As of 2026-06-11 the live pipeline
averages each prediction over both corner orderings (`predict_ensemble_symmetric`).
This section re-scores the same matched fights with symmetrized probabilities
`p_sym = (p(row) + 1 − p(mirror(row))) / 2`, where `mirror` swaps R/B columns,
negates the dif features, and swaps the `better_rank` label — so the table below
reflects what the live system actually serves.""")
)

CELLS.append(
    code("""from ufc_pred.features.static_v1 import _swap_red_blue

mirrored = _swap_red_blue(matched_fights)
Xm, _, _, _ = prepare(mirrored, augment_symmetry=False, one_hot=False)
Xm = Xm.reindex(columns=payload['columns'], fill_value=None)
for c in payload.get('cat_features', []):
    Xm[c] = Xm[c].fillna('__missing__').astype(str)
p_mir = payload['model'].predict_proba(Xm)[:, 1]
p_sym = 0.5 * (p_red + 1.0 - p_mir)

gap = p_red - (1.0 - p_mir)
print(f'orientation gap |p_fwd - (1-p_rev)|: mean={np.abs(gap).mean():.4f}  '
      f'median={np.median(np.abs(gap)):.4f}  max={np.abs(gap).max():.4f}')

m_sym = evaluate(y_red, p_sym, label='kalshi_subset_sym')
print(f"symmetric:  log_loss={m_sym['log_loss']:.4f}  brier={m_sym['brier']:.4f}  "
      f"ece={m_sym['ece']:.4f}  acc={m_sym['accuracy_argmax']:.3f}")
print(f"original :  log_loss={metrics['log_loss']:.4f}  brier={metrics['brier']:.4f}  "
      f"ece={metrics['ece']:.4f}  acc={metrics['accuracy_argmax']:.3f}")

sym_rows = []
for thr in (0.03, 0.05):
    for label, pp in (('original', p_red), ('symmetric', p_sym)):
        r = evaluate_bets(pp, y_red, R_kal_odds, B_kal_odds,
                          edge_threshold=thr, fee_rate=0.07, use_no_vig=False)
        sym_rows.append({'edge_thr': f'{int(thr*100)}%', 'preds': label, 'n_bets': r.n_bets,
                         'roi_pct': r.roi_pct, 'hit_rate': r.hit_rate,
                         'ci95_lo': r.ci95_roi_pct[0], 'ci95_hi': r.ci95_roi_pct[1]})
pd.DataFrame(sym_rows)
""")
)

CELLS.append(
    md("""## Flat-stake ROI on Kalshi prices — across fee scenarios and edge thresholds

Kalshi prices are **NOT no-vig** (last-trade on each side can cross the midpoint), so `use_no_vig=False` and the EV calculation uses the raw closing prices.

Three fee scenarios:
- **0% fee** — idealized; no trading costs.
- **2% fee** — for comparison with Polymarket's effective cost.
- **7% fee** — **Kalshi's actual fee structure** (~7% on winnings). This is the realistic deployment number.

**Watch the CI95 lower bound** — that's the credibility signal.
""")
)

CELLS.append(
    code("""rows = []
for fee in [0.0, 0.02, 0.07]:
    for thr in [0.03, 0.05, 0.10]:
        r = evaluate_bets(
            p_red, y_red, R_kal_odds, B_kal_odds,
            edge_threshold=thr, fee_rate=fee, use_no_vig=False,
        )
        rows.append({
            'fee_pct': f'{int(fee*100)}%',
            'edge_thr': f'{int(thr*100)}%',
            'n_bets': r.n_bets,
            'roi_pct': r.roi_pct,
            'ci95_low': r.ci95_roi_pct[0],
            'ci95_high': r.ci95_roi_pct[1],
            'hit_rate': r.hit_rate,
            'mean_ev_pct': r.mean_ev_pct,
        })
flat_df = pd.DataFrame(rows)
flat_df.style.format({
    'roi_pct': '{:+.3f}', 'ci95_low': '{:+.3f}', 'ci95_high': '{:+.3f}',
    'hit_rate': '{:.3f}', 'mean_ev_pct': '{:+.2f}',
})
""")
)

CELLS.append(
    md("""## Side-by-side: Kalshi vs BestFightOdds (test eval)

Same model, same fights — different market pricing assumption. The BFO-Kalshi-like number in notebook 04 was a synthetic estimate; this is what real Kalshi closing prices give us.
""")
)

CELLS.append(
    code("""r_bfo = evaluate_bets(
    p_red, y_red, matched_fights['R_odds'], matched_fights['B_odds'],
    edge_threshold=0.05, fee_rate=0.07, use_no_vig=True,
)
r_kal = evaluate_bets(
    p_red, y_red, R_kal_odds, B_kal_odds,
    edge_threshold=0.05, fee_rate=0.07, use_no_vig=False,
)
compare = pd.DataFrame([{
    'market': 'BestFightOdds Kalshi-like (no-vig + 7% fee)',
    'n_bets': r_bfo.n_bets, 'roi_pct': r_bfo.roi_pct,
    'ci95_low': r_bfo.ci95_roi_pct[0], 'ci95_high': r_bfo.ci95_roi_pct[1],
    'hit_rate': r_bfo.hit_rate, 'mean_ev_pct': r_bfo.mean_ev_pct,
}, {
    'market': 'Kalshi (real prices + 7% fee)',
    'n_bets': r_kal.n_bets, 'roi_pct': r_kal.roi_pct,
    'ci95_low': r_kal.ci95_roi_pct[0], 'ci95_high': r_kal.ci95_roi_pct[1],
    'hit_rate': r_kal.hit_rate, 'mean_ev_pct': r_kal.mean_ev_pct,
}])
compare.style.format({
    'roi_pct': '{:+.3f}', 'ci95_low': '{:+.3f}', 'ci95_high': '{:+.3f}',
    'hit_rate': '{:.3f}', 'mean_ev_pct': '{:+.2f}',
})
""")
)

CELLS.append(md("""## Kelly bankroll grid on Kalshi prices (7% fee)"""))

CELLS.append(
    code("""FRACTIONS = [0.10, 0.25, 0.50, 1.00]
CAPS = [0.01, 0.02, 0.05, 0.10, 1.00]
FEE_RATE = 0.07   # Kalshi fee
EDGE_THRESHOLD = 0.03

grid = {}
for f in FRACTIONS:
    for c in CAPS:
        grid[(f, c)] = evaluate_bets_kelly(
            p_red, y_red, R_kal_odds, B_kal_odds,
            edge_threshold=EDGE_THRESHOLD, fee_rate=FEE_RATE, use_no_vig=False,
            kelly_fraction=f, max_bet_fraction=c, starting_bankroll=1.0,
        )

def grid_to_df(key):
    data = {('no cap' if c >= 1 else f'{int(c*100)}%'): [grid[(f, c)][key] for f in FRACTIONS] for c in CAPS}
    df = pd.DataFrame(data, index=[f'{int(f*100)}%-K' for f in FRACTIONS])
    df.index.name = 'Kelly fraction'; df.columns.name = 'per-bet cap'
    return df

bk = grid_to_df('final_bankroll')
dd = grid_to_df('max_drawdown_pct')

print('FINAL BANKROLL ($1 → ?) on Kalshi prices, 7% fee, edge ≥ 3%')
display(bk.style.format('${:,.2f}').background_gradient(cmap='RdYlGn', vmin=0.5, vmax=50, axis=None))

print('\\nMAX DRAWDOWN (%)')
display(dd.style.format('{:.1f}%').background_gradient(cmap='RdYlGn_r', vmin=0, vmax=100, axis=None))
""")
)

CELLS.append(
    md("""## Deploy-config trajectories on Kalshi

Same three accounts as DEPLOY.md, priced against Kalshi. Starting bankroll $300 each.
""")
)

CELLS.append(
    code("""payload_c = joblib.load(ROOT / 'artifacts/models/v3_full2000_no_skill_corrupted_trainval.joblib')
Xc = X.copy()
p_corrupt = payload_c['model'].predict_proba(Xc)[:, 1]

START = 300.0

def simulate(p, kelly, cap):
    return evaluate_bets_kelly(
        p, y_red, R_kal_odds, B_kal_odds,
        edge_threshold=EDGE_THRESHOLD, fee_rate=FEE_RATE, use_no_vig=False,
        kelly_fraction=kelly, max_bet_fraction=cap, starting_bankroll=START,
    )

sims = {
    'A': simulate(p_red, 0.10, 0.10),
    'B': simulate(p_red, 0.25, 1.00),
    'C': simulate(p_corrupt, 0.25, 1.00),
}
for tag, sim in sims.items():
    print(f'Account {tag}: ${START:.0f} → ${sim["final_bankroll"]:,.2f}  '
          f'max DD {sim["max_drawdown_pct"]:.1f}%  n_bets={sim["n_bets"]}')
""")
)

CELLS.append(
    code("""colors = {'A': '#1a9641', 'B': '#1f78b4', 'C': '#d7191c'}
fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=False)
for ax, scale in zip(axes, ['linear', 'log']):
    for tag, sim in sims.items():
        ax.plot(sim['trajectory'], color=colors[tag], linewidth=2.0,
                label=f'Account {tag}: ${START:.0f} → ${sim["final_bankroll"]:,.2f}')
    ax.axhline(START, color='black', alpha=0.4, linewidth=0.8)
    ax.set_yscale(scale)
    ax.set_xlabel('Bet # (chronological through Kalshi-matched test fights)')
    ax.set_ylabel(f'Bankroll ($) — {scale}')
    ax.set_title(f'Deploy accounts on Kalshi prices ({scale} scale)')
    ax.legend(loc='upper left' if scale == 'linear' else 'lower right')
    ax.grid(alpha=0.3, which='both')
plt.tight_layout(); plt.show()
""")
)

CELLS.append(
    md("""## Side-by-side: deploy-config bankrolls (BFO-Kalshi-like vs real Kalshi)

What did the synthetic Kalshi-like simulation (notebook 04) project vs what real Kalshi pricing delivers on the matched subset?
""")
)

CELLS.append(
    code("""def simulate_bfo(p, kelly, cap):
    return evaluate_bets_kelly(
        p, y_red, matched_fights['R_odds'], matched_fights['B_odds'],
        edge_threshold=0.03, fee_rate=0.07, use_no_vig=True,
        kelly_fraction=kelly, max_bet_fraction=cap, starting_bankroll=START,
    )
sims_bfo = {
    'A': simulate_bfo(p_red, 0.10, 0.10),
    'B': simulate_bfo(p_red, 0.25, 1.00),
    'C': simulate_bfo(p_corrupt, 0.25, 1.00),
}
tbl = pd.DataFrame([{
    'account': tag,
    'config': ('10%-K + 10% cap' if tag == 'A' else '¼-K + no cap'),
    'BFO Kalshi-like ($300 → ?)': sims_bfo[tag]['final_bankroll'],
    'Real Kalshi ($300 → ?)':   sims[tag]['final_bankroll'],
    'ratio': sims_bfo[tag]['final_bankroll'] / max(sims[tag]['final_bankroll'], 1e-9),
} for tag in ['A','B','C']])
tbl.style.format({
    'BFO Kalshi-like ($300 → ?)': '${:,.2f}',
    'Real Kalshi ($300 → ?)':   '${:,.2f}',
    'ratio': '{:.2f}×',
})
""")
)

CELLS.append(md("""## Per-fight-night Kalshi P&L for each account"""))

CELLS.append(
    code("""def simulate_log(p, kelly, cap):
    R = R_kal_odds.to_numpy(); B = B_kal_odds.to_numpy()
    dec_R = 1.0 / np.where(np.asarray([r['a_is_red'] for r in matches]),
                            [r['closing_price_a'] for r in matches],
                            [r['closing_price_b'] for r in matches])
    dec_B = 1.0 / np.where(np.asarray([r['a_is_red'] for r in matches]),
                            [r['closing_price_b'] for r in matches],
                            [r['closing_price_a'] for r in matches])
    eff_R = 1.0 + (1.0 - FEE_RATE) * (dec_R - 1.0)
    eff_B = 1.0 + (1.0 - FEE_RATE) * (dec_B - 1.0)
    p_R = np.asarray(p); p_B = 1.0 - p_R
    ev_R = p_R * eff_R - 1.0; ev_B = p_B * eff_B - 1.0
    bet_red = ev_R >= ev_B
    chosen_ev = np.where(bet_red, ev_R, ev_B)
    chosen_dec = np.where(bet_red, eff_R, eff_B)
    chosen_p = np.where(bet_red, p_R, p_B)
    bets_mask = chosen_ev > EDGE_THRESHOLD
    won_full = np.where(bet_red, y_red == 1, y_red == 0)
    bankroll = START; rows = []
    for i in range(len(p_R)):
        if not bets_mask[i]: continue
        b = chosen_dec[i] - 1.0
        pi = chosen_p[i]; qi = 1.0 - pi
        fk = (b*pi - qi) / b
        if fk <= 0: continue
        stake_frac = min(kelly * fk, cap)
        stake = bankroll * stake_frac
        if won_full[i]: bankroll += stake * (chosen_dec[i] - 1.0)
        else: bankroll -= stake
        rows.append({'date': matches[i]['date'], 'won': int(won_full[i]),
                     'stake': stake, 'bankroll_after': bankroll})
    return pd.DataFrame(rows)

logs = {
    'A': simulate_log(p_red, 0.10, 0.10),
    'B': simulate_log(p_red, 0.25, 1.00),
    'C': simulate_log(p_corrupt, 0.25, 1.00),
}

def per_night(log):
    g = log.groupby('date', sort=True)
    out = g.agg(n_bets=('won','size'), n_wins=('won','sum'),
                bankroll_end=('bankroll_after','last')).reset_index()
    prev = np.concatenate([[START], out['bankroll_end'].iloc[:-1].to_numpy()])
    out['pnl'] = out['bankroll_end'] - prev
    return out

nights = {tag: per_night(log) for tag, log in logs.items()}
for tag, n in nights.items():
    print(f'Account {tag}: {len(n)} fight nights, ${START:.0f} → ${n["bankroll_end"].iloc[-1]:,.2f}')
""")
)

CELLS.append(
    code("""fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
for ax, scale in zip(axes, ['linear', 'log']):
    for tag, n in nights.items():
        ax.plot(n['date'], n['bankroll_end'], color=colors[tag], linewidth=1.8,
                label=f'Account {tag}: ${START:.0f} → ${n["bankroll_end"].iloc[-1]:,.2f}')
    ax.axhline(START, color='black', alpha=0.4, linewidth=0.8)
    ax.set_yscale(scale)
    ax.set_ylabel(f'Bankroll ($) — {scale}')
    ax.set_title(f'Per-fight-night bankroll on Kalshi prices ({scale})')
    ax.legend(loc='upper left' if scale == 'linear' else 'lower right')
    ax.grid(alpha=0.3, which='both')
axes[1].set_xlabel('Date (fight night)')
plt.tight_layout(); plt.show()
""")
)

CELLS.append(
    code("""# Top 5 best and worst nights per account
out = []
for tag, n in nights.items():
    best = n.nlargest(5, 'pnl').assign(rank='best', account=tag)
    worst = n.nsmallest(5, 'pnl').assign(rank='worst', account=tag)
    out.append(pd.concat([best, worst]))
moves = pd.concat(out, ignore_index=True)[['account','rank','date','n_bets','n_wins','pnl','bankroll_end']]
moves['pnl'] = moves['pnl'].round(2)
moves['bankroll_end'] = moves['bankroll_end'].round(2)
moves
""")
)

CELLS.append(
    md("""## Calibration on Kalshi-matched bets

Same overconfidence finding as nb05; expected to repeat since it's a model property.
""")
)

CELLS.append(
    code("""r_kal_5 = evaluate_bets(
    p_red, y_red, R_kal_odds, B_kal_odds,
    edge_threshold=0.05, fee_rate=FEE_RATE, use_no_vig=False,
)
bd = r_kal_5.bets_df
bins = [(0.5,0.6),(0.6,0.7),(0.7,0.8),(0.8,0.9),(0.9,1.0)]
rows = []
for lo, hi in bins:
    m = (bd['model_prob_chosen'] >= lo) & (bd['model_prob_chosen'] < hi)
    if m.sum() == 0: continue
    rows.append({
        'bin': f'[{lo:.1f}, {hi:.1f})',
        'n_bets': int(m.sum()),
        'avg_claimed_p': bd.loc[m, 'model_prob_chosen'].mean(),
        'realized_hit': bd.loc[m, 'won'].mean(),
        'gap_pp': (bd.loc[m, 'won'].mean() - bd.loc[m, 'model_prob_chosen'].mean()) * 100,
    })
calib_df = pd.DataFrame(rows)
calib_df.style.format({
    'avg_claimed_p': '{:.3f}', 'realized_hit': '{:.3f}', 'gap_pp': '{:+.1f}',
})
""")
)

CELLS.append(
    md("""## Walkthrough — first 10 bets on Kalshi (Account B: ¼-Kelly, no cap, real model)

Same column legend as nb05 cell 24: per-bet inputs and Kelly math.
""")
)

CELLS.append(
    code("""START = 300.0
FEE = FEE_RATE
KELLY_FRAC = 0.25
CAP = 1.0
EDGE_THR = 0.03

pa = np.array([m['closing_price_a'] for m in matches])
pb = np.array([m['closing_price_b'] for m in matches])
a_is_red = np.array([m['a_is_red'] for m in matches])

dec_R = np.where(a_is_red, 1.0/pa, 1.0/pb)
dec_B = np.where(a_is_red, 1.0/pb, 1.0/pa)
eff_R = 1.0 + (1.0 - FEE) * (dec_R - 1.0)
eff_B = 1.0 + (1.0 - FEE) * (dec_B - 1.0)

p_R = p_red; p_B = 1.0 - p_R
ev_R = p_R * eff_R - 1.0; ev_B = p_B * eff_B - 1.0
bet_red = ev_R >= ev_B
chosen_ev = np.where(bet_red, ev_R, ev_B)
chosen_dec_eff = np.where(bet_red, eff_R, eff_B)
chosen_p_model = np.where(bet_red, p_R, p_B)
chosen_p_kal  = np.where(bet_red,
                          np.where(a_is_red, pa, pb),
                          np.where(a_is_red, pb, pa))
chosen_name = np.where(bet_red,
                       matched_fights['R_fighter'].to_numpy(),
                       matched_fights['B_fighter'].to_numpy())
won_full = np.where(bet_red,
                    matched_fights['Winner'].to_numpy() == 'Red',
                    matched_fights['Winner'].to_numpy() == 'Blue').astype(int)
bets_mask = chosen_ev > EDGE_THR

bankroll = START
log = []
bet_idx = 0
for i in range(len(matches)):
    if not bets_mask[i]: continue
    b = chosen_dec_eff[i] - 1.0
    p = chosen_p_model[i]; q = 1.0 - p
    full_kelly = (b*p - q) / b
    if full_kelly <= 0: continue
    stake_frac = min(KELLY_FRAC * full_kelly, CAP)
    stake = bankroll * stake_frac
    realized = stake * (chosen_dec_eff[i] - 1.0) if won_full[i] else -stake
    bankroll += realized
    log.append({
        'bet_idx': bet_idx,
        'date': matched_fights['date'].iloc[i].date(),
        'chosen': chosen_name[i],
        'model_p': p, 'kal_p': chosen_p_kal[i],
        'dec': 1.0/chosen_p_kal[i], 'eff_dec': chosen_dec_eff[i],
        'EV%': chosen_ev[i] * 100, 'full_K': full_kelly,
        'qK_pct': stake_frac * 100, 'stake': stake,
        'won': int(won_full[i]), 'realized': realized, 'bankroll': bankroll,
    })
    bet_idx += 1

log_df = pd.DataFrame(log)
print(f'Total bets placed: {len(log_df)}')
if len(log_df) >= 10:
    print(f'Bankroll at end of first 10 bets: ${log_df["bankroll"].iloc[9]:,.2f}')
print(f'Final bankroll: ${log_df["bankroll"].iloc[-1]:,.2f}' if len(log_df) else 'no bets')

if len(log_df):
    log_df.head(10).style.format({
        'model_p': '{:.3f}', 'kal_p': '{:.3f}', 'dec': '{:.3f}', 'eff_dec': '{:.3f}',
        'EV%': '{:+.2f}%', 'full_K': '{:.3f}', 'qK_pct': '{:.2f}%',
        'stake': '${:,.2f}', 'realized': '${:+,.2f}', 'bankroll': '${:,.2f}',
    })
""")
)

CELLS.append(md("""## Walkthrough — Account C (corrupted model) first 10 bets"""))

CELLS.append(
    code("""p_R_c = p_corrupt; p_B_c = 1.0 - p_R_c
ev_R_c = p_R_c * eff_R - 1.0; ev_B_c = p_B_c * eff_B - 1.0
bet_red_c = ev_R_c >= ev_B_c
chosen_ev_c = np.where(bet_red_c, ev_R_c, ev_B_c)
chosen_dec_eff_c = np.where(bet_red_c, eff_R, eff_B)
chosen_p_model_c = np.where(bet_red_c, p_R_c, p_B_c)
chosen_p_kal_c = np.where(bet_red_c,
                          np.where(a_is_red, pa, pb),
                          np.where(a_is_red, pb, pa))
chosen_name_c = np.where(bet_red_c,
                         matched_fights['R_fighter'].to_numpy(),
                         matched_fights['B_fighter'].to_numpy())
won_full_c = np.where(bet_red_c,
                      matched_fights['Winner'].to_numpy() == 'Red',
                      matched_fights['Winner'].to_numpy() == 'Blue').astype(int)
bets_mask_c = chosen_ev_c > EDGE_THR

bankroll_c = START
log_c = []
bet_idx_c = 0
for i in range(len(matches)):
    if not bets_mask_c[i]: continue
    b = chosen_dec_eff_c[i] - 1.0
    p = chosen_p_model_c[i]; q = 1.0 - p
    fk = (b*p - q) / b
    if fk <= 0: continue
    stake_frac = min(KELLY_FRAC * fk, CAP)
    stake = bankroll_c * stake_frac
    realized = stake * (chosen_dec_eff_c[i] - 1.0) if won_full_c[i] else -stake
    bankroll_c += realized
    log_c.append({
        'bet_idx': bet_idx_c,
        'date': matched_fights['date'].iloc[i].date(),
        'chosen': chosen_name_c[i],
        'model_p': p, 'kal_p': chosen_p_kal_c[i],
        'dec': 1.0/chosen_p_kal_c[i], 'eff_dec': chosen_dec_eff_c[i],
        'EV%': chosen_ev_c[i] * 100, 'full_K': fk,
        'qK_pct': stake_frac * 100, 'stake': stake,
        'won': int(won_full_c[i]), 'realized': realized, 'bankroll': bankroll_c,
    })
    bet_idx_c += 1

log_c_df = pd.DataFrame(log_c)
print(f'Total bets placed (Account C): {len(log_c_df)}')
if len(log_c_df) >= 10:
    print(f'Bankroll at end of first 10 bets: ${log_c_df["bankroll"].iloc[9]:,.2f}')
print(f'Final bankroll: ${log_c_df["bankroll"].iloc[-1]:,.2f}' if len(log_c_df) else 'no bets')

if len(log_c_df):
    log_c_df.head(10).style.format({
        'model_p': '{:.3f}', 'kal_p': '{:.3f}', 'dec': '{:.3f}', 'eff_dec': '{:.3f}',
        'EV%': '{:+.2f}%', 'full_K': '{:.3f}', 'qK_pct': '{:.2f}%',
        'stake': '${:,.2f}', 'realized': '${:+,.2f}', 'bankroll': '${:,.2f}',
    })
""")
)

CELLS.append(
    code("""# Side-by-side: same first 10 bets, Account B (real) vs Account C (corrupted)
def fmt_money(x): return f'${x:,.2f}'
n = min(10, len(log_df), len(log_c_df))
side_by_side = pd.DataFrame({
    'date':    log_df['date'].head(n).astype(str),
    'chosen_B': log_df['chosen'].head(n).str.slice(0, 18),
    'B_model_p': log_df['model_p'].head(n).round(3),
    'B_qK%':   log_df['qK_pct'].head(n).round(2),
    'B_stake': log_df['stake'].head(n).apply(fmt_money),
    'B_bank':  log_df['bankroll'].head(n).apply(fmt_money),
    'chosen_C': log_c_df['chosen'].head(n).str.slice(0, 18),
    'C_model_p': log_c_df['model_p'].head(n).round(3),
    'C_qK%':   log_c_df['qK_pct'].head(n).round(2),
    'C_stake': log_c_df['stake'].head(n).apply(fmt_money),
    'C_bank':  log_c_df['bankroll'].head(n).apply(fmt_money),
})
side_by_side
""")
)

CELLS.append(
    md("""## Takeaways

Final numbers depend on backfill size and match rate. Compare with notebook 05 (Polymarket) to choose deployment venue:

| Dimension | Polymarket (nb05) | Kalshi (this nb) |
|---|---|---|
| Effective per-trade fee | ~2% | ~7% |
| Closing-price form | no-vig (sum to 1) | last-trade (sum ≠ 1, spread visible) |
| Available in CT/MA | ❌ blocked | ✅ |
| Historical data quality | scraped via Gamma API | official REST API, includes OI |
| Bet count (matched test fights) | 615 | see top of notebook |

The CI95 lower bound on the 7% fee scenario is the headline credibility number — if it's positive, the model's edge survives Kalshi's higher fee + spread cost; if it crosses zero, the venue is marginal.

This notebook is regeneratable: re-run `scripts/kalshi_backfill.py` to refresh historical data, then rerun all cells.
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
    out = ROOT / "notebooks/08_kalshi_evaluation.ipynb"
    out.write_text(json.dumps(nb, indent=1))
    print(f"wrote {out} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()

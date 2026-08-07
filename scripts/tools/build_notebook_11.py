"""Build notebooks/11_cap_large_edges.ipynb from the precomputed results in
artifacts/metrics/cap_large_edges_sim.json (produced by scripts/cap_large_edges_sim.py).

The notebook only LOADS and renders the JSON, so it executes in seconds and does
not retrain. The heavy lifting (10-seed ensembles, symmetrize+sharpen, per-account
Kelly, edge-cap sims, sweep) lives in the script and is documented in §0.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "11_cap_large_edges.ipynb"

nb = nbf.v4.new_notebook()
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s))


def code(s):
    cells.append(nbf.v4.new_code_cell(s))


md(r"""# 11 — Capping large model edges: does it help or hurt?

**Question.** The live bet tool sometimes sizes a big stake on a fight where the
model disagrees hugely with the market — e.g. **Bolanos** (model 53%, market 23¢,
**+30pp edge**, $73 stake) or **Garcia** on 06‑14 (model 74%, market 43¢, +31pp,
the night's largest position — which *lost*). The deployment never caps a bet by
edge size: the seed-grid backtests bet every +EV fight regardless of edge and
still compounded. This notebook tests, directly, whether **capping large edges**
would have helped on two real-priced windows.

**Four base simulations** (2 venues × {uncapped, capped}), each run for all three
deploy accounts, with the cap applied two ways (exclude / clip), plus a full cap
sweep:

| Venue | Window | Prices | Model artifact (leakage-free) |
|---|---|---|---|
| **Polymarket** | 2025‑07 → 2026‑05 (11 mo, 435 fights) | real Polymarket no‑vig CLOB | 10‑seed ensemble @ **cutoff 2025‑01‑01** (6‑mo‑stale analog; the live 2025‑11‑30 artifact would leak on Jul–Nov 2025) |
| **Kalshi** | 2026‑01‑24 → 2026‑05 (41 fights) | real Kalshi pre‑fight closing prices | the **actual live‑deployed 10‑seed ensemble @ cutoff 2025‑11‑30** (fully out‑of‑sample on 2026 fights) |

**Everything else matches the deployed implementation** (DEPLOY.md §3): orientation‑
symmetrized predictions (2026‑06‑11 fix), `sharpen_T=1.25`, per‑account Kelly
(A = 10%‑K/10%‑cap real, B = 25%‑K/no‑cap real, C = 25%‑K/no‑cap corrupted),
3% edge threshold, and the **quadratic venue fee** (Kalshi `0.07·p·(1−p)`,
Polymarket `0.03·p·(1−p)`).

**The cap.** It acts on the per‑bet **probability edge** = `model_p_chosen −
market_price_chosen` (the same "+Xc edge" the live tool prints). Two modes:
- **exclude** — skip any fight whose edge exceeds the cap (the "don't bet Bolanos" rule).
- **clip** — still bet, but size Kelly as if the edge equalled the cap (keeps exposure, kills the oversizing).

Headline cap = **15pp**; a sweep over 5→50pp is included.
""")

md(r"""## §0 Provenance / reproduce

All numbers below are loaded from `artifacts/metrics/cap_large_edges_sim.json`,
produced by:

```bash
PYTHONPATH=src .conda/bin/python scripts/cap_large_edges_sim.py   # ~5 min (trains the Polymarket ensemble)
```

That script is the single source of truth for the simulation math; this notebook
is a thin rendering layer so it re-runs in seconds.
""")

code(r"""import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd().parent
R = json.loads((ROOT / 'artifacts/metrics/cap_large_edges_sim.json').read_text())
cfg = R['config']
print('config:', json.dumps(cfg, indent=1))
for venue in ('kalshi', 'polymarket'):
    v = R[venue]
    print(f"\n{venue:11}: {v['n_fights']:>3} fights  {v['date_range'][0]}..{v['date_range'][1]}"
          f"   large-edge(>{int(cfg['headline_cap']*100)}pp): {v['large_edge_summary']['n']} bets, "
          f"hit rate {v['large_edge_summary']['hit_rate']}")
""")

md(r"""## §1 Headline — the four simulations (×3 accounts, ×2 cap modes)

`$300 → final` for each account, uncapped vs capped at 15pp. `n_large` = how many
of the placed bets exceed the 15pp cap (i.e. how much of the book the cap touches).""")

code(r"""def headline(venue):
    v = R[venue]; rows = []
    for a in 'ABC':
        h = v['headline'][a]
        rows.append(dict(
            account=a,
            uncapped=h['uncapped']['final'],
            exclude_15=h['exclude']['final'],
            clip_15=h['clip']['final'],
            dd_unc=h['uncapped']['max_dd'],
            n_bets=h['uncapped']['n_bets'],
            n_large=h['uncapped']['n_large'],
        ))
    return pd.DataFrame(rows).set_index('account')

for venue in ('kalshi', 'polymarket'):
    v = R[venue]
    print(f"\n===== {venue.upper()}  ({v['n_fights']} fights, {v['date_range'][0]}..{v['date_range'][1]}) =====")
    df = headline(venue)
    display(df.style.format({'uncapped': '${:,.0f}', 'exclude_15': '${:,.0f}',
                             'clip_15': '${:,.0f}', 'dd_unc': '{:.0f}%'}))
""")

md(r"""**Read:** capping *reduces* the final bankroll for every account on both venues,
under both modes. `exclude` is the most destructive (it removes the bets entirely);
`clip` keeps some upside but still costs most of the gain. Note `n_large` ≈ **the
majority** of placed bets — at 15pp the cap is not trimming a few outliers, it is
cutting more than half the book.""")

md(r"""## §2 Cap sweep — does *any* cap level help? (Account B)

Lower cap = more aggressive trimming. If large edges were noise, tightening the cap
would *raise* the curve. It does the opposite — monotonically.""")

code(r"""fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, venue in zip(axes, ('kalshi', 'polymarket')):
    v = R[venue]; sw = v['sweep']['B']
    unc = v['headline']['B']['uncapped']['final']
    caps = [s['cap'] * 100 for s in sw['exclude']]
    ax.plot(caps, [s['final'] for s in sw['exclude']], 'o-', label='exclude', color='#d7191c')
    ax.plot(caps, [s['final'] for s in sw['clip']], 's-', label='clip', color='#2c7bb6')
    ax.axhline(unc, color='black', ls='--', alpha=0.6, label=f'uncapped (${unc:,.0f})')
    ax.axvline(cfg['headline_cap'] * 100, color='gray', ls=':', alpha=0.6)
    ax.set_yscale('log'); ax.set_xlabel('edge cap (pp)')
    ax.set_ylabel('Account B final $ (log)')
    ax.set_title(f"{venue} — cap sweep (B, $300 start)")
    ax.legend(); ax.grid(alpha=0.3, which='both')
plt.tight_layout(); plt.show()
""")

md(r"""## §3 Bankroll trajectories — uncapped vs capped (Account B)""")

code(r"""fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, venue in zip(axes, ('kalshi', 'polymarket')):
    h = R[venue]['headline']['B']
    for mode, color in (('uncapped', '#1a9641'), ('clip', '#2c7bb6'), ('exclude', '#d7191c')):
        ax.plot(h[mode]['trajectory'], color=color, lw=1.8,
                label=f"{mode}: ${h[mode]['final']:,.0f}")
    ax.axhline(cfg['start'], color='black', alpha=0.4, lw=0.8)
    ax.set_yscale('log'); ax.set_xlabel('bet #'); ax.set_ylabel('bankroll $ (log)')
    ax.set_title(f"{venue} — Account B"); ax.legend(); ax.grid(alpha=0.3, which='both')
plt.tight_layout(); plt.show()
""")

md(r"""## §4 Where do the large edges come from — and do they win?

A "large edge" splits into two kinds: **favorites the market underpriced**
(model loves a side priced < what the model says) and **underdogs the model backs**
(market price < 0.5). Bolanos/Garcia are the second kind. Hit rates below.""")

code(r"""rows = []
for venue in ('kalshi', 'polymarket'):
    led = pd.DataFrame(R[venue]['large_edge_ledger'])
    fav = led[led['price'] >= 0.5]; dog = led[led['price'] < 0.5]
    rows.append(dict(venue=venue, n_large=len(led),
                     fav_n=len(fav), fav_hit=round(fav['won'].mean(), 3) if len(fav) else None,
                     dog_n=len(dog), dog_hit=round(dog['won'].mean(), 3) if len(dog) else None,
                     all_hit=round(led['won'].mean(), 3)))
display(pd.DataFrame(rows).set_index('venue'))

# The biggest underdog edges (the Bolanos/Garcia archetype) and how they resolved
print("\nLargest underdog edges (price < 0.5) — the Bolanos/Garcia archetype:")
for venue in ('kalshi', 'polymarket'):
    led = pd.DataFrame(R[venue]['large_edge_ledger'])
    dog = led[led['price'] < 0.5].sort_values('edge_pp', ascending=False).head(6)
    print(f"\n  {venue}:")
    for _, r in dog.iterrows():
        print(f"    {r['date']}  {r['fight'][:40]:40}  +{r['edge_pp']:.0f}pp  "
              f"p={r['model_p']:.2f} px={r['price']:.2f}  {'WON' if r['won'] else 'lost'}")
""")

md(r"""## §5 Is the edge real, or compounding luck? — flat-stake ROI by edge bucket

Compounded bankrolls are sequence-sensitive (a few early large-edge wins dominate;
DEPLOY/audit flag the headline multiples as ~2× inflated). **Flat-stake realized
ROI** removes that confound — $1 on every bet, no compounding. If the large-edge
buckets still show positive ROI, the edge is *real in this window*, not a Kelly artifact.""")

code(r"""BINS = [(-100, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 200)]
def blabel(lo, hi): return (f'<{hi}' if lo < 0 else f'{lo}+' if hi >= 200 else f'{lo}-{hi}') + 'pp'

def buckets(venue, kind='book_real'):
    b = pd.DataFrame(R[venue][kind])
    b['unit'] = np.where(b['won'] == 1, b['dec'] - 1.0, -1.0)
    b['e'] = b['edge'] * 100
    out = []
    for lo, hi in BINS:
        m = (b['e'] >= lo) & (b['e'] < hi)
        if m.sum() == 0: continue
        out.append(dict(bucket=blabel(lo, hi), n=int(m.sum()),
                        hit=round(b.loc[m, 'won'].mean(), 2),
                        avg_price=round(b.loc[m, 'price'].mean(), 2),
                        flat_roi_pct=round(b.loc[m, 'unit'].mean() * 100, 1)))
    out.append(dict(bucket='ALL', n=len(b), hit=round(b['won'].mean(), 2),
                    avg_price=round(b['price'].mean(), 2),
                    flat_roi_pct=round(b['unit'].mean() * 100, 1)))
    return pd.DataFrame(out).set_index('bucket')

for venue in ('kalshi', 'polymarket'):
    print(f"\n===== {venue.upper()} — flat-stake ROI by edge bucket (Account B, real model, uncapped) =====")
    display(buckets(venue))

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
for ax, venue in zip(axes, ('kalshi', 'polymarket')):
    bt = buckets(venue).drop('ALL')
    colors = ['#d7191c' if x < 0 else '#1a9641' for x in bt['flat_roi_pct']]
    ax.bar(bt.index, bt['flat_roi_pct'], color=colors)
    ax.axhline(0, color='black', lw=0.8)
    ax.axvline(2.5, color='gray', ls=':', alpha=0.7)  # 15pp cap boundary (after 10-15 bucket)
    ax.set_title(f"{venue} — flat ROI by edge bucket")
    ax.set_ylabel('flat-stake ROI %'); ax.set_xlabel('edge bucket')
    ax.grid(alpha=0.3, axis='y')
plt.tight_layout(); plt.show()
print("\nGreen = profitable bucket, red = loss. The cap would remove the buckets to the RIGHT.")
""")

md(r"""## §6 Takeaways

**1. The backtest says capping large edges *hurts* — on every account, both venues,
both cap modes, and every cap level.** Tightening the cap monotonically lowers the
final bankroll (§2). At the 15pp headline, Account B goes from **\$1,174 → \$243
(exclude) / \$540 (clip)** on Kalshi and **\$2.69M → \$1,386 / \$42,858** on
Polymarket. This is the user's original observation confirmed: the deployment bets
every +EV edge and the large ones are doing the heavy lifting.

**2. It is not just a compounding artifact.** Flat‑stake realized ROI (§5, no
compounding) shows the **same shape**: the `<10pp` buckets are **negative** on both
venues (Kalshi −51%/−52%, Polymarket −10%/−13%), while the **large‑edge buckets are
positive** and the **30+pp tier — the Bolanos/Garcia archetype — is the single best
bucket** (+113% Kalshi, +67% Polymarket flat ROI). In this window the model's real
edge *lives in the large disagreements*; the small edges lose money. Capping keeps
the losers and discards the winners.

**3. At 15pp you are not trimming outliers — you are gutting the book.** `n_large`
is the majority of bets (Kalshi 23/37, Polymarket 222/399). On Kalshi the model
disagrees with the market by >15pp on ~60% of fights. "Cap large edges" at 15pp is
effectively "only bet near‑coinflips," which §5 shows are the unprofitable bets.

**4. The honest caveats — why this is not a license to fire at will.**
   - **Spent test set / friendly calendar.** Both windows overlap the period the
     model+strategy were selected on (DEPLOY §5/§6, audit 2026‑06‑11). The large‑edge
     hit rates (Kalshi 78%, Polymarket 60%) are in‑sample‑flavored and may not hold live.
   - **The large‑edge bucket is also where model *error* concentrates.** High edge =
     high model–market disagreement = either real alpha *or* a bad feature row. The
     backtest can't tell which prospectively.
   - **Live evidence is thin and mixed.** The first true large‑edge underdog live —
     **Garcia, 06‑14, +31pp — lost**, and it was the biggest position of the night.
     One bet isn't a verdict, but it's the failure mode the cap was meant to prevent.
   - **Small Kalshi buckets.** Kalshi per‑bucket n is 3–9; treat its bucket ROIs as
     directional, not precise. Polymarket buckets (n=33–88) are firmer.

**5. So what should change live?** On *expected value*, the data says **don't cap by
edge** — keep betting large edges. The real risk in Bolanos/Garcia wasn't the edge,
it was **single‑bet variance**: $73–$111 on one underdog. The lever that addresses
that without throwing away EV is **position sizing / concentration**, not an edge cap
— and the deployment already has those knobs (Account A's 10% cap, the combined
liquidity cap, the 50‑bet hit‑rate kill switch). A defensible tweak would be a
**per‑fight bankroll cap on no‑cap accounts** (limit any single bet to e.g. 8–10% of
that account) — which trims tail variance on the Garcia case while leaving the
large‑edge EV intact. That is a *sizing* cap, categorically different from the
*edge* cap tested here, which the data rejects.

> Bottom line: my earlier "skip Bolanos" advice was risk‑management instinct, but the
> data (both compounded and flat‑stake) does not support an **edge** cap — it supports
> a **per‑bet size** cap. The edge itself is where the money is.
""")

nb["cells"] = cells
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
OUT.write_text(nbf.writes(nb))
print(f"wrote {OUT}  ({len(cells)} cells)")

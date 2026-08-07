"""Build notebooks/10_balance_lag_simulation.ipynb.

Simulates the first-fight-week balance-capture lag bug (sizing bankroll lagged
~3 fights) versus the correct no-lag compounding, on two windows:
  A. frozen-polymarket  (seed grid, strategy frozen_2025_07)
  B. kalshi             (T-90min snapshot, ensemble inference)
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "10_balance_lag_simulation.ipynb"

nb = nbf.v4.new_notebook()
cells = []


def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))


def code(src):
    cells.append(nbf.v4.new_code_cell(src))


# ---------------------------------------------------------------------------
md(r"""# 10 — Balance-capture lag simulation

**The bug (first live card, 2026-06-06).** Notifications/captures fire at T−90min.
On the live card, for the first ~3–4 fights the captured account balance was
*stale* — it still showed the money from **~3 fights prior**. Settlements were
landing in the captured balance with a **~3-fight delay**. Because every bet is
**fractional-Kelly sized off the captured balance**, those early fights were
sized off a bankroll that hadn't yet absorbed the prior fights' wins/losses.

**What this notebook does.** Re-runs the compounding bankroll simulation with a
*sizing lag* and compares it to the correct (no-lag) version, on two windows:

| Window | Data | Period |
|---|---|---|
| **A. Frozen polymarket** | seed grid, strategy `frozen_2025_07` | 2025-07 → 2026-05 |
| **B. Kalshi** | T−90min per-fight snapshot, ensemble inference | 2026-01-24 → 2026-05-30 |

### Lag model (precise)
Within each **card** (fights sharing a date):

- The **true bankroll** always realizes the *actual* win/loss of every bet immediately.
- The bankroll **used to size** fight *j* (0-indexed within the card) reflects only
  settlements through fight `j − lag`. For `j < lag` it is the **card-opening balance**
  (exactly the observed bug: "first ~3 fights show the money from 3 fights ago").
- The lag **resets at each card boundary** — between cards the balance reconciles.
  So the simulation answers: *"what if this same ~3-fight lag happened on every card?"*

`lag = 0` reproduces the canonical compounding exactly. The reported bug is `lag = 3`.
Accounts: **A** = 10% Kelly (10% stake cap), **B** = 25% Kelly (real model),
**C** = 25% Kelly (corrupted model).""")

# ---------------------------------------------------------------------------
code(r"""import sys
from pathlib import Path
from itertools import groupby

import numpy as np
import pandas as pd

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()

pd.set_option("display.float_format", lambda v: f"{v:,.2f}")

# Account specs (must match deployment / strategy_grid.ACCOUNT_SPECS)
#   name, kelly_frac, stake_cap, model_kind
ACCOUNTS = {
    "A": dict(kfrac=0.10, cap=0.10, kind="real"),
    "B": dict(kfrac=0.25, cap=1.00, kind="real"),
    "C": dict(kfrac=0.25, cap=1.00, kind="corrupt"),
}
START = 300.0
LAGS = [0, 1, 2, 3, 4]
BUG_LAG = 3  # the lag reported on the live card
print("ROOT:", ROOT)""")

# ---------------------------------------------------------------------------
md(r"""## The simulator

`simulate_with_lag` walks bets in chronological order, grouped by card. Each bet
is a dict with `date`, `fk` (full-Kelly fraction), `b` (net profit per $1 staked on
a win), and `won` (0/1). The stake for fight *j* uses the bankroll lagged by `lag`
settlements *within the card*; the true bankroll always books the real outcome.""")

code(r"""def simulate_with_lag(bets, kfrac, cap, lag, start=START, min_stake=0.0):
    # Compounding fractional-Kelly sim with a within-card sizing lag.
    # bets: chronological list of dicts {date, fk, b, won}.
    # Returns (final_bank, per_bet_records).
    true_bank = float(start)
    records = []
    # stable groupby on date, preserving chronological order
    for date, grp in groupby(bets, key=lambda x: x["date"]):
        grp = list(grp)
        card_open = true_bank
        cum = [card_open]              # cum[j] = true bank after j settled bets this card
        for j, bet in enumerate(grp):
            sizing_bank = cum[max(0, j - lag)]      # j<lag -> card_open (stale)
            frac = min(kfrac * bet["fk"], cap)
            stake = sizing_bank * frac
            if stake < min_stake:                    # bet dropped under this sizing bank
                cum.append(cum[-1])
                records.append({**bet, "stake": 0.0, "placed": False,
                                "sizing_bank": sizing_bank, "true_bank_before": cum[-2] if len(cum) > 1 else card_open})
                continue
            pnl = stake * bet["b"] if bet["won"] else -stake
            records.append({**bet, "stake": stake, "placed": True,
                            "sizing_bank": sizing_bank, "true_bank_before": cum[-1],
                            "pnl": pnl})
            cum.append(cum[-1] + pnl)
        true_bank = cum[-1]
    return true_bank, records""")

# ---------------------------------------------------------------------------
md(r"""## Window A — Frozen polymarket (`frozen_2025_07`)

Per-bet records come straight from the seed-grid backtest
(`data/interim/all_accounts_seed_grid_bets.parquet`), which stores `fk`, `dec_odds`
and `won` for every placed bet in chronological order. `b = dec_odds − 1`.
10 seeds → we report the median final and the per-seed lag/no-lag ratio so the
lag effect is separated from seed noise. lag=0 is validated against the stored
`bank_after`.""")

code(r"""grid = pd.read_parquet(ROOT / "data/interim/all_accounts_seed_grid_bets.parquet")
STRAT = "frozen_2025_07"
seeds = sorted(grid["seed"].unique())

def bets_for(acct, seed):
    sub = grid[(grid.account == acct) & (grid.strategy == STRAT) & (grid.seed == seed)]
    out = []
    for _, r in sub.iterrows():
        out.append(dict(date=r.fight_id.split("|")[0], fk=float(r.fk),
                        b=float(r.dec_odds) - 1.0, won=int(r.won)))
    return out

# --- validate lag=0 reproduces stored bank_after ---
ok = True
for acct, spec in ACCOUNTS.items():
    sub = grid[(grid.account == acct) & (grid.strategy == STRAT) & (grid.seed == 0)]
    final, _ = simulate_with_lag(bets_for(acct, 0), spec["kfrac"], spec["cap"], lag=0)
    stored = sub.bank_after.iloc[-1]
    match = abs(final - stored) < 1e-6
    ok &= match
    print(f"{acct}: lag0={final:14,.2f}  stored={stored:14,.2f}  match={match}")
print("\nVALIDATION:", "PASS" if ok else "FAIL")""")

code(r"""# Run the lag sweep across all seeds, per account.
rowsA = []
ratioA = []  # per-seed lag/no-lag ratios for the bug lag
for acct, spec in ACCOUNTS.items():
    for seed in seeds:
        bets = bets_for(acct, seed)
        finals = {lag: simulate_with_lag(bets, spec["kfrac"], spec["cap"], lag)[0] for lag in LAGS}
        for lag in LAGS:
            rowsA.append(dict(account=acct, seed=seed, lag=lag, final=finals[lag]))
        ratioA.append(dict(account=acct, seed=seed,
                           ratio=finals[BUG_LAG] / finals[0],
                           delta=finals[BUG_LAG] - finals[0]))

dfA = pd.DataFrame(rowsA)
ratioA = pd.DataFrame(ratioA)

# Median final per (account, lag)
medA = dfA.groupby(["account", "lag"])["final"].median().unstack("lag")
medA.columns = [f"lag{c}" for c in medA.columns]
print("Median final bankroll across 10 seeds (start $300):")
medA""")

code(r"""# Impact of the bug lag (lag=3) vs no-lag, in median terms.
summaryA = []
for acct in ACCOUNTS:
    m0 = medA.loc[acct, "lag0"]
    m3 = medA.loc[acct, f"lag{BUG_LAG}"]
    r = ratioA[ratioA.account == acct]["ratio"]
    summaryA.append(dict(account=acct,
                         median_lag0=m0, median_lag3=m3,
                         median_delta=m3 - m0,
                         median_pct=(m3 / m0 - 1) * 100,
                         per_seed_ratio_median=r.median(),
                         per_seed_ratio_min=r.min(), per_seed_ratio_max=r.max()))
summaryA = pd.DataFrame(summaryA).set_index("account")
print(f"Window A — frozen polymarket: lag={BUG_LAG} vs lag=0")
summaryA""")

code(r"""import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
for ax, acct in zip(axes, ACCOUNTS):
    g = dfA[dfA.account == acct].groupby("lag")["final"]
    med = g.median()
    lo = g.quantile(0.25); hi = g.quantile(0.75)
    ax.bar(med.index, med.values, color="#4c72b0")
    ax.errorbar(med.index, med.values, yerr=[med - lo, hi - med], fmt="none",
                ecolor="black", capsize=4)
    ax.axvline(BUG_LAG, color="red", ls="--", lw=1, alpha=.6)
    ax.set_title(f"Account {acct}")
    ax.set_xlabel("sizing lag (fights)")
    ax.set_xticks(LAGS)
axes[0].set_ylabel("median final bankroll ($)")
fig.suptitle("Window A — frozen polymarket: final bankroll vs sizing lag (median, IQR across 10 seeds)")
fig.tight_layout()
plt.show()""")

# ---------------------------------------------------------------------------
md(r"""## Window B — Kalshi (T−90min snapshot)

Per-bet records are generated by running the deployment ensemble over the
T−90min per-fight snapshot and applying the **Kalshi** Kelly math
(`fee = 0.07·p·(1−p)`, taker). The chosen side, full-Kelly fraction `fk`, and
payout `b` are all **bankroll-independent**, so they're computed once per fight
and cached to `data/interim/kalshi_window_lag_bets.parquet`. The first run does
the ensemble inference (a few minutes; NUTS skill posteriors fit once per month);
reruns load the cache.""")

code(r"""def snap_path(buf):   # buf like "T-90min", "T-15min"
    return ROOT / f"data/raw/kalshi/snapshots/historical_{buf}_perfight.parquet"

def bets_cache(buf):
    return ROOT / f"data/interim/kalshi_window_lag_bets_{buf}.parquet"

EDGE_THR = 0.03

NAME_FIX = {
    "Yadong Song": "Song Yadong", "Qileng Aori": "Aori Qileng",
    "Harris Carlston": "Carlston Harris", "Mingyang Zhang": "Zhang Mingyang",
    "Cong Wang": "Wang Cong",
}

def kalshi_fk_b(p_model, p_market):
    # Return (full_kelly, b) for a Kalshi bet, or None if no edge.
    if p_model - p_market < EDGE_THR:
        return None
    if p_market <= 0.01 or p_market >= 0.99:
        return None
    fee_per = 0.07 * p_market * (1 - p_market)
    cost_per = p_market + fee_per
    b = (1.0 - cost_per) / cost_per
    fk = (b * p_model - (1 - p_model)) / b
    if fk <= 0:
        return None
    return fk, b""")

code(r"""def generate_kalshi_bets(buf):
    # Run the ensemble over the given snapshot buffer; cache per-fight fk/b/side/won.
    from ufc_pred.inference.ensemble_predict import predict_ensemble
    from ufc_pred.inference.skill_for_upcoming import attach_skill_for_upcoming
    from ufc_pred.inference.upcoming_builder import build_upcoming_row, resolve_fighter_name
    from ufc_pred.ingest.rankings_attach import load_rankings
    from ufc_pred.paths import PROCESSED

    snap = pd.read_parquet(snap_path(buf)).sort_values(["fight_date", "close_time"]).reset_index(drop=True)
    fights = pd.read_parquet(PROCESSED / "fights.parquet")
    rankings = load_rankings()

    recs = []
    for _, f in snap.iterrows():
        date = pd.Timestamp(f["fight_date"]).strftime("%Y-%m-%d")
        pa, pb, winner = f["close_yes_price_a"], f["close_yes_price_b"], f["winner"]
        if pd.isna(pa) or pd.isna(pb) or winner not in ("A", "B"):
            continue
        na = f["canon_a"] if pd.notna(f["canon_a"]) else f["fighter_a"]
        nb = f["canon_b"] if pd.notna(f["canon_b"]) else f["fighter_b"]
        try:
            ca = resolve_fighter_name(NAME_FIX.get(na, na), fights)
            cb = resolve_fighter_name(NAME_FIX.get(nb, nb), fights)
        except Exception:
            continue
        sub = pd.concat([
            fights[fights["R_fighter"].isin([ca, cb])]["weight_class"],
            fights[fights["B_fighter"].isin([ca, cb])]["weight_class"],
        ])
        wc = sub.mode().iat[0] if not sub.empty else "Bantamweight"
        gender = "FEMALE" if str(wc).startswith("Women") else "MALE"
        try:
            row = build_upcoming_row(fighter_a=ca, fighter_b=cb,
                                     fight_date=pd.Timestamp(date), weight_class=wc,
                                     fights=fights, gender=gender, rankings=rankings)
            row = attach_skill_for_upcoming(row, fights)
            pred = predict_ensemble(row)
        except Exception as e:
            print(f"  SKIP {ca} vs {cb} ({date}): {e}")
            continue

        for acct, spec in ACCOUNTS.items():
            p_model = pred.real_mean if spec["kind"] == "real" else pred.corrupted_mean
            best = None  # (frac_key, side, fk, b)
            for side, p_side, p_mkt in [("A", p_model, pa), ("B", 1 - p_model, pb)]:
                r = kalshi_fk_b(p_side, p_mkt)
                if r is None:
                    continue
                fk, b = r
                frac = min(spec["kfrac"] * fk, spec["cap"])   # bankroll-independent argmax
                if best is None or frac > best[0]:
                    best = (frac, side, fk, b)
            if best is None:
                continue
            _, side, fk, b = best
            recs.append(dict(account=acct, date=date, fighter_a=ca, fighter_b=cb,
                             side=side, fk=fk, b=b, won=int(side == winner)))
    return pd.DataFrame(recs)


def get_kalshi_bets(buf):
    cache = bets_cache(buf)
    if cache.exists():
        df = pd.read_parquet(cache)
        print(f"[{buf}] loaded cached kalshi bets: {len(df)} rows")
    else:
        print(f"[{buf}] generating kalshi bets (ensemble inference) ...")
        df = generate_kalshi_bets(buf)
        df.to_parquet(cache, index=False)
        print(f"[{buf}] wrote {cache}  ({len(df)} rows)")
    return df

kbets = get_kalshi_bets("T-90min")   # primary window-B capture
print("cards:", kbets["date"].nunique(), "| date range:", kbets["date"].min(), "->", kbets["date"].max())
print("bets per account:\n", kbets.groupby("account").size())""")

code(r"""# Kalshi lag sweep. Single ensemble path per account (no seeds).
# min_stake=0.50 matches deployment's per-bet floor.
rowsB = []
detailB = {}
for acct, spec in ACCOUNTS.items():
    sub = kbets[kbets.account == acct].sort_values("date")
    bets = [dict(date=r.date, fk=float(r.fk), b=float(r.b), won=int(r.won)) for _, r in sub.iterrows()]
    for lag in LAGS:
        final, recs = simulate_with_lag(bets, spec["kfrac"], spec["cap"], lag, min_stake=0.50)
        rowsB.append(dict(account=acct, lag=lag, final=final,
                          n_placed=sum(x["placed"] for x in recs)))
        detailB[(acct, lag)] = recs

dfB = pd.DataFrame(rowsB)
pivB = dfB.pivot(index="account", columns="lag", values="final")
pivB.columns = [f"lag{c}" for c in pivB.columns]
print("Final bankroll (start $300):")
pivB""")

code(r"""summaryB = []
for acct in ACCOUNTS:
    f0 = pivB.loc[acct, "lag0"]
    f3 = pivB.loc[acct, f"lag{BUG_LAG}"]
    summaryB.append(dict(account=acct, lag0=f0, lag3=f3,
                         delta=f3 - f0, pct=(f3 / f0 - 1) * 100))
summaryB = pd.DataFrame(summaryB).set_index("account")
print(f"Window B — kalshi: lag={BUG_LAG} vs lag=0")
summaryB""")

code(r"""fig, ax = plt.subplots(figsize=(8, 4.5))
w = 0.15
x = np.arange(len(ACCOUNTS))
for i, lag in enumerate(LAGS):
    vals = [pivB.loc[a, f"lag{lag}"] for a in ACCOUNTS]
    ax.bar(x + (i - 2) * w, vals, w, label=f"lag {lag}",
           color="red" if lag == BUG_LAG else None, alpha=.9 if lag == BUG_LAG else .7)
ax.set_xticks(x); ax.set_xticklabels(list(ACCOUNTS))
ax.set_ylabel("final bankroll ($)"); ax.set_xlabel("account")
ax.set_title("Window B — kalshi: final bankroll by sizing lag (red = reported lag 3)")
ax.legend(ncol=5, fontsize=8)
fig.tight_layout(); plt.show()""")

# ---------------------------------------------------------------------------
md(r"""## Combined summary — lag 3 (the bug) vs no lag""")

code(r"""combined = pd.DataFrame({
    ("frozen_poly", "lag0"): summaryA["median_lag0"],
    ("frozen_poly", "lag3"): summaryA["median_lag3"],
    ("frozen_poly", "%"):   summaryA["median_pct"],
    ("kalshi", "lag0"):     summaryB["lag0"],
    ("kalshi", "lag3"):     summaryB["lag3"],
    ("kalshi", "%"):        summaryB["pct"],
})
combined.columns = pd.MultiIndex.from_tuples(combined.columns)
print("Final bankroll: no-lag (lag0) vs reported 3-fight lag (lag3)")
print("frozen_poly = median across 10 seeds; kalshi = single ensemble path\n")
combined""")

code(r"""# Plain-language readout.
print("=" * 72)
print(f"IMPACT OF THE ~{BUG_LAG}-FIGHT BALANCE-SIZING LAG (per card)")
print("=" * 72)
for acct in ACCOUNTS:
    a = summaryA.loc[acct]; b = summaryB.loc[acct]
    print(f"\nAccount {acct}:")
    print(f"  frozen-poly (median): ${a.median_lag0:,.0f} -> ${a.median_lag3:,.0f}"
          f"  ({a.median_pct:+.1f}%)  per-seed ratio {a.per_seed_ratio_min:.3f}..{a.per_seed_ratio_max:.3f}")
    print(f"  kalshi:               ${b.lag0:,.2f} -> ${b.lag3:,.2f}"
          f"  ({b.pct:+.1f}%)")
print("\nNote: lag resets each card, so this is the impact IF the ~3-fight lag")
print("recurred on every card in the window. On a single card it is far smaller.")""")

# ---------------------------------------------------------------------------
md(r"""## Variant — size the **whole fight night off the card-opening balance**

Instead of a fixed 3-fight lag, this is the case where every bet on a card is
Kelly-sized off the **balance at the start of that night** — i.e. no within-card
compounding at all (the captured balance never updates until the next card). In
the lag model this is just a lag larger than any card, so every fight uses
`card_open`. The true bankroll still books real outcomes; only sizing is frozen
to the night's opening balance. Reported as `card_open` vs `lag0` (per-fight
compounding).""")

code(r"""BIG = 10**9   # lag >= card length  ->  every fight sized off card-opening balance

# --- Window A: frozen polymarket (median + per-seed ratio across 10 seeds) ---
rA = []
for acct, spec in ACCOUNTS.items():
    for seed in seeds:
        bets = bets_for(acct, seed)
        f0 = simulate_with_lag(bets, spec["kfrac"], spec["cap"], 0)[0]
        fco = simulate_with_lag(bets, spec["kfrac"], spec["cap"], BIG)[0]
        rA.append(dict(account=acct, seed=seed, lag0=f0, card_open=fco, ratio=fco / f0))
coA = pd.DataFrame(rA)
sumA_co = coA.groupby("account").agg(
    median_lag0=("lag0", "median"), median_card_open=("card_open", "median"),
    ratio_median=("ratio", "median"), ratio_min=("ratio", "min"), ratio_max=("ratio", "max"))
sumA_co["median_pct"] = (sumA_co["median_card_open"] / sumA_co["median_lag0"] - 1) * 100
print("Window A — frozen polymarket: card-opening-balance sizing vs per-fight compounding")
sumA_co""")

code(r"""# --- Window B: kalshi (single ensemble path) ---
rB = []
for acct, spec in ACCOUNTS.items():
    sub = kbets[kbets.account == acct].sort_values("date")
    bets = [dict(date=r.date, fk=float(r.fk), b=float(r.b), won=int(r.won)) for _, r in sub.iterrows()]
    f0 = simulate_with_lag(bets, spec["kfrac"], spec["cap"], 0, min_stake=0.50)[0]
    fco = simulate_with_lag(bets, spec["kfrac"], spec["cap"], BIG, min_stake=0.50)[0]
    rB.append(dict(account=acct, lag0=f0, card_open=fco, delta=fco - f0, pct=(fco / f0 - 1) * 100))
sumB_co = pd.DataFrame(rB).set_index("account")
print("Window B — kalshi: card-opening-balance sizing vs per-fight compounding")
sumB_co""")

code(r"""# Combined readout for the card-opening-balance variant.
print("=" * 72)
print("SIZING THE WHOLE NIGHT OFF THE CARD-OPENING BALANCE  (vs per-fight compounding)")
print("=" * 72)
for acct in ACCOUNTS:
    a = sumA_co.loc[acct]; b = sumB_co.loc[acct]
    print(f"\nAccount {acct}:")
    print(f"  frozen-poly (median): ${a.median_lag0:,.0f} -> ${a.median_card_open:,.0f}"
          f"  ({a.median_pct:+.1f}%)  per-seed ratio {a.ratio_min:.3f}..{a.ratio_max:.3f}")
    print(f"  kalshi:               ${b.lag0:,.2f} -> ${b.card_open:,.2f}  ({b.pct:+.1f}%)")
print("\nThis sits at the far end of the lag sweep (lag>=card length); the lag3")
print("column in the tables above is the partial version of the same effect.")""")

code(r"""# Full picture: lag 0..4 plus the card-open extreme, side by side.
allA = medA.copy()
allA["card_open"] = sumA_co["median_card_open"]
print("Window A (frozen poly) median final by sizing rule:")
display(allA)

allB = pivB.copy()
allB["card_open"] = sumB_co["card_open"]
print("\nWindow B (kalshi) final by sizing rule:")
allB""")

# ---------------------------------------------------------------------------
md(r"""## Window B — capture-time ladder: **T−90 / T−60 / T−30 / T−15**

Re-runs the whole Kalshi window at each cached capture buffer and compares final
bankrolls under every sizing rule. All cached snapshots (`historical_T-Nmin_perfight.parquet`)
are aligned to the **same fight universe** as T−90 so only the *price* changes.

**How to read the buffer (important).** `--buffer-min N` captures the last trade `N`
minutes before the detected **fight END**, not fight start. The price stays a clean
*pre-fight* consensus only while `N` exceeds the fight's full duration (walkouts +
rounds + any eye-poke/doctor stoppages). So:

| Buffer | Stays pre-fight if fight lasts under… | Verdict |
|---|---|---|
| T−90 | ~90 min | always clean (canonical) |
| **T−60** | ~60 min | clean even for a prolonged 5-rounder (~25 min + walkouts) |
| T−30 | ~30 min | borderline — a long/delayed 5-round main event can leak in-play |
| T−15 | ~15 min | **contaminated** — mid-fight for almost any decision |

This is exactly your 45-min instinct: a 5-round fight stretched by an eye poke can
push toward ~35-40 min of total clock, so **T−60 keeps a safe margin while T−45/T−30
start risking the very contamination you're worried about.** The diagnostic at the
bottom measures how many fights actually move at each buffer.""")

code(r"""def kalshi_sweep(kdf, lags=LAGS):
    # Per-account final bankroll under each sizing rule for a kalshi bet set.
    rows = {}
    for acct, spec in ACCOUNTS.items():
        sub = kdf[kdf.account == acct].sort_values("date")
        bets = [dict(date=r.date, fk=float(r.fk), b=float(r.b), won=int(r.won)) for _, r in sub.iterrows()]
        rec = {f"lag{l}": simulate_with_lag(bets, spec["kfrac"], spec["cap"], l, min_stake=0.50)[0] for l in lags}
        rec["card_open"] = simulate_with_lag(bets, spec["kfrac"], spec["cap"], 10**9, min_stake=0.50)[0]
        rec["n_bets"] = len(bets)
        rows[acct] = rec
    return pd.DataFrame(rows).T

common_dates = set(kbets["date"])   # T-90 fight universe
BUFFERS = ["T-90min", "T-60min", "T-30min", "T-15min"]
sweeps = {"T-90min": kalshi_sweep(kbets)}
kbets_by_buf = {"T-90min": kbets}
for buf in BUFFERS:
    if buf == "T-90min":
        continue
    full = get_kalshi_bets(buf)
    aligned = full[full["date"].isin(common_dates)].copy()   # price-only comparison
    kbets_by_buf[buf] = aligned
    sweeps[buf] = kalshi_sweep(aligned)
    print(f"{buf}: {aligned['date'].nunique()} cards, {len(aligned)} bets "
          f"(dropped {full['date'].nunique() - aligned['date'].nunique()} card vs T-90 universe)")""")

code(r"""# Per-fight (lag0 = correct compounding) final bankroll across capture times.
lag0 = pd.DataFrame({buf: sweeps[buf]["lag0"] for buf in BUFFERS})
lag0pct = pd.DataFrame({buf: (sweeps[buf]["lag0"] / sweeps["T-90min"]["lag0"] - 1) * 100 for buf in BUFFERS})
print("Final bankroll by capture time — per-fight compounding (start $300):")
display(lag0)
print("\nvs T-90 (%):")
display(lag0pct.round(1))""")

code(r"""# Same ladder for the whole-night card-open sizing rule.
co = pd.DataFrame({buf: sweeps[buf]["card_open"] for buf in BUFFERS})
print("Final bankroll by capture time — whole-night card-open sizing:")
display(co)

fig, ax = plt.subplots(figsize=(9.5, 4.5))
x = np.arange(len(ACCOUNTS)); w = 0.2
colors = {"T-90min": "#4c72b0", "T-60min": "#55a868", "T-30min": "#dd8452", "T-15min": "#c44e52"}
for i, buf in enumerate(BUFFERS):
    ax.bar(x + (i - 1.5) * w, [sweeps[buf]["lag0"].loc[a] for a in ACCOUNTS], w,
           label=buf, color=colors[buf])
ax.set_xticks(x); ax.set_xticklabels(list(ACCOUNTS))
ax.set_ylabel("final bankroll ($)"); ax.set_xlabel("account")
ax.set_title("Kalshi window: final bankroll by capture time (per-fight sizing)")
ax.legend(fontsize=9); fig.tight_layout(); plt.show()""")

# ---------------------------------------------------------------------------
md(r"""### Contamination check — how many fights actually move at each buffer

For each buffer we count the fights whose price differs from T−90 by >2c, and the
mean shift **toward the eventual winner** on those fights. A clean tighter capture
should move *few* fights and shift *toward* the winner (sharper line). In-play
contamination instead moves price toward whoever led **mid-fight**, often *away*
from the final winner — the negative shifts below.""")

code(r"""def winner_prob_shift(r, suf):
    w = r["winner_90"]
    p90 = r["close_yes_price_a_90"]; pX = r[f"close_yes_price_a{suf}"]
    wp90 = p90 if w == "A" else 1 - p90
    wpX = pX if w == "A" else 1 - pX
    return wpX - wp90

s90 = pd.read_parquet(snap_path("T-90min"))
diag = []
detail_frames = {}
for buf in BUFFERS:
    if buf == "T-90min":
        continue
    sX = pd.read_parquet(snap_path(buf))
    mm = s90.merge(sX, on="event_ticker", suffixes=("_90", "_X"))
    mm["move"] = (mm["close_yes_price_a_90"] - mm["close_yes_price_a_X"]).abs()
    moved = mm[mm["move"] > 0.02].copy()
    moved["winner_prob_shift"] = moved.apply(lambda r: winner_prob_shift(r, "_X"), axis=1)
    detail_frames[buf] = moved.sort_values("move", ascending=False)
    diag.append(dict(buffer=buf, n_moved=len(moved), n_total=len(mm),
                     max_move=mm["move"].max(),
                     mean_shift_toward_winner=moved["winner_prob_shift"].mean() if len(moved) else 0.0))
diag = pd.DataFrame(diag).set_index("buffer")
print("Price moves vs the clean T-90 line:")
display(diag.round(3))

# Worst contaminated fights at the tightest buffer.
print("\nLargest moves at T-15 (negative shift = captured while mid-fight leader led):")
detail_frames["T-15min"][["event_title_90", "winner_90", "close_yes_price_a_90",
                          "close_yes_price_a_X", "move", "winner_prob_shift"]].head(12)""")

# ---------------------------------------------------------------------------
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
OUT.parent.mkdir(exist_ok=True)
nbf.write(nb, str(OUT))
print("Wrote", OUT)

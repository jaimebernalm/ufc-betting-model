"""Resumable BestFightOdds scrape -> UFC odds dataset.

Usage (from worktree root):
    PYTHONPATH=src python scripts/research/r2026_10/claude_bfo_scrape.py events
    PYTHONPATH=src python scripts/research/r2026_10/claude_bfo_scrape.py fighters [--max N]
    PYTHONPATH=src python scripts/research/r2026_10/claude_bfo_scrape.py match

Every page is cached (data/raw/bfo/...), so any step can be killed and re-run.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

from ufc_pred.ingest.identity import ALIASES, normalise
from ufc_pred.paths import ROOT
from ufc_pred.research import bfo

OUT = ROOT / "artifacts" / "r2026_10" / "claude" / "bfo"
TODAY = "2026-10-13"
EXCHANGES = {"Kalshi", "Polymarket"}
TRAIN = ROOT / "artifacts" / "shadow_candidate_2026_09_14_v2" / "training_features.parquet"


def log(*a):
    print(*a, flush=True)


def _priority(ev):
    d = ev["lastmod"]
    if "2024-01-01" <= d <= TODAY:
        return (0, d)
    if "2013-01-01" <= d < "2024-01-01":
        return (1, d)
    if d < "2013-01-01":
        return (2, d)
    return (9, d)  # future / placeholder pages


# ---------------------------------------------------------------- events
def cmd_events(args):
    evs = sorted(bfo.list_event_urls(), key=lambda e: (_priority(e)[0], [-ord(c) for c in e["lastmod"]]))
    evs = [e for e in evs if _priority(e)[0] < 9 and e["lastmod"] >= args.since]
    log(f"{len(evs)} UFC event pages to process")
    rows, metas = [], []
    for i, ev in enumerate(evs):
        try:
            page = bfo.fetch(ev["url"])
        except bfo.BlockedError as exc:
            log("BLOCKED, stopping:", exc)
            break
        except Exception as exc:  # noqa: BLE001 - keep going, record failure
            log("fail", ev["slug"], exc)
            metas.append({"event_slug": ev["slug"], "error": str(exc)})
            continue
        parsed = bfo.parse_event(page)
        m = parsed["meta"]
        metas.append(
            {
                "event_slug": ev["slug"],
                "event_title": m["event_title"],
                "event_date": m["event_date"],
                "sitemap_date": ev["lastmod"],
                "n_matchups": len({r["matchup_id"] for r in parsed["rows"]}),
                "n_book_lines": sum(r["book_id"] is not None for r in parsed["rows"]),
                "books": json.dumps(m.get("books", {})),
            }
        )
        for r in parsed["rows"]:
            rows.append(
                {
                    "event_slug": ev["slug"],
                    "event_date": m["event_date"] or ev["lastmod"],
                    "event_title": m["event_title"],
                    **r,
                }
            )
        if i % 25 == 0:
            log(i, ev["slug"], m["event_date"], len(rows))
    df = pd.DataFrame(rows)
    df["book_id"] = df["book_id"].astype("Int64")
    df["american_odds"] = df["american_odds"].astype("Int64")
    cols = [
        "event_slug",
        "event_date",
        "event_title",
        "matchup_id",
        "fighter_name",
        "fighter_slug",
        "fighter_pos",
        "opponent_name",
        "book_id",
        "book_name",
        "american_odds",
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    df[cols].to_parquet(OUT / "bfo_event_odds_long.parquet", index=False)
    pd.DataFrame(metas).to_parquet(OUT / "bfo_events_meta.parquet", index=False)
    log("wrote", len(df), "rows,", df.event_slug.nunique(), "events")


# ---------------------------------------------------------------- fighters
def cmd_fighters(args):
    ev = pd.read_parquet(OUT / "bfo_event_odds_long.parquet")
    ev = ev[ev.event_date >= args.since]
    need = set(ev.matchup_id.unique())
    by_fighter = ev.groupby("fighter_slug").matchup_id.apply(set).to_dict()
    recent = ev.groupby("fighter_slug").event_date.max().to_dict()
    covered, rows, n_new = set(), [], 0
    done = set()

    def ingest(slug, page):
        recs = bfo.parse_fighter(page)
        for r in recs:
            r["source_page"] = slug
        rows.extend(recs)
        covered.update(r["matchup_id"] for r in recs if r.get("matchup_id"))
        done.add(slug)

    # cached pages first (free)
    for slug in by_fighter:
        url = f"{bfo.BASE}/fighters/{slug}"
        if bfo.is_cached(url):
            ingest(slug, bfo.fetch(url))
    log(f"{len(need)} matchups; {len(covered & need)} covered by {len(done)} cached pages")
    # greedy set cover: newest-first among fighters with most uncovered matchups
    while True:
        best, best_gain = None, 0
        for slug, mus in by_fighter.items():
            if slug in done:
                continue
            gain = len(mus - covered)
            key = (gain, recent.get(slug, ""))
            if gain and (best is None or key > best_gain):
                best, best_gain = slug, key
        if best is None or (args.max and n_new >= args.max):
            break
        try:
            page = bfo.fetch(f"{bfo.BASE}/fighters/{best}")
            n_new += 1
        except bfo.BlockedError as exc:
            log("BLOCKED, stopping:", exc)
            break
        except Exception as exc:  # noqa: BLE001
            log("fail", best, exc)
            done.add(best)
            continue
        ingest(best, page)
        if n_new % 50 == 0:
            log(f"fetched {n_new}; covered {len(covered & need)}/{len(need)}")
            _write_fighters(rows)
    _write_fighters(rows)
    log(f"done: fetched {n_new} new pages; covered {len(covered & need)}/{len(need)}")


def _write_fighters(rows):
    df = pd.DataFrame(rows)
    if df.empty:
        return
    df = df.drop_duplicates(["matchup_id", "fighter_slug"])
    for c in ("open_odds", "close_low", "close_high", "matchup_id", "fighter_pos"):
        df[c] = df[c].astype("Int64")
    df["event_slug_or_title"] = df["event_slug"].fillna(df["event_title"])
    df["event_date_if_available"] = df["event_date"]
    cols = [
        "fighter_slug",
        "fighter_name",
        "opponent_name",
        "opponent_slug",
        "matchup_id",
        "event_slug_or_title",
        "event_title",
        "event_date_if_available",
        "open_odds",
        "close_low",
        "close_high",
        "source_page",
    ]
    df[cols].to_parquet(OUT / "bfo_fighter_open.parquet", index=False)


# ---------------------------------------------------------------- matching
_ALIAS_N = {normalise(k): normalise(v) for k, v in ALIASES.items()}


def nkey(name):
    n = normalise(name)
    parts = n.split()
    if len(parts) == 2 and parts[0] == parts[1]:  # BFO: "Sumudaerji Sumudaerji"
        n = parts[0]
    return _ALIAS_N.get(n, n)


def nkey_loose(name):
    """Space-insensitive key (e.g. 'Loneer' vs "Lone'er" -> both 'loneer')."""
    parts = nkey(name).split()
    while len(parts) > 2 and parts[-1] in {"jr", "sr", "ii", "iii", "iv"}:
        parts = parts[:-1]
    return "".join(parts)


def initial_key(name):
    parts = nkey(name).split()
    return f"{parts[0][0]} {parts[-1]}" if len(parts) >= 2 else None


def _fights_table():
    t = pd.read_parquet(TRAIN)[["date", "R_fighter", "B_fighter", "R_odds", "B_odds"]].copy()
    t["source"] = "training_features"
    t["date"] = pd.to_datetime(t["date"])
    extra = []
    try:
        f = pd.read_parquet(
            ROOT.parent / "UFC pred" / "data" / "processed" / "fights.parquet",
            columns=["date", "R_fighter", "B_fighter", "R_odds", "B_odds"],
        )
        f["date"] = pd.to_datetime(f["date"])
        f["source"] = "fights.parquet"
        extra.append(f)
    except Exception as exc:  # noqa: BLE001
        log("fights.parquet unavailable:", exc)
    try:
        h = pd.read_parquet(
            ROOT.parent
            / "UFC pred"
            / "data"
            / "processed"
            / "shadow_candidate_2026_09_14_v2"
            / "history.parquet"
        )
        if {"date", "R_fighter", "B_fighter"} <= set(h.columns):
            h = h[["date", "R_fighter", "B_fighter"]].copy()
            h["date"] = pd.to_datetime(h["date"])
            h = h[h.date > t.date.max()]
            h["source"] = "shadow_history"
            extra.append(h)
    except Exception as exc:  # noqa: BLE001
        log("shadow history unavailable:", exc)
    allf = pd.concat([t, *extra], ignore_index=True)
    allf["k"] = [frozenset((nkey(a), nkey(b))) for a, b in zip(allf.R_fighter, allf.B_fighter)]
    allf = allf.drop_duplicates(["date", "k"], keep="first").drop(columns="k")
    return allf.reset_index(drop=True)


def _bfo_matchups(ev, fo):
    """One record per BFO matchup with both fighters and per-book closing odds by pos."""
    ev = ev.copy()
    ev["event_date"] = pd.to_datetime(ev["event_date"])
    base = ev.drop_duplicates(["matchup_id", "fighter_pos"]).pivot(
        index="matchup_id", columns="fighter_pos", values=["fighter_name", "fighter_slug"]
    )
    base.columns = [f"{a}_{b}" for a, b in base.columns]
    meta = ev.drop_duplicates("matchup_id").set_index("matchup_id")[
        ["event_slug", "event_date", "event_title"]
    ]
    mu = meta.join(base).dropna(subset=["fighter_name_1", "fighter_name_2"]).reset_index()
    lines = ev.dropna(subset=["book_id"])
    odds = {}
    for (m, pos, book), v in (
        lines.groupby(["matchup_id", "fighter_pos", "book_name"]).american_odds.first().items()
    ):
        odds.setdefault(m, {})[(pos, book)] = float(v)
    mu["odds"] = mu.matchup_id.map(lambda m: odds.get(m, {}))
    fo = fo.dropna(subset=["matchup_id"])
    op = {(int(r.matchup_id), r.fighter_slug): r for r in fo.itertuples()}
    mu["open_1"] = [
        getattr(op.get((m, s)), "open_odds", np.nan) for m, s in zip(mu.matchup_id, mu.fighter_slug_1)
    ]
    mu["open_2"] = [
        getattr(op.get((m, s)), "open_odds", np.nan) for m, s in zip(mu.matchup_id, mu.fighter_slug_2)
    ]
    for side in ("low", "high"):
        mu[f"close_{side}_1"] = [
            getattr(op.get((m, s)), f"close_{side}", np.nan) for m, s in zip(mu.matchup_id, mu.fighter_slug_1)
        ]
        mu[f"close_{side}_2"] = [
            getattr(op.get((m, s)), f"close_{side}", np.nan) for m, s in zip(mu.matchup_id, mu.fighter_slug_2)
        ]
    return mu


def _to_float(x):
    try:
        return float(x) if x is not None and x is not pd.NA else np.nan
    except (TypeError, ValueError):
        return np.nan


def cmd_match(args):
    ev = pd.read_parquet(OUT / "bfo_event_odds_long.parquet")
    fo_path = OUT / "bfo_fighter_open.parquet"
    fo = (
        pd.read_parquet(fo_path)
        if fo_path.exists()
        else pd.DataFrame(columns=["matchup_id", "fighter_slug", "open_odds", "close_low", "close_high"])
    )
    mu = _bfo_matchups(ev, fo)
    books = sorted(ev.book_name.dropna().unique())
    fights = _fights_table()

    # index BFO matchups by (date, key) for each key type
    idx = {"exact": defaultdict(list), "loose": defaultdict(list), "initial": defaultdict(list)}
    for i, r in mu.iterrows():
        a, b = r.fighter_name_1, r.fighter_name_2
        keys = {
            "exact": (nkey(a), nkey(b)),
            "loose": (nkey_loose(a), nkey_loose(b)),
            "initial": (initial_key(a), initial_key(b)),
        }
        for kind, (ka, kb) in keys.items():
            if ka and kb:
                idx[kind][(r.event_date.date(), frozenset((ka, kb)))].append(i)

    used, out = set(), []
    for f in fights.itertuples():
        hit, method = None, None
        for kind, fn in (("exact", nkey), ("loose", nkey_loose), ("initial", initial_key)):
            kr, kb = fn(f.R_fighter), fn(f.B_fighter)
            if not kr or not kb:
                continue
            for off in (0, 1, -1):
                d = (f.date + pd.Timedelta(days=off)).date()
                cands = [c for c in idx[kind].get((d, frozenset((kr, kb))), []) if c not in used]
                if len(cands) > 1:  # duplicate pages: prefer the one with most lines
                    cands.sort(key=lambda c: -len(mu.at[c, "odds"]))
                if cands:
                    hit = cands[0]
                    method = kind if off == 0 else f"{kind}_date{off:+d}"
                    break
            if hit is not None:
                break
        rec = {
            "date": f.date,
            "R_fighter": f.R_fighter,
            "B_fighter": f.B_fighter,
            "fights_source": f.source,
            "kaggle_R_odds": _to_float(f.R_odds),
            "kaggle_B_odds": _to_float(f.B_odds),
        }
        if hit is None:
            rec["match_method"] = "unmatched"
            out.append(rec)
            continue
        used.add(hit)
        m = mu.loc[hit]
        r_is_1 = (
            (nkey_loose(f.R_fighter) == nkey_loose(m.fighter_name_1))
            if method.startswith(("exact", "loose"))
            else (initial_key(f.R_fighter) == initial_key(m.fighter_name_1))
        )
        pr, pb = (1, 2) if r_is_1 else (2, 1)
        rec.update(
            {
                "match_method": method,
                "bfo_event_slug": m.event_slug,
                "bfo_event_date": m.event_date,
                "bfo_matchup_id": int(m.matchup_id),
                "bfo_R_name": m[f"fighter_name_{pr}"],
                "bfo_B_name": m[f"fighter_name_{pb}"],
                "R_open": _to_float(m[f"open_{pr}"]),
                "B_open": _to_float(m[f"open_{pb}"]),
                "R_close_low": _to_float(m[f"close_low_{pr}"]),
                "R_close_high": _to_float(m[f"close_high_{pr}"]),
                "B_close_low": _to_float(m[f"close_low_{pb}"]),
                "B_close_high": _to_float(m[f"close_high_{pb}"]),
            }
        )
        for bk in books:
            col = bk.replace(" ", "")
            rec[f"R_close_{col}"] = m.odds.get((pr, bk), np.nan)
            rec[f"B_close_{col}"] = m.odds.get((pb, bk), np.nan)
        out.append(rec)
    df = pd.DataFrame(out)

    # summaries
    book_cols = [b.replace(" ", "") for b in books]
    sb = [b for b in book_cols if b not in EXCHANGES]
    qs = (
        np.column_stack(
            [
                [
                    bfo.devig_pair(r, b)
                    for r, b in zip(df.get(f"R_close_{c}", np.nan), df.get(f"B_close_{c}", np.nan))
                ]
                for c in sb
            ]
        )
        if sb
        else np.full((len(df), 1), np.nan)
    )
    df["n_books"] = np.isfinite(qs).sum(1)
    with np.errstate(all="ignore"):
        df["q_close_consensus_R"] = np.where(
            df.n_books > 0, np.nanmean(np.where(np.isfinite(qs), qs, np.nan), 1), np.nan
        )
        df["q_close_disp_R"] = np.where(df.n_books > 1, np.nanstd(qs, 1, ddof=1), np.nan)
    df["q_open_R"] = [
        bfo.devig_pair(a, b) for a, b in zip(df.get("R_open", np.nan), df.get("B_open", np.nan))
    ]
    # closing-range midpoint (fighter pages): available for all eras, sportsbooks only
    mid = lambda lo, hi: (bfo.american_to_prob(lo) + bfo.american_to_prob(hi)) / 2  # noqa: E731
    pr_ = np.array([mid(a, b) for a, b in zip(df.get("R_close_low", np.nan), df.get("R_close_high", np.nan))])
    pb_ = np.array([mid(a, b) for a, b in zip(df.get("B_close_low", np.nan), df.get("B_close_high", np.nan))])
    df["q_close_range_mid_R"] = pr_ / (pr_ + pb_)
    for ex in EXCHANGES:
        df[f"q_{ex.lower()}_close_R"] = [
            bfo.devig_pair(a, b)
            for a, b in zip(df.get(f"R_close_{ex}", np.nan), df.get(f"B_close_{ex}", np.nan))
        ]
    df["q_kaggle_close_R"] = [bfo.devig_pair(a, b) for a, b in zip(df.kaggle_R_odds, df.kaggle_B_odds)]
    df.to_parquet(OUT / "bfo_matched.parquet", index=False)
    report(df, mu, fights)


def report(df, mu, fights):
    df = df.copy()
    df["year"] = df.date.dt.year
    m = df.match_method != "unmatched"
    rep = {
        "n_fights": int(len(df)),
        "n_matched": int(m.sum()),
        "match_method_counts": df.match_method.value_counts().to_dict(),
    }
    by = df.groupby("year")
    rep["by_year"] = {}
    for y, g in by:
        gm = g[g.match_method != "unmatched"]
        row = {
            "fights": int(len(g)),
            "matched": int(len(gm)),
            "match_rate": round(len(gm) / len(g), 4),
            "pct_open": round(float(gm.q_open_R.notna().mean()), 4) if len(gm) else None,
            "pct_book_close": round(float((gm.n_books > 0).mean()), 4) if len(gm) else None,
            "pct_range_close": round(float(gm.q_close_range_mid_R.notna().mean()), 4) if len(gm) else None,
            "pct_kalshi": round(float(gm.q_kalshi_close_R.notna().mean()), 4) if len(gm) else None,
            "pct_polymarket": round(float(gm.q_polymarket_close_R.notna().mean()), 4) if len(gm) else None,
        }
        for col in ("q_close_consensus_R", "q_close_range_mid_R"):
            both = gm[[col, "q_kaggle_close_R"]].dropna()
            if len(both) >= 10:
                row[f"{col}_vs_kaggle"] = {
                    "n": int(len(both)),
                    "corr": round(float(both.corr().iloc[0, 1]), 4),
                    "mae": round(float((both[col] - both.q_kaggle_close_R).abs().mean()), 4),
                }
        for ex in ("kalshi", "polymarket"):
            both = gm[[f"q_{ex}_close_R", "q_close_consensus_R"]].dropna()
            if len(both) >= 10:
                row[f"{ex}_vs_consensus"] = {
                    "n": int(len(both)),
                    "corr": round(float(both.corr().iloc[0, 1]), 4),
                    "mae": round(float((both.iloc[:, 0] - both.iloc[:, 1]).abs().mean()), 4),
                    "share_absdiff_gt_0.25": round(
                        float(((both.iloc[:, 0] - both.iloc[:, 1]).abs() > 0.25).mean()), 4
                    ),
                }
        rep["by_year"][int(y)] = row
    um = df[~m]
    rep["unmatched_examples"] = (
        um.sort_values("date", ascending=False)
        .head(60)[["date", "R_fighter", "B_fighter"]]
        .astype(str)
        .to_dict("records")
    )
    matched_ids = set(df.bfo_matchup_id.dropna().astype(int)) if "bfo_matchup_id" in df else set()
    orphan = mu[
        ~mu.matchup_id.isin(matched_ids) & (mu.event_date >= "2010-03-01") & (mu.event_date <= "2026-09-13")
    ]
    rep["bfo_matchups_unmatched_count"] = int(len(orphan))
    rep["bfo_matchups_unmatched_examples"] = (
        orphan.sort_values("event_date", ascending=False)
        .head(40)[["event_slug", "event_date", "fighter_name_1", "fighter_name_2"]]
        .astype(str)
        .to_dict("records")
    )
    (OUT / "match_report.json").write_text(json.dumps(rep, indent=1, default=str))
    log(
        json.dumps(
            {
                k: v
                for k, v in rep.items()
                if k not in ("unmatched_examples", "bfo_matchups_unmatched_examples")
            },
            indent=1,
            default=str,
        )
    )


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["events", "fighters", "match"])
    p.add_argument("--since", default="2000-01-01")
    p.add_argument("--max", type=int, default=0, help="max new fighter pages to fetch")
    args = p.parse_args(argv)
    {"events": cmd_events, "fighters": cmd_fighters, "match": cmd_match}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())

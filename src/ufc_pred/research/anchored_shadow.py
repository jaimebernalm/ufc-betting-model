"""Paper-only prospective shadow for the market-anchored candidate F1 (R6_mkt).

It never places orders, never sends notifications and never touches the live
watchdog, its configs or its bundle. It reads the feature rows and Kalshi
snapshots that the existing v2 shadow runner already archives, fetches the
BestFightOdds event page *at scoring time* (sportsbook moneylines + method
props), and logs what F1 would have done.

Temporal contract: a capture is scored only if the BFO page is fetched within
`--max-lag-min` minutes AFTER the Kalshi capture (measured after the network
fetch) and before the scheduled start; otherwise it is logged as an abstention
and never scored with later prices. Debutants are skipped. Records are signals,
not executions (`executed: false`).

  python -m ufc_pred.research.anchored_shadow fit --before 2026-10-13
  python -m ufc_pred.research.anchored_shadow score --captures <dir> [--max-lag-min 10]
  python -m ufc_pred.research.anchored_shadow report
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ufc_pred.paths import ROOT
from ufc_pred.research import anchored as A
from ufc_pred.research import bfo
from ufc_pred.research.harness import KEYS, SPORTSBOOK_PRIORITY, american_to_prob, load_base, logit

BUNDLE = ROOT / "artifacts/r2026_10/anchored_shadow_bundle"
OUTDIR = ROOT / "data/processed/anchored_shadow_2026_10_13"
MKT_EXTRA = ROOT / "artifacts/r2026_10/claude/mkt_extra.parquet"
# read-only: the live shadow runner's history, used only to recognise debutants
LIVE_HISTORY = Path("/Users/jaimebernal/Desktop/pred_project/UFC pred/data/processed/shadow_candidate_2026_09_14_v2/history.parquet")
POLICY = dict(ev_thr=0.02, kelly_frac=0.25, cap_per_fight=0.02, cap_per_card=0.15, edge_ceiling=0.10, fee_coeff=0.07)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return " ".join(s.lower().replace("-", " ").replace(".", "").replace("'", "").split())


def _model_fn():
    # identical recipe to the frozen F1 (manifest.json -> frozen_finalists.F1)
    return A.make_resid_fn(logit_q_feature=True, flb_baseline=True)


# ------------------------------------------------------------------ fit


def fit_bundle(before: str, seeds: int = 10) -> Path:
    base = load_base(fill_bfo_from="2024-01-01")
    base = base.merge(pd.read_parquet(MKT_EXTRA), on=KEYS, how="left", validate="one_to_one")
    cutoff = pd.Timestamp(before)
    train = base[base.date < cutoff].reset_index(drop=True)
    fn = _model_fn()
    if (BUNDLE / "manifest.json").exists():
        raise FileExistsError(f"{BUNDLE} already holds a bundle; never overwrite a shadow bundle in place")
    BUNDLE.mkdir(parents=True, exist_ok=True)
    a_used, cat_used = None, None
    for s in range(seeds):
        m, a, cat = fn.fit(train, cutoff - pd.Timedelta(days=1), s)
        joblib.dump(m, BUNDLE / f"seed{s}.joblib")
        a_used, cat_used = a, cat
    code = [ROOT / "src/ufc_pred/research/anchored.py", ROOT / "src/ufc_pred/research/harness.py",
            ROOT / "src/ufc_pred/research/anchored_shadow.py", ROOT / "src/ufc_pred/research/bfo.py"]
    manifest = {
        "candidate": "F1 = R6_mkt (frozen 2026-10-13, see artifacts/r2026_10/manifest.json)",
        "status": "PAPER SHADOW ONLY - not confirmed on holdout (G1 upper +0.0002); no real money",
        "fit_before": before, "n_train": int(train.has_odds.sum()), "seeds": seeds,
        "flb_slope_a": a_used, "cat_features": cat_used, "policy": POLICY,
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "code_sha256": {str(p.relative_to(ROOT)): _sha(p) for p in code},
        "data_sha256": {str(MKT_EXTRA.relative_to(ROOT)): _sha(MKT_EXTRA)},
        "model_sha256": {f"seed{s}.joblib": _sha(BUNDLE / f"seed{s}.joblib") for s in range(seeds)},
    }
    (BUNDLE / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return BUNDLE


def load_bundle(check_code: bool = True):
    """Load and verify: model files, frozen data, policy and (by default) code must match the manifest."""
    manifest = json.loads((BUNDLE / "manifest.json").read_text())
    for name, digest in manifest["model_sha256"].items():
        if _sha(BUNDLE / name) != digest:
            raise RuntimeError(f"bundle file changed: {name}")
    for rel, digest in manifest["data_sha256"].items():
        if _sha(ROOT / rel) != digest:
            raise RuntimeError(f"frozen data changed: {rel}")
    if manifest["policy"] != POLICY:
        raise RuntimeError("policy differs from the bundle manifest")
    if check_code:
        for rel, digest in manifest["code_sha256"].items():
            if rel.endswith("anchored_shadow.py"):
                continue  # this file is the runner; its hash is recorded per bundle for traceability
            if _sha(ROOT / rel) != digest:
                raise RuntimeError(f"model code changed since the bundle was fit: {rel}")
    models = [joblib.load(BUNDLE / f"seed{s}.joblib") for s in range(manifest["seeds"])]
    return manifest, models


# ------------------------------------------------------------------ BFO live parsing

# data-li of a prop cell = [book, yes(1), matchup_id, prop_type, fighter_pos]; verified on cached pages
_PROP_TYPES = {8: "ko", 9: "sub", 11: "dec"}  # "<X> wins by TKO/KO" / "by submission" / "by decision"


def parse_method_props(page: str) -> dict[int, dict[int, dict[str, float]]]:
    """{matchup_id: {fighter_pos: {'ko'|'sub'|'dec': implied prob of the median decimal price}}}.

    Sportsbooks only (header names in SPORTSBOOK_PRIORITY), never Kalshi/Polymarket.
    The cross-book median differs from the single-source Kaggle props used in training
    (documented source difference in the prospective protocol).
    """
    start = page.find('<table class="odds-table">')
    if start < 0:
        return {}
    table = page[start: page.find("</table>", start)]
    thead = table.partition("<tbody")[0]
    books = {int(b): bfo._text(inner.split("<br", 1)[0])
             for b, inner in re.findall(r'<th scope="col" data-b="(\d+)">(.*?)</th>', thead, re.S)}
    allowed = {b for b, name in books.items() if name in SPORTSBOOK_PRIORITY}
    out: dict[int, dict[int, dict[str, list]]] = {}
    for book, yes, mu, ptype, pos, odds in re.findall(
        r'data-li="\[(\d+),(\d),(\d+),(\d+),(\d)\]"><span[^>]*>([^<]*)</span>', table
    ):
        kind = _PROP_TYPES.get(int(ptype))
        if kind is None or yes != "1" or int(book) not in allowed:
            continue
        v = bfo._american(odds)
        if v is None or abs(v) < 100:
            continue
        dec = 1 + (100 / -v if v < 0 else v / 100)
        out.setdefault(int(mu), {}).setdefault(int(pos), {}).setdefault(kind, []).append(dec)
    return {mu: {pos: {k: 1 / float(np.median(v)) for k, v in kinds.items()} for pos, kinds in d.items()}
            for mu, d in out.items()}


def find_bfo_event(card_date: pd.Timestamp, names: tuple[str, str]):
    """Fetch (fresh) candidate UFC event pages near the card date and return (slug, page, rows, matchup)."""
    events = bfo.list_event_urls(refresh=True)
    cands = []
    for e in events:
        try:
            lm = pd.Timestamp(e["lastmod"][:10])
        except Exception:
            continue
        if abs((lm - card_date).days) <= 10:
            cands.append(e)
    want = {_norm(names[0]), _norm(names[1])}
    for e in sorted(cands, key=lambda e: e["lastmod"], reverse=True):
        page = bfo.fetch(e["url"], refresh=True)
        parsed = bfo.parse_event(page)
        rows = pd.DataFrame(parsed["rows"])
        if rows.empty:
            continue
        for mu, g in rows.groupby("matchup_id"):
            if {_norm(n) for n in g.fighter_name.unique()} == want:
                return e["slug"], page, g, int(mu)
    return None, None, None, None


def book_pair(rows: pd.DataFrame, red: str):
    """First sportsbook in the fixed priority with both sides quoted -> (book, R_odds, B_odds)."""
    for book in SPORTSBOOK_PRIORITY:
        g = rows[rows.book_name.eq(book) & rows.american_odds.notna()]
        r = g[g.fighter_name.map(_norm) == _norm(red)].american_odds
        b = g[g.fighter_name.map(_norm) != _norm(red)].american_odds
        if len(r) == 1 and len(b) == 1 and abs(r.iloc[0]) >= 100 and abs(b.iloc[0]) >= 100:
            return book, float(r.iloc[0]), float(b.iloc[0])
    return None, np.nan, np.nan


def props_features(props: dict, red_pos: int, q: float) -> dict:
    """Same definitions as mkt_extra: logit(props-implied P(red)) - logit(q) and P(inside distance)."""
    blue_pos = 3 - red_pos
    if not all(k in props.get(s, {}) for s in (red_pos, blue_pos) for k in ("ko", "sub", "dec")):
        return {"mkt_props_dif": np.nan, "mkt_finish_prob": np.nan}
    r = sum(props[red_pos].values())
    b = sum(props[blue_pos].values())
    fin = props[red_pos]["ko"] + props[red_pos]["sub"] + props[blue_pos]["ko"] + props[blue_pos]["sub"]
    return {"mkt_props_dif": float(logit(r / (r + b)) - logit(q)), "mkt_finish_prob": float(fin / (r + b))}


# ------------------------------------------------------------------ scoring


def kalshi_asks_for(cap: dict, red: str) -> dict:
    """Map the snapshot asks to the feature row's Red fighter BY NAME.

    Snapshot asks follow Kalshi's ticker_a/ticker_b; ticker_a is the UFC fighter_b when
    kalshi_swap_vs_ufc is True. The archived feature row itself follows Kalshi order, so
    never assume either orientation -- match names.
    """
    ks, fight = cap["kalshi_snapshot"], cap["fight"]
    swap = bool(fight.get("kalshi_swap_vs_ufc"))
    tick_a = fight["fighter_b_ufc"] if swap else fight["fighter_a_ufc"]
    tick_b = fight["fighter_a_ufc"] if swap else fight["fighter_b_ufc"]
    if _norm(red) == _norm(tick_a):
        return {"R_ask": ks["a_ask"], "B_ask": ks["b_ask"]}
    if _norm(red) == _norm(tick_b):
        return {"R_ask": ks["b_ask"], "B_ask": ks["a_ask"]}
    raise ValueError(f"cannot orient Kalshi asks: {red!r} not in {tick_a!r}/{tick_b!r}")


def _valid_prob(x) -> bool:
    return x is not None and np.isfinite(x) and 0.0 < float(x) < 1.0


def decide(p_red: float, q: float, asks: dict) -> dict:
    if not (_valid_prob(p_red) and _valid_prob(q)):
        return {"signal": "invalid_probability"}
    fee = POLICY["fee_coeff"]
    sides = []
    for side, p, ask, mkt in [("R", p_red, asks["R_ask"], q), ("B", 1 - p_red, asks["B_ask"], 1 - q)]:
        if not _valid_prob(ask):
            continue
        cost = ask + fee * ask * (1 - ask)
        ev = p / cost - 1
        b = 1 / cost - 1
        kelly = max(0.0, (p * b - (1 - p)) / b)
        sides.append(dict(side=side, p=p, ask=ask, cost=cost, ev=ev, kelly=kelly, edge_vs_book=p - mkt))
    if not sides:
        return {"signal": "no_price"}
    best = max(sides, key=lambda s: s["ev"])
    if best["ev"] <= POLICY["ev_thr"]:
        return {"signal": "none", "best": best}
    if best["edge_vs_book"] > POLICY["edge_ceiling"]:
        return {"signal": "rejected_implausible_edge", "best": best}
    # per-fight Kelly fraction before the per-card cap (applied in report(), across the card)
    raw_frac = min(POLICY["kelly_frac"] * best["kelly"], POLICY["cap_per_fight"])
    return {"signal": "would_bet", "best": best, "raw_kelly_frac_unexecuted": raw_frac}


def _prior_ufc_bouts(name: str, before: pd.Timestamp) -> int:
    """UFC appearances strictly before `before`: frozen verified history + the live shadow history (read-only)."""
    frames = [pd.read_parquet(ROOT / "artifacts/shadow_candidate_2026_09_14_v2/verified_history.parquet")[KEYS]]
    if LIVE_HISTORY.exists():
        frames.append(pd.read_parquet(LIVE_HISTORY)[KEYS])
    h = pd.concat(frames, ignore_index=True).drop_duplicates()
    h = h[h.date < before]
    n = _norm(name)
    return int((h.R_fighter.map(_norm).eq(n) | h.B_fighter.map(_norm).eq(n)).sum())


def score_capture(path: Path, manifest: dict, models: list, max_lag_min: float) -> dict:
    """Score one archived v2 capture under the paper contract.

    Abstains (never scores) when: the capture is in the future or at/after the scheduled
    start, the BFO page could not be observed within `max_lag_min` of the Kalshi capture
    (lag measured AFTER the network fetch), either fighter is a UFC debutant, or prices
    are missing. Output is a signal log, not an execution record.
    """
    cap = json.loads(path.read_text())
    captured = pd.Timestamp(cap["captured_at_utc"])
    sched = pd.Timestamp(cap["fight_date_utc"]) if cap.get("fight_date_utc") else None
    rec = {"capture_file": path.name, "captured_at_utc": str(captured),
           "bundle_fit_before": manifest["fit_before"], "paper_only": True, "executed": False}
    now = pd.Timestamp(dt.datetime.now(dt.UTC))
    if captured > now:
        return {**rec, "status": "abstain_future_timestamp"}
    if sched is not None and captured >= sched:
        return {**rec, "status": "abstain_capture_at_or_after_scheduled_start"}
    if (now - captured).total_seconds() / 60 > max_lag_min:  # cheap pre-check; re-checked after fetch
        return {**rec, "status": "abstain_stale_capture"}
    fwd = (cap.get("features") or {}).get("forward") or []
    if not fwd or not cap.get("kalshi_snapshot"):
        return {**rec, "status": "abstain_no_features_or_snapshot"}
    row = pd.DataFrame([fwd[0]])
    row["date"] = pd.to_datetime(row.date)
    red, blue = row.R_fighter.iloc[0], row.B_fighter.iloc[0]
    card_date = row.date.iloc[0]
    if min(_prior_ufc_bouts(red, card_date), _prior_ufc_bouts(blue, card_date)) == 0:
        return {**rec, "status": "abstain_debut", "R_fighter": red, "B_fighter": blue}
    slug, page, rows, mu = find_bfo_event(card_date, (red, blue))
    fetched = pd.Timestamp(dt.datetime.now(dt.UTC))
    if page is not None:  # immutable copy + hash: the shared BFO cache is overwritten by refresh fetches
        import gzip

        digest = hashlib.sha256(page.encode()).hexdigest()
        ev_dir = OUTDIR / "bfo_html"
        ev_dir.mkdir(parents=True, exist_ok=True)
        (ev_dir / f"{path.stem}.{digest[:16]}.html.gz").write_bytes(gzip.compress(page.encode()))
        rec["bfo_html_sha256"] = digest
    lag = (fetched - captured).total_seconds() / 60
    rec.update(bfo_fetched_utc=str(fetched), lag_bfo_vs_kalshi_min=lag, bfo_event=slug, bfo_matchup=mu)
    if lag > max_lag_min:
        return {**rec, "status": "abstain_markets_not_contemporaneous"}
    if sched is not None and fetched >= sched:
        return {**rec, "status": "abstain_bfo_fetched_after_scheduled_start"}
    if rows is None:
        return {**rec, "status": "abstain_no_bfo_match"}
    book, r_odds, b_odds = book_pair(rows, red)
    if book is None:
        return {**rec, "status": "abstain_no_book_pair"}
    pr, pb = american_to_prob(r_odds), american_to_prob(b_odds)
    q = float(pr / (pr + pb))
    red_pos = int(rows[rows.fighter_name.map(_norm) == _norm(red)].fighter_pos.iloc[0])
    feats = props_features(parse_method_props(page).get(mu, {}), red_pos, q)
    row = row.assign(R_odds=r_odds, B_odds=b_odds, q=q, has_odds=True, **feats)
    row = row.drop(columns=[c for c in ["empty_arena"] if c in row])
    fn = _model_fn()
    preds = [fn.predict(m, manifest["flb_slope_a"], manifest["cat_features"], row)[0] for m in models]
    p = float(np.mean(preds))
    asks = kalshi_asks_for(cap, red)
    return {**rec, "status": "scored", "card_date": str(card_date.date()), "R_fighter": red, "B_fighter": blue,
            "book": book, "R_odds": r_odds, "B_odds": b_odds, "q_book": q, **feats,
            "p_red": p, "p_red_seed_std": float(np.std(preds)), **asks, **decide(p, q, asks)}


def score_dir(captures: Path, max_lag_min: float) -> list[dict]:
    manifest, models = load_bundle()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    done = []
    for path in sorted(captures.glob("*.json")):
        out = OUTDIR / f"{path.stem}.anchored.json"
        if out.exists():
            continue
        try:
            rec = score_capture(path, manifest, models, max_lag_min)
        except Exception as exc:  # keep evidence of every failure, never retry silently with later prices
            rec = {"capture_file": path.name, "status": "error", "error": repr(exc)}
        out.write_text(json.dumps(rec, indent=1, default=str))
        done.append(rec)
    return done


def report() -> pd.DataFrame:
    """All signal records; for would_bet rows, scale per-fight fractions so each card sums <= cap_per_card."""
    recs = [json.loads(p.read_text()) for p in sorted(OUTDIR.glob("*.anchored.json"))]
    d = pd.DataFrame(recs)
    if d.empty or "signal" not in d:
        return d
    d["kelly_frac_after_card_cap"] = 0.0
    wb = d.signal.eq("would_bet")
    for _, g in d[wb].groupby("card_date"):
        tot = g.raw_kelly_frac_unexecuted.sum()
        scale = min(1.0, POLICY["cap_per_card"] / tot) if tot > 0 else 0.0
        d.loc[g.index, "kelly_frac_after_card_cap"] = g.raw_kelly_frac_unexecuted * scale
    return d


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--before", required=True)
    f.add_argument("--seeds", type=int, default=10)
    s = sub.add_parser("score")
    s.add_argument("--captures", required=True, type=Path)
    s.add_argument("--max-lag-min", type=float, default=10.0)
    sub.add_parser("report")
    args = ap.parse_args()
    if args.cmd == "fit":
        print(fit_bundle(args.before, args.seeds))
    elif args.cmd == "score":
        for r in score_dir(args.captures, args.max_lag_min):
            print(r.get("capture_file"), r.get("status"), r.get("signal", ""))
    else:
        d = report()
        print(d.status.value_counts() if not d.empty else "no records")


if __name__ == "__main__":
    main()

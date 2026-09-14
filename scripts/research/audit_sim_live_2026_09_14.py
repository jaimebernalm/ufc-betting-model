"""Offline, non-mutating audit of the simulation/live boundary.

Run from the repo: .conda/bin/python scripts/research/audit_sim_live_2026_09_14.py
Optional: --fit-ablation trains two diagnostic seed-0 models in memory only.
Only writes new files under artifacts/audit_2026_09_14; no network, orders,
notifications, production cache updates, model replacement or bankroll writes.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from ufc_pred.backtest.bet_eval import _effective_decimal, evaluate_bets
from ufc_pred.backtest.strategy_grid import ModelBundle, predict
from ufc_pred.backtest.universe import add_prior_fight_counts
from ufc_pred.features.joins import flip_signed_columns, join_skill_v3
from ufc_pred.features.skill_v3 import build_index
from ufc_pred.features.static_v1 import _swap_red_blue, prepare
from ufc_pred.inference import sizing
from ufc_pred.inference.skill_for_upcoming import _prior_hash
from ufc_pred.ingest.rankings_attach import WEIGHT_CLASSES, _rank_at
from ufc_pred.models._harness import build_matrices
from ufc_pred.models.baseline_v3 import SPEC
from ufc_pred.ops import bankroll
from ufc_pred.paths import ROOT

OUT = ROOT / "artifacts/audit_2026_09_14"
KEYS = ["date", "R_fighter", "B_fighter"]


def sharpen(p, t=1.25):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return 1 / (1 + np.exp(-t * np.log(p / (1 - p))))


def ensembles(frame):
    both = pd.concat([frame, _swap_red_blue(frame)], ignore_index=True)
    n = len(frame)
    result = {}
    for kind in ("real", "corrupted"):
        ps = []
        for seed in range(10):
            d = joblib.load(ROOT / f"artifacts/models/v3_{kind}_2025_11_30_seed{seed}.joblib")
            ps.append(predict(ModelBundle(d["model"], d["columns"], d["cat_features"]), both))
        mean = np.mean(ps, axis=0)
        result[kind] = {"forward": mean[:n], "symmetric": (mean[:n] + 1 - mean[n:]) / 2}
    return result


def simulate(frame, p, *, gate="probability", fee_model="kalshi", fee=0.07, kelly=0.25, simultaneous=False):
    a, b = frame.price_red.to_numpy(float), frame.price_blue.to_numpy(float)
    ea = _effective_decimal(1 / a, fee, fee_model)
    eb = _effective_decimal(1 / b, fee, fee_model)
    ar, ab = (p - a, 1 - p - b) if gate == "probability" else (p * ea - 1, (1 - p) * eb - 1)
    red = ar >= ab
    pc, dec = np.where(red, p, 1 - p), np.where(red, ea, eb)
    fk = (pc * dec - 1) / (dec - 1)
    take = (np.maximum(ar, ab) >= 0.03) & (fk > 0)
    won = np.where(red, frame.Winner.eq("Red"), frame.Winner.eq("Blue"))
    pnl = np.where(won, dec - 1, -1)
    frac = kelly * np.maximum(0, fk)
    bank, peak, dd = 1.0, 1.0, 0.0
    if simultaneous:
        for _, indices in frame.groupby("date", sort=True).groups.items():
            ii = np.asarray(indices)[take[np.asarray(indices)]]
            stakes = frac[ii] / max(1.0, frac[ii].sum())
            bank *= 1 + np.sum(stakes * pnl[ii])
            peak = max(peak, bank)
            dd = max(dd, 1 - bank / peak)
    else:
        for f, profit in zip(frac[take], pnl[take], strict=True):
            bank *= 1 + f * profit
            peak = max(peak, bank)
            dd = max(dd, 1 - bank / peak)
    return {
        "fights": len(frame),
        "bets": int(take.sum()),
        "wins": int(won[take].sum()),
        "flat_roi_pct": float(100 * pnl[take].mean()),
        "bank_multiple": float(bank),
        "max_drawdown_pct": float(100 * dd),
        "mean_selected_probability": float(pc[take].mean()),
        "take": take,
        "red": red,
    }


def compact(sim):
    return {k: v for k, v in sim.items() if k not in ("take", "red")}


def isolated_failures():
    sample = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01"]),
            "R_fighter": ["a"],
            "B_fighter": ["b"],
            "Winner": ["Red"],
            "skill_diff_mean": [2.0],
            "skill_diff_std": [0.3],
            "win_dif": [3.0],
        }
    )
    x, *_ = prepare(sample, one_hot=False)
    bad = flip_signed_columns(x, ("skill_diff_mean",))
    assert x.skill_diff_mean.tolist() == [2.0, -2.0]
    assert bad.skill_diff_mean.tolist() == [2.0, 2.0]
    account = sizing.AccountConfig("A", 1000.0, 0.25, None, "real")
    rec = sizing.size_bet(account, 0.531, [(0.50, 1.0), (0.53, 10000.0)], [(0.51, 10000.0)])
    production_rec = sizing.size_bets_combined(
        [account], {"A": 0.531}, [(0.50, 1.0), (0.53, 10000.0)], [(0.51, 10000.0)]
    )[0]
    assert production_rec.decision == rec.decision
    assert production_rec.avg_fill_price == rec.avg_fill_price
    assert production_rec.stake_usd == rec.stake_usd
    cost = rec.avg_fill_price + 0.07 * rec.avg_fill_price * (1 - rec.avg_fill_price)
    assert rec.decision == "BET" and 0.531 / cost - 1 < 0
    zero_error = None
    try:
        sizing.size_bets_combined(
            sizing.default_accounts({"A": 0.0, "B": 300.0, "C": 300.0}),
            {"A": 0.7, "B": 0.7, "C": 0.7},
            [(0.5, 10000.0)],
            [(0.51, 10000.0)],
        )
    except KeyError as exc:
        zero_error = repr(exc)
    assert zero_error is not None
    # Exercise the real settlement function, but only in a temporary directory
    # with a fake client. The second side is actually producible by the sizer.
    recs = sizing.size_bets_combined(
        sizing.default_accounts({"A": 300.0, "B": 300.0, "C": 300.0}),
        {"A": 0.8, "B": 0.8, "C": 0.2},
        [(0.5, 10000.0)],
        [(0.5, 10000.0)],
    )
    assert {r.side for r in recs} == {"A", "B"}
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        bp = base / "bankrolls.json"
        initial = {"A": 300.0, "B": 300.0, "C": 300.0, "card_state": {"card_date": "2026-01-01"}}
        bp.write_text(json.dumps(initial))
        orders = {}
        for side in ("A", "B"):
            entries = [asdict(r) for r in recs if r.side == side]
            orders[side] = {"token": side, "avg_fill_price": 0.5, "per_account": entries}
        record = {"dry_run": False, "recommendation": {"status": "ok", "orders": orders}}
        (base / "2026-01-01_test.json").write_text(json.dumps(record))

        class FakeClient:
            def get_market(self, ticker):
                return {"market": {"status": "finalized", "result": "yes" if ticker == "A" else "no"}}

        with patch.object(bankroll, "BANKROLLS_PATH", bp), patch.object(bankroll, "NOTIF_DIR", base):
            actual = bankroll.sync(client=FakeClient(), ufc_results=[])
        expected_c = 300.0 - next(r.stake_usd for r in recs if r.account == "C")
        assert actual["C"] == 300.0 and expected_c < 300.0
        # The audit created NO fill ledger: A/B still booked paper PnL.
        assert actual["A"] != 300.0
        from ufc_pred.cli import bet_runner

        with patch.object(bet_runner, "NOTIF_DIR", base):
            file = bet_runner.write_record(pd.Timestamp("2026-01-01"), "overwrite", {"dry_run": False})
            bet_runner.write_record(pd.Timestamp("2026-01-01"), "overwrite", {"dry_run": True})
            assert json.loads(file.read_text())["dry_run"]
    return {
        "double_flip": {
            "prepare": x.skill_diff_mean.tolist(),
            "after_harness_flip": bad.skill_diff_mean.tolist(),
        },
        "slippage_negative_ev": {
            **asdict(rec),
            "model_probability": 0.531,
            "ev_at_avg_fill_pct": 100 * (0.531 / cost - 1),
        },
        "zero_bankroll_error": zero_error,
        "two_side_settlement": {
            "actual_C": actual["C"],
            "expected_C": expected_c,
            "paper_A_delta_without_fills": actual["A"] - 300.0,
        },
        "dry_run_overwrites_real_notification": True,
    }


def rank_snapshot(frame, ranks):
    out = frame.copy()
    lookup = ranks.copy()
    lookup["fighter"] = lookup.fighter.str.strip()
    groups = {wc: x for wc, x in lookup.groupby("weightclass")}

    def get(name, wc, date):
        sub = groups.get(wc)
        if sub is None:
            return np.nan
        sub = sub[sub.date < date]
        if sub.empty:
            return np.nan
        snap = sub[sub.date == sub.date.max()]
        hit = snap[snap.fighter.eq(name.strip())]
        return float(hit.iloc[0]["rank"]) if len(hit) else np.nan

    for side in ("R", "B"):
        for wc in WEIGHT_CLASSES:
            out[f"{side}_{wc}_rank"] = [
                get(
                    r[f"{side}_fighter"],
                    ("Women's Pound-for-Pound" if r.gender == "FEMALE" else "Men's Pound-for-Pound")
                    if wc == "Pound-for-Pound"
                    else wc,
                    r.date,
                )
                for _, r in out.iterrows()
            ]
        out[f"{side}_match_weightclass_rank"] = [
            get(r[f"{side}_fighter"], r.weight_class, r.date) for _, r in out.iterrows()
        ]
    rr, bb = out.R_match_weightclass_rank, out.B_match_weightclass_rank
    out["better_rank"] = np.where(
        rr.isna() & bb.isna(), "neither", np.where(bb.isna() | (rr < bb), "Red", "Blue")
    )
    return out


def operational_audit():
    """Additional offline identity, execution and historical-evaluator checks."""
    from bs4 import BeautifulSoup

    from ufc_pred.inference.upcoming_builder import _ALIASES
    from ufc_pred.ingest import ufc_schedule
    from ufc_pred.ingest.ufcstats_state import UFCStatsStateSource, _normalise_name

    try:
        from .venue_priced_evaluation import evaluate, swap_corners
    except ImportError:
        from venue_priced_evaluation import evaluate, swap_corners

    result = {}
    fights = pd.read_parquet(ROOT / "data/processed/fights.parquet")
    names_r = fights.R_fighter.map(_normalise_name)
    names_b = fights.B_fighter.map(_normalise_name)
    # An actual saved event page that the live parser fails to recognize.
    page = ufc_schedule.CACHE_DIR / "https_www.ufc.com_event_ufc-fight-night-september-12-2026.html"
    html = page.read_text()
    soup = BeautifulSoup(html, "html.parser")
    with patch.object(ufc_schedule, "_fetch_html", return_value=html):
        live = ufc_schedule.fetch_card_schedule("unused", main_card_ts=1789257600, prelims_ts=1789250400)
    suffixed = [
        n
        for n in soup.find_all(id=True)
        if n["id"] in ("main-card--2", "prelims-card--2", "early-prelims--2")
    ]
    result["schedule_actual_archive"] = {
        "page": str(page.relative_to(ROOT)),
        "live_count": len(live),
        "visible_fight_blocks": sum(len(s.find_all(class_="c-listing-fight")) for s in suffixed),
    }
    assert not live and result["schedule_actual_archive"]["visible_fight_blocks"] == 13
    # Name collisions in both the historical model index and raw profile lookup.
    bruno = fights[fights.R_fighter.eq("Bruno Silva") | fights.B_fighter.eq("Bruno Silva")]
    result["bruno_identity_collision"] = {
        "fights_by_class": bruno.weight_class.value_counts().to_dict(),
        "index_entries": sum(n == "Bruno Silva" for n in build_index(fights).fighter_to_id),
    }

    def row(url, nickname):
        return f'<tr><td><a href="{url}">Bruno</a></td><td><a href="{url}">Silva</a></td><td><a href="{url}">{nickname}</a></td></tr>'

    # Fixture demonstrates deterministic overwrite; it does NOT claim which
    # of the two profiles is last in today's live UFCStats listing.
    first, last = "http://ufcstats.com/fighter-details/person1", "http://ufcstats.com/fighter-details/person2"
    listing = f"<table>{row(first, 'Bulldog')}{row(last, 'Blindado')}</table>"
    with UFCStatsStateSource(html_getter=lambda url: listing) as source:
        result["bruno_identity_collision"]["duplicate_profile_fixture_returns"] = source.fighter_url(
            "Bruno Silva"
        )
    assert result["bruno_identity_collision"]["duplicate_profile_fixture_returns"] == last
    # This historical evaluator subtracts fees only in the winning branch.
    df = pd.DataFrame({"px_a": [0.5, 0.5], "px_b": [0.5, 0.5], "won_a": [True, False]})
    out = evaluate(df, np.array([0.6, 0.6]), 0.03)
    wrong, right = out.pnl.to_numpy(), np.array([1 / 0.5175 - 1, -1])
    assert np.isclose(wrong[0], 0.965) and not np.isclose(wrong[0], right[0])
    result["legacy_venue_evaluator_fee"] = {
        "actual_returns": wrong.tolist(),
        "correct_returns_per_cash_dollar": right.tolist(),
        "roi_actual_pct": 100 * wrong.mean(),
        "roi_correct_pct": 100 * right.mean(),
    }
    original = pd.DataFrame({"better_rank": ["Red"], "Winner": ["Red"], "R_wins": [3], "B_wins": [1]})
    result["legacy_venue_swap_keeps_better_rank"] = swap_corners(original).better_rank.iloc[0]
    assert result["legacy_venue_swap_keeps_better_rank"] == "Red"
    # Ground truth: prioritize archived venue settlement; for later fills only,
    # use an exact local fight + exact alias-resolved fighter, and label source.
    outcomes = {}
    for path in sorted((ROOT / "data/raw/kalshi").rglob("*.parquet")):
        snap = pd.read_parquet(path)
        for side in ("a", "b"):
            for _, s in snap.iterrows():
                settled = s.get(f"settle_result_{side}")
                if settled in ("yes", "no"):
                    outcomes[s[f"ticker_{side}"]] = (settled == "yes", "archived_kalshi_settlement")
    notified = {}
    for path in sorted((ROOT / "data/processed/bet_notifications").glob("*.json")):
        d = json.loads(path.read_text())
        fm = d.get("fight", {})
        fa, fb = fm.get("fighter_a_ufc"), fm.get("fighter_b_ufc")
        if not fa or not fb:
            continue
        fa, fb = _ALIASES.get(fa, fa), _ALIASES.get(fb, fb)
        date = pd.Timestamp(path.name[:10])
        key_a, key_b = _normalise_name(fa), _normalise_name(fb)
        hit = fights[
            (fights.date == date)
            & ((names_r.eq(key_a) & names_b.eq(key_b)) | (names_r.eq(key_b) & names_b.eq(key_a)))
        ]
        if len(hit) == 1 and hit.iloc[0].Winner in ("Red", "Blue"):
            r = hit.iloc[0]
            winner = r.R_fighter if r.Winner == "Red" else r.B_fighter
            swap = fm.get("kalshi_swap_vs_ufc", False)
            for letter, name in (("a", fb if swap else fa), ("b", fa if swap else fb)):
                ticker = fm.get(f"kalshi_ticker_{letter}")
                if ticker and ticker not in outcomes:
                    outcomes[ticker] = (
                        _normalise_name(winner) == _normalise_name(name),
                        "local_fight_result_exact_normalized_match",
                    )
        for side, o in d.get("recommendation", {}).get("orders", {}).items():
            notified[o["token"]] = {
                "file": path.name,
                "dry_run": d.get("dry_run"),
                "stake": o["total_stake_usd"],
                "avg": o["avg_fill_price"],
                "shares": o["total_shares"],
            }
            if o["token"] in outcomes or len(hit) != 1:
                continue
            r = hit.iloc[0]
            if r.Winner not in ("Red", "Blue"):
                continue
            winner = r.R_fighter if r.Winner == "Red" else r.B_fighter
            betname = _ALIASES.get(o["side_name"], o["side_name"])
            if betname in (r.R_fighter, r.B_fighter):
                outcomes[o["token"]] = (winner == betname, "local_fight_result_exact_match")
    fills = json.loads((ROOT / "data/processed/kalshi_fills.json").read_text())
    rows, unresolved = [], []
    for f in fills.values():
        ticker = f.get("ticker") or f["market_ticker"]
        if ticker not in outcomes or f["action"] != "buy" or f["side"] != "yes":
            unresolved.append(ticker)
            continue
        won, source = outcomes[ticker]
        shares = float(f["count_fp"])
        price = float(f["yes_price_dollars"])
        fee = float(f["fee_cost"])
        cost = shares * price + fee
        rows.append(
            {
                "ticker": ticker,
                "card": pd.to_datetime(ticker.split("-")[1][:7], format="%y%b%d").strftime("%Y-%m-%d"),
                "shares": shares,
                "price": price,
                "fee": fee,
                "cash_cost": cost,
                "won": bool(won),
                "pnl": shares * int(won) - cost,
                "outcome_source": source,
            }
        )
    ledger = pd.DataFrame(rows)
    ledger.to_csv(OUT / "actual_fills_recheck.csv", index=False)
    cards = []
    for day, g in ledger.groupby("card"):
        cards.append(
            {
                "card": day,
                "fills": len(g),
                "bets": g.ticker.nunique(),
                "wins": int(g.drop_duplicates("ticker").won.sum()),
                "cost": float(g.cash_cost.sum()),
                "pnl": float(g.pnl.sum()),
                "roi_pct": float(100 * g.pnl.sum() / g.cash_cost.sum()),
            }
        )
    result["actual_fill_ledger"] = {
        "unresolved_tickers": unresolved,
        "cards": cards,
        "total_cost": float(ledger.cash_cost.sum()),
        "total_pnl": float(ledger.pnl.sum()),
        "roi_pct": float(100 * ledger.pnl.sum() / ledger.cash_cost.sum()),
        "outcome_source_counts": ledger.outcome_source.value_counts().to_dict(),
    }
    differences = []
    for ticker, g in ledger.groupby("ticker"):
        if ticker not in notified:
            continue
        n = notified[ticker]
        differences.append(
            {
                "ticker": ticker,
                "file": n["file"],
                "dry_run": n["dry_run"],
                "recommended_cost": n["stake"],
                "actual_cost": float(g.cash_cost.sum()),
                "recommended_avg": n["avg"],
                "actual_avg": float((g.shares * g.price).sum() / g.shares.sum()),
            }
        )
    pd.DataFrame(differences).to_csv(OUT / "recommendation_fill_comparison.csv", index=False)
    result["execution_differences"] = {
        "matched_markets": len(differences),
        "absolute_price_gap_mean": float(
            np.mean([abs(r["actual_avg"] - r["recommended_avg"]) for r in differences])
        ),
        "price_gap_max": float(max(abs(r["actual_avg"] - r["recommended_avg"]) for r in differences)),
        "filled_despite_saved_dry_run": [r["file"] for r in differences if r["dry_run"]],
    }
    comp = ledger.merge(pd.DataFrame(differences)[["ticker", "recommended_avg"]], on="ticker")
    result["execution_differences"]["actual_minus_recommended_price_cost_usd"] = float(
        ((comp.price - comp.recommended_avg) * comp.shares).sum()
    )
    (OUT / "operational_results.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, indent=2, default=str))


def headline_audit():
    """Reprice the historical test artifact; never fit on or overwrite test."""
    from ufc_pred.backtest.bet_eval import evaluate_bets_kelly

    try:
        from .test_set_evaluation import DEPLOY_MODEL, load_test
        from .test_set_evaluation import predict as test_predict
    except ImportError:
        from test_set_evaluation import DEPLOY_MODEL, load_test
        from test_set_evaluation import predict as test_predict
    frame = load_test()
    p = test_predict(DEPLOY_MODEL, frame)
    y = frame.Winner.eq("Red").to_numpy(int)
    result = {}
    for universe, mask in [("all", pd.Series(True, index=frame.index)), ("no_debut", ~frame.has_debut)]:
        for fee_model in ("winnings", "kalshi"):
            args = (p[mask], y[mask], frame.loc[mask, "R_odds"], frame.loc[mask, "B_odds"])
            flat = evaluate_bets(
                *args, edge_threshold=0.05, fee_rate=0.07, use_no_vig=True, fee_model=fee_model
            )
            kelly = evaluate_bets_kelly(
                *args,
                edge_threshold=0.03,
                fee_rate=0.07,
                use_no_vig=True,
                fee_model=fee_model,
                max_bet_fraction=1.0,
            )
            result[f"{universe}_{fee_model}"] = {
                "n_bets": flat.n_bets,
                "roi_pct": flat.roi_pct,
                "ci95": flat.ci95_roi_pct,
                "kelly_multiple": kelly["final_bankroll"],
                "max_dd_pct": kelly["max_drawdown_pct"],
            }
    (OUT / "test_headline_fee_recheck.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-ablation", action="store_true")
    ap.add_argument("--operational-only", action="store_true")
    ap.add_argument("--headline-only", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.headline_only:
        headline_audit()
        return
    if args.operational_only:
        operational_audit()
        return
    result = {"isolated_failures": isolated_failures()}
    print("Isolated failures reproduced", flush=True)
    fights = pd.read_parquet(ROOT / "data/processed/fights.parquet")
    fights = fights[fights.Winner.isin(["Red", "Blue"])].copy()
    fights = add_prior_fight_counts(join_skill_v3(fights))
    ranks = pd.read_csv(ROOT / "data/raw/martj42_rankings/rankings_history.csv", parse_dates=["date"])
    result["data"] = {
        "fights": len(fights),
        "last_fight": str(fights.date.max()),
        "rank_last_date": str(ranks.date.max()),
        "rank_age_days_at_audit": (pd.Timestamp("2026-09-14") - ranks.date.max()).days,
        "p4p_literal_rows": int(ranks.weightclass.eq("Pound-for-Pound").sum()),
        "p4p_actual_class_counts": ranks[ranks.weightclass.str.contains("Pound")]
        .weightclass.value_counts()
        .to_dict(),
    }
    matrices = build_matrices(SPEC)
    xm = matrices["X_train"]
    n = len(xm) // 2
    a, b = xm.skill_diff_mean.to_numpy()[:n], xm.skill_diff_mean.to_numpy()[n:]
    valid = np.isfinite(a) & (a != 0)
    result["full_matrix_double_flip"] = {
        "original_train_rows": n,
        "nonzero_skill_rows": int(valid.sum()),
        "swapped_wrong_sign": int(np.isclose(a[valid], b[valid]).sum()),
    }
    # Modal weight class is a model input, not just a harmless integer index.
    cutoff = pd.Timestamp("2025-11-30")
    prior = fights[fights.date < cutoff]
    full_i, prior_i = build_index(fights), build_index(prior)
    fw, pw = {v: k for k, v in full_i.wc_to_id.items()}, {v: k for k, v in prior_i.wc_to_id.items()}
    changes = [
        {
            "fighter": name,
            "prior_wc": pw[prior_i.fighter_wc[idx]],
            "full_wc": fw[full_i.fighter_wc[full_i.fighter_to_id[name]]],
        }
        for name, idx in prior_i.fighter_to_id.items()
        if pw[prior_i.fighter_wc[idx]] != fw[full_i.fighter_wc[full_i.fighter_to_id[name]]]
    ]
    changed_prior = prior.copy()
    changed_prior.loc[changed_prior.index[0], "weight_class"] = "Different"
    assert _prior_hash(prior) == _prior_hash(changed_prior)
    result["skill_future_dependency"] = {"modal_class_changes": changes, "hash_ignores_weight_class": True}
    prices = pd.read_parquet(ROOT / "data/interim/polymarket_matched_to_kaggle_v2.parquet")
    prices.date = pd.to_datetime(prices.date)
    frame = prices[KEYS + ["polymarket_p_red", "polymarket_p_blue"]].merge(
        fights, on=KEYS, validate="one_to_one"
    )
    frame = (
        frame[(frame.date >= cutoff) & (frame.date <= "2026-05-31")]
        .sort_values("date", kind="stable")
        .reset_index(drop=True)
    )
    frame = frame.rename(columns={"polymarket_p_red": "price_red", "polymarket_p_blue": "price_blue"})
    preds = ensembles(frame)
    result["protocol_ladder"] = {}
    for kind, pred in preds.items():
        rows = {}
        for label, p, gate, fee_model, fee in [
            ("legacy_forward_ev_winnings2", pred["forward"], "ev", "winnings", 0.02),
            ("symmetric_ev_winnings2", pred["symmetric"], "ev", "winnings", 0.02),
            ("symmetric_sharpen_ev_winnings2", sharpen(pred["symmetric"]), "ev", "winnings", 0.02),
            ("symmetric_sharpen_prob_winnings2", sharpen(pred["symmetric"]), "probability", "winnings", 0.02),
            ("symmetric_sharpen_prob_kalshi7", sharpen(pred["symmetric"]), "probability", "kalshi", 0.07),
        ]:
            rows[label] = compact(simulate(frame, p, gate=gate, fee_model=fee_model, fee=fee))
        p = sharpen(pred["symmetric"])
        dep = ~frame.has_debut.to_numpy()
        rows["above_no_debut"] = compact(simulate(frame[dep].reset_index(drop=True), p[dep]))
        rows["above_no_debut_card_open"] = compact(
            simulate(frame[dep].reset_index(drop=True), p[dep], simultaneous=True)
        )
        evg, prg = simulate(frame, p, gate="ev"), simulate(frame, p)
        rows["threshold_only_changed_bets"] = int(np.sum(evg["take"] != prg["take"]))
        result["protocol_ladder"][kind] = rows
    print("Stored deployment protocol ladder complete", flush=True)
    # Rank lookup counterexamples: only dates covered by the local archive.
    examples = []
    latest = ranks[ranks.date == ranks.date.max()]
    for name in ["Tony Ferguson", "Conor McGregor", "Dominick Cruz", "Henry Cejudo"]:
        for wc in ["Lightweight", "Featherweight", "Bantamweight", "Flyweight"]:
            old = _rank_at(ranks, name, wc, pd.Timestamp("2026-05-20"))
            active = latest[(latest.fighter.str.strip() == name) & (latest.weightclass == wc)]
            if old is not None and active.empty:
                examples.append(
                    {
                        "fighter": name,
                        "wc": wc,
                        "stale_rank_returned": old,
                        "last_appearance": str(
                            ranks[(ranks.fighter == name) & (ranks.weightclass == wc)].date.max()
                        ),
                    }
                )
    fwd = pd.read_parquet(ROOT / "data/interim/feature_parity_live_raw_fwd.parquet")
    meta = pd.read_parquet(ROOT / "data/interim/feature_parity_live_raw_meta.parquet")
    ledger = pd.read_parquet(ROOT / "artifacts/metrics/feature_parity_live_raw.parquet")
    order = np.argsort(meta.file.to_numpy())
    fwd = fwd.iloc[order].reset_index(drop=True)
    ledger = ledger.sort_values("file").reset_index(drop=True)
    assert meta.iloc[order].file.reset_index(drop=True).equals(ledger.file)
    corrected = rank_snapshot(fwd, ranks)
    old_p, new_p = ensembles(fwd), ensembles(corrected)
    market = pd.DataFrame(
        {
            "date": fwd.date,
            "Winner": np.where(ledger.winA, "Red", "Blue"),
            "price_red": ledger.askA,
            "price_blue": ledger.askB,
        }
    )
    result["rank_audit"] = {"stale_examples": examples, "models": {}}
    rankcols = [c for c in fwd if c.endswith("_rank") or c == "better_rank"]
    changed = ~(fwd[rankcols].eq(corrected[rankcols]) | (fwd[rankcols].isna() & corrected[rankcols].isna()))
    result["rank_audit"]["changed_columns_counts"] = changed.sum().to_dict()
    for kind in old_p:
        p, q = sharpen(old_p[kind]["symmetric"]), sharpen(new_p[kind]["symmetric"])
        before, after = simulate(market, p), simulate(market, q)
        result["rank_audit"]["models"][kind] = {
            "mean_abs_probability_change": float(np.mean(np.abs(p - q))),
            "max_abs_probability_change": float(np.max(np.abs(p - q))),
            "before": compact(before),
            "after": compact(after),
            "changed_decisions": int(
                np.sum((before["take"] != after["take"]) | (before["red"] != after["red"]))
            ),
        }
    print("Rank sensitivity complete", flush=True)
    # Recompute canonical published results directly from persisted predictions;
    # no claims of retraining these source models in this run.
    canonical = pd.read_parquet(ROOT / "artifacts/metrics/strict_multivenue_postcutoff_fights.parquet")
    result["canonical_recheck"] = {}
    for venue in ("polymarket", "kalshi"):
        for period in ("early", "late"):
            mask = canonical.universe.eq(venue) & ~canonical.has_debut
            mask &= canonical.date.le("2026-05-16") if period == "early" else canonical.date.gt("2026-05-16")
            sub = canonical[mask].reset_index(drop=True)
            if sub.empty:
                continue
            for kind in ("real", "corrupted"):
                p = sub[f"p_red_{kind}"].to_numpy()
                # Per-row actual archived fee schedule.
                red = p - sub.price_red.to_numpy() >= 1 - p - sub.price_blue.to_numpy()
                price = np.where(red, sub.price_red, sub.price_blue)
                fee = sub.fee_rate.to_numpy() * (price * (1 - price)) ** sub.fee_exponent.to_numpy()
                cost = price + fee
                take = np.maximum(p - sub.price_red, 1 - p - sub.price_blue) >= 0.03
                won = np.where(red, sub.Winner.eq("Red"), sub.Winner.eq("Blue"))
                pnl = np.where(won, 1 / cost - 1, -1)
                result["canonical_recheck"][f"{venue}_{period}_{kind}"] = {
                    "fights": len(sub),
                    "bets": int(take.sum()),
                    "wins": int(won[take].sum()),
                    "flat_roi_pct": float(100 * pnl[take].mean()),
                }
    # Actual fills vs notifications, matched by ticker. Report aggregate only.
    fills = list(json.loads((ROOT / "data/processed/kalshi_fills.json").read_text()).values())
    ftickers = {f.get("ticker", f.get("market_ticker")) for f in fills}
    missing, dual, dry, negative_ev = [], [], [], []
    for file in sorted((ROOT / "data/processed/bet_notifications").glob("*.json")):
        d = json.loads(file.read_text())
        orders = d.get("recommendation", {}).get("orders", {})
        if len(orders) > 1:
            dual.append(file.name)
        if d.get("dry_run") and orders:
            dry.append(file.name)
        for side, o in orders.items():
            if not d.get("dry_run") and o.get("token") not in ftickers:
                missing.append({"file": file.name, "stake": o["total_stake_usd"]})
            for entry in o.get("per_account", []):
                model = d.get("model") or {}
                p = model.get("p_a_corrupted" if entry["account"] == "C" else "p_a_real")
                if p is None:
                    continue
                p = p if side == "A" else 1 - p
                avg = o["avg_fill_price"]
                ev = p / (avg + 0.07 * avg * (1 - avg)) - 1
                if ev < 0:
                    negative_ev.append({"file": file.name, "account": entry["account"], "ev": ev})
    result["ledger_archive"] = {
        "fill_records": len(fills),
        "fill_tickers": len(ftickers),
        "recommendations_no_archived_fill": missing,
        "dual_side_records": dual,
        "dry_run_order_records": dry,
        "negative_ev_at_recommended_average": negative_ev,
    }
    # Repository inventory and script-root regressions, without importing
    # historical scripts (several train or write at import time).
    inventory, bad_roots = [], []
    for folder in ("src", "scripts", "tests", "notebooks", "experiments"):
        for file in sorted((ROOT / folder).rglob("*")):
            if not file.is_file() or file.suffix not in (".py", ".ipynb", ".md"):
                continue
            source = file.read_text()
            item = {
                "path": str(file.relative_to(ROOT)),
                "sha256": hashlib.sha256(source.encode()).hexdigest(),
            }
            if file.suffix == ".py":
                tree = ast.parse(source)
                item["functions"] = [
                    node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                ]
                if folder == "scripts" and (
                    "resolve().parents[1]" in source or "resolve().parent.parent" in source
                ):
                    bad_roots.append(item["path"])
            elif file.suffix == ".ipynb":
                notebook = json.loads(source)
                item["cell_count"] = len(notebook.get("cells", []))
                item["headings"] = [
                    line
                    for cell in notebook.get("cells", [])
                    if cell.get("cell_type") == "markdown"
                    for line in "".join(cell.get("source", [])).splitlines()
                    if line.startswith("#")
                ]
            inventory.append(item)
    result["script_root_candidates"] = bad_roots
    if args.fit_ablation:
        from catboost import CatBoostClassifier, Pool

        from ufc_pred.models._spec import BASE_CATBOOST_PARAMS
        from ufc_pred.utils.time_splits import recency_weights

        result["double_flip_ablation"] = {}
        for label, xt in [
            ("current_double_flip", xm),
            ("single_correct_flip", flip_signed_columns(xm, ("skill_diff_mean",))),
        ]:
            print(f"Fitting diagnostic: {label}", flush=True)
            model = CatBoostClassifier(**BASE_CATBOOST_PARAMS, random_seed=0, thread_count=4)
            model.fit(
                Pool(
                    xt,
                    matrices["y_train"],
                    cat_features=matrices["cat_features"],
                    weight=recency_weights(matrices["d_train"]),
                )
            )
            p = model.predict_proba(matrices["X_val"])[:, 1]
            from sklearn.metrics import log_loss

            val = matrices["splits"].val
            ev = evaluate_bets(
                p,
                matrices["y_val"],
                val.R_odds,
                val.B_odds,
                edge_threshold=0.05,
                fee_rate=0.07,
                use_no_vig=True,
                fee_model="kalshi",
            )
            result["double_flip_ablation"][label] = {
                "val_log_loss": log_loss(matrices["y_val"], p),
                "bets": ev.n_bets,
                "roi_pct": ev.roi_pct,
                "ci95": ev.ci95_roi_pct,
            }
    (OUT / "inventory.json").write_text(json.dumps(inventory, indent=2))
    (OUT / "audit_results.json").write_text(
        json.dumps(result, indent=2, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()

import json

import numpy as np
import pandas as pd
import pytest

from ufc_pred.inference.sizing import default_accounts, size_bets_combined
from ufc_pred.ingest.identity import canonical
from ufc_pred.ingest.rankings_attach import _rank_at
from ufc_pred.ops.fill_ledger import migration_anchor, reconcile


def test_removed_fighter_and_renamed_pound_for_pound():
    rankings = pd.DataFrame(
        [
            ["2026-01-01", "Lightweight", "Old Fighter", 1],
            ["2026-01-08", "Lightweight", "New Fighter", 1],
            ["2026-01-08", "Men's Pound-for-Pound", "New Fighter", 2],
        ],
        columns=["date", "weightclass", "fighter", "rank"],
    )
    assert _rank_at(rankings, "Old Fighter", "Lightweight", pd.Timestamp("2026-01-08")) == 1
    assert _rank_at(rankings, "Old Fighter", "Lightweight", pd.Timestamp("2026-01-09")) is None
    assert _rank_at(rankings, "New Fighter", "Pound-for-Pound", pd.Timestamp("2026-01-09")) == 2


def test_identity_homonym_and_alias():
    assert canonical("Bruno Silva", "Middleweight") != canonical("Bruno Silva", "Flyweight")
    assert canonical("Bruno Silva", "Bantamweight") == canonical("Bruno Silva", "Flyweight")
    with pytest.raises(ValueError):
        canonical("Bruno Silva")
    assert canonical("Bobby Green") == canonical("King Green")


def test_posterior_prior_only_and_cache_recipe(monkeypatch, tmp_path):
    from ufc_pred.inference import skill_for_upcoming as m

    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01", "2027-01-01"]),
            "R_fighter": ["One", "One"],
            "B_fighter": ["Two", "Future"],
            "Winner": ["Red", "Blue"],
            "weight_class": ["Lightweight", "Heavyweight"],
        }
    )
    monkeypatch.setattr(m, "POSTERIOR_CACHE", tmp_path)
    seen = []

    def fit(a, b, y, index, **kwargs):
        seen.append(index)
        return {"skill": np.zeros((4, index.n_fighters))}

    monkeypatch.setattr(m, "fit_nuts", fit)
    m.fit_or_load_month_posterior(frame, pd.Timestamp("2026-01-01"), num_samples=5)
    assert "Future" not in seen[0].fighter_to_id
    assert "Heavyweight" not in seen[0].wc_to_id
    frame.loc[1, "weight_class"] = "Middleweight"
    m.fit_or_load_month_posterior(frame, pd.Timestamp("2026-01-01"), num_samples=5)
    assert len(seen) == 1
    m.fit_or_load_month_posterior(frame, pd.Timestamp("2026-01-01"), num_samples=6)
    assert len(seen) == 2
    frame.loc[0, "weight_class"] = "Welterweight"
    m.fit_or_load_month_posterior(frame, pd.Timestamp("2026-01-01"), num_samples=6)
    assert len(seen) == 3


def test_augmented_harness_owns_sign_once(monkeypatch):
    from types import SimpleNamespace

    from ufc_pred.models import _harness as h

    f = pd.DataFrame(
        {
            "date": ["2020-01-01"],
            "R_fighter": ["One"],
            "B_fighter": ["Two"],
            "Winner": ["Red"],
            "skill_diff_mean": [2.0],
            "skill_diff_std": [0.7],
        }
    )
    monkeypatch.setattr(h.pd, "read_parquet", lambda p: f)
    monkeypatch.setattr(h, "split", lambda df: SimpleNamespace(train=df, val=df))
    spec = SimpleNamespace(recipe=SimpleNamespace(join=None, flip_columns=("skill_diff_mean",)))
    m = h.build_matrices(spec)
    assert m["X_train"].skill_diff_mean.tolist() == [2.0, -2.0]
    assert m["X_train"].skill_diff_std.tolist() == [0.7, 0.7]


def test_adverse_depth_cannot_turn_edge_negative():
    accts = default_accounts({"A": 1000.0, "B": 0.0, "C": 0.0})
    recs = size_bets_combined(
        accts, {"A": 0.531, "B": 0.531, "C": 0.531}, [(0.5, 1), (0.53, 10000)], [(0.51, 10000)]
    )
    assert recs[0].decision == "BET"
    assert recs[0].limit_price == 0.5
    assert recs[0].shares <= 1
    assert recs[0].shares * 0.531 > recs[0].stake_usd
    assert recs[1].decision == recs[2].decision == "SKIP"


def test_shared_cash_and_two_opposite_sides():
    accts = default_accounts(dict.fromkeys("ABC", 1000.0))
    recs = size_bets_combined(
        accts, {"A": 0.8, "B": 0.8, "C": 0.2}, [(0.5, 10000)], [(0.5, 10000)], available_cash=12.0
    )
    assert {r.side for r in recs if r.decision == "BET"} == {"A", "B"}
    assert sum(r.stake_usd for r in recs) <= 12.0 + 1e-4


def fill(ticker, side="yes", action="buy", qty="10", price="0.40", fee="0.17", time="2026-09-15T12:00:00Z"):
    return {
        "ticker": ticker,
        "side": side,
        "action": action,
        "count_fp": qty,
        f"{side}_price_dollars": price,
        "fee_cost": fee,
        "created_time": time,
    }


def test_ledger_only_fills_both_sides_partial_idempotence_and_rollover():
    anchor = migration_anchor(dict.fromkeys("ABC", 100.0), "2026-09-14T00:00:00Z")
    rec = {
        "captured_at_utc": "2026-09-15T11:00:00Z",
        "recommendation": {
            "orders": {
                "A": {"token": "RED", "per_account": [{"account": "A", "shares": 100}]},
                "B": {"token": "BLUE", "per_account": [{"account": "C", "shares": 100}]},
            }
        },
    }
    empty = reconcile(anchor, {}, [rec], {"RED": {"yes_won": True, "final": True}})
    assert empty["equity"] == dict.fromkeys("ABC", 100.0)
    fills = {"1": fill("RED"), "2": fill("BLUE")}
    resolutions = {"RED": {"yes_won": True, "final": True}, "BLUE": {"yes_won": False, "final": True}}
    one = reconcile(anchor, fills, [rec], resolutions)
    assert one["equity"]["A"] == pytest.approx(105.83)
    assert one["equity"]["C"] == pytest.approx(95.83)
    two = reconcile(one, fills, [rec], resolutions)
    assert one["equity"] == two["equity"]


def test_ledger_provisional_not_spendable_and_overturn():
    anchor = migration_anchor(dict.fromkeys("ABC", 100.0), "2026-09-14T00:00:00Z")
    fills = {"1": fill("T")}
    prov = reconcile(anchor, fills, [], {"T": {"yes_won": True, "final": False}})
    assert sum(prov["cash"].values()) == pytest.approx(295.83)
    assert sum(prov["equity"].values()) == pytest.approx(305.83)
    final = reconcile(prov, fills, [], {"T": {"yes_won": False, "final": True}})
    assert sum(final["cash"].values()) == pytest.approx(295.83)
    assert sum(final["equity"].values()) == pytest.approx(295.83)


def test_ledger_sell_no_and_missing_fees():
    anchor = migration_anchor(dict.fromkeys("ABC", 100.0), "2026-09-14T00:00:00Z")
    fills = {
        "1": fill("T", side="no"),
        "2": fill("T", side="no", action="sell", qty="5", price=".6", fee=".1", time="2026-09-15T13:00:00Z"),
    }
    out = reconcile(anchor, fills, [], {"T": {"yes_won": False, "final": True}})
    assert sum(out["equity"].values()) == pytest.approx(303.73)
    fills["3"] = fill("OTHER")
    del fills["3"]["fee_cost"]
    assert "3" in reconcile(anchor, fills, [], {})["quarantine"]


def test_dry_run_cannot_overwrite_live(monkeypatch, tmp_path):
    from ufc_pred.cli import bet_runner as m

    monkeypatch.setattr(m, "NOTIF_DIR", tmp_path)
    p = m.write_record(pd.Timestamp("2026-01-01"), "one", {"dry_run": False, "value": 1})
    m.write_record(pd.Timestamp("2026-01-01"), "one", {"dry_run": True, "value": 2})
    m.write_record(pd.Timestamp("2026-01-01"), "one", {"dry_run": False, "value": 3})
    assert json.loads(p.read_text())["value"] == 1
    assert (tmp_path / "dry_run/one.json").exists()
    assert len(list((tmp_path / "revisions").glob("*.json"))) == 1


def test_schedule_suffix(monkeypatch):
    from ufc_pred.ingest import ufc_schedule as m

    html = """<div id="main-card--2"><div class="c-listing-fight">
    <span class="c-listing-fight__corner-name--red">One</span>
    <span class="c-listing-fight__corner-name--blue">Two</span></div></div>"""
    monkeypatch.setattr(m, "_fetch_html", lambda url: html)
    out = m.fetch_card_schedule("test", main_card_ts=1789400000, prelims_ts=1789390000)
    assert len(out) == 1 and out[0].fighter_a == "One"


def test_dwcs_does_not_enter_ufc_only_features():
    from ufc_pred.ingest.ufcstats_scraper import _is_ufc_history_event

    assert not _is_ufc_history_event("DWCS 2.6")
    assert not _is_ufc_history_event("Dana White's Contender Series")
    assert _is_ufc_history_event("UFC Fight Night: Chiesa vs. Magny")
    assert _is_ufc_history_event("Noche UFC: Silva vs. Delgado")


def test_raw_state_uses_only_completed_prior_days(tmp_path):
    from ufc_pred.ingest.strict_history import RawHistoryStateSource

    pd.DataFrame({"EVENT": ["UFC Past", "UFC Future"], "DATE": ["2020-01-01", "2022-01-01"]}).to_csv(
        tmp_path / "ufc_event_details.csv", index=False
    )
    pd.DataFrame(
        {
            "EVENT": ["UFC Past", "UFC Future"],
            "BOUT": ["Alpha vs. Beta"] * 2,
            "OUTCOME": ["W/L", "L/W"],
            "WEIGHTCLASS": ["Lightweight Bout"] * 2,
            "METHOD": ["KO/TKO ", "Submission "],
            "ROUND": [1, 1],
            "TIME": ["5:00", "5:00"],
        }
    ).to_csv(tmp_path / "ufc_fight_results.csv", index=False)
    stats = pd.DataFrame(
        {
            "EVENT": ["UFC Past"] * 2 + ["UFC Future"] * 2,
            "BOUT": ["Alpha vs. Beta"] * 4,
            "FIGHTER": ["Alpha", "Beta"] * 2,
            "SIG.STR.": ["10 of 20", "5 of 10", "999 of 999", "999 of 999"],
            "TD": ["1 of 2"] * 4,
            "SUB.ATT": [0, 0, 999, 999],
        }
    )
    stats.to_csv(tmp_path / "ufc_fight_stats.csv", index=False)
    pd.DataFrame(
        {
            "FIGHTER": ["Alpha", "Beta"],
            "HEIGHT": ["6' 0\""] * 2,
            "WEIGHT": ["155 lbs."] * 2,
            "REACH": ['72"'] * 2,
            "STANCE": ["Orthodox"] * 2,
            "DOB": ["Jan 01, 1990"] * 2,
            "URL": ["one", "two"],
        }
    ).to_csv(tmp_path / "ufc_fighter_tott.csv", index=False)
    source = RawHistoryStateSource(tmp_path)
    assert source("Alpha", "2020-01-01")["wins"] == 0
    state = source("Alpha", "2022-01-01")
    assert state["wins"] == 1 and state["losses"] == 0
    assert state["avg_SIG_STR_landed"] == 2.0
    assert state["win_by_KO/TKO"] == 1
    stats.loc[stats.EVENT.eq("UFC Future"), "SIG.STR."] = "1 of 99999"
    stats.to_csv(tmp_path / "ufc_fight_stats.csv", index=False)
    assert RawHistoryStateSource(tmp_path)("Alpha", "2022-01-01") == state


def test_shadow_execution_is_separate_and_idempotent(tmp_path):
    from ufc_pred.ops.shadow_evaluation import project

    (tmp_path / "captures").mkdir()
    protocol = {
        "frozen_at_utc": "2026-09-14T00:00:00+00:00",
        "reference_bankrolls": dict.fromkeys("ABC", 100.0),
    }
    record = {
        "mode": "shadow",
        "capture_trigger": "first_fight",
        "captured_at_utc": "2026-09-15T11:00:00+00:00",
        "fight_date_utc": "2026-09-15T12:00:00+00:00",
        "fight": {"kalshi_event": "event"},
        "model": {"p_a_real": 0.7, "p_a_corrupted": 0.2},
        "recommendation": {
            "status": "ok",
            "orders": {
                "A": {
                    "token": "T",
                    "avg_fill_price": 0.4,
                    "per_account": [{"account": "A", "shares": 10.0, "stake_usd": 4.17, "fee_usd": 0.17}],
                }
            },
        },
    }
    (tmp_path / "captures/one.json").write_text(json.dumps(record))
    first = project(tmp_path, protocol)
    assert first["cash"]["A"] == pytest.approx(95.83)
    assert first["equity"]["A"] == 100.0

    class SettledClient:
        def get_market(self, ticker):
            return {"status": "finalized", "result": "yes"}

    second = project(tmp_path, protocol, SettledClient())
    assert second["equity"]["A"] == pytest.approx(105.83)
    assert project(tmp_path, protocol)["equity"] == second["equity"]
    assert json.loads((tmp_path / "prospective_metrics.json").read_text())["realised_trading_roi"] is None


def test_verified_envelope_corrects_swapped_result_without_future_rows():
    from ufc_pred.ingest.identity import verified_envelope

    history = pd.DataFrame(
        {
            "date": ["2021-01-20", "2020-05-09"],
            "R_fighter": ["Mason Jones", "Uriah Hall"],
            "B_fighter": ["Mike Davis", "Jacare Souza"],
            "Winner": ["Red", "Red"],
            "weight_class": ["Lightweight", "Middleweight"],
        }
    )
    corrected = verified_envelope(history)
    assert len(corrected) == 1
    assert corrected.iloc[0].Winner == "Blue"
    assert history.iloc[0].Winner == "Red"


def test_shadow_wrapper_routes_all_outputs_and_disables_alerts(monkeypatch, tmp_path):
    import sys

    from ufc_pred.cli import bet_runner as r
    from ufc_pred.cli import shadow_runner as s
    from ufc_pred.ops import shadow_evaluation

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text("{}")
    (bundle / "shadow_protocol.json").write_text(
        json.dumps({"reference_bankrolls": dict.fromkeys("ABC", 100.0)})
    )
    shadow = tmp_path / "shadow"
    monkeypatch.setattr(s, "BUNDLE", bundle)
    monkeypatch.setattr(s, "SHADOW", shadow)
    monkeypatch.setattr(s, "verify_bundle", lambda: {})
    monkeypatch.setattr(s, "refresh_history", lambda: (shadow / "history.parquet").write_bytes(b"history"))
    monkeypatch.setattr(shadow_evaluation, "project", lambda *a, **kw: {"cash": dict.fromkeys("ABC", 100.0)})
    for attr in [
        "KalshiClient",
        "FIGHTS_PATH",
        "NOTIF_DIR",
        "IDEMPOTENCY_PATH",
        "TICKER_CACHE_PATH",
        "BANKROLLS_PATH",
        "notify",
        "write_audit_log",
        "write_record",
    ]:
        monkeypatch.setattr(r, attr, getattr(r, attr))
    from ufc_pred.ingest import kalshi_client

    monkeypatch.setattr(kalshi_client, "KalshiClient", type("ReadOnlyClient", (), {}))

    def run(args):
        assert args.shadow and args.dry_run and args.watchdog and not args.once
        assert args.models_dir == bundle / "models"
        assert r.KalshiClient().get_balance()["balance_dollars"] == "300.0"
        assert r.notify("must not send") is None
        assert r.write_audit_log(pd.Timestamp("2026-09-15"), {}, []).exists()
        path = r.write_record(pd.Timestamp("2026-09-15"), "test", {"mode": "shadow"})
        payload = json.loads(path.read_text())
        assert path.parent == shadow / "captures"
        assert payload["bundle_sha256"] and payload["history_sha256"]
        return 0

    monkeypatch.setattr(r, "run", run)
    monkeypatch.setattr(sys, "argv", ["shadow_runner"])
    assert s.main() == 0
    assert json.loads((shadow / "status.json").read_text())["status"] == "ok"

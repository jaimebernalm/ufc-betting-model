"""Synthetic tests of every abstention gate of the paper-only shadow logger (no network)."""

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import ufc_pred.research.anchored_shadow as S

pytestmark = pytest.mark.skipif(
    not (S.CANDIDATES["F1"]["bundle"] / "manifest.json").exists(), reason="needs the local shadow bundles"
)

NOW = pd.Timestamp(dt.datetime.now(dt.UTC))


def _capture(tmp_path: Path, captured, sched, card_date="2026-11-01", swap=False, name="cap.json") -> Path:
    base = pd.read_parquet(
        S.ROOT / "artifacts/shadow_candidate_2026_09_14_v2/training_features.parquet"
    ).iloc[-1]
    row = {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in base.to_dict().items()}
    row.update(date=card_date, R_fighter="Red Fighter", B_fighter="Blue Fighter")
    cap = {
        "captured_at_utc": None if captured is None else str(captured),
        "fight_date_utc": None if sched is None else str(sched),
        "fight": {
            "fighter_a_ufc": "Red Fighter",
            "fighter_b_ufc": "Blue Fighter",
            "kalshi_swap_vs_ufc": swap,
        },
        "kalshi_snapshot": {"a_ask": 0.40, "b_ask": 0.62, "a_bid": 0.39, "b_bid": 0.61},
        "features": {"forward": [row]},
    }
    p = tmp_path / name
    p.write_text(json.dumps(cap, default=str))
    return p


@pytest.fixture(scope="module")
def bundle():
    return S.load_bundles()


@pytest.fixture
def offline(monkeypatch, tmp_path):
    """No network and no writes outside tmp: fake BFO page/rows with one DraftKings pair and method props."""
    monkeypatch.setattr(S, "OUTDIR", tmp_path / "shadow_out")
    rows = pd.DataFrame(
        {
            "matchup_id": [1, 1],
            "fighter_name": ["Red Fighter", "Blue Fighter"],
            "fighter_pos": [1, 2],
            "book_name": ["DraftKings", "DraftKings"],
            "american_odds": [150.0, -170.0],
        }
    )
    page = (
        '<table class="odds-table"><thead><th scope="col" data-b="22">DraftKings</th></thead><tbody>'
        + "".join(
            f'<td data-li="[22,1,1,{t},{pos}]"><span>{o}</span></td>'
            for t, pos, o in [
                (8, 1, "+400"),
                (9, 1, "+800"),
                (11, 1, "+250"),
                (8, 2, "+200"),
                (9, 2, "+600"),
                (11, 2, "+120"),
            ]
        )
        + "</tbody></table>"
    )
    monkeypatch.setattr(S, "find_bfo_event", lambda d, n: ("fake-event", page, rows, 1))
    monkeypatch.setattr(S, "_prior_ufc_bouts", lambda name, before: 3)

    # B1/B2 for the synthetic target: all-missing columns with a fresh history (7 days)
    def fake_perf(row, card_date):
        cols = [f for f in S.load_bundle("F1B")[0]["feature_names"] if "perf" in f]
        return (
            pd.DataFrame({c: [np.nan] for c in cols}),
            {"source_history_lag_days": 7, "snapshot": "synthetic"},
        )

    monkeypatch.setattr(S, "target_performance", fake_perf)
    return rows


def _score(path, bundle, lag=10.0):
    return S.score_capture(path, bundle, lag)


@pytest.mark.parametrize(
    "captured,sched,status",
    [
        (None, NOW + pd.Timedelta(hours=1), "abstain_missing_timestamp"),
        (NOW - pd.Timedelta(minutes=1), None, "abstain_missing_timestamp"),
        (NOW + pd.Timedelta(minutes=5), NOW + pd.Timedelta(hours=1), "abstain_future_timestamp"),
        (
            NOW - pd.Timedelta(minutes=1),
            NOW - pd.Timedelta(minutes=2),
            "abstain_capture_at_or_after_scheduled_start",
        ),
        (NOW - pd.Timedelta(minutes=30), NOW + pd.Timedelta(hours=1), "abstain_stale_capture"),
    ],
)
def test_timestamp_gates(tmp_path, bundle, offline, captured, sched, status):
    assert _score(_capture(tmp_path, captured, sched), bundle)["status"] == status


def test_card_before_fit_date(tmp_path, bundle, offline):
    p = _capture(tmp_path, NOW - pd.Timedelta(minutes=1), NOW + pd.Timedelta(hours=1), card_date="2026-01-01")
    assert _score(p, bundle)["status"] == "abstain_card_before_bundle_fit_date"


def test_debut_gate(tmp_path, bundle, offline, monkeypatch):
    monkeypatch.setattr(S, "_prior_ufc_bouts", lambda name, before: 0 if name == "Blue Fighter" else 4)
    p = _capture(tmp_path, NOW - pd.Timedelta(minutes=1), NOW + pd.Timedelta(hours=1))
    assert _score(p, bundle)["status"] == "abstain_debut"


def test_invalid_lag_parameter(tmp_path, bundle, offline):
    p = _capture(tmp_path, NOW - pd.Timedelta(minutes=1), NOW + pd.Timedelta(hours=1))
    with pytest.raises(ValueError):
        _score(p, bundle, lag=float("nan"))


def test_happy_path_and_orientation(tmp_path, bundle, offline):
    p = _capture(tmp_path, NOW - pd.Timedelta(minutes=1), NOW + pd.Timedelta(hours=1))
    r = _score(p, bundle)
    assert r["status"] == "scored" and r["executed"] is False
    assert r["book"] == "DraftKings" and np.isfinite(r["mkt_props_dif"]) and r["R_ask"] == 0.40
    assert {"capture_sha256", "runner_sha256", "bundle_manifest_sha256"} <= set(r)
    for name in bundle:
        c = r["candidates"][name]
        assert c["status"] == "scored" and 0 < c["p_red"] < 1
    assert 0 < r["controls"]["p_mf"] < 1 and 0 < r["controls"]["q_kalshi_mid"] < 1
    # swapped Kalshi order: ticker_a is the UFC fighter_b -> asks must follow the NAME
    p2 = _capture(
        tmp_path, NOW - pd.Timedelta(minutes=1), NOW + pd.Timedelta(hours=1), swap=True, name="c2.json"
    )
    r2 = _score(p2, bundle)
    assert r2["R_ask"] == 0.62 and r2["B_ask"] == 0.40


def test_stale_b_history_only_blocks_f1b(tmp_path, bundle, offline, monkeypatch):
    if "F1B" not in bundle:
        pytest.skip("F1B bundle not fitted")
    monkeypatch.setattr(
        S, "target_performance", lambda row, d: (pd.DataFrame(index=[0]), {"source_history_lag_days": 40})
    )
    r = _score(_capture(tmp_path, NOW - pd.Timedelta(minutes=1), NOW + pd.Timedelta(hours=1)), bundle)
    assert r["candidates"]["F1B"]["status"] == "abstain_stale_b_history"
    assert r["candidates"]["F1"]["status"] == "scored"


def test_real_b_features_for_a_historical_target():
    """The CURRENT snapshot reproduces the frozen B1/B2 row exactly for a known fight."""
    if not S.SNAPSHOTS.exists():
        pytest.skip("no UFCStats snapshot")
    tf = pd.read_parquet(S.ROOT / "artifacts/shadow_candidate_2026_09_14_v2/training_features.parquet")
    row = tf[tf.date == tf.date.max()].head(1).reset_index(drop=True)
    perf, diag = S.target_performance(row, row.date.iloc[0])
    frozen = pd.read_parquet(S.PERF_B1).merge(pd.read_parquet(S.PERF_B2), on=S.KEYS)
    ref = frozen.merge(row[S.KEYS], on=S.KEYS).drop(columns=S.KEYS)
    assert np.allclose(perf[ref.columns].to_numpy(float), ref.to_numpy(float), equal_nan=True)
    assert diag["source_history_lag_days"] <= 14


def test_decide_rejects_non_finite():
    assert S.decide(np.nan, 0.5, {"R_ask": 0.4, "B_ask": 0.6})["signal"] == "invalid_probability"
    assert S.decide(0.6, np.inf, {"R_ask": 0.4, "B_ask": 0.6})["signal"] == "invalid_probability"

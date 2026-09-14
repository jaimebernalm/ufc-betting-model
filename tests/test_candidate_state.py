import hashlib
import json

import pandas as pd
import pytest

from ufc_pred.ingest.candidate_state import CandidateStateSource, state_source_for_models
from ufc_pred.ingest.strict_history import COUNTERS

URL = "http://ufcstats.com/fighter-details/example"
DETAIL = "http://ufcstats.com/fight-details/example"


def history_row(date, link=DETAIL):
    values = ["win", "", "", "", "", "", "", "", "", "", "", "UFC Test", date, "KO/TKO", "", "1", "5:00"]
    return (
        f'<tr class="b-fight-details__table-row" data-link="{link}"><td>'
        + "".join(f'<p class="b-fight-details__table-text">{v}</p>' for v in values)
        + "</td></tr>"
    )


def detail_html():
    values = [
        f'<a href="{URL}">Known Fighter</a>',
        "",
        "<p>10 of 20</p>",
        "",
        "",
        "<p>1 of 2</p>",
        "",
        "<p>1</p>",
    ]
    return (
        "<p>Method: TKO - Doctor's Stoppage</p><table><tbody class='b-fight-details__table-body'><tr>"
        + "".join(f"<td>{v}</td>" for v in values)
        + "</tr></tbody></table>"
    )


@pytest.fixture
def contract(tmp_path):
    base = dict(
        key="known fighter",
        date="2026-01-01",
        outcome="W",
        rounds=2,
        method="KO/TKO",
        title=False,
        sig_landed=60,
        sig_attempted=120,
        td_landed=2,
        td_attempted=4,
        sub_attempted=1,
        seconds=600,
    )
    external = {
        **base,
        "date": "2025-01-01",
        "outcome": "L",
        "rounds": 3,
        **dict.fromkeys(
            ["sig_landed", "sig_attempted", "td_landed", "td_attempted", "sub_attempted", "seconds"], 0
        ),
    }
    p = tmp_path / "candidate_inputs.json"
    p.write_text(
        json.dumps(
            dict(
                last_ufc_event="2026-01-01",
                entries=[external, base],
                profiles={
                    "known fighter": dict(
                        Stance="Orthodox",
                        Height_cms=180,
                        Reach_cms=185,
                        Weight_lbs=155,
                        DOB="2000-01-01",
                        URL=URL,
                    )
                },
            )
        )
    )
    return p


def test_prospective_rates_counters_doctor_and_strict_day(contract, tmp_path):
    seen = []
    html = (
        '<span class="b-content__title-highlight">Known Fighter</span><table>'
        + history_row("Feb. 01, 2026")
        + history_row("Mar. 01, 2026", DETAIL + "future")
        + "</table>"
    )

    def get(url):
        seen.append(url)
        return html if url == URL else detail_html()

    source = CandidateStateSource(contract, live=True, html_getter=get, capture_dir=tmp_path / "sources")
    same_day = source("Known Fighter", "2026-02-01T23:59:00Z")
    assert same_day["wins"] == 1 and same_day["losses"] == 1
    assert same_day["total_rounds_fought"] == 5
    assert same_day["avg_SIG_STR_landed"] == 6
    assert seen == [URL]
    after = source("Known Fighter", "2026-03-01")
    assert after["wins"] == 2 and after["total_rounds_fought"] == 6
    assert after["win_by_KO/TKO"] == 1
    assert after["win_by_TKO_Doctor_Stoppage"] == 1
    assert after["avg_SIG_STR_landed"] == 4.67
    assert after["avg_TD_landed"] == 3
    assert after["Height_cms"] == 180 and after["Weight_lbs"] == 155
    assert seen == [URL, DETAIL]
    assert source("Known Fighter", "2026-03-01") == after
    assert seen == [URL, DETAIL]
    for digest in source.observations.values():
        assert hashlib.sha256((tmp_path / "sources" / (digest + ".html")).read_bytes()).hexdigest() == digest
    source.close()


def test_future_frozen_records_do_not_change_prior_state(contract):
    before = CandidateStateSource(contract)("Known Fighter", "2026-02-01")
    data = json.loads(contract.read_text())
    data["entries"].append({**data["entries"][-1], "date": "2026-02-01", "outcome": "L", "sig_landed": 99999})
    contract.write_text(json.dumps(data))
    after = CandidateStateSource(contract)("Known Fighter", "2026-02-01")
    assert before == after
    assert set(COUNTERS) <= set(after)


def test_unknown_identity_and_wrong_profile_block(contract):
    source = CandidateStateSource(
        contract,
        live=True,
        html_getter=lambda url: '<span class="b-content__title-highlight">Someone Else</span>',
    )
    with pytest.raises(ValueError, match="outside frozen"):
        source("Unknown Fighter", "2026-02-01")
    with pytest.raises(ValueError, match="identity mismatch"):
        source("Known Fighter", "2026-02-01")
    source.close()


def test_incomplete_future_totals_block(contract):
    html = (
        '<span class="b-content__title-highlight">Known Fighter</span><table>'
        + history_row("Feb. 01, 2026")
        + "</table>"
    )
    source = CandidateStateSource(
        contract, live=True, html_getter=lambda url: html if url == URL else "<html></html>"
    )
    with pytest.raises(ValueError, match="Missing totals"):
        source("Known Fighter", pd.Timestamp("2026-02-02"))
    source.close()


def test_missing_declared_candidate_contract_cannot_fall_back(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"input_constructor": "candidate_v2"}))
    with pytest.raises(FileNotFoundError):
        state_source_for_models(tmp_path / "models")


def test_shadow_verifies_candidate_contract_and_runtime(monkeypatch, tmp_path):
    from ufc_pred.cli import shadow_runner

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    contract = bundle / "candidate_inputs.json"
    contract.write_text("{}")
    runtime = tmp_path / "runtime.py"
    runtime.write_text("frozen = True\n")
    (bundle / "runtime_manifest.json").write_text(
        json.dumps({"runtime.py": hashlib.sha256(runtime.read_bytes()).hexdigest()})
    )
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "input_constructor": "candidate_v2",
                "files": {"bundle/candidate_inputs.json": hashlib.sha256(contract.read_bytes()).hexdigest()},
            }
        )
    )
    monkeypatch.setattr(shadow_runner, "ROOT", tmp_path)
    monkeypatch.setattr(shadow_runner, "BUNDLE", bundle)
    assert shadow_runner.verify_bundle()["input_constructor"] == "candidate_v2"
    contract.write_text('{"changed": true}')
    with pytest.raises(ValueError, match="Frozen bundle changed"):
        shadow_runner.verify_bundle()
    contract.write_text("{}")
    runtime.write_text("frozen = False\n")
    with pytest.raises(ValueError, match="Frozen shadow runtime changed"):
        shadow_runner.verify_bundle()

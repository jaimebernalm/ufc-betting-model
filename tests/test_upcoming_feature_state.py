from __future__ import annotations

from datetime import datetime

import pandas as pd
from bs4 import BeautifulSoup

from ufc_pred.inference.upcoming_builder import _last_fighter_stats
from ufc_pred.ingest.ufcstats_scraper import (
    EventMeta,
    FighterStats,
    FightInfo,
    _pre_event_career_rates,
    _row,
)


def test_previous_row_fallback_advances_all_recoverable_counters():
    fights = pd.DataFrame(
        [
            {
                "date": "2026-01-01",
                "R_fighter": "A Fighter",
                "B_fighter": "B Fighter",
                "Winner": "Red",
                "title_bout": True,
                "finish": "U-DEC",
                "finish_round": 5,
                "R_wins": 4,
                "R_losses": 1,
                "R_current_win_streak": 2,
                "R_current_lose_streak": 0,
                "R_longest_win_streak": 3,
                "R_total_rounds_fought": 12,
                "R_total_title_bouts": 1,
                "R_win_by_Decision_Unanimous": 1,
                "R_avg_SIG_STR_landed": 3.25,
            }
        ]
    )

    state, _ = _last_fighter_stats("A Fighter", fights)

    assert state["wins"] == 5
    assert state["current_win_streak"] == 3
    assert state["longest_win_streak"] == 3
    assert state["total_rounds_fought"] == 17
    assert state["total_title_bouts"] == 2
    assert state["win_by_Decision_Unanimous"] == 2
    # Not recoverable from the wide row; the raw state source handles it.
    assert state["avg_SIG_STR_landed"] == 3.25


def test_raw_career_rates_exclude_target_bout_and_use_attempt_denominators():
    profile_html = """
    <table><tr class="b-fight-details__table-row">
      <td><a href="http://ufcstats.com/fight-details/prior">win</a></td>
      <td><p class="b-fight-details__table-text">win</p></td>
      <td><p class="b-fight-details__table-text">Fighter A</p><p class="b-fight-details__table-text">Fighter B</p></td>
      <td><p class="b-fight-details__table-text">0</p><p class="b-fight-details__table-text">0</p></td>
      <td><p class="b-fight-details__table-text">30</p><p class="b-fight-details__table-text">10</p></td>
      <td><p class="b-fight-details__table-text">2</p><p class="b-fight-details__table-text">0</p></td>
      <td><p class="b-fight-details__table-text">1</p><p class="b-fight-details__table-text">0</p></td>
      <td><p class="b-fight-details__table-text">Event</p></td>
      <td><p class="b-fight-details__table-text">Jan. 01, 2025</p></td>
      <td><p class="b-fight-details__table-text">U-DEC</p></td>
      <td><p class="b-fight-details__table-text">unused</p></td>
      <td><p class="b-fight-details__table-text">1</p></td>
      <td><p class="b-fight-details__table-text">5:00</p></td>
    </tr></table>
    """
    detail_html = """
    <table><tbody class="b-fight-details__table-body"><tr>
      <td><p><a href="http://ufcstats.com/fighter-details/a">Fighter A</a></p>
          <p><a href="http://ufcstats.com/fighter-details/b">Fighter B</a></p></td>
      <td><p>0</p><p>0</p></td>
      <td><p>30 of 60</p><p>10 of 25</p></td>
      <td><p>50%</p><p>40%</p></td>
      <td><p>40 of 70</p><p>20 of 35</p></td>
      <td><p>2 of 4</p><p>0 of 1</p></td>
      <td><p>50%</p><p>0%</p></td>
      <td><p>1</p><p>0</p></td>
    </tr></tbody></table>
    """

    rates = _pre_event_career_rates(
        BeautifulSoup(profile_html, "html.parser"),
        datetime(2026, 1, 1),
        "http://ufcstats.com/fighter-details/a",
        lambda _url: detail_html,
    )

    assert rates == (6.0, 0.5, 6.0, 0.5, 3.0)


def test_event_row_does_not_subtract_current_title_bout_twice():
    stats = FighterStats(
        stance="Orthodox",
        height_cms=180.0,
        reach_cms=182.0,
        weight_lbs=155,
        age=30,
        avg_sig_str_landed=3.0,
        avg_sig_str_pct=0.5,
        avg_td_landed=1.0,
        avg_td_pct=0.4,
        avg_sub_att=0.5,
        wins=5,
        losses=1,
        draws=0,
        current_win_streak=2,
        current_lose_streak=0,
        longest_win_streak=3,
        total_rounds_fought=10,
        total_title_bouts=1,
        win_by_dec_majority=0,
        win_by_dec_split=0,
        win_by_dec_unanimous=2,
        win_by_ko=2,
        win_by_sub=1,
        win_by_tko_doctor=1,
    )
    event = EventMeta(datetime(2026, 1, 1), "Las Vegas, USA", "USA", [])
    fight = FightInfo(
        "A",
        "B",
        "a-url",
        "b-url",
        "Red",
        "Lightweight",
        True,
        5,
        "U-DEC",
        None,
        5.0,
        "5:00",
    )

    built = _row(event, fight, stats, stats)

    assert built["R_total_title_bouts"] == 1
    assert built["B_total_title_bouts"] == 1
    assert built["total_title_bout_dif"] == 0
    assert built["R_win_by_TKO_Doctor_Stoppage"] == 1

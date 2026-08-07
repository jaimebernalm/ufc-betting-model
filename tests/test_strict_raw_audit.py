import pandas as pd

from scripts.research.strict_raw_retrain_audit import _build_counter_states


def test_strict_counter_states_are_shifted_before_target_bout():
    results = pd.DataFrame(
        [
            {
                "BOUT": "A Fighter vs. B Fighter",
                "OUTCOME": "W/L",
                "date": pd.Timestamp("2025-01-01"),
                "ROUND": 3,
                "WEIGHTCLASS": "Lightweight Bout",
                "METHOD": "Decision - Unanimous",
            },
            {
                "BOUT": "A Fighter vs. C Fighter",
                "OUTCOME": "W/L",
                "date": pd.Timestamp("2026-01-01"),
                "ROUND": 2,
                "WEIGHTCLASS": "Lightweight Title Bout",
                "METHOD": "TKO - Doctor's Stoppage",
            },
            {
                "BOUT": "A Fighter vs. D Fighter",
                "OUTCOME": "L/W",
                "date": pd.Timestamp("2026-06-01"),
                "ROUND": 1,
                "WEIGHTCLASS": "Lightweight Bout",
                "METHOD": "Submission",
            },
        ]
    )

    states = _build_counter_states(results)
    a = states[states["fighter_key"] == "a fighter"].set_index("date")

    before_title = a.loc[pd.Timestamp("2026-01-01")]
    assert before_title["wins"] == 1
    assert before_title["total_rounds_fought"] == 3
    assert before_title["total_title_bouts"] == 0
    assert before_title["win_by_TKO_Doctor_Stoppage"] == 0

    after_title = a.loc[pd.Timestamp("2026-06-01")]
    assert after_title["wins"] == 2
    assert after_title["current_win_streak"] == 2
    assert after_title["total_rounds_fought"] == 5
    assert after_title["total_title_bouts"] == 1
    assert after_title["win_by_TKO_Doctor_Stoppage"] == 1

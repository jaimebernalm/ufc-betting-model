import numpy as np
import pandas as pd

from scripts.research.strict_multivenue_postcutoff import _effective_decimal
from scripts.research.temporal_market_edge_audit import add_venue_metrics, calendar_coordinate


def test_market_specific_fee_schedule_changes_effective_payout():
    price = np.array([0.5, 0.5])
    rate = np.array([0.0, 0.05])
    exponent = np.array([1.0, 1.0])

    payout = _effective_decimal(price, rate, exponent)

    assert payout[0] == 2.0
    assert payout[1] == 1.0 / (0.5 + 0.05 * 0.5 * 0.5)
    assert payout[1] < payout[0]


def test_venue_detail_uses_probability_edge_and_fee_adjusted_return():
    frame = pd.DataFrame(
        {
            "Winner": ["Red", "Blue"],
            "price_red": [0.50, 0.50],
            "price_blue": [0.50, 0.50],
            "fee_rate": [0.05, 0.05],
            "fee_exponent": [1.0, 1.0],
            "p_red_real": [0.60, 0.51],
        }
    )

    result = add_venue_metrics(frame, "real")

    assert result["bet"].tolist() == [True, False]
    assert result.loc[0, "bet_red"]
    assert (
        result.loc[0, "pnl"]
        == _effective_decimal(np.array([0.5]), np.array([0.05]), np.array([1.0]))[0] - 1.0
    )
    assert np.isnan(result.loc[1, "pnl"])


def test_calendar_coordinate_spans_zero_to_one():
    dates = pd.Series(pd.to_datetime(["2026-01-01", "2026-01-06", "2026-01-11"]))

    coordinate = calendar_coordinate(dates)

    np.testing.assert_allclose(coordinate, [0.0, 0.5, 1.0])

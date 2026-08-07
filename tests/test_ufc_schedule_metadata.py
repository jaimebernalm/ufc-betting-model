import pandas as pd
from bs4 import BeautifulSoup

from ufc_pred.ingest.ufc_schedule import _parse_section_fights


def _fight(red: str, blue: str, class_text: str) -> str:
    return f"""
    <div class="c-listing-fight">
      <div class="c-listing-fight__corner-name--red">{red}</div>
      <div class="c-listing-fight__corner-name--blue">{blue}</div>
      <div class="details-content__class">{class_text}</div>
    </div>
    """


def test_schedule_preserves_title_and_five_round_main_event_metadata():
    # Display order is main event first; parser reverses to chronology.
    html = (
        '<section id="main-card">'
        + _fight("Champion", "Challenger", "UFC Lightweight Title Bout")
        + _fight("Feature Red", "Feature Blue", "Welterweight Bout")
        + "</section>"
    )
    section = BeautifulSoup(html, "html.parser").find("section")

    fights = _parse_section_fights(
        section,
        section_start_utc=pd.Timestamp("2026-01-01T00:00:00Z"),
        card_position="main_card",
    )

    assert fights[0].fighter_a == "Feature Red"
    assert fights[0].weight_class == "Welterweight"
    assert fights[0].no_of_rounds == 3
    assert fights[1].fighter_a == "Champion"
    assert fights[1].weight_class == "Lightweight"
    assert fights[1].title_bout is True
    assert fights[1].no_of_rounds == 5

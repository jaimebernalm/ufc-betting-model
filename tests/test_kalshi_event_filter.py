"""Regression tests for `_is_real_ufc`.

Kalshi dropped the "UFC " prefix from event titles around 2026-06-20. The
original prefix-only check then rejected every event from that date onward
(99 of 572), silently disabling the snapshot-backfill path. These tests pin
both the shapes that must be accepted and the non-UFC promotion that must not.
"""

import pytest

from ufc_pred.ingest.kalshi_history import _is_real_ufc


@pytest.mark.parametrize(
    "title",
    [
        # legacy, prefixed
        "UFC Fight Night: Brener vs Ribovics",
        "UFC 324: Gaethje vs Pimblett",
        "UFC Freedom 250: Someone vs Someone",
        # post-2026-06-20, prefix dropped
        "Fight Night: Blachowicz vs Stirling",
        "329: Ankalaev vs Guskov",
        "Krylov vs Guskov",
        "McGregor vs. Holloway 2",
    ],
)
def test_accepts_real_ufc_titles(title):
    assert _is_real_ufc({"title": title}) is True


@pytest.mark.parametrize(
    "title",
    [
        "Netflix MMA Special: Rousey vs Carano",
        "",
        "Some Random Boxing Card",
    ],
)
def test_rejects_non_ufc(title):
    assert _is_real_ufc({"title": title}) is False


def test_missing_title_key():
    assert _is_real_ufc({}) is False


def test_denylist_beats_structural_accept():
    """A non-UFC promotion still contains ' vs' — the denylist must win."""
    assert _is_real_ufc({"title": "Netflix MMA Special: A vs B"}) is False

"""How long a dead wall keeps its label.

The grace is proportional to the visible span with a floor, and the floor is
the part that bites: at 30s it governed every live view, because a 4-minute
span is only 12s at 5%. A label then marked a level the market had left --
price moves a median 19 cents in 30s on SOXL -- while holding one of the
top_n slots that a live wall would otherwise take.

These pin the arithmetic rather than the rendering: which of the two terms
wins at each span, and that a long replay is untouched by the change.
"""

import pytest

from analysis.liq_hm_window import (          # noqa: E402
    _WALL_MAX_EMPTY_FRAC, _WALL_MAX_EMPTY_SECS,
)


def grace(span_secs):
    """The window's own rule: max(floor, span * frac)."""
    return max(_WALL_MAX_EMPTY_SECS, span_secs * _WALL_MAX_EMPTY_FRAC)


class TestTheProportionGoverns:
    def test_a_live_four_minute_span(self):
        """The case the floor used to override."""
        assert grace(4 * 60) == pytest.approx(12.0)

    def test_a_ten_minute_span(self):
        assert grace(10 * 60) == pytest.approx(30.0)

    def test_an_hour(self):
        assert grace(3600) == pytest.approx(180.0)

    def test_a_day_long_replay_keeps_its_history(self):
        """Unchanged by lowering the floor -- the proportion already won."""
        assert grace(6.5 * 3600) == pytest.approx(1170.0)


class TestTheFloorStillProtects:
    def test_a_tightly_zoomed_span_does_not_cut_to_nothing(self):
        assert grace(30) == _WALL_MAX_EMPTY_SECS
        assert grace(30) > 0

    def test_the_crossover_is_around_two_minutes(self):
        below = 100
        above = 120
        assert grace(below) == _WALL_MAX_EMPTY_SECS
        assert grace(above) > _WALL_MAX_EMPTY_SECS


class TestItIsFasterThanItWas:
    def test_the_live_span_lost_the_thirty_second_floor(self):
        """Guards the regression: a floor at or above 12 reinstates it."""
        assert _WALL_MAX_EMPTY_SECS < 4 * 60 * _WALL_MAX_EMPTY_FRAC

    def test_the_floor_is_still_long_enough_to_read(self):
        assert _WALL_MAX_EMPTY_SECS >= 3.0

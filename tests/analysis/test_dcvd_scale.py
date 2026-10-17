"""The DCVD panel's scale has to follow its window.

DCVD is sum(buy-sell)/sum(buy+sell) over N bars, bounded by construction,
so the panel is deliberately NOT auto-ranged: 0.4 has to mean the same thing
on a quiet day as a busy one. But it was pinned at +/-1.05 with guides at
+/-0.5 for every window, and at the shipped 10-bar setting the median
|DCVD| is 0.148 -- the series used a seventh of the panel and reached a
guide line 2.4% of the time.

So the range is fixed PER WINDOW from measured quantiles. These tests pin
the two properties that matter: it shrinks as the window grows, and it
always leaves the typical reading visible without clipping the extremes.
"""
import pytest

from analysis.trade_viewer_qt import DCVD_SCALE, dcvd_scale


class TestMeasuredTable:
    def test_every_entry_is_ordered_p75_p90_p99(self):
        for n, (p75, p90, p99) in DCVD_SCALE.items():
            assert 0 < p75 < p90 < p99 < 1, f"window {n} is not ordered"

    def test_a_longer_window_is_always_flatter(self):
        """Averaging more bars cannot widen the distribution."""
        keys = sorted(DCVD_SCALE)
        for a, b in zip(keys, keys[1:]):
            for i in range(3):
                assert DCVD_SCALE[b][i] < DCVD_SCALE[a][i], \
                    f"window {b} is not flatter than {a} at quantile {i}"


class TestInterpolation:
    def test_measured_windows_come_back_exactly(self):
        for n, want in DCVD_SCALE.items():
            assert dcvd_scale(n) == want

    def test_between_two_measured_windows_it_interpolates(self):
        lo, hi = dcvd_scale(5), dcvd_scale(10)
        mid = dcvd_scale(7)
        for i in range(3):
            assert hi[i] < mid[i] < lo[i]

    def test_outside_the_table_it_clamps(self):
        assert dcvd_scale(1) == dcvd_scale(0)
        assert dcvd_scale(30) == dcvd_scale(500)

    @pytest.mark.parametrize("n", [1, 2, 3, 5, 7, 10, 15, 22, 30])
    def test_the_axis_leaves_the_typical_reading_visible(self, n):
        """The axis is p99*1.05. The p75 guide has to sit well inside it --
        the old fixed +/-1.05 put the typical reading in the middle 14%."""
        p75, p90, p99 = dcvd_scale(n)
        axis = p99 * 1.05
        assert p75 / axis > 0.25, \
            f"window {n}: the p75 guide is only {100*p75/axis:.0f}% up the panel"
        assert p90 < axis, f"window {n}: the p90 guide falls outside the axis"


class TestDefault:
    def test_the_default_window_is_short_enough_to_discriminate(self):
        """Measured on the drill set: a 1-bar reading separates outcomes by
        13.1 points, 5 bars by 7.2, 15 by 1.5. The shipped default was 10,
        which sits in the part of that range where most of the signal is
        already gone."""
        from analysis.trade_viewer_qt import DCVD_LOOKBACK
        assert DCVD_LOOKBACK <= 5

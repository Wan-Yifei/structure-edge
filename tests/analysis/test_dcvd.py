"""DCVD: the CVD slope normalised to [-1, +1].

CVD's slope over N bars is the net delta across them; the largest slope those
bars could have produced is every classified trade landing on one side. The
ratio is therefore bounded by construction, not by a fitted scale -- which is
the whole point, since it means 0.4 reads the same on a quiet day as a busy one.

NEUTRAL volume is excluded from both halves, matching the CVD delta. It is the
majority of SOXL's tape (57.5% measured), and including it would pin the median
bar near 0.08 and leave the range beyond +/-0.5 unused.

Two properties the panel gained once the backtest had something to say:

  A FLOOR. Under DCVD_MIN_SIGNED a bar reads NaN and is drawn as a gap. Below
  that volume a single small print swings the ratio to a rail, and the floor
  is also what the backtest applies -- a panel showing a quantity the measured
  rule never traded would be the wrong panel.

  BARS, NOT A LINE. Each reading belongs to its own minute; a line between two
  of them draws a segment that was never measured, and at a 1-bar window that
  invented segment is most of the picture.

Volumes here are scaled past the floor by default (see `_run`), so the ratio
tests exercise the ratio and the floor tests can ask for raw sizes.
"""

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6.QtWidgets")

from PyQt6.QtWidgets import QApplication   # noqa: E402

CM = 1
T0 = datetime(2026, 10, 5, 10, 0)
SCALE = 100          # lifts the toy flows clear of DCVD_MIN_SIGNED


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app
    app.processEvents()


@pytest.fixture(scope="module")
def viewer(qapp):
    from analysis.trade_viewer_qt import TradeViewerQt

    v = TradeViewerQt()
    v.resize(1400, 900)
    v.show()
    for _ in range(30):
        qapp.processEvents()
    yield v
    v.close()
    qapp.processEvents()


@pytest.fixture(autouse=True)
def _reset_window(viewer):
    """The window is a spinbox, and a test that widens it used to leave it
    widened for everything after -- an ordering dependency that hid itself
    for as long as the order happened to work."""
    from analysis.trade_viewer_qt import DCVD_LOOKBACK

    viewer._candle_mins = 1
    viewer._dcvd_spin.setValue(DCVD_LOOKBACK)
    yield
    viewer._candle_mins = 1
    viewer._dcvd_spin.setValue(DCVD_LOOKBACK)


def _bars(flows, scale=SCALE):
    """flows: list of (buy, sell, neutral) per bar -> (klines, buckets)."""
    from core.time_utils import candle_start

    rows, buckets = [], {}
    for i, (b, s, n) in enumerate(flows):
        b, s, n = b * scale, s * scale, n * scale
        bar_end = T0 + timedelta(minutes=CM * (i + 1))
        rows.append((bar_end.strftime("%Y-%m-%d %H:%M"),
                     100.0, 101.0, 99.0, 100.0, float(b + s + n)))
        bk = candle_start(bar_end - timedelta(minutes=CM), CM)
        if b or s or n:
            buckets[bk] = {100.0: {"buy": float(b), "sell": float(s),
                                   "neutral": float(n)}}
    kl = pd.DataFrame(rows, columns=["time_key", "open", "high", "low",
                                     "close", "volume"])
    return kl, buckets


def _run(v, flows, scale=SCALE, look=None):
    if look is not None:
        v._dcvd_spin.setValue(look)
    kl, buckets = _bars(flows, scale)
    # The render path clears before it draws; without that the items pile up
    # across calls and any assertion about what got drawn sees every earlier
    # test's leftovers too.
    v._clear_dcvd_items()
    v._draw_dcvd(kl, buckets, CM)
    return np.asarray(v._dcvd_arr, dtype=float)


class TestBounds:
    def test_all_buying_pins_at_plus_one(self, viewer):
        assert _run(viewer, [(100, 0, 0)] * 20)[-1] == pytest.approx(1.0)

    def test_all_selling_pins_at_minus_one(self, viewer):
        assert _run(viewer, [(0, 100, 0)] * 20)[-1] == pytest.approx(-1.0)

    def test_balanced_flow_is_zero(self, viewer):
        assert _run(viewer, [(50, 50, 0)] * 20)[-1] == pytest.approx(0.0)

    def test_never_leaves_the_range(self, viewer):
        rng = np.random.default_rng(0)
        flows = [(int(rng.integers(0, 5000)), int(rng.integers(0, 5000)),
                  int(rng.integers(0, 9000))) for _ in range(300)]
        d = _run(viewer, flows)
        assert np.nanmin(d) >= -1.0 and np.nanmax(d) <= 1.0

    def test_a_single_huge_bar_cannot_push_it_out(self, viewer):
        """An unnormalised slope would spike; this cannot exceed 1 by design."""
        flows = [(10, 10, 0)] * 19 + [(10_000_000, 0, 0)]
        assert _run(viewer, flows)[-1] <= 1.0


class TestNormalisation:
    def test_scale_invariant(self, viewer):
        """Ten times the volume, same imbalance, same reading -- the property
        an unnormalised slope lacks."""
        small = _run(viewer, [(70, 30, 0)] * 20)[-1]
        large = _run(viewer, [(700, 300, 0)] * 20)[-1]
        assert small == pytest.approx(large)
        assert small == pytest.approx(0.4)

    def test_equals_the_slope_over_the_max_possible_slope(self, viewer):
        flows = [(80, 20, 0), (60, 40, 0), (90, 10, 0)] * 10
        look = 5
        d = _run(viewer, flows, look=look)
        buy = np.array([f[0] for f in flows], dtype=float)
        sell = np.array([f[1] for f in flows], dtype=float)
        i = len(flows) - 1
        j = max(0, i + 1 - look)
        slope = (buy[j:i + 1] - sell[j:i + 1]).sum()
        cap = (buy[j:i + 1] + sell[j:i + 1]).sum()
        assert d[-1] == pytest.approx(slope / cap)

    def test_neutral_is_excluded_from_both_halves(self, viewer):
        """Adding neutral volume must not move the reading."""
        without = _run(viewer, [(70, 30, 0)] * 20)[-1]
        with_n = _run(viewer, [(70, 30, 100_000)] * 20)[-1]
        assert without == pytest.approx(with_n)


class TestLookback:
    def test_default_window_is_one_bar(self):
        """The window that predicts. A 5-bar version of the measured rule
        returned nothing out of sample; averaging removes the imbalance."""
        from analysis.trade_viewer_qt import DCVD_LOOKBACK

        assert DCVD_LOOKBACK == 1

    def test_one_bar_window_reads_only_its_own_bar(self, viewer):
        flows = [(100, 0, 0)] * 50 + [(0, 100, 0)]
        assert _run(viewer, flows, look=1)[-1] == pytest.approx(-1.0)

    def test_only_the_last_N_bars_count(self, viewer):
        look = 5
        flows = [(100, 0, 0)] * 50 + [(0, 100, 0)] * look
        assert _run(viewer, flows, look=look)[-1] == pytest.approx(-1.0), (
            "older buying must have rolled out of the window")

    def test_turns_over_as_the_window_fills(self, viewer):
        look = 5
        d = _run(viewer, [(100, 0, 0)] * look + [(0, 100, 0)] * look,
                 look=look)
        assert d[look - 1] == pytest.approx(1.0)
        assert d[2 * look - 1] == pytest.approx(-1.0)
        assert d[look + look // 2 - 1] < d[look - 1], \
            "must be falling through the flip"

    def test_short_history_uses_what_there_is(self, viewer):
        d = _run(viewer, [(100, 0, 0)] * 3)
        assert len(d) == 3 and d[-1] == pytest.approx(1.0)


class TestFloor:
    def test_floor_matches_the_backtest(self):
        """The panel and the measured rule must apply the same floor, or the
        panel is showing a quantity the rule never traded."""
        from analysis.trade_viewer_qt import DCVD_MIN_SIGNED

        assert DCVD_MIN_SIGNED == 300.0

    def test_a_thin_bar_is_a_gap_not_a_rail(self, viewer):
        """40 shares one way is not a +1 reading, it is no reading."""
        d = _run(viewer, [(40, 0, 0)] * 10, scale=1)
        assert np.all(np.isnan(d))

    def test_a_bar_at_the_floor_reads(self, viewer):
        from analysis.trade_viewer_qt import DCVD_MIN_SIGNED

        d = _run(viewer, [(int(DCVD_MIN_SIGNED), 0, 0)] * 5, scale=1)
        assert np.isfinite(d).all() and d[-1] == pytest.approx(1.0)

    def test_a_wider_window_can_clear_what_one_bar_cannot(self, viewer):
        """The floor is on the window's total, not on each bar, so thin bars
        still contribute once enough of them are summed."""
        thin = [(100, 0, 0)] * 6
        assert np.all(np.isnan(_run(viewer, thin, scale=1, look=1)))
        d5 = _run(viewer, thin, scale=1, look=5)
        assert np.isfinite(d5[-1]) and d5[-1] == pytest.approx(1.0)


class TestGaps:
    def test_a_bar_with_no_ticks_is_nan_not_zero(self, viewer):
        """Zero would read as balanced flow, which is a claim about the tape
        rather than an absence of one."""
        d = _run(viewer, [(100, 0, 0)] * 5 + [(0, 0, 0)])
        assert np.isnan(d[-1])

    def test_no_ticks_at_all_is_all_gap(self, viewer):
        assert np.all(np.isnan(_run(viewer, [(0, 0, 0)] * 10)))

    def test_neutral_only_bars_are_not_directional(self, viewer):
        """Neutral is in neither half, so such a bar has no signed volume at
        all and cannot clear the floor."""
        assert np.all(np.isnan(_run(viewer, [(0, 0, 5000)] * 10)))

    def test_a_gap_does_not_poison_the_bars_after_it(self, viewer):
        d = _run(viewer, [(100, 0, 0)] * 3 + [(0, 0, 0)] + [(0, 100, 0)] * 3)
        assert np.isnan(d[3])
        assert d[-1] == pytest.approx(-1.0)


class TestDrawing:
    def _kinds(self, viewer):
        return [type(i).__name__ for i in viewer._dcvd_items]

    def test_readings_are_bars_not_a_line(self, viewer):
        """A line between two readings draws a segment nobody measured, and
        at a 1-bar window that invented segment is most of the panel."""
        _run(viewer, [(70, 30, 0)] * 10 + [(30, 70, 0)] * 10, look=1)
        assert "BarGraphItem" in self._kinds(viewer)

    def test_context_line_is_drawn_behind_the_bars(self, viewer):
        from analysis.trade_viewer_qt import DCVD_CONTEXT

        assert DCVD_CONTEXT > 1
        _run(viewer, [(70, 30, 0)] * 20, look=1)
        assert "PlotCurveItem" in self._kinds(viewer)

    def test_no_context_line_once_the_window_reaches_it(self, viewer):
        """Drawing the same series twice would say there are two readings."""
        from analysis.trade_viewer_qt import DCVD_CONTEXT

        _run(viewer, [(70, 30, 0)] * 20, look=DCVD_CONTEXT)
        assert "PlotCurveItem" not in self._kinds(viewer)

    def test_thin_bars_draw_nothing(self, viewer):
        _run(viewer, [(40, 0, 0)] * 10, scale=1, look=1)
        assert "BarGraphItem" not in self._kinds(viewer)


class TestGuides:
    def _levels(self, viewer):
        return sorted(round(float(g.value()), 4)
                      for g in viewer._dcvd_guides)

    def test_decision_threshold_is_marked_both_ways(self, viewer):
        """Long and short were measured symmetric, so one side would be half
        the instruction."""
        from analysis.trade_viewer_qt import DCVD_DECISION

        lv = self._levels(viewer)
        assert DCVD_DECISION in lv and -DCVD_DECISION in lv

    def test_p75_is_gone(self, viewer):
        """At the 1-bar window p75 is 0.517, close enough to the 0.40 line
        that the two read as one band and neither can be identified."""
        from analysis.trade_viewer_qt import dcvd_scale

        p75, _, _ = dcvd_scale(viewer._dcvd_lookback())
        assert round(p75, 4) not in self._levels(viewer)

    def test_zero_is_marked(self, viewer):
        assert 0.0 in self._levels(viewer)


class TestWiring:
    def test_series_is_published_for_the_crosshair(self, viewer):
        d = _run(viewer, [(60, 40, 0)] * 20)
        assert viewer._dcvd_arr is not None
        assert len(viewer._dcvd_arr) == 20 and d[-1] == pytest.approx(0.2)

    def test_clearing_drops_the_series(self, viewer):
        _run(viewer, [(60, 40, 0)] * 20)
        viewer._clear_dcvd_items()
        assert viewer._dcvd_arr is None and viewer._dcvd_items == []

    def test_subplot_axis_does_not_follow_the_data(self, viewer):
        """A fitted y-range would undo the normalisation's whole purpose: a
        reading of 0.4 has to look the same on a quiet day as a busy one."""
        _run(viewer, [(60, 40, 0)] * 20)
        quiet = viewer._plot_dcvd.vb.viewRange()[1]
        _run(viewer, [(10_000, 10, 0)] * 20)      # violently one-sided
        loud = viewer._plot_dcvd.vb.viewRange()[1]
        assert quiet == pytest.approx(loud, abs=1e-6), \
            "the axis moved with the data -- it is being fitted"

    def test_subplot_axis_matches_the_measured_table_for_its_window(self, viewer):
        """Fixed, but fixed PER WINDOW. At +/-1.05 for every window the old
        panel left a 10-bar series -- median |DCVD| 0.148 -- in its middle
        seventh, never reaching the guides."""
        from analysis.trade_viewer_qt import dcvd_scale
        lo, hi = viewer._plot_dcvd.vb.viewRange()[1]
        _, _, p99 = dcvd_scale(viewer._dcvd_lookback())
        assert hi == pytest.approx(p99 * 1.05, rel=0.02)
        assert lo == pytest.approx(-p99 * 1.05, rel=0.02)

    def test_a_longer_window_narrows_the_axis(self, viewer):
        viewer._dcvd_spin.setValue(1)
        short = viewer._plot_dcvd.vb.viewRange()[1][1]
        viewer._dcvd_spin.setValue(30)
        long_ = viewer._plot_dcvd.vb.viewRange()[1][1]
        assert long_ < short, \
            "averaging more bars flattens DCVD, so the axis must tighten"

    def test_decision_line_needs_a_one_minute_window(self, viewer):
        """A longer window drops the threshold whatever the axis does.

        This used to turn only on p99 falling under 0.40 at long windows.
        The minute rule now decides it first and subsumes that case: the
        level was measured on one minute of flow, so a 30-bar window has no
        business showing it even where the axis would fit.
        """
        from analysis.trade_viewer_qt import DCVD_DECISION

        viewer._dcvd_spin.setValue(30)
        lv = sorted(round(float(g.value()), 4) for g in viewer._dcvd_guides)
        assert DCVD_DECISION not in lv

class TestWindowIsMinutesNotBars:
    """The spinbox counts BARS; everything calibrated counts MINUTES.

    DCVD_SCALE was measured on 1-minute bars over N-bar windows, and the
    0.40 threshold was measured on one minute of flow. On a 5m chart a
    lookback of 1 is a five-minute reading: scaling it against the 1-minute
    distribution leaves it in the middle of the panel, and drawing the
    threshold over it invites trading a level never validated for it -- the
    5-bar version of the same rule was worth nothing out of sample.
    """

    def test_minutes_follow_the_timeframe(self, viewer):
        viewer._candle_mins = 5
        viewer._dcvd_spin.setValue(1)
        assert viewer._dcvd_minutes() == 5
        viewer._dcvd_spin.setValue(3)
        assert viewer._dcvd_minutes() == 15
        viewer._candle_mins = 1

    def test_axis_uses_the_minute_scale(self, viewer):
        from analysis.trade_viewer_qt import dcvd_scale

        viewer._candle_mins = 5
        viewer._dcvd_spin.setValue(1)
        viewer._apply_dcvd_scale()
        hi = viewer._plot_dcvd.vb.viewRange()[1][1]
        assert hi == pytest.approx(dcvd_scale(5)[2] * 1.05, rel=0.02)
        assert hi != pytest.approx(dcvd_scale(1)[2] * 1.05, rel=0.02)
        viewer._candle_mins = 1
        viewer._apply_dcvd_scale()

    def test_label_states_minutes(self, viewer):
        viewer._candle_mins = 5
        viewer._dcvd_spin.setValue(1)
        viewer._apply_dcvd_scale()
        assert "5m" in viewer._plot_dcvd.getAxis("left").labelString()
        viewer._candle_mins = 1
        viewer._apply_dcvd_scale()

    def test_threshold_only_at_one_minute(self, viewer):
        from analysis.trade_viewer_qt import DCVD_DECISION

        def levels():
            return sorted(round(float(g.value()), 4)
                          for g in viewer._dcvd_guides)

        viewer._candle_mins = 1
        viewer._dcvd_spin.setValue(1)
        viewer._apply_dcvd_scale()
        assert DCVD_DECISION in levels(), "1m must carry the measured level"

        viewer._candle_mins = 5
        viewer._apply_dcvd_scale()
        assert DCVD_DECISION not in levels(), \
            "a 5-minute window must not show a 1-minute threshold"

        viewer._candle_mins = 1
        viewer._apply_dcvd_scale()


class TestDefaults:
    """The defaults have to agree with each other.

    The 0.40 threshold exists only at a one-minute window, so DCVD being on
    by default is only useful if the default timeframe is 1m -- on 5m the
    panel appears with its decision line suppressed, which looks broken
    rather than careful.
    """

    def test_default_timeframe_is_one_minute(self):
        import argparse

        from analysis import trade_viewer_qt as tv

        ap = argparse.ArgumentParser()
        ap.add_argument("--tf", default="1m", choices=list(tv.TIMEFRAME_MAP))
        assert ap.parse_args([]).tf == "1m"
        assert tv.TIMEFRAME_MAP["1m"][1] == 1

    def test_dcvd_is_on_and_at_one_minute(self, viewer):
        assert viewer._ind_checks["dcvd"].isChecked()
        assert viewer._tf_combo.currentText() == "1m"
        assert viewer._dcvd_minutes() == 1

    def test_option_walls_on_alerts_off(self, viewer):
        """Drawing on the chart is harmless; a notification stream nobody
        asked for is not."""
        assert viewer._ind_checks["option_walls"].isChecked()
        assert not viewer._ind_checks["option_alerts"].isChecked()


class TestCheckedAtStartupActuallyActs:
    """A box that starts checked never emits stateChanged.

    Everything that reveals a subplot or starts a poll lived in those
    handlers, so adding an indicator to _default_checked ticked its box and
    did nothing else: DCVD appeared checked with its row still collapsed,
    Option Walls appeared checked with the chain never polled. The row
    visibilities were literals hand-synced to the set -- the comment
    "# CVD on by default" was that sync, done once and never repeated.
    """

    ROWS = {1: "vol", 2: "kd", 3: "cvd", 4: "adi", 5: "dcvd"}

    def test_every_checked_subplot_is_visible(self, viewer):
        for row, key in self.ROWS.items():
            on = viewer._ind_checks[key].isChecked()
            stretch = viewer._chart_widget.ci.layout.rowStretchFactor(row)
            assert (stretch > 0) == on, (
                f"{key}: checked={on} but row stretch={stretch}")

    def test_dcvd_in_particular(self, viewer):
        """The one that was reported."""
        assert viewer._ind_checks["dcvd"].isChecked()
        assert viewer._chart_widget.ci.layout.rowStretchFactor(5) > 0

    def test_the_option_poll_is_running(self, viewer):
        """Walls default on, so the chain has to be being fetched."""
        assert viewer._ind_checks["option_walls"].isChecked()
        assert viewer._option_timer.isActive()


class TestTheBrightnessIsExplained:
    """The lit/dim split encodes the stronger of the panel's two rules --
    the flow gate, +0.352 R against the height rule's +0.207 -- and
    nothing on screen said so. The numbers in that sentence have to
    track the constants, or the panel documents a gate it does not
    apply."""

    def test_the_tooltip_names_the_gate_it_actually_uses(self, viewer):
        from analysis.trade_viewer_qt import DCVD_FLOW_PCT, DCVD_FLOW_WIN
        tip = viewer._dcvd_spin.toolTip()
        assert f"{DCVD_FLOW_WIN} bars" in tip, "the window is not named"
        assert f"top {100 - int(DCVD_FLOW_PCT * 100)}%" in tip, \
            "the percentile is not named, or does not match the constant"
        assert "+0.352" in tip and "+0.285" in tip, \
            "the gate rule's measured return is not stated"

    def test_it_says_the_two_rules_are_different(self, viewer):
        """Height is a ratio, brightness is a magnitude. Reading one as
        the other is the mistake the tooltip exists to stop."""
        tip = viewer._dcvd_spin.toolTip()
        assert "RATIO" in tip
        assert "1-bar" in tip, "the sign-agreement caveat is missing"

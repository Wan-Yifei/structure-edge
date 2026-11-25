"""The Opt delta-Flow row's view state, not its arithmetic.

Everything this file pins was found by opening the app, not by a failing
assertion. The row's data layer has 28 tests and eight killed mutants, and
it still shipped five bugs in one afternoon -- a locked Y axis, a note
drawn thousands of bars off-screen, an axis scaled to the whole session, a
crosshair that stopped one row short, and a cached frame drawn under the
wrong symbol. None of them is a wrong number. All of them are "the number
is right and you cannot see it".

So these tests ask where things are and what state the view is in, which
is the question no other test in the repo asks.
"""
import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6.QtWidgets")


@pytest.fixture(scope="module")
def viewer():
    from PyQt6.QtWidgets import QApplication
    from analysis.trade_viewer_qt import TradeViewerQt

    app = QApplication.instance() or QApplication([])
    v = TradeViewerQt()
    # Nothing here wants a live poll overwriting the state under test.
    v._opt_flow_timer.stop()
    v._option_timer.stop()
    yield v
    v.close()
    app.processEvents()


# ── T2: the cached frame has to be about what is on screen ───────────────

class TestTheCacheMustDescribeThisChart:
    """A frame carries a symbol, a bar size and a Range. All three are
    recorded when it ARRIVES, from its own payload, so they say what the
    frame is -- not what was last asked for."""

    def _arm(self, v, code="US.SOXL", cm=1, rng="1d"):
        v._opt_flow_data = {"bars": pd.DataFrame(), "range": rng}
        v._opt_flow_code = code
        v._opt_flow_minutes = cm
        v._opt_flow_range = rng

    def test_a_matching_frame_is_current(self, viewer, monkeypatch):
        monkeypatch.setattr(viewer, "_get_range_val", lambda: "1d")
        self._arm(viewer)
        assert viewer._opt_flow_cache_is_current("US.SOXL", 1)

    def test_another_symbol_is_not(self, viewer, monkeypatch):
        monkeypatch.setattr(viewer, "_get_range_val", lambda: "1d")
        self._arm(viewer)
        assert not viewer._opt_flow_cache_is_current("US.QQQ", 1), (
            "SOXL's option flow would be drawn under QQQ's candles")

    def test_another_bar_size_is_not(self, viewer, monkeypatch):
        monkeypatch.setattr(viewer, "_get_range_val", lambda: "1d")
        self._arm(viewer)
        assert not viewer._opt_flow_cache_is_current("US.SOXL", 5), (
            "a 1-minute reduction drawn on 5-minute candles lands about a "
            "fifth of its bars and leaves the rest NaN, with nothing said")

    def test_another_range_is_not(self, viewer, monkeypatch):
        """The sequence this exists for: hide the row, change Range (the
        hook returns early while hidden), show it again. The toggle fires a
        fetch; if the new Range were stamped onto the OLD frame at that
        moment, this check would pass it."""
        self._arm(viewer, rng="1d")
        monkeypatch.setattr(viewer, "_get_range_val", lambda: "3d")
        assert not viewer._opt_flow_cache_is_current("US.SOXL", 1)

    def test_no_frame_is_never_current(self, viewer, monkeypatch):
        monkeypatch.setattr(viewer, "_get_range_val", lambda: "1d")
        viewer._opt_flow_data = None
        assert not viewer._opt_flow_cache_is_current("US.SOXL", 1)


# ── T3: the note has to be where the user is looking ─────────────────────

class TestTheNoteLandsInsideTheView:
    """The note is the row's only way of saying why it is blank. Pinned at
    bar 0 it sat thousands of bars off the left edge; pinned before
    setYRange moved the view it sat at y=-32035 inside a +/-1 view. Both
    times the panel looked empty for no stated reason, which is the exact
    failure the note exists to prevent."""

    def test_it_is_inside_the_view_after_a_real_data_range(self, viewer):
        vb = viewer._plot_oflow.vb
        viewer._clear_opt_flow_items()
        # The state that broke it: the panel had just been showing a real
        # series, so the view is nowhere near the +/-1 the note will set.
        vb.setXRange(3850, 4000, padding=0)
        vb.setYRange(-127758, 63688, padding=0)

        viewer._opt_flow_note("no option tape for US.QQQ in this range")

        lbl = viewer._opt_flow_note_item
        assert lbl is not None
        (x0, x1), (y0, y1) = vb.viewRange()
        pos = lbl.pos()
        assert x0 <= pos.x() <= x1, (pos.x(), x0, x1)
        assert y0 <= pos.y() <= y1, (pos.y(), y0, y1)

    def test_it_is_inside_the_view_when_the_chart_is_scrolled_right(self, viewer):
        """A chart holding thousands of bars opens at the newest ones, so a
        position computed in data coordinates from a fixed index is off
        screen by however far the user has scrolled."""
        vb = viewer._plot_oflow.vb
        viewer._clear_opt_flow_items()
        vb.setXRange(9000, 9200, padding=0)
        # Y as well, though only X is asserted below. Inherited, it comes
        # from whatever the previous test left -- harmless today and a
        # silent order dependency the moment anyone adds a Y assertion.
        # Same reason test_subplot_xaxis clears its ticks before measuring.
        vb.setYRange(-5000, 5000, padding=0)

        viewer._opt_flow_note("Opt Δ-Flow: loading…")

        (x0, x1), _ = vb.viewRange()
        assert x0 <= viewer._opt_flow_note_item.pos().x() <= x1

    def test_clearing_forgets_the_note(self, viewer):
        viewer._opt_flow_note("x")
        assert viewer._opt_flow_note_item is not None
        viewer._clear_opt_flow_items()
        assert viewer._opt_flow_note_item is None


# ── T1c: the stored series is cleared with the items it describes ────────

class TestTheStoredSeriesIsClearedWithTheItems:
    """_autorange_cumulative_subplots ranges each row from a stored array.
    Left behind after the items are gone, a pan scales the axis to a series
    that is not on screen -- and fights the note's own +/-1 range.

    CVD, DCVD and the A/D line are the controls: all three already cleared
    theirs, and without them a parametrised test over the one row that was
    broken proves nothing about the convention it claims to enforce."""

    @pytest.mark.parametrize("attr,clear", [
        ("_cvd_arr", "_clear_cvd_items"),
        ("_adi_arr", "_clear_adi_items"),
        ("_dcvd_arr", "_clear_dcvd_items"),
        ("_oflow_arr", "_clear_opt_flow_items"),
    ])
    def test_the_array_is_none_after_clearing(self, viewer, attr, clear):
        setattr(viewer, attr, np.arange(10, dtype=float))
        getattr(viewer, clear)()
        assert getattr(viewer, attr) is None, (
            f"{clear}() left {attr} behind")


# ── T1b: the crosshair reaches every visible row ─────────────────────────

class TestEverySubplotHasItsOwnCrosshair:
    """Each subplot carries its OWN InfiniteLine -- they are separate items,
    not one line spanning the layout -- so a new row needs registering in
    FOUR places: create+addItem, the in_* hit test, the in_any chain, and
    the setPos/setVisible update. The Opt delta row was added with none of
    them and the line simply stopped one panel short.

    The structural check below covers the first. The hover checks cover
    the other three, which is where the reported symptom actually lived:
    an earlier version of this class asserted only that the line existed
    and never called _on_mouse_move, so deleting the update line left it
    entirely green.
    """

    LINES = {
        "_vline_v": "_plot_v", "_vline_kd": "_plot_kd",
        "_vline_cvd": "_plot_cvd", "_vline_adi": "_plot_adi",
        "_vline_dcvd": "_plot_dcvd", "_vline_oflow": "_plot_oflow",
    }

    def test_each_row_has_a_line_and_it_belongs_to_that_row(self, viewer):
        for name, plot_attr in self.LINES.items():
            line = getattr(viewer, name, None)
            assert line is not None, f"{name} was never created"
            plot = getattr(viewer, plot_attr)
            assert line in plot.items, f"{name} was not added to {plot_attr}"

    def test_hovering_the_option_row_moves_its_own_line(self, viewer):
        """The three registrations a structural check cannot see."""
        from PyQt6.QtWidgets import QApplication

        viewer._ind_checks["opt_flow"].setChecked(True)
        viewer.resize(1400, 1000)
        viewer.show()
        QApplication.processEvents()
        try:
            rect = viewer._plot_oflow.sceneBoundingRect()
            if rect.width() < 10 or rect.height() < 10:
                pytest.skip("offscreen layout gave the row no area")
            viewer._on_mouse_move(rect.center())
            assert viewer._vline_oflow.isVisible(), (
                "the crosshair did not reach the Opt delta row -- check the "
                "in_any chain and the setPos/setVisible block")
            want = viewer._plot_oflow.vb.mapSceneToView(rect.center()).x()
            # 0.1 of a BAR, not a float epsilon. The handler maps every
            # non-candle row through one shared viewbox, so the measured
            # difference is ~0.02 bars -- the same bar, not the same float.
            # The tolerance still has to stay well under 1, because "off by
            # one bar" is the failure this is here to catch.
            assert abs(viewer._vline_oflow.value() - want) < 0.1
        finally:
            viewer.hide()

    def test_leaving_every_panel_hides_the_option_row_line(self, viewer):
        from PyQt6.QtCore import QPointF
        from PyQt6.QtWidgets import QApplication

        viewer._ind_checks["opt_flow"].setChecked(True)
        viewer.resize(1400, 1000)
        viewer.show()
        QApplication.processEvents()
        try:
            rect = viewer._plot_oflow.sceneBoundingRect()
            if rect.width() < 10:
                pytest.skip("offscreen layout gave the row no area")
            viewer._on_mouse_move(rect.center())
            viewer._on_mouse_move(QPointF(-5000, -5000))
            assert not viewer._vline_oflow.isVisible(), (
                "the line stayed up after the pointer left every panel -- "
                "check the hide list")
        finally:
            viewer.hide()


# ── D1: the row had bars and no way to read a value off them ─────────────

class TestTheOptionRowHasAHoverReadout:
    """Like DCVD's, with two differences that are the whole point.

    The series is NaN wherever no option print landed in the bar, so a
    number there would be an invention; and the Cumulative toggle changes
    WHICH quantity the number is, not merely its scale, so the readout
    says which.

    These call _on_mouse_move for real. A structural check -- "the label
    exists and is in the plot" -- passes with the update block deleted,
    which is exactly how the crosshair shipped one row short.
    """

    @staticmethod
    def _klines(n=400):
        """A minimal OHLCV frame.

        Required, not decoration: the whole readout block -- DCVD's as well
        as this one -- sits inside `if self._klines is not None and not
        self._klines.empty`, so without a frame every "is it visible"
        assertion reads False and every "is it hidden" assertion passes for
        the wrong reason. Found exactly that way: the four positive tests
        failed and the two negative ones went green.
        """
        return pd.DataFrame({
            "time_key": pd.date_range("2026-11-23 09:30", periods=n,
                                      freq="1min").strftime("%Y-%m-%d %H:%M"),
            "open": np.full(n, 43.0), "high": np.full(n, 43.2),
            "low": np.full(n, 42.8), "close": np.full(n, 43.1),
            "volume": np.full(n, 10_000),
        })

    def _hover(self, viewer, arr, cum=False):
        """Put a series on the row, hover its centre, return the label."""
        from PyQt6.QtWidgets import QApplication

        viewer._klines = self._klines(len(arr))
        viewer._ind_checks["opt_flow"].setChecked(True)
        if "opt_flow_cum" in viewer._ind_checks:
            viewer._ind_checks["opt_flow_cum"].setChecked(cum)
        viewer._oflow_arr = np.asarray(arr, dtype=float)
        viewer.resize(1400, 1000)
        viewer.show()
        QApplication.processEvents()
        rect = viewer._plot_oflow.sceneBoundingRect()
        if rect.width() < 10 or rect.height() < 10:
            viewer.hide()
            pytest.skip("offscreen layout gave the row no area")
        viewer._on_mouse_move(rect.center())
        return viewer._oflow_label

    def test_the_label_exists_and_belongs_to_the_option_row(self, viewer):
        assert viewer._oflow_label is not None
        assert viewer._oflow_label in viewer._plot_oflow.items

    def test_hovering_a_covered_bar_shows_a_number(self, viewer):
        try:
            lbl = self._hover(viewer, [1_500.0] * 400)
            assert lbl.isVisible(), (
                "no readout on a covered bar -- check the update block in "
                "_on_mouse_move, not just that the item exists")
            # 1.5K, not "2K". Measured on the real tape: 53% of
            # one-minute bars with flow sit between 1K and 10K (median
            # 3,165), so a whole-number K loses a third of the reading
            # right where this panel spends most of its time.
            assert lbl.toPlainText() == "+1.5K", lbl.toPlainText()
        finally:
            viewer.hide()

    def test_a_positive_value_carries_its_plus(self, viewer):
        """On a flow panel the sign is the entire reading, and
        _fmt_signed_short omits the plus."""
        try:
            lbl = self._hover(viewer, [2_000.0] * 400)
            assert lbl.toPlainText().startswith("+"), lbl.toPlainText()
        finally:
            viewer.hide()

    def test_a_negative_value_reads_negative(self, viewer):
        try:
            lbl = self._hover(viewer, [-2_000.0] * 400)
            assert lbl.toPlainText() == "-2.0K", lbl.toPlainText()
        finally:
            viewer.hide()

    def test_an_uncovered_bar_shows_nothing_rather_than_zero(self, viewer):
        """NaN means no signable print landed in the bar. A "+0" would say
        the flow was balanced, which is a claim about data that is absent."""
        try:
            lbl = self._hover(viewer, [float("nan")] * 400)
            assert not lbl.isVisible(), (
                f"readout on an uncovered bar: {lbl.toPlainText()!r}")
        finally:
            viewer.hide()

    def test_the_cumulative_toggle_marks_the_reading(self, viewer):
        """Per-bar net and a running sum differ by orders of magnitude and
        the axis label cannot say which is on."""
        if "opt_flow_cum" not in viewer._ind_checks:
            pytest.skip("no cumulative toggle in this build")
        try:
            plain = self._hover(viewer, [3_000.0] * 400, cum=False).toPlainText()
            cum = self._hover(viewer, [3_000.0] * 400, cum=True).toPlainText()
            assert "Σ" not in plain, plain
            assert "Σ" in cum, cum
            # Same number, different label -- so the marker is what changed,
            # not the formatting.
            assert plain.lstrip("+") in cum
        finally:
            viewer.hide()
            viewer._ind_checks["opt_flow_cum"].setChecked(False)

    def test_leaving_every_panel_hides_the_readout(self, viewer):
        from PyQt6.QtCore import QPointF
        try:
            lbl = self._hover(viewer, [1_000.0] * 400)
            assert lbl.isVisible()
            viewer._on_mouse_move(QPointF(-5000, -5000))
            assert not lbl.isVisible(), "check the hide list"
        finally:
            viewer.hide()

    def test_no_series_means_no_readout(self, viewer):
        from PyQt6.QtWidgets import QApplication
        try:
            viewer._klines = self._klines()
            viewer._ind_checks["opt_flow"].setChecked(True)
            viewer._oflow_arr = None
            viewer.resize(1400, 1000)
            viewer.show()
            QApplication.processEvents()
            rect = viewer._plot_oflow.sceneBoundingRect()
            if rect.width() < 10:
                pytest.skip("offscreen layout gave the row no area")
            viewer._on_mouse_move(rect.center())
            assert not viewer._oflow_label.isVisible()
        finally:
            viewer.hide()

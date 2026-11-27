"""The paint counter has to be right about the one number it reports.

WHAT IT IS FOR. analysis/qt_axis_cache.py pays unless more than 87.9% of
frames invalidate an axis, and that share is a fact about how the
application is used. On a cache hit Qt blits and never calls
AxisItem.paint, so counting paint calls counts invalidations exactly.

WHY THESE GROUPS. Each guards a way the instrument could report a
plausible number that is not the truth:

  * it could be on in the shipped application, which would make every
    later CPU reading carry an instrument nobody asked for;
  * installing twice would wrap the wrapper and double every count,
    turning a 40% hit rate into 80% with nothing to show for it;
  * it could count paints without counting frames, or count frames
    without noticing which of them repainted an axis -- the ratio is
    the whole answer, so both halves are pinned separately;
  * it could key counts by window TITLE, which changes with the symbol
    mid-session and would split one window's run across two rows;
  * it could lose the closing reading, because the viewer ends on
    os._exit and no atexit handler runs;
  * it could cost enough to matter, which would make the run it
    instruments unusable for anything else.

The frame/dirty logic is tested directly rather than through Qt's
caching: whether DeviceCoordinateCache suppresses a paint offscreen is
Qt's business and is already measured elsewhere. What must hold here is
that a paint during a frame marks that frame and a paint outside one
does not.
"""

import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6.QtWidgets")

import pyqtgraph as pg                                        # noqa: E402
from PyQt6.QtGui import QImage, QPainter                      # noqa: E402
from PyQt6.QtWidgets import QApplication                      # noqa: E402

import analysis.qt_axis_probe as P                            # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def probe(qapp):
    """Installed, counters zeroed, and always removed again.

    Without the uninstall every later test in the session would run
    through the wrapper, and a failure here would be charged to them.
    """
    P.uninstall()
    P.reset()
    # Snapshot Qt's own methods, and put them back ourselves at the end
    # rather than trusting uninstall() to. uninstall() restores whatever
    # P._orig_paint_event holds, and the tests below replace that
    # attribute with stand-ins -- one left behind would become Qt's paint
    # path for the whole session.
    saved = (pg.AxisItem.paint, pg.GraphicsView.paintEvent)
    P.install(interval_s=0)          # no timer: the test drives the reads
    yield P
    P.uninstall()
    pg.AxisItem.paint, pg.GraphicsView.paintEvent = saved
    P.reset()


class _FakeView:
    """Enough of a GraphicsView for the counters, and nothing more.

    It carries objectName() as well as window() because _view_tag asks
    for both; a stub that is missing one makes the probe look broken
    when it is the stub that is thin.
    """

    def window(self):
        return self

    def objectName(self):
        return ""


def _plot(qapp):
    """A shown, exposed plot, with the counters back at zero.

    Qt paints nothing into a widget that has not been mapped, so the
    show() has to be flushed before repaint() does anything -- that
    flush is itself a frame, which is why the counters are zeroed after
    it rather than in the fixture alone. Every test therefore starts
    from a widget where one repaint() is exactly one frame.
    """
    w = pg.PlotWidget()
    w.resize(240, 180)
    w.getPlotItem().showGrid(x=True, y=True)
    w.plot([0, 1, 2, 3], [1, 3, 2, 4])
    w.show()
    qapp.processEvents()
    P.reset()
    return w


def _render(widget) -> None:
    """Force exactly one real paintEvent, synchronously.

    QWidget.render() does NOT go through it -- it paints into the given
    device by another route, and a test built on it counts zero frames
    while looking like it drew something. repaint() on the viewport is
    the path Qt itself uses, and it returns only once the event has been
    delivered, so one call is one frame.
    """
    widget.viewport().repaint()


class TestItIsOffUnlessAsked:
    """The shipped application must carry a no-op."""

    def test_maybe_install_does_nothing_without_the_variable(self, qapp,
                                                             monkeypatch):
        monkeypatch.delenv(P.ENV_ON, raising=False)
        P.uninstall()
        before = pg.AxisItem.paint
        assert P.maybe_install() is False
        assert pg.AxisItem.paint is before
        assert P._installed is False

    @pytest.mark.parametrize("val", ["", "0", "false", "no", "NO", " 0 "])
    def test_falsey_values_are_off(self, qapp, monkeypatch, val):
        monkeypatch.setenv(P.ENV_ON, val)
        P.uninstall()
        assert P.maybe_install() is False
        P.uninstall()

    def test_a_truthy_value_turns_it_on(self, qapp, monkeypatch):
        monkeypatch.setenv(P.ENV_ON, "1")
        monkeypatch.setenv(P.ENV_SECS, "0")
        P.uninstall()
        try:
            assert P.maybe_install() is True
            assert pg.AxisItem.paint is P._axis_paint
        finally:
            P.uninstall()

    def test_final_report_is_silent_when_off(self, qapp, capsys):
        P.uninstall()
        P.final_report()
        assert capsys.readouterr().out == ""


class TestInstallingTwiceCannotDoubleCount:
    """Wrapping the wrapper would inflate every count by exactly 2x."""

    def test_second_install_is_refused(self, probe):
        assert P.install(interval_s=0) is False

    def test_the_method_is_wrapped_once(self, probe):
        P.install(interval_s=0)
        assert pg.AxisItem.paint is P._axis_paint
        # The saved original must still be pyqtgraph's, not our wrapper.
        assert P._orig_axis_paint is not P._axis_paint
        assert P._orig_paint_event is not P._paint_event

    def test_uninstall_restores_what_was_there(self, qapp):
        P.uninstall()
        a, g = pg.AxisItem.paint, pg.GraphicsView.paintEvent
        P.install(interval_s=0)
        P.uninstall()
        assert pg.AxisItem.paint is a
        assert pg.GraphicsView.paintEvent is g


class TestTheRatioIsCountedCorrectly:
    """Frames, and which of them repainted an axis. Both halves."""

    def test_a_real_render_moves_both_counters(self, probe, qapp):
        w = _plot(qapp)
        _render(w)
        assert sum(P._frames.values()) >= 1, "no paintEvent seen"
        assert P._axis_paints >= 1, "no AxisItem.paint seen"
        # Not _dirty: nothing cached this widget's axes, and only a
        # cached axis can invalidate. The cached case is below.
        assert sum(P._dirty.values()) == 0

    def test_a_cached_axis_painting_does_mark_the_frame(self, probe, qapp):
        from analysis.qt_axis_cache import cache_grid_axes

        w = pg.PlotWidget()
        w.resize(200, 150)
        w.getPlotItem().showGrid(x=True, y=True)
        w.show()
        qapp.processEvents()
        assert cache_grid_axes([w.getPlotItem()], who="test") is True
        P.reset()
        # A range change is the thing that invalidates a cached axis.
        w.getPlotItem().setYRange(0, 99)
        _render(w)
        assert P._cached_paints >= 1, "a cached axis never repainted"
        assert sum(P._dirty.values()) >= 1

    # The three below drive _paint_event with a stand-in for the real
    # paint, so they must go through monkeypatch and never assign
    # P._orig_paint_event directly: uninstall() puts whatever that
    # attribute holds back onto pg.GraphicsView.paintEvent, so a stub
    # left behind becomes Qt's paint path for the rest of the session --
    # every later test then renders nothing and blames its own code.

    def test_a_frame_that_paints_no_axis_is_not_dirty(self, probe, qapp,
                                                      monkeypatch):
        """The exact case the cache is bought for.

        Driven directly instead of relying on Qt to suppress the paint:
        the question here is whether the bookkeeping marks the frame, not
        whether DeviceCoordinateCache works offscreen.
        """
        calls = {"n": 0}
        monkeypatch.setattr(
            P, "_orig_paint_event",
            lambda self, ev: calls.__setitem__("n", calls["n"] + 1))
        P._paint_event(_FakeView(), None)
        assert calls["n"] == 1
        assert sum(P._frames.values()) == 1
        assert sum(P._dirty.values()) == 0, "a quiet frame was marked dirty"

    def test_a_frame_that_paints_an_axis_is_dirty(self, probe, qapp,
                                                  monkeypatch):
        def _inner(self, ev):
            P._cached_paints += 1        # stand in for a CACHED axis painting

        monkeypatch.setattr(P, "_orig_paint_event", _inner)
        P._paint_event(_FakeView(), None)
        assert sum(P._frames.values()) == 1
        assert sum(P._dirty.values()) == 1

    def test_an_uncached_axis_painting_does_not_mark_the_frame(self, probe,
                                                               qapp,
                                                               monkeypatch):
        """Only a cached axis can invalidate.

        The profile panels show the grid on one orientation, so their
        left axes are never cached. Counting their paints reads 100% on a
        panel where the cache is neither paying nor costing -- measured
        on a real TradeViewerQt, both panels read exactly that before
        this rule went in.
        """
        def _inner(self, ev):
            P._axis_paints += 1          # an UNCACHED axis painted

        monkeypatch.setattr(P, "_orig_paint_event", _inner)
        P._paint_event(_FakeView(), None)
        assert sum(P._frames.values()) == 1
        assert sum(P._dirty.values()) == 0

    def test_the_frame_is_counted_even_if_the_paint_raises(self, probe, qapp,
                                                           monkeypatch):
        """A lost frame silently lowers the ratio -- the wrong direction."""
        def _boom(self, ev):
            raise RuntimeError("paint failed")

        monkeypatch.setattr(P, "_orig_paint_event", _boom)
        with pytest.raises(RuntimeError):
            P._paint_event(_FakeView(), None)
        assert sum(P._frames.values()) == 1

    def test_per_axis_totals_add_up_to_the_global_count(self, probe, qapp):
        w = _plot(qapp)
        _render(w)
        _render(w)
        assert sum(P._per_axis.values()) == P._axis_paints


class TestLabelsDoNotSplitOrCollide:
    """A label that moves mid-session splits one window across two rows."""

    def test_two_views_in_one_window_stay_two_rows(self, probe, qapp):
        """The flaw this keying exists to avoid.

        The main chart window holds the seven stacked plots in one
        GraphicsLayoutWidget and each profile panel in its own
        PlotWidget. The panels call setXRange on every rebuild, so they
        invalidate their own axes on every frame they draw. Keyed by
        window, their ~100% would be averaged into the seven plots' and
        the reading would say the cache never pays.
        """
        from PyQt6.QtWidgets import QVBoxLayout, QWidget

        host = QWidget()
        lay = QVBoxLayout(host)
        a, b = pg.PlotWidget(), pg.PlotWidget()
        for p in (a, b):
            p.getPlotItem().showGrid(x=True, y=True)
            lay.addWidget(p)
        host.resize(240, 240)
        host.show()
        qapp.processEvents()
        P.reset()

        a.viewport().repaint()
        assert len(P._frames) == 1, f"expected one row, got {dict(P._frames)}"
        b.viewport().repaint()
        assert len(P._frames) == 2, (
            f"both views collapsed into one row: {dict(P._frames)}")
        assert all(k.startswith("QWidget/") for k in P._frames), dict(P._frames)

    def test_a_named_view_is_reported_by_its_name(self, probe, qapp):
        """The viewer names its three views; without this the report
        numbers them in construction order and the reader has to guess
        which row is the chart and which are the profile panels."""
        w = pg.PlotWidget()
        w.setObjectName("range-profile")
        w.getPlotItem().showGrid(x=True, y=True)
        w.resize(200, 150)
        w.show()
        qapp.processEvents()
        P.reset()
        _render(w)
        assert list(P._frames) == ["PlotWidget/range-profile"], dict(P._frames)
        assert "#" not in next(iter(P._frames)), "named view still numbered"

    def test_the_viewer_names_all_three_of_its_views(self):
        import inspect

        from analysis import trade_viewer_qt as V

        src = inspect.getsource(V.TradeViewerQt)
        for name in ("chart", "tick-profile", "range-profile"):
            assert f'setObjectName("{name}")' in src, name

    def test_the_window_tag_is_the_class_not_the_title(self, probe, qapp):
        w = _plot(qapp)
        w.setWindowTitle("Trade Viewer Qt - US.SOXL 1m Live")
        # The tag is also asserted at its source, because _view_tag
        # memoises on the view: a tag built from the title would still
        # look stable here, having been computed once before the rename.
        assert P._window_tag(w) == type(w).__name__
        _render(w)
        first = dict(P._frames)
        w.setWindowTitle("Trade Viewer Qt - US.QQQ 30m Live")
        assert P._window_tag(w) == type(w).__name__, "tag follows the title"
        _render(w)
        assert set(P._frames) == set(first), (
            "renaming the window started a new row")
        assert sum(P._frames.values()) == sum(first.values()) + 1

    def test_an_axis_label_is_stable_across_frames(self, probe, qapp):
        w = _plot(qapp)
        _render(w)
        labels = set(P._per_axis)
        _render(w)
        assert set(P._per_axis) == labels

    def test_the_label_names_the_orientation(self, probe, qapp):
        w = _plot(qapp)
        _render(w)
        ends = {lbl.rsplit("/", 1)[-1] for lbl in P._per_axis}
        assert ends and ends <= {"left", "right", "top", "bottom"}


class TestTheReport:
    def test_it_says_so_when_nothing_has_been_drawn(self, probe):
        assert "no frames yet" in P.report()

    def test_it_carries_the_break_even_beside_the_number(self, probe, qapp):
        _render(_plot(qapp))
        out = P.report()
        # The literal, not f"{P.BREAK_EVEN:.1%}": formatting the constant
        # and then looking for it asks the object under test to answer
        # for itself, and passes for any value it happens to hold.
        assert "87.9%" in out, (
            "a hit rate printed without the number it must be compared to")
        assert "invalidating" in out

    def test_the_break_even_equals_the_milliseconds_it_came_from(self):
        """The one constant the whole instrument exists to compare
        against, tied to its derivation.

        Two mutations of it -- 0.879 to 0.500 and to 0.050 -- survived
        the suite, because the only test that mentioned it formatted the
        constant and searched for its own output.
        """
        save = 50.90 - 3.57        # crosshair frame, qt_axis_cache docstring
        loss = 76.62 - 70.12       # setTicks / setXRange frame
        assert abs(P.BREAK_EVEN - save / (save + loss)) < 1e-6
        assert abs(P.BREAK_EVEN - 0.87925) < 1e-4

    def test_no_verdict_is_printed(self, probe, qapp):
        """The regime the 87.9% was calibrated in is not this one.

        Forced full-viewport repaints put a clean frame's saving at
        48.13 ms; under MinimalViewportUpdate, which is what pyqtgraph
        sets and nothing here changes, it is 0.93 ms, and the break-even
        moves from ~92% to ~27%. A share in between would have printed
        "paying" while the cache cost.
        """
        _render(_plot(qapp))
        out = P.report()
        assert "no verdict" in out
        assert "MinimalViewportUpdate" in out
        assert "under break-even" not in out
        assert "OVER break-even" not in out

    def test_a_view_with_no_cached_axes_says_so(self, probe, qapp):
        w = _plot(qapp)                 # nothing here calls cache_grid_axes
        _render(w)
        assert "no cached axes" in P.report()

    def test_it_carries_what_is_needed_to_convert_the_share(self, probe, qapp):
        """The share alone cannot be turned into milliseconds.

        What a clean frame saves scales with the exposed area, and what
        an invalidating frame costs scales with how many axes actually
        redrew. Without both, a whole session's reading cannot be priced
        afterwards and the day has to be run again.
        """
        from analysis.qt_axis_cache import cache_grid_axes

        w = pg.PlotWidget()
        w.resize(200, 150)
        w.getPlotItem().showGrid(x=True, y=True)
        w.show()
        qapp.processEvents()
        cache_grid_axes([w.getPlotItem()], who="test")
        P.reset()
        w.getPlotItem().setYRange(0, 77)
        _render(w)
        out = P.report()
        assert "per invalidating frame" in out
        assert "mean exposed" in out
        tag = next(iter(P._frames))
        assert P._dirty_axes[tag] >= 1
        assert P._area[tag][1] >= 1
        assert 0.0 < P._area[tag][0] <= P._area[tag][1] * 1.01

    def test_a_closed_window_keeps_its_cached_mark(self, probe, qapp):
        """The heatmap is WA_DeleteOnClose and is opened and closed all
        day. Read live, its row downgrades to "no cached axes" the
        moment it is shut -- and the closing report is exactly when that
        is read, so a whole window's reading would be thrown away."""
        from analysis.qt_axis_cache import cache_grid_axes

        w = pg.PlotWidget()
        w.resize(200, 150)
        w.getPlotItem().showGrid(x=True, y=True)
        w.show()
        qapp.processEvents()
        cache_grid_axes([w.getPlotItem()], who="test")
        P.reset()
        _render(w)
        assert "no cached axes" not in P.report()
        w.close()
        qapp.processEvents()
        assert "no cached axes" not in P.report(final=True), (
            "the window's reading was downgraded when it closed")

    def test_paints_outside_any_frame_are_flagged(self, probe, qapp):
        """QWidget.render() and QGraphicsScene.render() paint axes
        without a paintEvent -- measured at 14 paints and 0 frames.
        Neither is used here today, but a missed frame LOWERS the share,
        which is the direction that flatters the cache, so it is counted
        rather than assumed away."""
        from analysis.qt_axis_cache import cache_grid_axes

        w = pg.PlotWidget()
        w.resize(200, 150)
        w.getPlotItem().showGrid(x=True, y=True)
        w.show()
        qapp.processEvents()
        cache_grid_axes([w.getPlotItem()], who="test")
        P.reset()
        _render(w)
        assert P._orphans == 0, "an ordinary frame was called an orphan"
        assert "OUTSIDE any frame" not in P.report()

        # The real bypass, not a hand-set counter: scene.render() paints
        # the axes with no paintEvent anywhere.
        img = QImage(w.size(), QImage.Format.Format_ARGB32)
        p = QPainter(img)
        try:
            w.scene().render(p)
        finally:
            p.end()
        assert P._orphans > 0, "a paint outside every frame went uncounted"
        out = P.report()
        assert "OUTSIDE any frame" in out
        assert "too low" in out

    def test_a_cached_axis_that_never_painted_is_listed_at_zero(self, probe,
                                                                qapp):
        """A 100% hit rate shows up as an absence, and an absence reads
        the same as an axis that was never there."""
        from analysis.qt_axis_cache import cache_grid_axes

        w = pg.PlotWidget()
        w.resize(200, 150)
        w.getPlotItem().showGrid(x=True, y=True)
        w.show()
        qapp.processEvents()
        assert cache_grid_axes([w.getPlotItem()], who="test") is True
        P.reset()
        _render(w)
        out = P.report(detail=True)
        # The hidden orientations are the clean case: PlotItem enables the
        # grid on all four, a hidden QGraphicsItem is never painted, so
        # top and right are cached with exactly zero paints for the whole
        # session. If a zero row can be dropped, so can a real 100% hit
        # rate -- and an axis reading 0 would then be indistinguishable
        # from one that was never cached at all.
        rows = [ln.split() for ln in out.splitlines()
                if "/top" in ln or "/right" in ln]
        assert rows, f"no hidden-axis rows at all:\n{out}"
        zeros = [r for r in rows if r[-2] == "0" and r[-1] == "cached"]
        assert zeros, f"a cached axis with no paints was left out:\n{out}"

    def test_the_periodic_reading_leaves_the_axis_table_out(self, probe, qapp):
        """36 rows a minute would bury the three lines that answer the
        question; the table comes with the closing reading."""
        _render(_plot(qapp))
        assert "per axis" not in P.report()
        assert "per axis" in P.report(final=True)
        assert "per axis" in P.report(detail=True)

    def test_the_interval_line_appears_only_after_a_second_read(self, probe,
                                                               qapp):
        _render(_plot(qapp))
        first = P.report()
        assert "this interval" in first        # first read: whole run so far
        second = P.report()
        assert "this interval" not in second   # nothing happened in between

    def test_final_report_prints(self, probe, qapp, capsys):
        _render(_plot(qapp))
        P.final_report()
        assert "[axis-probe] final" in capsys.readouterr().out


class TestTheViewerActuallyCallsIt:
    """Without this, deleting the two lines in main() leaves tests green."""

    def test_main_installs_and_reports(self):
        import inspect

        from analysis import trade_viewer_qt as V

        src = inspect.getsource(V.main)
        assert "qt_axis_probe.maybe_install()" in src
        # os._exit runs no atexit handler, so the closing reading has to be
        # taken explicitly before it.
        assert "qt_axis_probe.final_report()" in src
        assert (src.index("qt_axis_probe.final_report()")
                < src.index("os._exit(ret)"))


class TestItDoesNotCostEnoughToMatter:
    """An instrument that perturbs the run cannot measure it."""

    def test_the_wrapper_overhead_is_small_against_a_frame(self, probe, qapp):
        P._orig_paint_event = lambda self, ev: None
        v = _FakeView()
        n = 20000
        t0 = time.perf_counter()
        for _ in range(n):
            P._paint_event(v, None)
        per_frame_us = (time.perf_counter() - t0) / n * 1e6
        # A real frame on this chart costs 50-76 ms (see qt_axis_cache).
        # Anything under 50 us is below 0.1% of that; the assert is loose
        # because it runs on whatever machine CI has, and the point is the
        # order of magnitude, not the number.
        assert per_frame_us < 50, f"{per_frame_us:.1f} us of bookkeeping"

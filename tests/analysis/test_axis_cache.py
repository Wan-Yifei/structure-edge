"""The grid axes are rendered once and blitted, and that must stay bounded.

WHY THE CACHE EXISTS. A grid-enabled AxisItem's boundingRect is unioned
with its linked view's, so its QPicture is replayed over the whole plot
area on every repaint: 13.7 ms for the heatmap's price axis against
0.138 ms for a non-grid 43x512 strip holding a near-identical number of
painter commands. The cost is the area, not the drawing.
DeviceCoordinateCache turns the replay into a blit. Per frame class on a
7-plot chart: -93% when the frame does not touch an axis, +9.3% when it
does, so it pays unless more than 87.9% of frames invalidate one.

WHY THESE GROUPS. Each guards a way this could stop working with nothing
visibly breaking:

  * the pixels could change -- so they are compared;
  * the budget could stop being process-wide, which is how two windows
    end up oversubscribing the shared QPixmapCache: every paint a miss
    that renders a bitmap AND blits it, measured at 86.8 ms a frame
    against 62.0 uncached, a regression that looks like nothing from
    outside;
  * it could stop reclaiming when a window closes, which is the same
    regression arriving a few minutes later -- the heatmap window is
    rebuilt on every open and destroyed on close, so a budget that only
    grows turns itself off mid-session;
  * someone could go back to setting the cache mode on pg.AxisItem
    globally, which is that regression by a third route;
  * the cache could be invalidated every frame, which costs more than
    not caching and shows up nowhere except a benchmark.

NOTE ON THE MAIN CHART. An earlier version of this file guarded against
the main chart ever being cached. That was backwards -- it rested on a
measurement taken under Qt's 10 MB default ceiling, and with the ceiling
raised, caching both windows is the largest win available. The main
chart is cached deliberately now; what is guarded is that the two
windows' demands are added together before either is granted.
"""

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6.QtWidgets")

import pyqtgraph as pg                                        # noqa: E402
from PyQt6.QtGui import QPixmapCache                          # noqa: E402
from PyQt6.QtWidgets import QApplication, QGraphicsItem       # noqa: E402

import analysis.qt_axis_cache as A                            # noqa: E402
from analysis.liq_hm_window import LiqHmWindow                # noqa: E402

CACHED = QGraphicsItem.CacheMode.DeviceCoordinateCache
NOCACHE = QGraphicsItem.CacheMode.NoCache


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def win(qapp):
    w = LiqHmWindow()
    w.resize(1500, 950)
    w.show()
    w._timer.stop()
    for _ in range(20):
        qapp.processEvents()
    yield w
    from qt_window_teardown import destroy_window
    destroy_window(w)


@pytest.fixture()
def alone(monkeypatch):
    """Pin the process to "no other window is cached".

    The budget counts the windows that already hold cached axes, and in a
    test session that set is whatever earlier tests left behind. The
    refusal thresholds below are about the screen, so the window count is
    pinned here and tested on its own in TestTheBudgetIsProcessWide.
    """
    monkeypatch.setattr(A, "_windows_holding_cached_axes",
                        lambda exclude=None: [])


def _axes(win):
    pi = win._plot_widget.getPlotItem()
    return {s: pi.getAxis(s) for s in ("left", "bottom", "right", "top")}


class TestItIsSetOnTheRightAxesAndOnlyThose:

    def test_every_grid_axis_is_cached(self, win):
        for side, ax in _axes(win).items():
            if ax.grid is not False:
                assert ax.cacheMode() == CACHED, f"{side} is not cached"

    def test_the_hidden_grid_axes_are_cached_too(self, win):
        """They cost nothing and inherit the cache if ever shown.

        PlotItem.updateGrid sets a grid on all four orientations, so top
        and right report grid is not False while being hidden. A hidden
        QGraphicsItem is never painted and the bitmap is allocated inside
        the paint path, so these hold no memory: probing the cache by
        eviction reads 9.28 MB for a 7-plot window whether the cache mode
        is set on two axes or on four. Setting it anyway means a folded
        row that is later opened is already cached.
        """
        hidden = [s for s, ax in _axes(win).items() if not ax.isVisible()]
        assert hidden, "fixture is wrong: every axis is visible"
        for side in hidden:
            assert _axes(win)[side].cacheMode() == CACHED

    def test_a_plot_without_a_grid_gets_nothing_cached(self, qapp, alone):
        """A non-grid axis is a 43x512 strip costing 0.138 ms to paint. A
        bitmap for it buys nothing and still spends the shared budget.

        Asserted against a plot built here rather than against the
        heatmap's own axes: showGrid puts a grid on all four of those, so
        a loop over them has no negative case and would pass whatever the
        function did.
        """
        pw = pg.PlotWidget()
        try:
            pi = pw.getPlotItem()
            A.cache_grid_axes([pi], who="test")
            for side in ("left", "bottom", "right", "top"):
                ax = pi.getAxis(side)
                assert ax.grid is False, "fixture is wrong: grid is on"
                assert ax.cacheMode() != CACHED, (
                    f"{side} has no grid and should not hold a bitmap")
        finally:
            pw.deleteLater()
            qapp.processEvents()

    def test_the_cache_mode_is_not_set_globally(self, win):
        """A fresh AxisItem belonging to no window must be uncached.

        Setting it on pg.AxisItem itself was the first shape this was
        measured in, and it is a 45% regression as soon as a second
        window is open: the two together oversubscribe the process-wide
        pixmap cache.
        """
        loose = pg.AxisItem(orientation="left")
        try:
            assert loose.cacheMode() != CACHED, (
                "AxisItem is cached by default -- someone patched the class")
        finally:
            loose.deleteLater()


class TestOneCallPerWindowNotPerPlot:
    """The budget is per window, so a window asks once for all its plots.

    Calling it per plot is what the first version did, and the budget did
    not survive being summed: each call estimated a whole screen's worth
    of bitmaps for its own plot, and only the first call raised the
    ceiling, because by the second the ceiling was already that high.
    Seven plots got one plot's reservation.
    """

    @staticmethod
    def _patch_screen(monkeypatch, w, h, dpr):
        from PyQt6.QtCore import QSize
        from PyQt6.QtGui import QGuiApplication

        class _Screen:
            def devicePixelRatio(self):
                return dpr

            def size(self):
                return QSize(w, h)

        monkeypatch.setattr(QGuiApplication, "primaryScreen",
                            staticmethod(lambda: _Screen()))

    def test_seven_plots_cost_what_one_window_costs(self, qapp, monkeypatch,
                                                    alone):
        """A window's plots partition its area, so seven of them want
        about what one full-window plot wants -- measured by eviction
        probe at 1500x950: 10.27 MB for one plot, 9.28 MB for seven,
        8.53 MB for eleven. Charging each plot separately would ask for
        seven times the screen and be refused."""
        self._patch_screen(monkeypatch, 2560, 1440, 1.0)
        plots = [pg.PlotItem() for _ in range(7)]
        try:
            for pi in plots:
                pi.showGrid(x=True, y=True)
            assert A.cache_grid_axes(plots, who="test") is True
            for pi in plots:
                assert pi.getAxis("left").cacheMode() == CACHED
        finally:
            # No close(): PlotItem.close() walks into AxisItem.close(),
            # which dereferences a scene these items never had.
            del plots
            qapp.processEvents()


class TestTheBudgetIsProcessWide:
    """Two windows are added together before either is granted.

    This is the invariant the whole module exists for. Oversubscribing
    does not degrade to "uncached": every paint becomes a miss that
    renders a bitmap AND blits it, strictly more work than drawing
    directly, and invisible from outside.
    """

    _patch_screen = staticmethod(TestOneCallPerWindowNotPerPlot._patch_screen)

    def test_the_ceiling_was_raised(self, win):
        assert QPixmapCache.cacheLimit() >= A._PIXMAP_CACHE_KB

    def test_a_second_window_is_counted_and_can_be_refused(self, qapp,
                                                           monkeypatch):
        """3840x2160 fits one window and not two.

        Same screen, same axes; the only difference is that a window is
        already cached. If the second window's budget ignored the first,
        both would be granted and both would miss on every paint.
        """
        self._patch_screen(monkeypatch, 3840, 2160, 1.0)
        monkeypatch.setattr(A, "_windows_holding_cached_axes",
                            lambda exclude=None: [])
        first = pg.PlotItem()
        try:
            first.showGrid(x=True, y=True)
            assert A.cache_grid_axes([first], who="test") is True, (
                "one window at 3840x2160 wants 66 MB and must fit")
        finally:
            del first
            qapp.processEvents()

        monkeypatch.setattr(A, "_windows_holding_cached_axes",
                            lambda exclude=None: [object()])
        second = pg.PlotItem()
        try:
            second.showGrid(x=True, y=True)
            assert A.cache_grid_axes([second], who="test") is False, (
                "two windows want 133 MB and the second must be refused")
            assert second.getAxis("left").cacheMode() != CACHED
        finally:
            del second
            qapp.processEvents()

    def test_it_reclaims_when_a_window_closes(self, qapp, monkeypatch):
        """The count is read live, not accumulated.

        The heatmap window is constructed fresh on every open and
        destroyed on close, so a running total that only grows counts it
        again on each reopen: with the main chart plus two cycles the
        total passes the ceiling and the cache turns itself off in the
        middle of a session, silently. Here the same window count goes up
        and then back down, and the second grant must succeed.
        """
        self._patch_screen(monkeypatch, 3840, 2160, 1.0)
        monkeypatch.setattr(A, "_windows_holding_cached_axes",
                            lambda exclude=None: [object()])
        blocked = pg.PlotItem()
        try:
            blocked.showGrid(x=True, y=True)
            assert A.cache_grid_axes([blocked], who="test") is False
        finally:
            del blocked
            qapp.processEvents()

        monkeypatch.setattr(A, "_windows_holding_cached_axes",
                            lambda exclude=None: [])
        reopened = pg.PlotItem()
        try:
            reopened.showGrid(x=True, y=True)
            assert A.cache_grid_axes([reopened], who="test") is True, (
                "the other window is gone and this one must be granted "
                "again; a monotonic total would still refuse it")
        finally:
            del reopened
            qapp.processEvents()

    @staticmethod
    def _cached_windows(qapp):
        """Top-level windows the live counter can see, via the real walk."""
        return A._windows_holding_cached_axes()

    def test_the_counter_sees_a_second_real_window(self, win, qapp):
        """The counter above is mocked everywhere else, so prove the real
        one works rather than asserting that it does.

        The first attempt at a walk like this missed that findChildren
        does not include the widget itself, and a bare PlotWidget shown
        on its own IS the GraphicsView -- which is exactly how a sum like
        this ends up blind to the window it is meant to notice.
        """
        before = len(self._cached_windows(qapp))
        pw = pg.PlotWidget()
        try:
            pw.resize(1200, 800)
            pw.show()
            pi = pw.getPlotItem()
            pi.showGrid(x=True, y=True)
            A.cache_grid_axes([pi], who="test")
            for _ in range(10):
                qapp.processEvents()
            assert len(self._cached_windows(qapp)) > before, (
                "a second window's cached axes were not counted, so the "
                "budget is blind to exactly what it guards")
        finally:
            pw.close()
            pw.deleteLater()
            qapp.processEvents()

    def test_a_window_does_not_count_itself(self, qapp):
        """Otherwise the first window in the process pays for two."""
        pw = pg.PlotWidget()
        try:
            pw.resize(800, 600)
            pw.show()
            pi = pw.getPlotItem()
            pi.showGrid(x=True, y=True)
            A.cache_grid_axes([pi], who="test")
            for _ in range(10):
                qapp.processEvents()
            assert pw.window() not in A._windows_holding_cached_axes(
                exclude=pw.window())
        finally:
            pw.close()
            pw.deleteLater()
            qapp.processEvents()

    def test_what_the_process_holds_fits_the_ceiling(self, win, qapp):
        """The live sum, in device pixels, against the live ceiling."""
        total = 0.0
        for w in qapp.topLevelWidgets():
            for sc in A._scenes_of(w):
                for it in sc.items():
                    if (isinstance(it, pg.AxisItem)
                            and it.cacheMode() == CACHED and it.isVisible()):
                        br = it.boundingRect()
                        views = sc.views()
                        dpr = (views[0].devicePixelRatioF() or 1.0) \
                            if views else 1.0
                        total += br.width() * br.height() * (dpr ** 2) * 4
        budget = QPixmapCache.cacheLimit() * 1024
        assert total <= budget, (
            f"cached axes across the whole process hold "
            f"{total / 1024**2:.1f} MB of a {budget / 1024**2:.0f} MB "
            f"ceiling. Oversubscribing does not fall back to uncached -- "
            f"every paint becomes a miss that renders a bitmap AND blits "
            f"it, measured at 40% slower than not caching at all.")


class TestItRefusesWhenItWouldNotFit:
    """A cache that does not fit is slower than no cache, so it is not set.

    The appetite is two plot-area bitmaps per WINDOW, in device pixels,
    so it follows the screen: 9.4 MB at 1500x950, 16.6 at 1920x1080,
    29.5 at 2560x1440, 66.4 at 3840x2160, and all of those again x2.25 at
    150% scaling. The first version of this had no check at all and would
    have gone into the oversubscribed state -- measured at 40% slower
    than not caching -- with nothing said.

    A bare PlotItem is used rather than a PlotWidget because the budget is
    taken from the window's own screen when there is one, and only falls
    back to the primary screen otherwise. With no scene there is no
    window, which is what makes the patched screen the one consulted.
    """

    _patch_screen = staticmethod(TestOneCallPerWindowNotPerPlot._patch_screen)

    def test_a_huge_screen_turns_it_off(self, qapp, monkeypatch, capsys,
                                        alone):
        self._patch_screen(monkeypatch, 7680, 4320, 2.0)
        pi = pg.PlotItem()
        try:
            pi.showGrid(x=True, y=True)
            assert A.cache_grid_axes([pi], who="test") is False
            for side in ("left", "bottom", "right", "top"):
                assert pi.getAxis(side).cacheMode() != CACHED, (
                    f"{side} was cached although the bitmaps cannot fit")
            assert "cache OFF" in capsys.readouterr().out, (
                "it declined silently; the next person sees only that it "
                "got slower")
        finally:
            # No close(): PlotItem.close() walks into AxisItem.close(),
            # which dereferences a scene this item never had.
            del pi
            qapp.processEvents()

    def test_an_ordinary_screen_still_caches(self, qapp, monkeypatch, alone):
        """The control. Without it the test above would pass just as well
        if the function had stopped caching anything at all."""
        self._patch_screen(monkeypatch, 1920, 1080, 1.0)
        pi = pg.PlotItem()
        try:
            pi.showGrid(x=True, y=True)
            assert A.cache_grid_axes([pi], who="test") is True
            assert pi.getAxis("left").cacheMode() == CACHED
        finally:
            del pi
            qapp.processEvents()

    def test_the_scaling_factor_is_what_decides_it(self, qapp, monkeypatch,
                                                   alone):
        """3840x2160 fits at 100% and does not at 150%.

        Same screen, same axes; only the device pixel ratio differs. The
        bitmaps are device pixels, so leaving the ratio out of the budget
        under-counts by its square -- and that mistake is invisible on
        this machine, which runs at 100%.
        """
        self._patch_screen(monkeypatch, 3840, 2160, 1.0)
        pi = pg.PlotItem()
        try:
            pi.showGrid(x=True, y=True)
            assert A.cache_grid_axes([pi], who="test") is True, (
                "3840x2160 at 100% wants 66 MB and should fit")
        finally:
            del pi
            qapp.processEvents()

        self._patch_screen(monkeypatch, 3840, 2160, 1.5)
        pi = pg.PlotItem()
        try:
            pi.showGrid(x=True, y=True)
            assert A.cache_grid_axes([pi], who="test") is False, (
                "the same screen at 150% wants 2.25x the bitmaps and must "
                "be refused")
            assert pi.getAxis("left").cacheMode() != CACHED
        finally:
            del pi
            qapp.processEvents()

    def test_the_ceiling_is_raised_to_fit_a_bigger_screen(self, qapp,
                                                          monkeypatch, alone):
        """Between the two: 2560x1440 wants 29.5 MB, which does not fit
        the 32 MB default once the headroom is added but is well inside
        the maximum."""
        self._patch_screen(monkeypatch, 2560, 1440, 1.0)
        before = QPixmapCache.cacheLimit()
        pi = pg.PlotItem()
        try:
            pi.showGrid(x=True, y=True)
            assert A.cache_grid_axes([pi], who="test") is True
            assert pi.getAxis("left").cacheMode() == CACHED
            assert QPixmapCache.cacheLimit() >= before
            assert QPixmapCache.cacheLimit() > A._PIXMAP_CACHE_KB, (
                "29.5 MB plus headroom does not fit the 32 MB floor, so "
                "the ceiling had to move")
            assert QPixmapCache.cacheLimit() <= A._PIXMAP_CACHE_MAX_KB
        finally:
            del pi
            qapp.processEvents()


class TestThePixelsAreUnchanged:
    """Byte for byte on the heatmap, no tolerance.

    Only at device pixel ratio 1, which is what this machine runs. Under
    fractional scaling it is NOT identical -- 1.5 showed 4-5% of pixels
    differing on this window, and 2.8-7.1% on the main chart -- which is
    also why pyqtgraph has this same line commented out upstream
    (AxisItem.py:148) beside a note about a Qt 4.6 bug showing cached
    results incorrectly.

    THE MAIN CHART IS NOT BYTE-IDENTICAL EVEN AT RATIO 1: 53 of 1.73 M
    pixels differ (0.0031%), grid lines one colour step down plus the
    occasional single axis-line pixel dropping to the background. An A/A
    control differs by 0, so those 53 are the cache and not the capture.
    That is recorded, not asserted here, and it is the one claim from the
    heatmap's version of this change that does not carry over.
    """

    @staticmethod
    def _shot(win, qapp, cached):
        for ax in _axes(win).values():
            if ax.grid is not False:
                ax.setCacheMode(CACHED if cached else NOCACHE)
        win._plot_widget.viewport().update()
        for _ in range(6):
            qapp.processEvents()
        img = win._plot_widget.grab().toImage()
        img = img.convertToFormat(img.Format.Format_RGBA8888)
        ptr = img.bits()
        ptr.setsize(img.sizeInBytes())
        return np.frombuffer(ptr, np.uint8).reshape(
            img.height(), img.bytesPerLine() // 4, 4).copy()

    def _same(self, win, qapp):
        if win._plot_widget.devicePixelRatioF() != 1.0:
            pytest.skip("only claimed at device pixel ratio 1")
        off = self._shot(win, qapp, False)
        on = self._shot(win, qapp, True)
        assert off.shape == on.shape
        bad = int(np.count_nonzero(off != on))
        assert bad == 0, (
            f"{bad} bytes differ ({100 * bad / off.size:.2f}% of the image)")

    def test_at_the_default_view(self, win, qapp):
        self._same(win, qapp)

    def test_after_a_zoom(self, win, qapp):
        win._plot_widget.getPlotItem().vb.scaleBy((0.4, 0.4))
        self._same(win, qapp)

    def test_after_a_fractional_pan(self, win, qapp):
        """Half a pixel, so the drawing does not land on device pixels."""
        win._plot_widget.getPlotItem().vb.translateBy((0.5, 0.5))
        self._same(win, qapp)

    def test_after_a_resize(self, win, qapp):
        """A grid axis's boundingRect depends on the linked view's size,
        and AxisItem.linkedViewChanged clears the picture without calling
        prepareGeometryChange -- so a bitmap sized to the old rect is
        exactly what should surface here."""
        win.resize(1100, 700)
        for _ in range(10):
            qapp.processEvents()
        self._same(win, qapp)


class TestTheCacheActuallyHits:
    """Caching something that is invalidated every frame costs more than
    not caching it, and nothing outside a benchmark would show it.

    With the cache on, Qt calls paint() only on a miss, so counting paint
    calls against repaints counts misses. That is also the cheapest ruler
    available for checking this in the real application: 20 crosshair
    frames paint the axes 280 times uncached and 0 times cached, while 20
    redraw frames paint 280 against 140.
    """

    @staticmethod
    def _count(win, qapp, side, frames, render):
        ax = _axes(win)[side]
        assert ax.cacheMode() == CACHED
        calls = {"n": 0}
        orig = pg.AxisItem.paint

        def counting(self, *a, **k):
            if self is ax:
                calls["n"] += 1
            return orig(self, *a, **k)

        pg.AxisItem.paint = counting
        try:
            for _ in range(frames):
                if render:
                    win._render()
                win._plot_widget.viewport().update()
                qapp.processEvents()
        finally:
            pg.AxisItem.paint = orig
        return calls["n"]

    def test_the_price_axis_is_not_repainted_every_frame(self, win, qapp):
        assert self._count(win, qapp, "left", 20, render=False) <= 5

    def test_the_time_axis_under_real_renders(self, win, qapp):
        """Driving _render, which is what the window actually does.

        The first version of this test checked only the price axis and ran
        with the column timer stopped, so _render never fired. _render
        calls setTicks on the time axis every pass, and pyqtgraph's
        setTicks is `picture = None; update()` -- the time axis's cache is
        therefore missed on essentially every render, which is the exact
        failure this class exists to catch and which that version could
        not see. It is recorded rather than fixed: dropping the redundant
        setTicks calls was measured and made no difference beyond the
        bench's own noise, and the price axis's hits are where the saving
        is.
        """
        n = self._count(win, qapp, "bottom", 10, render=True)
        assert n <= 12, (
            f"the time axis repainted {n} times in 10 renders -- more than "
            f"one a frame means the cache is costing rather than saving")


class TestTheMainChartIsActuallyWired:
    """The call site exists and reaches every plot the window owns.

    Without this, deleting the one line in trade_viewer_qt.py that asks
    for the cache leaves all of the above green: everything else here
    builds its own plots. That exact gap -- a guard whose tests all aim
    at code other than the guard -- is how the first version of this
    change shipped three statements that were false.

    It costs one real TradeViewerQt construction, about 2 seconds, and
    its __init__ opens and closes an OpenD quote context.
    """

    def test_every_grid_axis_in_the_window_is_cached(self, qapp):
        from analysis.trade_viewer_qt import TradeViewerQt

        v = TradeViewerQt()
        try:
            v.resize(1200, 800)
            v.show()
            for _ in range(20):
                qapp.processEvents()
            plots = {
                "candles": v._plot_c, "volume": v._plot_v, "kd": v._plot_kd,
                "cvd": v._plot_cvd, "adi": v._plot_adi,
                "dcvd": v._plot_dcvd, "oflow": v._plot_oflow,
                "tick profile": v._tick_profile_widget.getPlotItem(),
                "session profile": v._profile_widget.getPlotItem(),
            }
            for name, pi in plots.items():
                grid = A._plot_axes(pi)
                assert grid, f"{name} has no grid axis; fixture is stale"
                for ax in grid:
                    assert ax.cacheMode() == CACHED, (
                        f"{name}'s {ax.orientation} grid axis is not "
                        f"cached -- the call in trade_viewer_qt.py is "
                        f"gone, or it was handed fewer plots than the "
                        f"window owns")
        finally:
            v.close()
            qapp.processEvents()

    def test_the_two_windows_are_counted_together(self, qapp):
        """Both open at once is the configuration the budget is for."""
        from analysis.trade_viewer_qt import TradeViewerQt

        v = TradeViewerQt()
        h = LiqHmWindow()
        try:
            v.resize(1200, 800)
            v.show()
            h.resize(900, 600)
            h.show()
            h._timer.stop()
            for _ in range(20):
                qapp.processEvents()
            seen = A._windows_holding_cached_axes()
            assert len(seen) >= 2, (
                f"the live counter sees {len(seen)} cached windows with "
                f"both open; a second window asking for a budget would "
                f"not know the first exists")
        finally:
            h.close()
            v.close()
            qapp.processEvents()


class TestTheTwoEdgesThatSurvivedMutation:
    """Both of these passed every test above while broken.

    A guard whose tests cannot fail is the failure mode this file keeps
    running into, so each of these was written against a mutant that
    survived and then checked to kill it.
    """

    _patch_screen = staticmethod(TestOneCallPerWindowNotPerPlot._patch_screen)

    def test_asking_twice_for_one_window_is_not_charged_twice(
            self, qapp, monkeypatch):
        """One window of this size fits; two do not. A window that counts
        ITSELF as a second one therefore refuses its own second call.

        No screen patch here, deliberately: a window that is SHOWN reads
        its own screen, not the primary one, so patching
        QGuiApplication.primaryScreen does nothing for it -- which is
        exactly how the first version of this test passed against the
        mutant it was written to kill. The window is sized instead, and
        the bound takes the larger of window and screen.

        Nothing calls this twice today, which is why the exclusion was
        dead code no test could reach. It stops being dead the moment a
        window gains a plot after construction.
        """
        pw = pg.PlotWidget()
        try:
            pw.show()
            pw.resize(3400, 3000)
            for _ in range(10):
                qapp.processEvents()
            if pw.window().width() < 3400 or pw.window().height() < 3000:
                pytest.skip("the platform clamped the window; nothing to say")
            # Only this window counts, so ambient windows left by other
            # tests cannot decide the outcome. If the exclusion is
            # dropped, the real walk finds this window on the second
            # call and the filter keeps it.
            real = A._windows_holding_cached_axes
            monkeypatch.setattr(
                A, "_windows_holding_cached_axes",
                lambda exclude=None: [w for w in real(exclude=exclude)
                                      if w is pw.window()])
            pi = pw.getPlotItem()
            pi.showGrid(x=True, y=True)
            assert A.cache_grid_axes([pi], who="test") is True, (
                "78 MB for one window of this size has to fit, or the "
                "second call below proves nothing")
            for _ in range(10):
                qapp.processEvents()
            assert A.cache_grid_axes([pi], who="test") is True, (
                "the second call charged this window for a window that "
                "is itself")
        finally:
            pw.close()
            pw.deleteLater()
            qapp.processEvents()

    def test_a_window_bigger_than_its_screen_is_measured_by_the_window(
            self, qapp, monkeypatch, capsys):
        """A window can exceed one screen -- dragged across two monitors
        on a desktop, and routinely under offscreen rendering, where the
        screen reports 800x800 and the real chart is built at 1733x1000
        and genuinely holds 8.48 MB against a 4.9 MB screen-sized
        reserve. Taking the screen alone under-reserves by the ratio of
        the two areas."""
        self._patch_screen(monkeypatch, 1920, 1080, 1.0)
        monkeypatch.setattr(A, "_windows_holding_cached_axes",
                            lambda exclude=None: [])
        pw = pg.PlotWidget()
        try:
            pw.show()
            pw.resize(6000, 4000)
            for _ in range(10):
                qapp.processEvents()
            if pw.window().width() < 6000 or pw.window().height() < 4000:
                pytest.skip("the platform clamped the window; nothing to say")
            pi = pw.getPlotItem()
            pi.showGrid(x=True, y=True)
            assert A.cache_grid_axes([pi], who="test") is False, (
                "183 MB of bitmaps were granted because only the 1920x1080 "
                "screen was measured")
            assert "cache OFF" in capsys.readouterr().out
        finally:
            pw.close()
            pw.deleteLater()
            qapp.processEvents()

"""The grid axes are rendered once and blitted, and that must stay exact.

WHY THE CACHE EXISTS. A grid-enabled AxisItem's boundingRect is unioned
with its linked view's, so its QPicture is replayed over the whole plot
area on every repaint: 18.4 ms for the heatmap's price axis against
0.138 ms for a non-grid axis holding a near-identical number of painter
commands. The cost is the area, not the drawing. DeviceCoordinateCache
turns the replay into a blit, and on scripts/bench/bench_liq_hm.py the
window goes from 112.1 s to 56.5 s of CPU per 300 simulated seconds.

WHY THESE FOUR GROUPS. Each guards a way this could stop working with
nothing visibly breaking:

  * the pixels could change -- so they are compared, with no tolerance;
  * someone could cache the main chart's eight grid axes too, which pushes
    the shared pixmap budget past its ceiling and makes every paint a miss
    (measured: 66.8 ms a frame uncached, 96.6 ms with both windows cached
    under the default 10 MB ceiling -- a 45% REGRESSION that looks like
    nothing from outside);
  * someone could go back to setting the cache mode on pg.AxisItem
    globally, which is that same regression by another route;
  * the cache could be invalidated every frame, which costs more than not
    caching and shows up nowhere except a benchmark.
"""

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6.QtWidgets")

import pyqtgraph as pg                                        # noqa: E402
from PyQt6.QtGui import QPixmapCache                          # noqa: E402
from PyQt6.QtWidgets import QApplication, QGraphicsItem       # noqa: E402

from analysis.liq_hm_window import (                          # noqa: E402
    LiqHmWindow, _PIXMAP_CACHE_KB,
)

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


def _axes(win):
    pi = win._plot_widget.getPlotItem()
    return {s: pi.getAxis(s) for s in ("left", "bottom", "right", "top")}


class TestItIsSetOnTheRightAxesAndOnlyThose:

    def test_every_grid_axis_is_cached(self, win):
        for side, ax in _axes(win).items():
            if ax.grid is not False:
                assert ax.cacheMode() == CACHED, f"{side} is not cached"

    def test_a_plot_without_a_grid_gets_nothing_cached(self, qapp):
        """A non-grid axis is a 43x512 strip costing 0.138 ms to paint. A
        bitmap for it buys nothing and still spends the shared budget.

        Asserted against a plot built here rather than against the
        heatmap's own axes: showGrid puts a grid on all four of those, so
        a loop over them has no negative case and would pass whatever the
        function did.
        """
        from analysis.liq_hm_window import _cache_grid_axes
        pw = pg.PlotWidget()
        try:
            pi = pw.getPlotItem()
            _cache_grid_axes(pi)
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
        measured in, and it is a 45% regression as soon as the main chart
        is open: the two windows together oversubscribe the process-wide
        pixmap cache.
        """
        loose = pg.AxisItem(orientation="left")
        try:
            assert loose.cacheMode() != CACHED, (
                "AxisItem is cached by default -- someone patched the class")
        finally:
            loose.deleteLater()


class TestTheBitmapsFitTheBudget:
    """The invariant that stops the regression, rather than measuring it.

    Qt keeps DeviceCoordinateCache bitmaps in the process-wide
    QPixmapCache. Oversubscribing it does not degrade to "uncached": every
    paint becomes a miss, which renders into a bitmap AND blits it --
    strictly more work than drawing directly, and invisible from outside.
    """

    def test_the_ceiling_was_raised(self, win):
        assert QPixmapCache.cacheLimit() >= _PIXMAP_CACHE_KB

    @staticmethod
    def _process_need_bytes(qapp):
        """Every cached, visible axis in the PROCESS, in device pixels.

        Summing only this window's axes was the first version, and it did
        not do what it claimed: a review gave a second window 28 cached
        grid axes worth 9 MB and the assertion still passed, because the
        sum never looked outside the heatmap. The cache is process-wide,
        so the invariant has to be too.
        """
        total = 0.0
        for w in qapp.topLevelWidgets():
            # findChildren does not include the widget itself, and a bare
            # PlotWidget shown on its own IS the GraphicsView -- leaving it
            # out is how a sum like this ends up blind to the window it is
            # supposed to notice.
            views = list(w.findChildren(pg.GraphicsView))
            if isinstance(w, pg.GraphicsView):
                views.append(w)
            for ax in views:
                sc = ax.scene()
                if sc is None:
                    continue
                for it in sc.items():
                    if (isinstance(it, pg.AxisItem)
                            and it.cacheMode() == CACHED and it.isVisible()):
                        br = it.boundingRect()
                        dpr = ax.devicePixelRatioF() or 1.0
                        total += br.width() * br.height() * (dpr ** 2) * 4
        return total

    def test_everything_cached_in_the_process_fits(self, win, qapp):
        need = self._process_need_bytes(qapp)
        budget = QPixmapCache.cacheLimit() * 1024
        assert need < budget / 2, (
            f"cached axes across the whole process want "
            f"{need / 1024**2:.1f} MB of a {budget / 1024**2:.0f} MB "
            f"ceiling. Oversubscribing does not fall back to uncached -- "
            f"every paint becomes a miss that renders a bitmap AND blits "
            f"it, measured at 47% slower than not caching at all.")

    def test_a_second_cached_window_is_counted(self, win, qapp):
        """The test above has to see outside this window, so this proves
        it does rather than asserting it."""
        from analysis.liq_hm_window import _cache_grid_axes
        before = self._process_need_bytes(qapp)
        pw = pg.PlotWidget()
        try:
            pw.resize(1200, 800)
            pw.show()
            pi = pw.getPlotItem()
            pi.showGrid(x=True, y=True)
            _cache_grid_axes(pi)
            for _ in range(10):
                qapp.processEvents()
            after = self._process_need_bytes(qapp)
            assert after > before, (
                "a second window's cached axes were not counted, so the "
                "budget invariant is blind to exactly what it guards")
        finally:
            pw.close()
            pw.deleteLater()
            qapp.processEvents()


class TestItRefusesWhenItWouldNotFit:
    """A cache that does not fit is slower than no cache, so it is not set.

    The appetite is one plot-area bitmap per grid axis, in device pixels,
    so it follows the screen: 9.4 MB at 1500x950, 25.6 at 2560x1440, 59.5
    at 3840x2160, and all of those again x2.25 at 150% scaling. The first
    version of this change had no check and would have gone into the
    oversubscribed state -- measured at 47% slower than not caching --
    with nothing said.

    A bare PlotItem is used rather than a PlotWidget because the budget is
    taken from the window's own screen when there is one, and only falls
    back to the primary screen otherwise. With no scene there is no
    window, which is what makes the patched screen the one consulted.
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

    def test_a_huge_screen_turns_it_off(self, qapp, monkeypatch, capsys):
        import analysis.liq_hm_window as L
        self._patch_screen(monkeypatch, 7680, 4320, 2.0)
        pi = pg.PlotItem()
        try:
            pi.showGrid(x=True, y=True)
            L._cache_grid_axes(pi)
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

    def test_an_ordinary_screen_still_caches(self, qapp, monkeypatch):
        """The control. Without it the test above would pass just as well
        if the function had stopped caching anything at all."""
        import analysis.liq_hm_window as L
        self._patch_screen(monkeypatch, 1920, 1080, 1.0)
        pi = pg.PlotItem()
        try:
            pi.showGrid(x=True, y=True)
            L._cache_grid_axes(pi)
            assert pi.getAxis("left").cacheMode() == CACHED
        finally:
            # No close(): PlotItem.close() walks into AxisItem.close(),
            # which dereferences a scene this item never had.
            del pi
            qapp.processEvents()

    def test_the_scaling_factor_is_what_decides_it(self, qapp, monkeypatch):
        """2560x1440 fits at 100% and does not at 200%.

        Same screen, same axes; only the device pixel ratio differs. The
        bitmaps are device pixels, so leaving the ratio out of the budget
        under-counts by its square -- and that mistake is invisible on
        this machine, which runs at 100%.
        """
        import analysis.liq_hm_window as L

        self._patch_screen(monkeypatch, 2560, 1440, 1.0)
        pi = pg.PlotItem()
        try:
            pi.showGrid(x=True, y=True)
            L._cache_grid_axes(pi)
            assert pi.getAxis("left").cacheMode() == CACHED, (
                "2560x1440 at 100% should fit")
        finally:
            del pi
            qapp.processEvents()

        self._patch_screen(monkeypatch, 2560, 1440, 2.0)
        pi = pg.PlotItem()
        try:
            pi.showGrid(x=True, y=True)
            L._cache_grid_axes(pi)
            assert pi.getAxis("left").cacheMode() != CACHED, (
                "the same screen at 200% wants four times the bitmaps and "
                "must be refused")
        finally:
            del pi
            qapp.processEvents()

    def test_the_ceiling_is_raised_to_fit_a_bigger_screen(self, qapp,
                                                          monkeypatch):
        """Between the two: 2560x1440 needs 25.6 MB, which does not fit
        the 32 MB default with margin but is well inside the maximum."""
        import analysis.liq_hm_window as L
        self._patch_screen(monkeypatch, 2560, 1440, 1.0)
        before = QPixmapCache.cacheLimit()
        pi = pg.PlotItem()
        try:
            pi.showGrid(x=True, y=True)
            L._cache_grid_axes(pi)
            assert pi.getAxis("left").cacheMode() == CACHED
            assert QPixmapCache.cacheLimit() >= before
            assert QPixmapCache.cacheLimit() <= L._PIXMAP_CACHE_MAX_KB
        finally:
            # No close(): PlotItem.close() walks into AxisItem.close(),
            # which dereferences a scene this item never had.
            del pi
            qapp.processEvents()


class TestThePixelsAreUnchanged:
    """Byte for byte, no tolerance.

    Only at device pixel ratio 1, which is what this machine runs. Under
    fractional scaling it is NOT identical -- 1.5 showed 4-5% of pixels
    differing, up to 22 levels on a channel -- which is also why pyqtgraph
    has this same line commented out upstream (AxisItem.py:148) beside a
    note about a Qt 4.6 bug showing cached results incorrectly.
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
    calls against repaints counts misses.
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
            f"the time axis painted {n} times in 10 renders -- more than "
            f"one per render means something new is invalidating it")

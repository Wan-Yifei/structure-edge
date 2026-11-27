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

    def test_what_is_cached_fits_with_room_to_spare(self, win):
        need = sum(
            ax.boundingRect().width() * ax.boundingRect().height() * 4
            for ax in _axes(win).values()
            if ax.cacheMode() == CACHED and ax.isVisible())
        budget = QPixmapCache.cacheLimit() * 1024
        assert need < budget / 2, (
            f"visible cached axes want {need / 1024**2:.1f} MB of a "
            f"{budget / 1024**2:.0f} MB budget; staying under half the "
            f"ceiling is what leaves room for another window to be given "
            f"the same treatment, and for Qt's own icons and styles")


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

    def test_the_price_axis_is_not_repainted_every_frame(self, win, qapp):
        ax = _axes(win)["left"]
        assert ax.cacheMode() == CACHED
        calls = {"n": 0}
        orig = type(ax).paint

        def counting(self, *a, **k):
            if self is ax:
                calls["n"] += 1
            return orig(self, *a, **k)

        type(ax).paint = counting
        try:
            for _ in range(20):
                win._plot_widget.viewport().update()
                qapp.processEvents()
        finally:
            type(ax).paint = orig
        assert calls["n"] <= 5, (
            f"the price axis repainted {calls['n']} times in 20 frames; the "
            f"cache is being invalidated and costs more than it saves")

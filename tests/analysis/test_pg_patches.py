"""A dying InfiniteLine must not take the process with it.

The full test suite aborted non-deterministically -- reaching roughly 100%
of runs on this machine by 2026-11-25 -- with no traceback, because
pytest's default fd capture swallowed it. Run with `--capture=no` and the
cause is printed in full:

    File ".../pyqtgraph/graphicsItems/InfiniteLine.py", line 336, boundingRect
    File ".../pyqtgraph/graphicsItems/InfiniteLine.py", line 322, _computeBoundingRect
        vs = self.getViewBox().size()
    AttributeError: 'NoneType' object has no attribute 'size'
    Fatal Python error: Aborted

The state that produces it: a window that was closed but not destroyed is
garbage collected, and inside the reference cycle the ViewBox is finalised
before the InfiniteLine that points at it. The line's `_cachedView` is
still warm, so GraphicsItem.viewRect() returns the cache without consulting
getViewBox() and the guard at the top of _computeBoundingRect cannot fire;
the unguarded call at the bottom then raises. Since boundingRect is a C++
virtual, the exception cannot propagate and PyQt6 calls qFatal.

These tests reproduce that state directly rather than by racing the
collector: cache warm, getViewBox() answering None. Without
analysis/pg_patches they raise AttributeError.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest                                               # noqa: E402
import pyqtgraph as pg                                      # noqa: E402
from pyqtgraph.Qt import QtCore, QtWidgets                  # noqa: E402

from analysis import pg_patches                             # noqa: E402,F401

# The real rectangle, read off the crashing object on 2026-11-25. It is a
# three-day Multi-VP window's price span -- a window a PREVIOUS test had
# closed, which is the point: the line outlives the test that made it.
CACHED = QtCore.QRectF(98.78959159829732, -3.1, 9.02405850883764, 3.2)


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


@pytest.fixture
def plot(qapp):
    """A real PlotWidget with a known range, so a line in it has a real
    boundingRect to lose."""
    def _make():
        w = pg.PlotWidget()
        w.resize(400, 300)
        w.setXRange(0, 10)
        w.setYRange(100, 110)
        return w
    return _make


def _orphaned_line():
    """A line in the exact state the collector leaves behind."""
    ln = pg.InfiniteLine(angle=90, movable=False)
    ln._cachedView = QtCore.QRectF(CACHED)     # viewRect() returns this
    ln.getViewBox = lambda: None               # the weakref is dead
    ln._boundingRect = None                    # force a recompute
    return ln


class TestTheOrphanedLineIsHarmless:

    def test_it_returns_an_empty_rect_instead_of_raising(self, qapp):
        assert _orphaned_line().boundingRect() == QtCore.QRectF()

    def test_the_cached_view_does_not_let_it_through(self, qapp):
        """The precondition, pinned: viewRect() must be returning the cache
        rather than None, or this test would pass against the unpatched
        library for the wrong reason -- pyqtgraph's own guard would have
        caught it and there would never have been a bug."""
        ln = _orphaned_line()
        assert ln.viewRect() == CACHED, (
            "viewRect returned something else, so the guard at the top of "
            "_computeBoundingRect would fire and this is not the real state")

    def test_the_unpatched_library_really_does_raise(self, qapp):
        """Without this, the two tests above could be passing because the
        bug never existed. Calls the original implementation directly."""
        ln = _orphaned_line()
        orig = pg.InfiniteLine._computeBoundingRect.__wrapped_original__
        with pytest.raises(AttributeError, match="has no attribute 'size'"):
            orig(ln)

    def test_a_deleted_cpp_object_is_also_survivable(self, qapp):
        """Observed on the same object, on the same teardown path:
        "wrapped C/C++ object of type InfiniteLine has been deleted"."""
        ln = _orphaned_line()

        def boom():
            raise RuntimeError(
                "wrapped C/C++ object of type InfiniteLine has been deleted")

        ln.getViewBox = boom
        assert ln.boundingRect() == QtCore.QRectF()


class TestAHealthyLineIsUntouched:
    """The patch must not quietly blank out lines that are fine -- that
    would trade a loud crash for a silently missing crosshair."""

    def test_a_line_in_a_real_plot_keeps_its_rectangle(self, qapp):
        plot = pg.PlotWidget()
        plot.resize(400, 300)
        plot.setXRange(0, 10)
        plot.setYRange(100, 110)
        ln = pg.InfiniteLine(angle=90, movable=False, pos=5.0)
        plot.addItem(ln)
        qapp.processEvents()
        br = ln.boundingRect()
        assert ln.getViewBox() is not None
        assert not br.isEmpty(), "a healthy line was blanked by the patch"

    def test_it_matches_what_the_library_would_have_returned(self, qapp):
        plot = pg.PlotWidget()
        plot.resize(400, 300)
        plot.setXRange(0, 10)
        plot.setYRange(100, 110)
        ln = pg.InfiniteLine(angle=90, movable=False, pos=5.0)
        plot.addItem(ln)
        qapp.processEvents()
        orig = pg.InfiniteLine._computeBoundingRect.__wrapped_original__
        assert ln._computeBoundingRect() == orig(ln)


class TestItIsInstalledWhereItIsNeeded:

    def test_applying_twice_does_not_stack_wrappers(self):
        first = pg.InfiniteLine._computeBoundingRect
        pg_patches.apply()
        assert pg.InfiniteLine._computeBoundingRect is first

    @pytest.mark.parametrize("module", [
        "analysis.dom_window", "analysis.liq_hm_window",
        "analysis.multi_vp_window", "analysis.replay_trainer",
        "analysis.trade_viewer_qt"])
    def test_every_window_module_pulls_the_patch_in(self, module):
        """Importing any window must install it, because pyqtgraph builds
        InfiniteLines we never see: LinearRegionItem makes two per region
        and this project has five of those, so patching our own call sites
        would have missed them."""
        import importlib
        importlib.import_module(module)
        assert pg_patches._APPLIED


class TestItKeepsTheRectItAlreadyReported:
    """Reverting the fallback to `return QtCore.QRectF()` left all twelve
    tests above green, which means the choice had no guard at all.

    The reason they could not catch it: `_orphaned_line()` never completes
    a successful compute, so its `_bounds` is still None (InfiniteLine.py
    initialises it there) and every assertion lands on the empty-rect
    branch -- the one case where both designs agree. The guard has to use
    a line that really did have a ViewBox first.
    """

    def _line_that_lost_its_view(self, plot_factory):
        plot = plot_factory()
        ln = pg.InfiniteLine(angle=90, movable=False, pos=5.0)
        plot.addItem(ln)
        QtWidgets.QApplication.instance().processEvents()
        healthy = ln.boundingRect()           # populates _bounds
        assert not healthy.isEmpty()
        ln.getViewBox = lambda: None          # now the weakref dies
        ln._boundingRect = None               # force a recompute
        return ln, healthy

    def test_the_degraded_rect_is_the_healthy_one(self, qapp, plot):
        ln, healthy = self._line_that_lost_its_view(plot)
        degraded = ln.boundingRect()
        assert degraded == healthy, (
            "a line that lost its ViewBox changed the geometry it reports")
        assert not degraded.isEmpty(), (
            "it collapsed to an empty rect -- the geometry changed without "
            "prepareGeometryChange(), which is the thing this avoids")

    def test_it_hands_back_a_copy_not_the_cache(self, qapp, plot):
        """Qt is free to mutate a QRectF it is given. Returning _bounds
        itself would let it edit pyqtgraph's own cache.

        The snapshot is taken with QRectF(...) on purpose. InfiniteLine's
        healthy path returns `self._bounds` ITSELF (InfiniteLine.py:332),
        so holding that reference and comparing against it later compares
        the cache with the cache -- the first version of this test did
        exactly that and could not fail.
        """
        ln, healthy = self._line_that_lost_its_view(plot)
        snapshot = QtCore.QRectF(healthy)
        assert ln._bounds == snapshot
        got = ln._computeBoundingRect()
        got.setWidth(got.width() + 1000)
        assert ln._bounds == snapshot, "the caller mutated pyqtgraph's cache"

    def test_a_line_that_never_had_a_view_still_degrades_to_empty(self,
                                                                 qapp):
        """_bounds is None until the first successful compute, so there is
        nothing to hand back and empty is the only honest answer."""
        ln = pg.InfiniteLine(angle=90, movable=False)
        assert getattr(ln, "_bounds", None) is None
        ln.getViewBox = lambda: None
        assert ln.boundingRect() == QtCore.QRectF()


class TestASwallowedFaultIsCounted:
    """Broad is defensible here -- an exception escaping a C++ virtual
    aborts the process. Broad AND silent is not: with a live ViewBox an
    unrelated AttributeError returns a healthy-looking rect and the line
    keeps drawing as though nothing happened."""

    def test_an_unrelated_fault_with_a_live_view_is_recorded(self, qapp,
                                                             plot, caplog):
        p = plot()
        ln = pg.InfiniteLine(angle=90, movable=False, pos=5.0)
        p.addItem(ln)
        qapp.processEvents()
        healthy = ln.boundingRect()
        assert ln.getViewBox() is not None, "the precheck must NOT fire"

        before = dict(pg_patches.SWALLOWED)

        # Raised from INSIDE the library's own implementation, and nothing
        # to do with teardown. pixelVectors is the first thing
        # _computeBoundingRect calls after its own None guard.
        #
        # Not by swapping __wrapped_original__: the wrapper closes over the
        # original as a local, so rebinding that attribute changes nothing
        # and the test would pass while measuring the unpatched path. That
        # is how the first version of this test failed.
        def boom(**_kw):
            raise AttributeError("'NoneType' object has no attribute 'pen'")

        ln.pixelVectors = boom
        ln._boundingRect = None
        with caplog.at_level("WARNING", logger="analysis.pg_patches"):
            got = ln.boundingRect()

        assert got == healthy, "the fault was not survived"
        assert (pg_patches.SWALLOWED.get("AttributeError", 0)
                > before.get("AttributeError", 0)), (
            "the fault was swallowed without being counted")

    def test_the_counter_starts_from_a_clean_process(self):
        """Pinned so the assertion above cannot pass on a stale number."""
        assert isinstance(pg_patches.SWALLOWED, dict)

    def test_a_deleted_cpp_object_is_counted_too(self, qapp):
        """The other swallowing branch. Without this, deleting its
        _swallowed() call leaves the suite green."""
        ln = _orphaned_line()

        def boom():
            raise RuntimeError(
                "wrapped C/C++ object of type InfiniteLine has been deleted")

        ln.getViewBox = boom
        before = pg_patches.SWALLOWED.get("RuntimeError", 0)
        assert ln.boundingRect() == QtCore.QRectF()
        assert pg_patches.SWALLOWED.get("RuntimeError", 0) > before


class TestTheKnownBugStaysQuiet:
    """The getViewBox() precheck is not redundant with the except branch,
    even though removing it leaves the rect assertions green.

    What it buys: the DOCUMENTED condition -- ViewBox gone, cache warm --
    is handled without going through the exception path, so it does not
    land in SWALLOWED. Route it through the except instead and every
    closing window logs a warning about an "unexpected" fault, which is
    the fastest way to make the counter worthless.
    """

    def test_a_dead_viewbox_does_not_register_as_a_swallowed_fault(self,
                                                                   qapp,
                                                                   plot):
        p = plot()
        ln = pg.InfiniteLine(angle=90, movable=False, pos=5.0)
        p.addItem(ln)
        qapp.processEvents()
        ln.boundingRect()
        ln.getViewBox = lambda: None
        ln._boundingRect = None

        before = dict(pg_patches.SWALLOWED)
        ln.boundingRect()
        assert pg_patches.SWALLOWED == before, (
            "the known teardown condition was logged as an unexpected "
            "fault; the precheck is what keeps the counter meaningful")

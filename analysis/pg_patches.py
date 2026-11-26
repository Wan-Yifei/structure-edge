"""Work around pyqtgraph bugs that abort the process.

Import this before building any pyqtgraph scene. `apply()` is idempotent,
so every window module can call it at import time.

Why a patch on the library class and not a subclass of our own: pyqtgraph
builds InfiniteLines itself. LinearRegionItem.py:108 constructs two of
them per region, and this project has five LinearRegionItems across
trade_viewer_qt.py and replay_trainer.py. A subclass swapped in at our own
call sites -- about fifty of them -- would miss exactly those, while being
a far wider edit.
"""
from __future__ import annotations

import logging
import os

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

_APPLIED = False

log = logging.getLogger(__name__)

#: How many times each guard has fired this process, by exception name.
#: A patch that hides faults has to at least count them -- read it from a
#: console, or assert on it in a test.
SWALLOWED: dict[str, int] = {}


def _swallowed(kind: str) -> None:
    """Count a swallowed fault, and say so the first time of each kind.

    Once per kind, not once per call: this sits on a paint path, and a
    warning per repaint would bury the console it is trying to inform.
    """
    first = kind not in SWALLOWED
    SWALLOWED[kind] = SWALLOWED.get(kind, 0) + 1
    if first:
        log.warning(
            "pg_patches: swallowed a %s inside InfiniteLine."
            "_computeBoundingRect and kept the last known rect. Letting it "
            "out would abort the process, so this is deliberate -- but it "
            "is NOT necessarily the pyqtgraph teardown bug this guard was "
            "written for. Further occurrences are counted in "
            "pg_patches.SWALLOWED, not logged.", kind, exc_info=True)


def _patch_infinite_line_bounding_rect() -> None:
    """Stop a dying InfiniteLine from taking the process with it.

    THE BUG (pyqtgraph 0.14.0, the current release as of 2026-11-25, and
    still unfixed on master):

        InfiniteLine._computeBoundingRect() ends with

            vs = self.getViewBox().size()

        with no None check, while the guard at the top of the same method
        checks something else:

            vr = self.viewRect()
            if vr is None:
                return QtCore.QRectF()

        GraphicsItem.viewRect() returns `self._cachedView` BEFORE it
        consults getViewBox(), so once that cache is populated the guard
        cannot fire. A line whose ViewBox weakref has died but whose cache
        is still warm therefore walks straight into the unguarded call.

    WHY IT KILLS THE PROCESS: boundingRect() is a C++ virtual. A Python
    exception raised inside one cannot propagate, so PyQt6 calls qFatal and
    the process aborts -- no traceback through pytest's default fd capture,
    just "Fatal Python error: Aborted".

    WHEN IT HAPPENS: garbage collection of a closed window. The finalisation
    order inside a reference cycle is arbitrary, so the ViewBox is collected
    before the line it owns in some fraction of runs and not in others.
    That is the whole source of the non-determinism.

    WHAT THIS IS AND IS NOT EVIDENCE FOR: the mechanism is measured -- see
    tests/analysis/test_pg_patches.py, which calls the library's own
    implementation and watches it raise. What is NOT measured is how often
    the live viewer would hit it. A/B over the test suite cannot tell you:
    once the suite destroys its windows deterministically, nothing there
    reaches this path at all (measured: patch on 3/8 aborts, patch off 1/8,
    p=0.57 -- no detectable effect either way). The case for shipping this
    is the mechanism plus the fact that the app really does leave its
    windows to the collector: a close() without WA_DeleteOnClose leaves
    sip.isdeleted(w) False, measured on a real window.
    """
    # _computeBoundingRect, not boundingRect: the bug is in the former, and
    # the latter has a fast path (`if self._boundingRect is None`) that
    # skips it entirely. Wrapping the outer method would add a getViewBox()
    # call to every paint; wrapping this one costs nothing until the cache
    # actually misses, which is where the fault lives.
    orig = pg.InfiniteLine._computeBoundingRect

    def _last_known(self):
        """The rect this item last reported, so the geometry never changes.

        `_bounds` is InfiniteLine's own cache of its last computed rect
        (InfiniteLine.py:325; None until the first successful compute, so
        a line that never had a ViewBox still degrades to empty).

        The honest reason, after two attempts at a better one both failed
        to reproduce: returning the SAME rect means there is no silent
        geometry change to reason about. An item that changes its
        boundingRect without prepareGeometryChange() is doing something
        Qt does not sanction, and a dying item is the worst place to find
        out whether Qt happens to tolerate it.

        Measured, so nobody re-derives it: a simple QGraphicsScene with
        BspTreeIndex answers the same as NoIndex after a silent change in
        BOTH directions -- enlarge and shrink -- so the "stale index entry"
        story this comment used to tell is NOT demonstrated, and neither is
        the "enlarging is the dangerous direction" story offered in its
        place. Keeping the rect constant costs one attribute read and makes
        the question moot, which is why it stays.
        """
        bounds = getattr(self, "_bounds", None)
        return QtCore.QRectF(bounds) if bounds is not None else QtCore.QRectF()

    def _computeBoundingRect(self):             # noqa: N802  (Qt naming)
        try:
            if self.getViewBox() is None:
                return _last_known(self)
            return orig(self)
        except RuntimeError:
            # "wrapped C/C++ object of type ... has been deleted" -- the
            # item is mid-teardown. Observed alongside the dead weakref
            # above, on the same object.
            _swallowed("RuntimeError")
            return _last_known(self)
        except AttributeError:
            # Deliberately broad, and ONLY here: anything escaping a C++
            # virtual is a process abort, which is strictly worse than a
            # line that fails to draw.
            #
            # But broad AND silent is a different thing. Measured: with a
            # live ViewBox, an AttributeError from anywhere inside orig()
            # makes this return a healthy-LOOKING rect, the line keeps
            # drawing, and nothing says a fault occurred. Note also that
            # the documented bug almost never arrives here -- the
            # getViewBox() precheck above catches it first -- so in
            # practice this branch catches things that are NOT that bug,
            # which is exactly when somebody needs to be told.
            _swallowed("AttributeError")
            return _last_known(self)

    _computeBoundingRect.__doc__ = _patch_infinite_line_bounding_rect.__doc__
    # Kept reachable so a test can call the library's own version and show
    # that the fault is real. Without it, a test asserting "no exception"
    # would pass just as happily against a version that never had the bug,
    # and nothing would be guarding anything.
    _computeBoundingRect.__wrapped_original__ = orig
    pg.InfiniteLine._computeBoundingRect = _computeBoundingRect


def apply() -> None:
    """Install every patch once. Safe to call from many modules.

    MOOMOO_NO_PG_PATCH is an EXPERIMENT switch, not a configuration knob:
    it exists so the guard can be A/B'd against itself, which is the only
    way to say what it is worth. Setting it in a viewer you actually use
    re-arms a process abort. The 2026-11-25 measurement it was built for:
    patch on 3/8 aborts, patch off 1/8, p=0.57 -- no detectable effect on
    the test suite, because the suite no longer leaves windows to the
    collector. The live viewer still does.
    """
    global _APPLIED
    if _APPLIED or os.environ.get("MOOMOO_NO_PG_PATCH"):
        return
    _patch_infinite_line_bounding_rect()
    _APPLIED = True


apply()

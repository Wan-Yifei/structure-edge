"""Count how often a cached grid axis actually has to repaint.

THE QUESTION THIS ANSWERS. analysis/qt_axis_cache.py trades a large win
on frames that leave the axes alone against a small loss on frames that
invalidate one:

    crosshair only                      -93.0%
    setData, axes untouched             -92.6%
    setTicks / setXRange / setYRange     +9.3%

so it pays unless more than 87.9% of frames invalidate an axis. That
share is a property of how the application is actually used, not of the
cache, and it is the one number the offscreen bench cannot supply.

AND 87.9% DOES NOT APPLY HERE, WHICH IS WHY NO VERDICT IS PRINTED. Those
three numbers were measured with the harness calling viewport().update()
every frame, which repaints the whole viewport. pyqtgraph's GraphicsView
runs MinimalViewportUpdate (GraphicsView.py:89) and nothing in this
application changes it, so a crosshair move exposes a sliver: on a
stand-in, only 2.26 of 14 axes redrew at all when uncached. A cache MISS
has no such discount -- Qt renders the item's whole boundingRect
whatever was exposed. Re-measured on that stand-in, twelve pairs each:

    forced   clean frame  -48.13 ms      break-even ~92.5%
    natural  clean frame   -0.93 ms      break-even ~27.0%

a 52x difference in what a clean frame is worth. The loss term's own
interval crosses zero in both regimes (+3.88 ms, [-2.04, +7.12]), so
even those break-evens are soft. A reading of, say, 60% would have
printed "under break-even -- paying" while the cache was costing. So
this module reports the share and the two quantities needed to convert
it into milliseconds later -- axes per invalidating frame, and mean
exposed area -- and leaves the verdict to a re-calibration done in this
application's own regime.

WHY A PAINT COUNTER AND NOT A PROFILER. On a cache hit Qt blits the
stored bitmap and never calls AxisItem.paint, so counting paint calls IS
counting invalidations -- exact, not sampled. A profiler reads the same
situation backwards: it would show AxisItem.paint near zero on the cached
arm, which is not evidence of CPU saved but of work moved outside every
Python frame. Chasing the difference through total process CPU means
fighting a bench whose medians drift 19% between batches.

WHAT IS COUNTED. A frame is one GraphicsView.paintEvent. An AxisItem
paints synchronously inside that event, so a frame is "invalidating" if
the axis-paint counter moved while it ran. Both numbers are per window,
because the heatmap and the main chart invalidate for different reasons
and averaging them answers nothing. Per-axis totals come out too: the two
profile panels call setXRange on every rebuild, which invalidates their
own axes every frame, and if their hit rate is near zero they are better
off not cached at all.

OFF UNLESS ASKED. Nothing is wrapped and no timer runs unless
MOOMOO_AXIS_PROBE is set, so the shipped application carries an import
and a function call that returns immediately.

    MOOMOO_AXIS_PROBE=1 uv run main.py trade_viewer_qt

MOOMOO_AXIS_PROBE_SECS sets the report interval (default 300). Reports
go to stdout with flush, because a Qt process under uv block-buffers it.
Do not set it to 0 for a real session: the whole day would then exist
only in the closing report, which a crash or a kill takes with it, and
the open and the quiet midday hours would not be distinguishable --
which is exactly where the share is expected to move.

WHAT IT COSTS WHEN ON. One Python call per axis paint and one per frame,
against frames that cost milliseconds. Measure it rather than trust it --
see tests/analysis/test_axis_probe.py, which asserts the wrapper's own
overhead against the unwrapped call. It is still an instrument, so a run
that carries it is a run for the hit rate, not for a CPU comparison.
"""

from __future__ import annotations

import atexit
import os
from collections import Counter

import pyqtgraph as pg
from PyQt6.QtWidgets import QGraphicsItem

_CACHED = QGraphicsItem.CacheMode.DeviceCoordinateCache

ENV_ON = "MOOMOO_AXIS_PROBE"
ENV_SECS = "MOOMOO_AXIS_PROBE_SECS"

# The share of invalidating frames above which the cache stops paying,
# AS MEASURED UNDER FORCED FULL-VIEWPORT REPAINTS. Kept because it is the
# only calibrated number that exists, and printed as a caveat rather than
# as a threshold -- see the docstring on why it does not apply to this
# application's regime. The arithmetic is pinned to the milliseconds it
# comes from by a test, so moving one without the other is a failure.
_SAVE_MS = 50.90 - 3.57          # crosshair frame, cached vs uncached
_LOSS_MS = 76.62 - 70.12         # setTicks/setXRange frame
BREAK_EVEN = _SAVE_MS / (_SAVE_MS + _LOSS_MS)

_installed = False
_timer = None

_axis_paints = 0                     # every AxisItem.paint, all windows
_cached_paints = 0                   # only those on a cached axis -- see below
_per_axis: Counter = Counter()       # label -> paints
_frames: Counter = Counter()         # view -> paintEvents
_dirty: Counter = Counter()          # view -> paintEvents that repainted a CACHED axis
_prev: dict[str, tuple[int, int]] = {}   # view -> (frames, dirty) at last report

# Added after review. The share of invalidating frames is not on its own
# convertible into milliseconds, because what a clean frame saves depends
# on how much of the viewport was exposed: pyqtgraph's GraphicsView runs
# in MinimalViewportUpdate (GraphicsView.py:89, and nothing here changes
# it), so a crosshair move exposes a sliver and only the axes under it
# redraw -- measured at 2.26 of 14 on a stand-in -- while a cache MISS
# always renders the item's whole boundingRect. Without these two, a
# session's reading cannot be turned into a cost afterwards and the day
# has to be run again.
_dirty_axes: Counter = Counter()     # view -> cached-axis paints inside frames
_area: dict[str, list] = {}          # view -> [summed exposed fraction, frames]

# Paints that happened outside any paintEvent. QWidget.render() and
# QGraphicsScene.render() both do this -- measured at 14 cached-axis
# paints and 0 frames -- and neither is used in this application today.
# Counted rather than assumed, because a frame that is missed lowers the
# share, which is the direction that flatters the cache.
_orphans = 0
_depth = 0

# Views that have held a cached axis at some point. The live walk cannot
# see a closed window -- the heatmap is WA_DeleteOnClose -- and without
# this the closing report silently downgrades it to "no cached axes",
# which is the row the whole exercise started from.
_ever_cached: set = set()
_ever_checks: Counter = Counter()

_orig_axis_paint = None
_orig_paint_event = None


def _window_tag(obj) -> str:
    """The window's CLASS name, not its title.

    A title carries the symbol and the timeframe and changes while the
    session runs, which would split one window's counts across several
    keys halfway through a reading.
    """
    try:
        win = obj.window()
    except Exception:                                   # noqa: BLE001
        return "?"
    return type(win).__name__ if win is not None else "?"


def _view_tag(view) -> str:
    """One row per GraphicsView, not per window.

    The main chart window holds THREE of them: the GraphicsLayoutWidget
    carrying the seven stacked plots, and a PlotWidget for each profile
    panel. Those panels call setXRange on every rebuild, which
    invalidates their own axes on every frame they draw, so folding them
    into the window's total pins it near 100% and hides whatever the
    seven plots are actually doing -- which is the number this exists to
    read. They are also the two candidates for being dropped from the
    cache, and that decision needs them counted apart.

    objectName is preferred when the widget has one, so naming a view
    later makes the report readable with no change here.
    """
    tag = getattr(view, "_mm_probe_view_tag", None)
    if tag is not None:
        return tag
    wtag = _window_tag(view)
    name = view.objectName()
    if name:
        tag = f"{wtag}/{name}"
    else:
        win = view.window()
        n = getattr(win, "_mm_probe_views", 0) + 1
        try:
            win._mm_probe_views = n
        except Exception:                               # noqa: BLE001
            pass
        tag = f"{wtag}/{type(view).__name__}#{n}"
    try:
        view._mm_probe_view_tag = tag
    except Exception:                                   # noqa: BLE001
        pass
    return tag


def _plot_tag(ax) -> str:
    """A stable short name for the plot an axis belongs to.

    Stored on the PlotItem rather than keyed by id(): the heatmap window
    is WA_DeleteOnClose and rebuilt on every open, and CPython reuses the
    id of a collected object, so an id-keyed table can hand a new plot the
    label of a dead one.
    """
    vb = ax.linkedView()
    pi = vb.parentItem() if vb is not None else None
    if pi is None:
        return "-"
    tag = getattr(pi, "_mm_probe_tag", None)
    if tag is not None:
        return tag
    tag = ""
    title = getattr(pi, "titleLabel", None)
    if title is not None:
        try:
            tag = (title.text or "").strip()
        except Exception:                               # noqa: BLE001
            tag = ""
    if not tag:
        try:
            tag = (pi.getAxis("left").labelText or "").strip()
        except Exception:                               # noqa: BLE001
            tag = ""
    if not tag:
        # Last resort: a per-window running index, so two unnamed plots
        # never collapse into one row.
        scene = pi.scene()
        n = getattr(scene, "_mm_probe_n", 0) + 1
        try:
            scene._mm_probe_n = n
        except Exception:                               # noqa: BLE001
            pass
        tag = f"plot{n}"
    tag = tag[:18]
    try:
        pi._mm_probe_tag = tag
    except Exception:                                   # noqa: BLE001
        pass
    return tag


def _axis_label(ax) -> str:
    lbl = getattr(ax, "_mm_probe_label", None)
    if lbl is not None:
        return lbl
    scene = ax.scene()
    views = scene.views() if scene is not None else []
    # The same view tag the frame counter uses, so an axis row can be
    # matched to the frames it was counted against without guessing.
    view = _view_tag(views[0]) if views else "?"
    lbl = f"{view}/{_plot_tag(ax)}/{ax.orientation}"
    try:
        ax._mm_probe_label = lbl
    except Exception:                                   # noqa: BLE001
        pass
    return lbl


def _axis_paint(self, p, opt, widget):
    """Count the paint, and separately count it if the axis is cached.

    ONLY A CACHED AXIS CAN INVALIDATE. The two profile panels show the
    grid on one orientation only, so their left axes are not cached by
    design -- they are 43x512 strips costing 0.138 ms, not the 13.7 ms
    full-area replay the cache exists for. Counting their paints as
    invalidations reads 100% on a panel where the cache is neither
    paying nor costing anything, which is a confident wrong answer
    rather than a missing one.
    """
    global _axis_paints, _cached_paints, _orphans
    _axis_paints += 1
    _per_axis[_axis_label(self)] += 1
    if self.cacheMode() == _CACHED:
        _cached_paints += 1
        if _depth == 0:
            _orphans += 1
    return _orig_axis_paint(self, p, opt, widget)


def _paint_event(self, ev):
    global _depth
    before = _cached_paints
    _depth += 1
    try:
        return _orig_paint_event(self, ev)
    finally:
        _depth -= 1
        tag = _view_tag(self)
        _frames[tag] += 1
        got = _cached_paints - before
        if got > 0:
            _dirty[tag] += 1
            _dirty_axes[tag] += got
        # Walked only on a view's first few frames, never on every one:
        # scene.items() is O(items) and the real chart holds thousands.
        # The cache is applied at window construction, before any frame,
        # so the first frames are enough -- a handful rather than one in
        # case a window paints before its constructor is finished.
        if tag not in _ever_cached and _ever_checks[tag] < 5:
            _ever_checks[tag] += 1
            if _ever_cached_here(self):
                _ever_cached.add(tag)
        slot = _area.setdefault(tag, [0.0, 0])
        slot[0] += _exposed_fraction(self, ev)
        slot[1] += 1


def _exposed_fraction(view, ev) -> float:
    """How much of the viewport this frame actually repainted.

    The number the cost model needs and the offscreen bench never had:
    its harness called viewport().update() every frame, which exposes
    everything, while the application exposes a sliver. A clean frame's
    saving scales with this; a miss does not.
    """
    try:
        r = ev.rect()
        vp = view.viewport().rect()
        total = vp.width() * vp.height()
        return (r.width() * r.height()) / total if total else 0.0
    except Exception:                                   # noqa: BLE001
        return 0.0


def _ever_cached_here(view) -> bool:
    try:
        sc = view.scene()
        if sc is None:
            return False
        return any(isinstance(it, pg.AxisItem) and it.cacheMode() == _CACHED
                   for it in sc.items())
    except Exception:                                   # noqa: BLE001
        return False


def _cache_state() -> tuple[dict[str, bool], Counter]:
    """Which labelled axes currently hold a device-coordinate cache.

    Read live at report time rather than recorded when the cache was set:
    cache_grid_axes can refuse (budget), and a window opened later can
    push the process over the ceiling, so what was true at construction
    need not be true now.
    """
    from PyQt6.QtWidgets import QApplication

    out: dict[str, bool] = {}
    per_view: Counter = Counter()
    try:
        tops = list(QApplication.topLevelWidgets())
    except Exception:                                   # noqa: BLE001
        # The final report can run after the application object is gone.
        # A reading without the cached/uncached column is still a reading;
        # losing the whole report to decorate it is not a trade worth making.
        return out, per_view
    for w in tops:
        views = list(w.findChildren(pg.GraphicsView))
        if isinstance(w, pg.GraphicsView):
            views.append(w)
        for v in views:
            sc = v.scene()
            if sc is None:
                continue
            for it in sc.items():
                if isinstance(it, pg.AxisItem):
                    is_cached = it.cacheMode() == _CACHED
                    out[_axis_label(it)] = is_cached
                    if is_cached:
                        per_view[_view_tag(v)] += 1
    return out, per_view


def report(final: bool = False, detail: bool | None = None) -> str:
    """One reading, as the text that is printed. Returned for testing.

    The per-axis table runs to one row per grid axis per orientation --
    36 of them on the main chart -- and a whole session at one reading a
    minute would bury the three lines that carry the answer. So the
    periodic reading is the per-view summary and the table comes with
    the closing one. Pass detail=True to force it.
    """
    if detail is None:
        detail = final
    lines = ["[axis-probe] final" if final else "[axis-probe]"]
    if not _frames:
        lines.append("  no frames yet")
        return "\n".join(lines)

    cached, _n_cached = ({}, Counter()) if not detail else _cache_state()
    w = max([len(k) for k in _frames] + [18])
    for view in sorted(_frames):
        f, d = _frames[view], _dirty[view]
        pf, pd = _prev.get(view, (0, 0))
        df, dd = f - pf, d - pd
        _prev[view] = (f, d)
        share = d / f if f else 0.0
        held = "cached" if view in _ever_cached else "no cached axes"
        a_sum, a_n = _area.get(view, (0.0, 0))
        per_dirty = _dirty_axes[view] / d if d else 0.0
        lines.append(
            f"  {view:<{w}s} frames {f:>7,d}  invalidating {d:>7,d} "
            f"({share:6.1%})  {held}")
        lines.append(
            f"  {'':<{w}s} per invalidating frame {per_dirty:4.1f} axes; "
            f"mean exposed {a_sum / a_n:6.1%} of the viewport"
            if a_n else f"  {'':<{w}s} (no area samples)")
        if df:
            lines.append(
                f"  {'':<{w}s} this interval: {df:,d} frames, {dd:,d} "
                f"invalidating ({dd / df:.1%})")
    if _orphans:
        lines.append(f"  WARNING {_orphans:,d} cached-axis paints happened "
                     f"OUTSIDE any frame -- the share above is too low")
    # NO VERDICT. The -93%/+9.3% that 87.9% comes from were measured with
    # the harness forcing a full viewport repaint every frame, and this
    # application runs MinimalViewportUpdate: on a stand-in, the saving on
    # a clean frame fell from 48.13 ms to 0.93 ms and the break-even with
    # it, from ~92% to ~27%, while the loss term's own interval crosses
    # zero. A share between those two numbers would print "paying" while
    # the cache was costing. Until save and loss are re-measured in this
    # application's own regime, the honest output is the share and the two
    # quantities needed to convert it -- not a verdict.
    lines.append(f"  no verdict: {BREAK_EVEN:.1%} was calibrated under "
                 f"forced full-viewport repaints; this app runs "
                 f"MinimalViewportUpdate. See the module docstring.")
    if not detail:
        return "\n".join(lines)
    lines.append("  per axis (paints, and whether it is cached now):")
    # Cached axes that never painted are the whole point -- a 100% hit
    # rate shows up as an absence, and an absence is indistinguishable
    # from an axis that does not exist. List them at zero.
    rows = dict(_per_axis)
    for lbl, is_cached in cached.items():
        if is_cached:
            rows.setdefault(lbl, 0)
    for lbl, n in sorted(rows.items(), key=lambda kv: (-kv[1], kv[0])):
        # "gone" means the live walk could not find the axis any more --
        # a closed WA_DeleteOnClose window, which is how the heatmap
        # ends. Its counts are still real; it is only the cache column
        # that cannot be read.
        mark = {True: "cached", False: "UNCACHED"}.get(cached.get(lbl), "gone")
        lines.append(f"     {lbl:<48s} {n:>8,d}  {mark}")
    return "\n".join(lines)


def _tick() -> None:
    print(report(), flush=True)


def install(interval_s: float | None = None) -> bool:
    """Wrap AxisItem.paint and GraphicsView.paintEvent. Idempotent.

    Call before any window is built. Returns False if already installed,
    so a second call cannot double-count by wrapping the wrapper.
    """
    global _installed, _orig_axis_paint, _orig_paint_event, _timer
    if _installed:
        return False
    from PyQt6.QtCore import QTimer

    _orig_axis_paint = pg.AxisItem.paint
    _orig_paint_event = pg.GraphicsView.paintEvent
    pg.AxisItem.paint = _axis_paint
    pg.GraphicsView.paintEvent = _paint_event
    _installed = True

    if interval_s is None:
        try:
            interval_s = float(os.environ.get(ENV_SECS, "300"))
        except ValueError:
            interval_s = 300.0
    if interval_s > 0:
        _timer = QTimer()
        _timer.timeout.connect(_tick)
        _timer.start(int(interval_s * 1000))
    # Belt for entry points that exit normally. The viewer is not one of
    # them -- it ends on os._exit, which runs no atexit handler -- so it
    # calls final_report() itself. Registering both is harmless: the
    # second call prints a second reading, it does not double any count.
    atexit.register(final_report)
    print(f"[axis-probe] on, reporting every {interval_s:.0f}s", flush=True)
    return True


def final_report() -> None:
    """Print the closing reading. Safe to call when the probe is off."""
    if not _installed:
        return
    print(report(final=True), flush=True)


def uninstall() -> None:
    """Put the original methods back. For tests; the app never calls it."""
    global _installed, _timer
    if not _installed:
        return
    pg.AxisItem.paint = _orig_axis_paint
    pg.GraphicsView.paintEvent = _orig_paint_event
    if _timer is not None:
        _timer.stop()
        _timer = None
    _installed = False


def reset() -> None:
    """Zero the counters without touching the wrapping. For tests."""
    global _axis_paints, _cached_paints, _orphans, _depth
    _axis_paints = 0
    _cached_paints = 0
    _orphans = 0
    _depth = 0
    _per_axis.clear()
    _frames.clear()
    _dirty.clear()
    _dirty_axes.clear()
    _area.clear()
    _ever_cached.clear()
    _ever_checks.clear()
    _prev.clear()


def maybe_install() -> bool:
    """Install only if MOOMOO_AXIS_PROBE is set to something truthy.

    This is what the application calls. Everything above stays inert --
    nothing is wrapped, no timer exists, no counter moves -- when the
    variable is unset, which is how it ships.
    """
    if os.environ.get(ENV_ON, "").strip().lower() in ("", "0", "false", "no"):
        return False
    return install()

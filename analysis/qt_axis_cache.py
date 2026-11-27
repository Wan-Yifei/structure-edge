"""Render pyqtgraph grid axes once and blit them, within a shared budget.

WHY. A grid-enabled AxisItem's boundingRect is unioned with its linked
view's (AxisItem.py:957), so its QPicture is replayed over the whole plot
area on every repaint -- 13.7 ms for the heatmap's price axis against
0.138 ms for a non-grid 43x512 strip holding a near-identical number of
painter commands. The cost is the area, not the drawing.
DeviceCoordinateCache turns that replay into a blit of a cached bitmap.

WHAT IT BUYS, measured per frame class on a 7-plot chart at 1600x1000,
four interleaved rounds, first discarded, medians:

    frame moves the crosshair only      50.90 ms -> 3.57 ms   -93.0%
    frame calls setData, axes untouched 49.68 ms -> 3.67 ms   -92.6%
    frame calls setTicks/setXRange/setYRange
                                        70.12 ms -> 76.62 ms   +9.3%

So it pays unless more than 87.9% of frames invalidate an axis. That is a
wide margin, and it is the reason this is worth doing -- not the 7x figure
quoted in v0.104.8, which came from a synthetic stand-in with no candles,
no crosshair and no overlays and does not extrapolate to a real window.

A FREE RULER FOR CHECKING IT LATER. On a cache hit Qt blits without
calling paint at all, so counting AxisItem.paint IS counting
invalidations. Measured on that same chart: 20 crosshair frames paint the
axes 280 times uncached and 0 times cached (100% hit rate); 20 redraw
frames, 280 against 140 (50%). Instrumenting paint in the real
application reads the hit rate directly, which is cheaper and sharper
than chasing an 8% CPU difference through a bench whose medians drift 19%
between batches.

WHY THERE IS A BUDGET AT ALL. Qt keeps these bitmaps in the process-wide
QPixmapCache, whose default ceiling is 10 MB. Oversubscribing it does NOT
degrade to "uncached": every paint becomes a miss that renders a bitmap
AND blits it, which is strictly more work than drawing directly.
Measured, both windows cached under the 10 MB default: 86.8 ms a frame
against 62.0 uncached -- a 40% regression that looks like nothing from
outside. So the choice is a cache that fits or no cache, never a cache
that nearly fits.
"""

from __future__ import annotations

from typing import Iterable

import pyqtgraph as pg
from PyQt6.QtWidgets import QGraphicsItem

_CACHED = QGraphicsItem.CacheMode.DeviceCoordinateCache

# The floor. Enough for one window at any ordinary size, so the common
# case never has to compute its way up from Qt's 10 MB default.
_PIXMAP_CACHE_KB = 32 * 1024

# The hard stop, and the one number here that is a policy rather than a
# measurement: raising the ceiling costs nothing by itself (Qt inserts on
# demand), so this is a statement about how much resident memory the
# bitmaps may occupy, not about how much is reserved.
#
# What it admits, at two grid axes per window-area (see _window_need_kb):
#
#     2560x1440 DPR 1.00, two windows    56.3 MB   ok
#     2560x1440 DPR 1.25, two windows    92.2 MB   ok
#     2560x1440 DPR 1.50, two windows   132.7 MB   refused
#     3840x2160 DPR 1.00, one window     66.4 MB   ok
#     3840x2160 DPR 1.00, two windows   132.7 MB   refused
#
# 96 MB was the previous value and refused the user's own monitor at 125%
# scaling, which is an ordinary setting rather than an exotic one.
_PIXMAP_CACHE_MAX_KB = 128 * 1024

# A fixed margin rather than a percentage. The appetite below is already
# an upper bound, and multiplying a growing total by 1.3 charges the
# slack again for every window -- the second window pays 30% of both.
# This covers Qt's own icons and styles, which share the cache.
_HEADROOM_KB = 8 * 1024

# Grid axes that actually allocate a bitmap, per plot.
#
# PlotItem.updateGrid calls setGrid on all four orientations, so a
# showGrid(x=True, y=True) plot reports four grid axes -- but top and
# right are hidden on every plot in this application, a hidden
# QGraphicsItem is never painted, and the DeviceCoordinateCache bitmap is
# allocated lazily inside the paint path. Verified two ways: the hidden
# axes' paint counters stay at 0 over 15 forced frames, and probing the
# cache by eviction (insert a 2 MiB sentinel, bisect the ceiling for the
# point where it survives) reads 9.28 MB for a 7-plot window whether the
# cache mode is set on two axes or on all four -- not one byte more.
#
# Counting four was what the first version did, and it is why it refused
# to cache anything at 3840x2160 or at 150% scaling: it asked for twice
# what it would have used.
_VISIBLE_GRID_AXES_PER_PLOT = 2


def _plot_axes(plot_item) -> list:
    """The grid-enabled axes of one plot, in any orientation."""
    axes = [plot_item.getAxis(s) for s in ("left", "bottom", "right", "top")]
    return [a for a in axes if a is not None and a.grid is not False]


def _scenes_of(widget) -> list:
    """Every QGraphicsScene reachable from one top-level widget.

    findChildren does not include the widget itself, and a bare
    PlotWidget shown on its own IS the GraphicsView -- leaving it out is
    how a sum like this ends up blind to the window it is meant to see.
    """
    views = list(widget.findChildren(pg.GraphicsView))
    if isinstance(widget, pg.GraphicsView):
        views.append(widget)
    scenes = []
    for v in views:
        sc = v.scene()
        if sc is not None and not any(sc is s for s in scenes):
            scenes.append(sc)
    return scenes


def _windows_holding_cached_axes(exclude=None) -> list:
    """Top-level windows that already have at least one cached grid axis.

    Read live rather than accumulated in a module-level total. The
    heatmap window is constructed fresh on every open and destroyed on
    close (WA_DeleteOnClose), so a running total that only ever grows
    counts the same window again each time it is reopened: with the
    main chart plus two open-and-close cycles of the heatmap the total
    passes the ceiling and the optimisation turns itself off, silently,
    in the middle of a session. Counting what is on screen right now
    costs one walk of the widget tree at construction time and cannot
    drift.
    """
    from PyQt6.QtWidgets import QApplication

    out = []
    for w in QApplication.topLevelWidgets():
        if exclude is not None and w is exclude:
            continue
        for sc in _scenes_of(w):
            if any(isinstance(it, pg.AxisItem) and it.cacheMode() == _CACHED
                   for it in sc.items()):
                out.append(w)
                break
    return out


def _window_need_kb(plot_item) -> tuple[float, int, int, float]:
    """Upper bound on one window's bitmaps, in KB, plus the screen it is on.

    THE MODEL: a window's plots partition its area, so the bitmaps a
    whole window wants are about (grid axes on one plot) x (window area),
    NOT (plots) x (axes) x (area). The main chart stacks seven plots and
    adds two profile panels beside them, and all of it together wants
    roughly what a single full-window plot wants.

    Measured by eviction probe, predicted against actual:

        1500x950   1 plot   10.27 MB   1.89 x window area
        1500x950   3 plots   9.90 MB
        1500x950   7 plots   9.28 MB   1.71 x
        1500x950  11 plots   8.53 MB   1.57 x
        1920x1080  7 plots  13.76 MB
        2560x1400  1 plot   26.33 MB
        real TradeViewerQt at 1733x1000: 8.59 MB against 8.48 predicted

    So two axes' worth of window area is an upper bound everywhere
    measured, by 6% to 35%. Folding rows does not change the total --
    1600x1000 with seven rows open is 8.78 MB and with three folded away
    8.90 MB, because the rows that remain grow taller.

    SIZED FOR THE SCREEN, NOT THE WINDOW, because the window is
    resizable and this runs once at construction -- but the larger of
    the two is taken, because a window CAN be bigger than one screen.
    Dragged across two monitors is the case on a desktop; offscreen
    rendering is the case in the test suite, where the screen reports
    800x800 and the real chart is built at 1733x1000 and genuinely
    holds 8.48 MB against a 4.9 MB screen-sized reserve. Taking the
    maximum is never worse than taking the screen, and it closes the
    hole for any window already oversized when this runs. A window
    dragged onto a second monitor afterwards is still outside the
    bound; at 2560x1440 DPR 1 the ceiling leaves 70 MB of slack, which
    covers it.

    NOT SIZED BY WHAT IS VISIBLE NOW, either. The main chart is built
    with its KD/CVD/ADI/DCVD rows folded away -- 8 of its 32 grid axes
    are visible at construction -- so summing visible axes would
    under-reserve and then oversubscribe the moment a row is opened.
    """
    from PyQt6.QtGui import QGuiApplication

    scene = plot_item.scene()
    views = scene.views() if scene is not None else []
    view = views[0] if views else None
    screen = (view.window().screen() if view is not None else None) \
        or QGuiApplication.primaryScreen()
    dpr = screen.devicePixelRatio() if screen is not None else 1.0
    size = screen.size() if screen is not None else None
    w, h = (size.width(), size.height()) if size is not None else (1920, 1080)
    if view is not None:
        win = view.window()
        w = max(w, win.width())
        h = max(h, win.height())
    need_kb = _VISIBLE_GRID_AXES_PER_PLOT * w * h * (dpr ** 2) * 4 / 1024
    return need_kb, w, h, dpr


def cache_grid_axes(plot_items: Iterable, *, who: str) -> bool:
    """Cache one window's grid axes, if the whole process still fits.

    Call this ONCE per window with every plot it owns. Calling it per
    plot is what the first version did, and the budget it computed did
    not survive being summed: each call estimated a whole screen's worth
    of bitmaps for its own plot, then raised the ceiling only on the
    first call, because the ceiling was already high enough by the
    second. Seven plots got one plot's reservation.

    Returns True if the axes were cached.

    THE CACHE MODE GOES ON ALL GRID AXES, including the hidden ones. They
    cost nothing while hidden (verified by eviction probe) and inherit
    the cache for free if a folded row is later opened.

    WHAT IT COSTS. The bitmaps themselves -- 8.6 MB for the main chart at
    1733x1000, 18.8 MB with every row open and maximised to 2560x1400 --
    and 9.3% on frames that invalidate an axis, against 93% saved on
    frames that do not.

    AND WHAT IT CHANGES. The output is NOT pixel-identical. On the main
    chart at DPR 1, 53 of 1,733,000 pixels differ (0.0031%): grid lines
    one colour step down, and the occasional single axis-line pixel
    dropping to the background. An A/A control (uncached against
    uncached, cached against cached) differs by 0 pixels, so those 53 are
    the cache and not the capture. At DPR 1.5 it is 2.8-7.1% of pixels,
    worse than the heatmap's 4-5%; pyqtgraph has this same line commented
    out upstream (AxisItem.py:148) beside a note about a Qt 4.6 bug
    showing cached results incorrectly, which is 2010 and this is Qt 6.11
    -- but it is why the pixel tests next to this exist.
    """
    from PyQt6.QtGui import QPixmapCache

    plot_items = list(plot_items)
    axes = [a for pi in plot_items for a in _plot_axes(pi)]
    if not axes:
        return False

    per_window_kb, w, h, dpr = _window_need_kb(plot_items[0])

    scene = plot_items[0].scene()
    views = scene.views() if scene is not None else []
    mine = views[0].window() if views else None
    others = len(_windows_holding_cached_axes(exclude=mine))
    need_kb = (others + 1) * per_window_kb

    want_kb = max(_PIXMAP_CACHE_KB, int(need_kb) + _HEADROOM_KB)
    if want_kb > _PIXMAP_CACHE_MAX_KB:
        # flush, because a Qt process under uv block-buffers stdout and a
        # line that never appears is the same as no line at all.
        print(f"[{who}] axis pixmap cache OFF: this window plus "
              f"{others} already cached would want {need_kb / 1024:.1f} MB "
              f"on a {w}x{h} screen at DPR {dpr:.2f}, over the "
              f"{_PIXMAP_CACHE_MAX_KB // 1024} MB ceiling -- an "
              f"oversubscribed cache is slower than none", flush=True)
        return False

    if QPixmapCache.cacheLimit() < want_kb:
        QPixmapCache.setCacheLimit(want_kb)
    for ax in axes:
        ax.setCacheMode(_CACHED)
    return True

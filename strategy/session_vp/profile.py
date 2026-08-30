"""Session-anchored volume profile: Point of Control / Value Area High-Low.

Builds on strategy.smc.fvg.compute_volume_profile (OHLCV-only, no tick-data
dependency -- see strategy/session_vp/README or the plan this package was
built from: not every session/date has tick coverage, so this profile is
deliberately kept independent of ticks.db).
"""

from __future__ import annotations

import numpy as np


def compute_value_area(
    edges: np.ndarray,
    bin_vols: np.ndarray,
    va_pct: float = 0.70,
) -> dict:
    """Return {"poc": float, "vah": float, "val": float} from a volume profile.

    Standard market-profile algorithm: start at POC (the bin with the most
    volume) and expand outward, at each step adding whichever side (above or
    below) has more volume, until va_pct of total volume is covered.

    A zero-volume neighbor is stepped through rather than treated as a wall
    -- treating it as a stopping point made VAH/VAL collapse onto POC
    whenever the immediately adjacent bin happened to be empty, even though
    real volume existed further out (this exact bug was found and fixed for
    trade_viewer_qt.py's _compute_poc_vah_val; this is a fresh
    implementation of the same corrected algorithm, not an import, since
    backtest/ has no dependency on the GUI module -- see -1.0 sentinel
    below, used instead of 0.0 so a genuine zero-volume bin still loses the
    comparison and expansion keeps moving).

    Returns all-zero when the profile is empty/degenerate (edges/bin_vols
    None or bin_vols sums to 0).
    """
    if edges is None or bin_vols is None or bin_vols.size == 0:
        return {"poc": 0.0, "vah": 0.0, "val": 0.0}

    centers = (edges[:-1] + edges[1:]) / 2
    total = float(bin_vols.sum())
    if total <= 0:
        return {"poc": 0.0, "vah": 0.0, "val": 0.0}

    poc_idx = int(np.argmax(bin_vols))
    # All bins equal (e.g. a single-bar or perfectly uniform window): argmax
    # would return index 0 (misleadingly "the lowest price"), use the median
    # bin instead -- mirrors _compute_poc_vah_val's same edge-case handling.
    if np.all(bin_vols == bin_vols[poc_idx]):
        poc_idx = len(bin_vols) // 2

    poc = float(centers[poc_idx])
    lo_idx = hi_idx = poc_idx
    cumvol = float(bin_vols[poc_idx])
    n = len(bin_vols)
    while cumvol / total < va_pct and (hi_idx + 1 < n or lo_idx - 1 >= 0):
        above = float(bin_vols[hi_idx + 1]) if hi_idx + 1 < n else -1.0
        below = float(bin_vols[lo_idx - 1]) if lo_idx - 1 >= 0 else -1.0
        if above >= below:
            hi_idx += 1
            cumvol += float(bin_vols[hi_idx])
        else:
            lo_idx -= 1
            cumvol += float(bin_vols[lo_idx])

    return {"poc": poc, "vah": float(centers[hi_idx]), "val": float(centers[lo_idx])}

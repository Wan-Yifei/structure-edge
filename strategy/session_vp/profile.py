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


def find_nodes(
    edges: np.ndarray,
    bin_vols: np.ndarray,
    min_peak_frac: float = 0.35,
    half: float = 0.5,
) -> list[tuple[float, float, float]]:
    """High-volume nodes as (low_edge, high_edge, share_of_volume), low first.

    A node is a peak in the profile together with the contiguous bins around
    it holding at least `half` of that peak's volume -- the conventional
    half-height width -- and overlapping spans are merged into one node.

    This is a different object from the value area and is not a substitute for
    it. The value area is a single band covering `va_pct` of volume around the
    POC; on a session that traded two separate shelves it stretches across the
    gap between them, so its VAL can sit at a price where nothing changed
    hands. A node is local: a session with two shelves has two of them, each
    with its own edges. Measured over the regular session, 40% of profiles
    carry more than one node, and on those the node's low edge and the
    statistical VAL differ by about 1.35 against 0.57 when there is only one.

    `min_peak_frac` keeps ripples from counting: without it every wobble in a
    thin profile becomes a node and the nearest one is always within a cent of
    any price you care to test. Both defaults were confirmed against charts by
    eye -- they reproduced the band and the edge a trader had drawn on two
    sessions, one single-node and one with a clear upper and lower shelf.

    Returns [] on an empty or degenerate profile, matching compute_value_area.
    """
    if edges is None or bin_vols is None or bin_vols.size < 3:
        return []
    v = np.asarray(bin_vols, dtype=float)
    total, top = float(v.sum()), float(v.max())
    if total <= 0 or top <= 0:
        return []
    if np.all(v == v[0]):
        # Nothing stands out, so nothing is a shelf. Without this a flat
        # profile reports one node spanning the whole range, because every
        # interior bin ties with its neighbours and counts as a peak.
        return []

    n = len(v)
    # Endpoints are candidates too: a shelf that runs to the edge of the
    # profile is still a shelf, and skipping them lost exactly the two-shelf
    # case this function exists for.
    peaks = [i for i in range(n)
             if v[i] >= min_peak_frac * top
             and v[i] >= (v[i - 1] if i else -1.0)
             and v[i] >= (v[i + 1] if i + 1 < n else -1.0)]
    # Tallest first, so a big shelf claims its bins before a ripple beside it
    # can. Extending from the small peak first let it swallow the large one.
    peaks.sort(key=lambda i: -v[i])

    spans: list[list] = []          # [low_edge, high_edge, share, peak_i, peak_h]
    claimed = np.zeros(n, dtype=bool)
    for i in peaks:
        if claimed[i]:
            continue
        cut = half * v[i]
        lo = i
        while lo > 0 and v[lo - 1] >= cut and v[lo - 1] <= v[lo]:
            lo -= 1
        hi = i
        while hi < n - 1 and v[hi + 1] >= cut and v[hi + 1] <= v[hi]:
            hi += 1
        # Stop at the valley, not at half height alone: `v[j] <= v[j-1]` keeps
        # the run from climbing back up into the next shelf.
        claimed[lo:hi + 1] = True
        spans.append([float(edges[lo]), float(edges[hi + 1]),
                      float(v[lo:hi + 1].sum() / total), i, float(v[i])])

    # Merge by prominence, not by whether the spans touch. Two peaks belong to
    # one shelf when the lowest bin between them still holds at least `half`
    # of the taller peak; a deeper valley than that separates two shelves.
    # Touch-based merging was tried first and could not tell the two apart:
    # it joined shelves either side of a real gap on one session while
    # splitting a single shelf on another.
    spans.sort(key=lambda s_: s_[0])
    merged: list[list] = []
    for span in spans:
        if merged:
            pa, pb = merged[-1][3], span[3]
            valley = float(v[min(pa, pb) + 1: max(pa, pb)].min()) if abs(pa - pb) > 1 else float(min(v[pa], v[pb]))
            if valley >= half * max(merged[-1][4], span[4]):
                merged[-1][1] = max(merged[-1][1], span[1])
                merged[-1][2] = max(merged[-1][2], span[2])
                if span[4] > merged[-1][4]:
                    merged[-1][3], merged[-1][4] = span[3], span[4]
                continue
        merged.append(span)
    return [(lo, hi, share) for lo, hi, share, _, _ in merged]


def in_node(price: float, nodes: list[tuple[float, float, float]]) -> bool:
    """Is *price* inside any node's span (edges inclusive)?"""
    return any(lo <= price <= hi for lo, hi, _ in nodes)

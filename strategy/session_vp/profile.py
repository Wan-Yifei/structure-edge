"""Session-anchored volume profile: Point of Control / Value Area High-Low.

Builds on strategy.smc.fvg.compute_volume_profile (OHLCV-only, no tick-data
dependency -- see strategy/session_vp/README or the plan this package was
built from: not every session/date has tick coverage, so this profile is
deliberately kept independent of ticks.db).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


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


class Node(NamedTuple):
    """One high-volume shelf. `peak` is the shelf's own POC."""

    lo: float
    hi: float
    peak: float
    share: float


# Bin width as a share of price. The profile must be binned finely enough
# that the bin width, not the bandwidth below, stops being what separates
# two shelves -- see bins_for_price().
BIN_PCT = 0.02
# Smoothing kernel width, also a share of price. This is the scale at which
# two busy prices stop being one shelf.
BW_PCT = 0.10
# How far the profile must fall between a peak and higher ground before the
# peak counts as its own shelf, as a fraction of that peak's own height.
PROMINENCE = 0.25
# A shelf must hold at least this much of the session's volume.
MIN_SHARE = 0.05


def bins_for_price(low: float, high: float, bin_pct: float = BIN_PCT) -> int:
    """Bin count giving bins `bin_pct` of price wide over [low, high].

    compute_volume_profile takes a bin *count*, which makes the bin width the
    session's range divided by that count: a quiet day and a wild one get
    resolutions that differ severalfold, and no peak rule can be consistent
    across them. Sizing the count from the price instead keeps the bin width
    fixed in the only unit that means anything here.
    """
    if not (high > low > 0):
        return 0
    return max(50, int(round((high - low) / (0.5 * (low + high) * bin_pct / 100))))


def find_nodes(
    edges: np.ndarray,
    bin_vols: np.ndarray,
    bw_pct: float = BW_PCT,
    prominence: float = PROMINENCE,
    min_share: float = MIN_SHARE,
) -> list[Node]:
    """High-volume shelves as Node(lo, hi, peak, share), low first.

    A shelf is a peak of the profile smoothed at `bw_pct` of price, kept when
    the profile falls at least `prominence` of that peak's height before
    reaching higher ground, and when the bins between its edges hold at least
    `min_share` of the session's volume. Edges run from the summit out to half
    height or to the valley, whichever comes first.

    This is a different object from the value area and is not a substitute for
    it. The value area is a single band covering `va_pct` of volume around the
    POC; on a session that traded two separate shelves it stretches across the
    gap between them, so its VAL can sit at a price where nothing changed
    hands. A shelf is local: a session with two of them has two nodes, each
    with its own edges.

    Three things this gets right that counting peaks in a raw histogram did
    not, each found by drawing SOXL 2026-10-12 and 2026-10-07 and looking:

    * **Bandwidth, not bin count.** The old version read peaks straight off a
      histogram binned into the session range, so the same session reported
      one shelf at 30 bins and two at 100. Smoothing at a width stated in
      price makes the bin count drop out: 100 through 800 bins now agree to
      within three cents. Bins coarser than the bandwidth defeat this, which
      is what bins_for_price() is for.

    * **Prominence, not height against the tallest peak.** The old floor
      required a peak to reach 35% of the session's busiest price, so a small
      but cleanly separated shelf was never a shelf. This is the same mistake
      as normalising the liquidity heatmap against its global maximum, which
      blacked out thin levels. Prominence is local and asks the right
      question: how far does the profile drop before it climbs again.

    * **Edges stop at the valley.** Half height alone lets a tall peak's skirt
      run through the dip and up into its neighbour; the spans then overlap
      and merge back together, so *lowering* the prominence threshold produced
      one shelf spanning $4 instead of several.

    `prominence` was set by eye against two sessions: 2026-10-12, which reads
    as one shelf with a shoulder above it, and 2026-10-07, which reads as
    three. Both are reproduced at 0.25.

    Returns [] on an empty or degenerate profile, matching compute_value_area.
    """
    if edges is None or bin_vols is None or bin_vols.size < 3:
        return []
    v = np.asarray(bin_vols, dtype=float)
    total = float(v.sum())
    if total <= 0 or float(v.max()) <= 0:
        return []

    if np.all(v == v[0]):
        # Nothing stands out, so nothing is a shelf. find_peaks reports the
        # middle of a plateau as a peak, so without this a perfectly uniform
        # profile returns one node spanning the whole range.
        return []

    mid = (edges[:-1] + edges[1:]) / 2
    bin_w = float(edges[1] - edges[0])
    sigma = (float(np.median(mid)) * bw_pct / 100) / bin_w if bin_w > 0 else 0.0
    # mode="nearest" rather than zero padding: outside the session's range
    # there is genuinely no volume, but zero padding also damps a shelf that
    # runs to the edge of the profile, and such a shelf is still a shelf.
    s = gaussian_filter1d(v, sigma, mode="nearest") if sigma > 0 else v

    # find_peaks ignores the first and last samples, so pad with a floor below
    # any real volume. For an edge peak this also makes prominence measure the
    # interior side only, which is the side that can separate it from anything.
    pad = np.concatenate(([-1.0], s, [-1.0]))
    idx, props = find_peaks(pad, prominence=0)

    out: list[Node] = []
    for i, prom in zip(idx - 1, props["prominences"]):
        if prom < prominence * s[i]:
            continue
        cut = 0.5 * s[i]
        lo = hi = int(i)
        while lo > 0 and s[lo - 1] >= cut and s[lo - 1] <= s[lo]:
            lo -= 1
        while hi < len(s) - 1 and s[hi + 1] >= cut and s[hi + 1] <= s[hi]:
            hi += 1
        share = float(v[lo:hi + 1].sum() / total)
        if share < min_share:
            continue
        out.append(Node(float(edges[lo]), float(edges[hi + 1]),
                        float(mid[i]), share))

    # The valley stop already keeps spans apart; this only guards the case of
    # two summits of equal height sharing one plateau.
    out.sort(key=lambda t: t.lo)
    merged: list[Node] = []
    for node in out:
        if merged and node.lo <= merged[-1].hi:
            prev = merged[-1]
            keep = prev if prev.share >= node.share else node
            merged[-1] = Node(prev.lo, max(prev.hi, node.hi),
                              keep.peak, max(prev.share, node.share))
        else:
            merged.append(node)
    return merged


def in_node(price: float, nodes: list[Node]) -> bool:
    """Is *price* inside any node's span (edges inclusive)?"""
    return any(n.lo <= price <= n.hi for n in nodes)

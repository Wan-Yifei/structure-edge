"""KD channel trend detector — alternative to BOS/CHoCH for HTF trend classification.

Indicator formula (mirrors the original EasyLanguage definition):
    UP1  = EMA(High, fast)
    LOW1 = EMA(Low,  fast)
    MID1 = (UP1 + LOW1) / 2          ← fast channel midpoint

    UP2  = EMA(High, slow)
    LOW2 = EMA(Low,  slow)
    MID2 = (UP2 + LOW2) / 2          ← slow channel midpoint

    spread = MID1 - MID2             ← fast vs slow positioning
    width  = spread - spread.shift(1) ← rate-of-change (Δspread per bar)

Trend classification — two modes:

  Adaptive (smooth > 0, default):
    Segments are defined by zero-crossings of lightly-smoothed width.
    At each crossing boundary the previous bar is conditionally moved into
    the new segment when its raw width already points the new direction
    (lag compensation).  Segments shorter than min_bars are merged into
    the preceding segment.  The current segment's mean width / mean ATR
    is compared against atr_threshold to determine bull / bear / flat.

  Fixed-window (smooth == 0, legacy):
    avg(width[-window:]) compared against flat_threshold / atr_threshold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_kd(
    klines: pd.DataFrame,
    fast: int = 25,
    slow: int = 90,
    atr_period: int = 14,
) -> pd.DataFrame:
    """Compute KD channel midlines, spread, width, and ATR for the given klines.

    Returns a DataFrame (same index as klines) with columns:
        up1, lo1, mid1  — fast EMA channel
        up2, lo2, mid2  — slow EMA channel
        spread          — MID1 - MID2
        width           — spread delta per bar (Δspread)
        atr             — Average True Range over atr_period bars
    """
    up1  = klines["high"].ewm(span=fast, adjust=False).mean()
    lo1  = klines["low"].ewm(span=fast, adjust=False).mean()
    mid1 = (up1 + lo1) / 2.0

    up2  = klines["high"].ewm(span=slow, adjust=False).mean()
    lo2  = klines["low"].ewm(span=slow, adjust=False).mean()
    mid2 = (up2 + lo2) / 2.0

    spread = mid1 - mid2
    width  = spread.diff()

    prev_close = klines["close"].shift(1)
    tr = pd.concat([
        klines["high"] - klines["low"],
        (klines["high"] - prev_close).abs(),
        (klines["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(atr_period, min_periods=1).mean()

    return pd.DataFrame(
        {"up1": up1, "lo1": lo1, "mid1": mid1,
         "up2": up2, "lo2": lo2, "mid2": mid2,
         "spread": spread, "width": width, "atr": atr},
        index=klines.index,
    )


def _adaptive_seg_start(
    smooth_w: pd.Series,
    raw_w: pd.Series,
    atr: pd.Series,
    atr_threshold: float,
    min_bars: int,
) -> int:
    """Return the index position (iloc) where the current segment starts.

    Implements conditional lag compensation and minimum segment length merge,
    matching the notebook exploration logic exactly.
    """
    n     = len(smooth_w)
    signs = np.sign(smooth_w.fillna(0))
    sw_v  = smooth_w.values
    rw_v  = raw_w.values
    at_v  = atr.values

    boundaries: list[int] = [0]
    for i in range(1, n):
        ps, cs = signs.iloc[i - 1], signs.iloc[i]
        if cs == 0 or ps == 0 or cs == ps:
            continue
        new_dir = cs
        # Don't shift into a flat segment
        flat = (atr_threshold > 0.0 and at_v[i] > 0.0
                and abs(sw_v[i]) / at_v[i] < atr_threshold)
        if flat:
            boundaries.append(i)
        elif np.sign(rw_v[i - 1]) == new_dir:
            boundaries.append(i - 1)   # raw width already turned: shift earlier
        else:
            boundaries.append(i)

    # Enforce minimum segment length (merge short segments backward)
    if len(boundaries) > 1 and boundaries[1] - boundaries[0] < min_bars:
        boundaries.pop(1)
    i = 1
    while i < len(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else n
        if end - boundaries[i] < min_bars:
            boundaries.pop(i)
        else:
            i += 1

    return boundaries[-1]  # start of the current (last) segment


def kd_trend(
    klines: pd.DataFrame,
    fast: int = 25,
    slow: int = 90,
    window: int = 10,
    flat_threshold: float = 0.0,
    atr_threshold: float = 0.036,  # p25 of |seg_avg_width|/ATR on SNDK 1545-bar HTF history
    atr_period: int = 14,
    smooth: int = 3,
    min_bars: int = 3,
) -> str | None:
    """Return HTF trend direction using the KD channel indicator.

    Args:
        klines:          OHLCV DataFrame (full history up to current bar).
        fast:            EMA span for the fast channel (default 25).
        slow:            EMA span for the slow channel (default 90).
        window:          Fixed-window size; used only when smooth == 0.
        flat_threshold:  Legacy price-unit filter; used only when smooth == 0.
        atr_threshold:   |seg_avg_width| / seg_avg_ATR below this → flat.
                         Scale-invariant; default 0.036 = p25 of the empirical
                         distribution (SNDK HTF history), filters weakest 25% of segments.
        atr_period:      ATR rolling period (default 14).
        smooth:          Pre-smoothing window for zero-crossing detection.
                         > 0 → adaptive segment mode (default 3).
                         0   → legacy fixed-window mode.
        min_bars:        Minimum bars per segment; short segments are merged
                         into the previous one (adaptive mode only).

    Returns:
        "bull" / "bear" / None
    """
    min_len = slow + (smooth if smooth > 0 else window)
    if len(klines) < min_len:
        return None

    kd = compute_kd(klines, fast, slow, atr_period)

    # ── Adaptive segment mode ─────────────────────────────────────────────────
    if smooth > 0:
        smooth_w = kd["width"].rolling(smooth, min_periods=1).mean()
        seg_start = _adaptive_seg_start(
            smooth_w, kd["width"], kd["atr"], atr_threshold, min_bars
        )
        seg_w   = float(smooth_w.iloc[seg_start:].mean())
        seg_atr = float(kd["atr"].iloc[seg_start:].mean())

        if np.isnan(seg_w) or np.isnan(seg_atr) or seg_atr == 0.0:
            return None
        if atr_threshold > 0.0 and abs(seg_w) / seg_atr < atr_threshold:
            return None
        return "bull" if seg_w > 0 else "bear"

    # ── Legacy fixed-window mode (smooth == 0) ────────────────────────────────
    avg_width = float(kd["width"].iloc[-window:].mean())
    if np.isnan(avg_width):
        return None
    if abs(avg_width) <= flat_threshold:
        return None
    if atr_threshold > 0.0:
        avg_atr = float(kd["atr"].iloc[-window:].mean())
        if not np.isnan(avg_atr) and avg_atr > 0.0:
            if abs(avg_width) / avg_atr < atr_threshold:
                return None
    return "bull" if avg_width > 0 else "bear"

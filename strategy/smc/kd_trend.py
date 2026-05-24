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

Trend classification:
    avg(width[-window:]) > +flat_threshold  → "bull"
    avg(width[-window:]) < -flat_threshold  → "bear"
    otherwise                               → None (flat / consolidation)

ATR-normalised filter (recommended over flat_threshold):
    |avg_width| / avg_ATR < atr_threshold   → None
    This makes the threshold scale-invariant across price levels and volatility.
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


def kd_trend(
    klines: pd.DataFrame,
    fast: int = 25,
    slow: int = 90,
    window: int = 10,
    flat_threshold: float = 0.0,
    atr_threshold: float = 0.0,
    atr_period: int = 14,
) -> str | None:
    """Return HTF trend direction using the KD channel indicator.

    Trend is determined by the average WIDTH (Δspread) over the last `window`
    bars — i.e. which direction the spread has been moving recently, not its
    absolute level.  Only recent bars are relevant to the current trend.

    Args:
        klines:          OHLCV DataFrame.
        fast:            EMA span for the fast channel (default 25).
        slow:            EMA span for the slow channel (default 90).
        window:          Number of recent bars to average WIDTH over.
        flat_threshold:  |avg_width| below this → no trend (price units/bar).
                         0 = no flat filter.
        atr_threshold:   |avg_width| / avg_ATR below this → no trend (dimensionless).
                         Preferred over flat_threshold — scale-invariant across stocks.
                         0 = disabled.
        atr_period:      ATR rolling period (default 14).

    Returns:
        "bull"  if avg WIDTH over window is sufficiently positive
        "bear"  if avg WIDTH over window is sufficiently negative
        None    if near flat or insufficient data
    """
    if len(klines) < slow + window:
        return None

    kd = compute_kd(klines, fast, slow, atr_period)
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

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
    spread > +threshold  → "bull"  (fast channel above slow)
    spread < -threshold  → "bear"
    |spread| ≤ threshold → None    (flat / consolidation)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_kd(
    klines: pd.DataFrame,
    fast: int = 25,
    slow: int = 90,
) -> pd.DataFrame:
    """Compute KD channel midlines, spread, and width for the given klines.

    Returns a DataFrame (same index as klines) with columns:
        up1, lo1, mid1  — fast EMA channel
        up2, lo2, mid2  — slow EMA channel
        spread          — MID1 - MID2
        width           — spread delta per bar (Δspread)
    """
    up1  = klines["high"].ewm(span=fast, adjust=False).mean()
    lo1  = klines["low"].ewm(span=fast, adjust=False).mean()
    mid1 = (up1 + lo1) / 2.0

    up2  = klines["high"].ewm(span=slow, adjust=False).mean()
    lo2  = klines["low"].ewm(span=slow, adjust=False).mean()
    mid2 = (up2 + lo2) / 2.0

    spread = mid1 - mid2
    width  = spread.diff()

    return pd.DataFrame(
        {"up1": up1, "lo1": lo1, "mid1": mid1,
         "up2": up2, "lo2": lo2, "mid2": mid2,
         "spread": spread, "width": width},
        index=klines.index,
    )


def kd_trend(
    klines: pd.DataFrame,
    fast: int = 25,
    slow: int = 90,
    flat_threshold: float = 0.0,
) -> str | None:
    """Return HTF trend direction using the KD channel indicator.

    Args:
        klines:          OHLCV DataFrame.
        fast:            EMA span for the fast channel (default 25).
        slow:            EMA span for the slow channel (default 90).
        flat_threshold:  Minimum |spread| to consider a directional trend.
                         Spread in absolute price units.  0 = no flat filter.

    Returns:
        "bull"  if MID1 is sufficiently above MID2
        "bear"  if MID1 is sufficiently below MID2
        None    if spread is near zero (flat) or insufficient data
    """
    if len(klines) < slow:
        return None

    kd = compute_kd(klines, fast, slow)
    last_spread = float(kd["spread"].iloc[-1])

    if np.isnan(last_spread):
        return None
    if abs(last_spread) <= flat_threshold:
        return None

    return "bull" if last_spread > 0 else "bear"

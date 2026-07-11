"""Wilder-smoothed Average True Range.

strategy/smc/kd_trend.py:compute_kd() also computes a column it calls "atr",
but that one is a simple rolling mean of True Range (adequate for its own use
as a trend-strength normalizer). Wilder's original ATR uses a specific
recursive smoothing (RMA), which is what the chandelier exit formula expects.
This is a separate, correct implementation; kd_trend.py is left untouched.
"""

from __future__ import annotations

import numpy as np


def wilder_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int) -> np.ndarray:
    """Return Wilder-smoothed ATR, same length as the inputs.

    True Range: max(high-low, |high-prev_close|, |low-prev_close|); the first
    bar has no prev_close so its TR is just high-low.

    Wilder RMA: seed ATR[period-1] = mean(TR[0:period]), then for t >= period:
        ATR[t] = (ATR[t-1] * (period-1) + TR[t]) / period

    Indices [0, period-2] are NaN (insufficient warmup), matching the
    pandas .rolling()-style convention used elsewhere in this repo.
    """
    n = len(highs)
    if n == 0:
        return np.array([], dtype=float)

    tr = np.empty(n, dtype=float)
    tr[0] = highs[0] - lows[0]
    prev_close = closes[:-1]
    tr[1:] = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(np.abs(highs[1:] - prev_close), np.abs(lows[1:] - prev_close)),
    )

    atr = np.full(n, np.nan, dtype=float)
    if n < period:
        return atr

    atr[period - 1] = tr[:period].mean()
    for t in range(period, n):
        atr[t] = (atr[t - 1] * (period - 1) + tr[t]) / period
    return atr

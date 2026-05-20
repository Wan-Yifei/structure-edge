"""SMC Fair Value Gap (FVG) detection.

An FVG is a 3-candle pattern where the first and third candles leave a price
gap that the middle candle's body does not fill.

Bullish FVG:  high[i-2] < low[i]   → gap from high[i-2] to low[i]
Bearish FVG:  low[i-2]  > high[i]  → gap from low[i-2]  to high[i]
"""

from __future__ import annotations
import pandas as pd


def detect_fvg(
    klines: pd.DataFrame,
    min_gap_pct: float = 0.001,
) -> list[dict]:
    """Return list of Fair Value Gaps.

    Each gap dict:
        {direction: 'bull'|'bear', top: float, bottom: float,
         idx: int,   # index of the third candle (where gap is confirmed)
         filled: bool}

    *min_gap_pct* filters out noise: gap must be at least this fraction of
    the middle candle's price.
    """
    highs  = klines["high"].values
    lows   = klines["low"].values
    closes = klines["close"].values
    n      = len(klines)
    gaps   = []

    for i in range(2, n):
        mid_price = closes[i - 1]

        # bullish: high of candle[i-2] < low of candle[i]
        if highs[i - 2] < lows[i]:
            gap_size = lows[i] - highs[i - 2]
            if gap_size / mid_price >= min_gap_pct:
                gaps.append({
                    "direction": "bull",
                    "top":       float(lows[i]),
                    "bottom":    float(highs[i - 2]),
                    "idx":       i,
                    "filled":    False,
                })

        # bearish: low of candle[i-2] > high of candle[i]
        elif lows[i - 2] > highs[i]:
            gap_size = lows[i - 2] - highs[i]
            if gap_size / mid_price >= min_gap_pct:
                gaps.append({
                    "direction": "bear",
                    "top":       float(lows[i - 2]),
                    "bottom":    float(highs[i]),
                    "idx":       i,
                    "filled":    False,
                })

    # mark filled: a later candle's close entered the gap zone
    for gap in gaps:
        for j in range(gap["idx"] + 1, n):
            c = closes[j]
            if gap["bottom"] <= c <= gap["top"]:
                gap["filled"] = True
                break

    return gaps

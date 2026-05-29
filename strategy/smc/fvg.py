"""SMC Fair Value Gap (FVG) detection.

An FVG is a 3-candle pattern where the first and third candles leave a price
gap that the middle candle's body does not fill.

Bullish FVG:  high[i-2] < low[i]   → gap from high[i-2] to low[i]
Bearish FVG:  low[i-2]  > high[i]  → gap from low[i-2]  to high[i]
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def detect_fvg(
    klines: pd.DataFrame,
    min_gap_pct: float = 0.001,
    require_displacement: bool = False,
) -> list[dict]:
    """Return list of Fair Value Gaps.

    Each gap dict:
        {direction: 'bull'|'bear', top: float, bottom: float,
         idx: int,   # index of the third candle (where gap is confirmed)
         filled: bool}

    Parameters
    ----------
    min_gap_pct:
        Minimum gap size as a fraction of the middle candle close price.
        Filters out micro-gaps from spread noise (default 0.1%).
    require_displacement:
        When True (default) the middle candle must pass the displacement
        test (large range + strong body vs. recent ATR).  This eliminates
        the vast majority of trivial 3-candle gaps and keeps only
        structurally meaningful FVGs driven by an impulse move.
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
                if require_displacement and not is_displacement_candle(klines, i):
                    continue
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
                if require_displacement and not is_displacement_candle(klines, i):
                    continue
                gaps.append({
                    "direction": "bear",
                    "top":       float(lows[i - 2]),
                    "bottom":    float(highs[i]),
                    "idx":       i,
                    "filled":    False,
                })

    # Mark filled: a later candle's close entered the gap zone.
    for gap in gaps:
        for j in range(gap["idx"] + 1, n):
            c = closes[j]
            if gap["bottom"] <= c <= gap["top"]:
                gap["filled"] = True
                break

    return gaps


def fvg_entry_depth(fvg: dict, price: float) -> float:
    """Fraction of the FVG that *price* has penetrated from the entry side.

    0.0 = price just touched the edge of the zone.
    1.0 = price fully traversed the zone.
    Returns 0.0 when price is outside the zone.

    Direction convention (SMC pullback/bounce into FVG):
        bull FVG: price pulls back from above → depth = (top - price) / size
        bear FVG: price bounces up from below → depth = (price - bottom) / size
    """
    top  = fvg["top"]
    bot  = fvg["bottom"]
    size = top - bot
    if size <= 0:
        return 0.0
    if fvg["direction"] == "bull":
        if price > top:
            return 0.0
        return min((top - price) / size, 1.0)
    else:
        if price < bot:
            return 0.0
        return min((price - bot) / size, 1.0)


def is_displacement_candle(
    klines: pd.DataFrame,
    fvg_idx: int,
    atr_mult: float = 1.5,
    body_ratio_min: float = 0.5,
    lookback: int = 5,
) -> bool:
    """Check whether the FVG's middle candle (fvg_idx - 1) is a displacement candle.

    Two conditions must both hold:
      1. Range: range_B >= atr_mult * mean(range of previous `lookback` candles)
      2. Body:  body_B / range_B >= body_ratio_min

    Using the mean of `lookback` preceding candles (instead of just the two
    immediate neighbours) gives a more stable baseline, closer to ATR.
    Falls back to True when there are fewer than 2 preceding candles.
    """
    mid = fvg_idx - 1
    if mid < 1 or fvg_idx >= len(klines):
        return True

    highs  = klines["high"].values.astype(float)
    lows   = klines["low"].values.astype(float)
    opens  = klines["open"].values.astype(float)
    closes = klines["close"].values.astype(float)

    range_b = highs[mid] - lows[mid]
    if range_b <= 0:
        return False

    # baseline: mean range of the `lookback` candles preceding the middle candle
    start      = max(0, mid - lookback)
    prev_ranges = highs[start:mid] - lows[start:mid]
    if len(prev_ranges) == 0:
        return True
    baseline = float(prev_ranges.mean())
    if baseline <= 0:
        return False

    if range_b < atr_mult * baseline:
        return False

    body_b = abs(closes[mid] - opens[mid])
    return (body_b / range_b) >= body_ratio_min


def compute_volume_profile(
    klines: pd.DataFrame,
    n_bins: int = 100,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Build a volume-at-price profile over the given kline window.

    Each bar's volume is distributed uniformly across the price bins it spans.
    Returns (edges, bin_vols) where edges has n_bins+1 elements.
    Returns (None, None) when the price range is degenerate.
    """
    lows  = klines["low"].values.astype(float)
    highs = klines["high"].values.astype(float)
    vols  = klines["volume"].values.astype(float)

    p_min, p_max = lows.min(), highs.max()
    if p_min >= p_max:
        return None, None

    edges    = np.linspace(p_min, p_max, n_bins + 1)
    bin_size = edges[1] - edges[0]
    bin_vols = np.zeros(n_bins)

    lo_bins = np.clip(((lows  - p_min) / bin_size).astype(int), 0, n_bins - 1)
    hi_bins = np.clip(((highs - p_min) / bin_size).astype(int), 0, n_bins - 1)

    for i in range(len(lows)):
        span = hi_bins[i] - lo_bins[i] + 1
        bin_vols[lo_bins[i] : hi_bins[i] + 1] += vols[i] / span

    return edges, bin_vols


def fvg_overlaps_lvn(
    fvg: dict,
    edges: np.ndarray | None,
    bin_vols: np.ndarray | None,
    lvn_threshold: float = 0.30,
) -> bool:
    """Return True if the FVG zone sits in a Low Volume Node.

    An LVN is a price band whose average bin volume is below
    lvn_threshold × max_bin_volume in the current window.
    Returns False when volume profile is unavailable.
    """
    if edges is None or bin_vols is None:
        return False

    mask = (edges[1:] > fvg["bottom"]) & (edges[:-1] < fvg["top"])
    if not mask.any():
        return False

    max_vol  = bin_vols.max()
    if max_vol <= 0:
        return False

    zone_vol = bin_vols[mask].mean()
    return zone_vol < lvn_threshold * max_vol

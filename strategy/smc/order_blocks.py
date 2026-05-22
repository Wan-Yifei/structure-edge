"""SMC Order Block detection.

An Order Block (OB) is the last opposing candle immediately before a
significant move (identified by a BOS/CHoCH signal).

Subtypes:
    regular    — price has not returned to the zone
    mitigation — price re-entered the zone but has not closed through it
    breaker    — price closed through the zone (former support → resistance)
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def detect_order_blocks(
    klines: pd.DataFrame,
    bos_signals: list[dict],
) -> list[dict]:
    """Return Order Blocks derived from BOS/CHoCH signals.

    Each block dict:
        {direction: 'bull'|'bear',
         subtype:   'regular'|'mitigation'|'breaker',
         top:    float, bottom: float,
         idx:    int,   # candle index of the OB
         bos_idx: int}  # candle index of the triggering BOS/CHoCH
    """
    opens   = klines["open"].values
    closes  = klines["close"].values
    highs   = klines["high"].values
    lows    = klines["low"].values
    volumes = klines["volume"].values
    n       = len(klines)
    blocks  = []

    for sig in bos_signals:
        bos_idx = sig["idx"]
        bull    = sig["direction"] == "bull"

        # OB is anchored at the reference swing (from_idx), not the break bar.
        # Scan backwards from the swing point to find the last opposing candle
        # in the structure that created that swing.
        anchor = min(sig.get("from_idx", bos_idx - 1), n - 1)
        ob_idx = None
        for k in range(anchor, max(-1, anchor - 8), -1):
            is_bear_candle = closes[k] < opens[k]
            is_bull_candle = closes[k] > opens[k]
            if bull and is_bear_candle:
                ob_idx = k
                break
            if not bull and is_bull_candle:
                ob_idx = k
                break
        # fallback: scan forward from swing toward break bar
        if ob_idx is None:
            for k in range(anchor + 1, bos_idx):
                is_bear_candle = closes[k] < opens[k]
                is_bull_candle = closes[k] > opens[k]
                if bull and is_bear_candle:
                    ob_idx = k
                    break
                if not bull and is_bull_candle:
                    ob_idx = k
                    break

        if ob_idx is None:
            continue

        # Volume filter: OB candle must exceed the median of the 5 preceding bars.
        # Low-volume OBs are more likely to be noise rather than institutional flow.
        vol_start  = max(0, ob_idx - 5)
        prior_vols = volumes[vol_start:ob_idx]
        if len(prior_vols) > 0 and float(volumes[ob_idx]) < float(np.median(prior_vols)):
            continue

        top    = float(highs[ob_idx])
        bottom = float(lows[ob_idx])

        # classify subtype based on price action after BOS
        subtype = "regular"
        for j in range(bos_idx + 1, n):
            c = closes[j]
            if bull:
                if c < bottom:          # closed below OB → breaker
                    subtype = "breaker"
                    break
                if bottom <= c <= top:  # entered zone → mitigation
                    subtype = "mitigation"
            else:
                if c > top:             # closed above OB → breaker
                    subtype = "breaker"
                    break
                if bottom <= c <= top:
                    subtype = "mitigation"

        blocks.append({
            "direction": sig["direction"],
            "subtype":   subtype,
            "top":       top,
            "bottom":    bottom,
            "idx":       ob_idx,
            "bos_idx":   bos_idx,
        })

    # keep only the most recent OBs to avoid cluttering the chart
    return blocks[-4:]

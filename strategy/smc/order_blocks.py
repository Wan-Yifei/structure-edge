"""SMC Order Block detection.

An Order Block (OB) is the last opposing candle immediately before a
significant move (identified by a BOS/CHoCH signal).

Subtypes:
    regular    — price has not returned to the zone
    mitigation — price re-entered the zone but has not closed through it
    breaker    — price closed through the zone (former support → resistance)
"""

from __future__ import annotations
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
    opens  = klines["open"].values
    closes = klines["close"].values
    highs  = klines["high"].values
    lows   = klines["low"].values
    n      = len(klines)
    blocks = []

    for sig in bos_signals:
        bos_idx = sig["idx"]
        bull    = sig["direction"] == "bull"

        # scan backwards from BOS to find last opposing candle
        ob_idx = None
        for k in range(bos_idx - 1, -1, -1):
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
    return blocks[-8:]

"""SMC market structure detection: swing points, BOS, CHoCH.

All functions take a pandas DataFrame with columns open/high/low/close/time_key
and return plain lists of dicts — no matplotlib, no GUI.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def find_swings(klines: pd.DataFrame, lookback: int = 2) -> list[dict]:
    """Return alternating swing highs and lows.

    Each swing: {kind: 'high'|'low', idx: int, price: float, time: str}
    A bar is a swing high if its close is >= all closes in [i-lookback, i+lookback].
    A bar is a swing low  if its close is <= all closes in [i-lookback, i+lookback].
    Price is set to the close (body), not the wick — consistent with body-only BOS/CHoCH.
    Highs and lows are detected independently then merged into alternating sequence.
    """
    closes = klines["close"].values.astype(float)
    times  = klines["time_key"].values
    n      = len(closes)
    highs_raw: list[dict] = []
    lows_raw:  list[dict] = []

    for i in range(lookback, n - lookback):
        c     = closes[i]
        c_win = closes[i - lookback : i + lookback + 1]

        if c >= c_win.max():
            highs_raw.append({"kind": "high", "idx": i, "price": c,
                               "time": str(times[i])})
        if c <= c_win.min():
            lows_raw.append({"kind": "low", "idx": i, "price": c,
                              "time": str(times[i])})

    # merge by index, then force strict alternation (keep more extreme on ties)
    merged = sorted(highs_raw + lows_raw, key=lambda s: s["idx"])
    cleaned: list[dict] = []
    for sw in merged:
        if not cleaned or cleaned[-1]["kind"] != sw["kind"]:
            cleaned.append(sw)
        else:
            prev = cleaned[-1]
            if sw["kind"] == "high" and sw["price"] >= prev["price"]:
                cleaned[-1] = sw
            elif sw["kind"] == "low" and sw["price"] <= prev["price"]:
                cleaned[-1] = sw

    return cleaned


def detect_bos_choch(klines: pd.DataFrame, lookback: int = 2) -> list[dict]:
    """Detect Break of Structure (BOS) and Change of Character (CHoCH).

    BOS: price breaks a swing level in the direction of the current trend.
    CHoCH: price breaks a swing level against the current trend (reversal signal).

    Returns list of dicts:
        {type: 'BOS'|'CHoCH', direction: 'bull'|'bear',
         idx: int, price: float, from_idx: int}
    """
    swings = find_swings(klines, lookback)
    if len(swings) < 4:
        return []

    highs = [s for s in swings if s["kind"] == "high"]
    lows  = [s for s in swings if s["kind"] == "low"]
    if len(highs) < 2 or len(lows) < 2:
        return []

    # initial trend: higher highs + higher lows → uptrend
    trend = ("up"
             if highs[1]["price"] > highs[0]["price"] or lows[1]["price"] > lows[0]["price"]
             else "down")

    closes = klines["close"].values
    n      = len(klines)
    signals: list[dict] = []
    processed_highs: set[int] = set()
    processed_lows:  set[int] = set()

    for i in range(2, len(swings)):
        sw = swings[i]

        if sw["kind"] == "high":
            prev_highs = [s for s in swings[:i] if s["kind"] == "high"]
            if not prev_highs:
                continue
            prev_high = prev_highs[-1]
            if prev_high["idx"] in processed_highs:
                continue
            for j in range(sw["idx"] + 1, n):
                if closes[j] > prev_high["price"]:
                    sig_type = "BOS" if trend == "up" else "CHoCH"
                    signals.append({
                        "type":      sig_type,
                        "direction": "bull",
                        "idx":       j,
                        "price":     prev_high["price"],
                        "from_idx":  prev_high["idx"],
                    })
                    processed_highs.add(prev_high["idx"])
                    if sig_type == "CHoCH":
                        trend = "up"
                    break

        else:  # kind == "low"
            prev_lows = [s for s in swings[:i] if s["kind"] == "low"]
            if not prev_lows:
                continue
            prev_low = prev_lows[-1]
            if prev_low["idx"] in processed_lows:
                continue
            for j in range(sw["idx"] + 1, n):
                if closes[j] < prev_low["price"]:
                    sig_type = "BOS" if trend == "down" else "CHoCH"
                    signals.append({
                        "type":      sig_type,
                        "direction": "bear",
                        "idx":       j,
                        "price":     prev_low["price"],
                        "from_idx":  prev_low["idx"],
                    })
                    processed_lows.add(prev_low["idx"])
                    if sig_type == "CHoCH":
                        trend = "down"
                    break

    return signals

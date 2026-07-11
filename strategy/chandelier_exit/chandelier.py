"""Chandelier-exit trailing stop simulator.

Long (direction="bull"):  stop = HighestHigh(period) - ATR(period) * multiplier
Short (direction="bear"): stop = LowestLow(period)  + ATR(period) * multiplier
The stop only ever ratchets in the trade's favor (up for bull, down for bear),
never loosens.

No-look-ahead: the stop level active *during* bar t must be derived from data
known before bar t opened. For the entry bar itself, the entry decision (and
therefore the initial stop) is allowed to use that bar's own high/low/ATR --
matching how backtest/engine.py enters at the triggering bar's close using
that same bar's data. For every bar after entry, the candidate stop is
derived from the *previous* bar's HH/LL/ATR only, so a bar's own new extreme
can never trigger a stop against itself.

Exit fills at the stop price (not the bar's open) -- matches
backtest/engine.py's _find_exit() fixed-SL convention (first-touch, no gap
modeling), so the two exit methods stay directly comparable. This is an
optimistic simplification versus real fills through a gap; documented here
and in the generated REPORT.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ChandelierResult:
    exit_bar:    int
    exit_price:  float
    exit_time:   str
    cause:       str     # "stopped" | "timeout"
    r_multiple:  float
    period:      int
    multiplier:  float
    stop_series: np.ndarray = field(repr=False)  # stop level per scanned bar (debugging/plotting)


def simulate_chandelier_exit(
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, times: np.ndarray,
    atr: np.ndarray, hh: np.ndarray, ll: np.ndarray,
    entry_idx: int, entry_price: float, direction: str,
    period: int, multiplier: float, risk_unit: float,
    max_bars: int = 200,
) -> ChandelierResult | None:
    """Simulate one trade's exit under the chandelier trailing stop.

    highs/lows/closes/times: full LTF arrays for the instrument.
    atr/hh/ll: full-length precomputed arrays for this `period` --
        atr = wilder_atr(highs, lows, closes, period)
        hh  = rolling max of `highs` over `period` bars ending at each index
        ll  = rolling min of `lows` over `period` bars ending at each index
    entry_idx: index into the above arrays where the trade opens.
    risk_unit: R-multiple denominator (the ORIGINAL engine trade's SL
        distance, not a chandelier-specific one -- see README for why).

    Returns None if atr[entry_idx] is NaN (insufficient warmup at entry).
    """
    n = len(highs)
    if np.isnan(atr[entry_idx]) or np.isnan(hh[entry_idx]) or np.isnan(ll[entry_idx]):
        return None

    end_idx = min(entry_idx + max_bars, n - 1)
    scan = np.arange(entry_idx, end_idx + 1)

    # Shifted source index per bar: entry bar uses its own hh/ll/atr; every
    # bar after uses the previous bar's, so a bar can't stop itself out on
    # its own newly-set extreme.
    src = np.where(scan == entry_idx, scan, scan - 1)

    if direction == "bull":
        cand = hh[src] - atr[src] * multiplier
        stop = np.maximum.accumulate(cand)
        touched = lows[scan] <= stop
    else:  # "bear"
        cand = ll[src] + atr[src] * multiplier
        stop = np.minimum.accumulate(cand)
        touched = highs[scan] >= stop

    if touched.any():
        hit_i      = int(np.argmax(touched))
        exit_bar   = int(scan[hit_i])
        exit_price = float(stop[hit_i])
        cause      = "stopped"
    else:
        exit_bar   = int(scan[-1])
        exit_price = float(closes[exit_bar])
        cause      = "timeout"

    if direction == "bull":
        r_multiple = (exit_price - entry_price) / risk_unit
    else:
        r_multiple = (entry_price - exit_price) / risk_unit

    return ChandelierResult(
        exit_bar=exit_bar, exit_price=exit_price, exit_time=str(times[exit_bar]),
        cause=cause, r_multiple=r_multiple, period=period, multiplier=multiplier,
        stop_series=stop,
    )


def rolling_extremes(highs: np.ndarray, lows: np.ndarray, period: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (hh, ll): rolling max(highs)/min(lows) over `period` bars ending
    at each index (inclusive). First `period-1` entries are NaN.
    """
    import pandas as pd
    hh = pd.Series(highs).rolling(period, min_periods=period).max().to_numpy()
    ll = pd.Series(lows).rolling(period, min_periods=period).min().to_numpy()
    return hh, ll

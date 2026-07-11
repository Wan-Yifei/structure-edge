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
    entry_stop_override: float | None = None,
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
    entry_stop_override: replace the candidate stop for both the entry bar
        AND the bar right after it (both read hh[entry_idx]/atr[entry_idx]
        under the shift convention, so overriding only the entry bar would
        let the very next bar's cummax revert to the un-overridden value one
        bar later) with this value instead. Normally that candidate is
        `hh[entry_idx] - atr[entry_idx]*multiplier`, anchored to the recent
        high/low, which can land on the wrong side of `entry_price` if price
        has already pulled back/rallied far from that extreme. The ratchet
        still trails via HH/LL from the second bar after entry onward exactly
        as before -- this only fixes the starting point. Pass None (default)
        to preserve the original indicator-style formula unchanged (this is
        what strategy/chandelier_exit/'s grid search and calculator.py both
        rely on).

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

    # entry_stop_override must replace every position whose src == entry_idx,
    # not just position 0 -- the shift convention means position 1 (the bar
    # right after entry) ALSO reads hh[entry_idx]/atr[entry_idx], so overriding
    # only cand[0] would let the very next bar's cummax immediately revert to
    # the un-overridden (possibly invalid) HH-anchored value one bar later.
    override_mask = (src == entry_idx)

    if direction == "bull":
        cand = hh[src] - atr[src] * multiplier
        if entry_stop_override is not None:
            cand[override_mask] = entry_stop_override
        stop = np.maximum.accumulate(cand)
        touched = lows[scan] <= stop
    else:  # "bear"
        cand = ll[src] + atr[src] * multiplier
        if entry_stop_override is not None:
            cand[override_mask] = entry_stop_override
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


def current_stop(
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
    period: int, multiplier: float, direction: str,
) -> dict | None:
    """Point-in-time chandelier stop as of the LATEST bar -- no ratchet memory,
    no entry/position needed. Shared by strategy/chandelier_exit/calculator.py
    (CLI) and analysis/trade_viewer_qt.py's corner-label overlay so the
    formula lives in exactly one place.

    Returns None if there aren't enough bars for `period` warmup. Otherwise a
    dict: {price, atr, hh, ll, stop, dist, dist_pct}, where `dist` is the
    distance from the latest close to `stop` (signed: positive means stop is
    below price for a long / above price for a short, as expected).
    """
    from strategy.chandelier_exit.atr import wilder_atr

    if len(highs) < period:
        return None

    atr    = wilder_atr(highs, lows, closes, period)
    hh, ll = rolling_extremes(highs, lows, period)
    if np.isnan(atr[-1]) or np.isnan(hh[-1]) or np.isnan(ll[-1]):
        return None

    price = float(closes[-1])
    atr_v = float(atr[-1])
    hh_v  = float(hh[-1])
    ll_v  = float(ll[-1])
    bull  = direction == "bull"
    stop  = hh_v - atr_v * multiplier if bull else ll_v + atr_v * multiplier
    dist  = abs(price - stop)
    pct   = dist / price * 100 if price else 0.0

    return {
        "price": price, "atr": atr_v, "hh": hh_v, "ll": ll_v,
        "stop": stop, "dist": dist, "dist_pct": pct,
    }

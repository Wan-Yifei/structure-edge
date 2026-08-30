"""VAL-touch-then-reversal signal detection, long-only, RSI-filtered.

Pure detector: klines DataFrame + a frozen VAL price in, raw structural
signals out. Entry/SL/TP/direction decisions are the caller's (the backtest
engine), matching the pure detect_X(klines, **params) -> list[dict]
contract already used by strategy/smc's detectors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_rsi(closes: pd.Series, period: int = 6) -> np.ndarray:
    """Wilder RSI of *closes* over *period* bars. First use of RSI in this repo."""
    from ta.momentum import RSIIndicator
    return RSIIndicator(close=closes, window=period).rsi().to_numpy()


def detect_val_reversal(
    klines: pd.DataFrame,
    val: float,
    rsi_period: int = 6,
    rsi_threshold: float = 30.0,
    start_idx: int = 0,
    end_idx: int | None = None,
    val_proximity_pct: float = 0.0,
    rsi: np.ndarray | None = None,
    lows: np.ndarray | None = None,
    highs: np.ndarray | None = None,
    closes: np.ndarray | None = None,
) -> list[dict]:
    """Scan klines[start_idx:end_idx] for VAL-touch-then-reversal signals, long-only.

    A signal at bar i+1 requires BOTH (a single AND, not either-or):
      1. Price action -- a plain 2-bar swing-low break: low[i] <= val * (1 +
         val_proximity_pct), and close[i+1] > high[i] and close[i+1] > val.
         Entry executes at close[i+1]: both bars are closed by decision
         time, so this has no lookahead and is unambiguous on raw OHLCV.
         val_proximity_pct (default 0.0, reproduces the original strict
         "must actually touch VAL" behavior) widens the touch test to fire
         when price gets within that fraction of VAL without necessarily
         breaking it -- the confirmation leg still requires close[i+1] > val,
         so a loosened touch never lets a trade in without price actually
         reclaiming VAL.
      2. RSI(rsi_period) < rsi_threshold evaluated at bar i (the touch/
         extreme bar -- by i+1 price is already recovering and RSI may
         have already turned up, so i is the right point for an "oversold"
         read, not i+1).

    Returns a list of {"entry_idx", "entry_price", "touch_idx", "rsi_at_touch"}
    dicts, one per raw signal found (may be more than one per session -- it's
    the caller's job to decide whether to act on more than the first, e.g.
    via an allow_multiple_entries_per_session flag).

    Caller responsibility: RSI(rsi_period) needs rsi_period+ prior bars to
    stop returning NaN, and a NaN RSI at bar i is always treated as "filter
    fails" (skipped, never a false pass). If klines is truncated to exactly
    a session's own bars, the first rsi_period-ish bars of every session
    would spuriously have no valid RSI at all and could never trigger --
    pass a klines slice that includes enough bars *before* start_idx for
    RSI to have warmed up by the time start_idx is reached.

    Pass precomputed `rsi`/`lows`/`highs`/`closes` arrays (same length as
    klines) when scanning the same klines repeatedly -- e.g. once per
    session occurrence in a backtest loop -- to avoid recomputing RSI and
    re-extracting numpy arrays from the full DataFrame on every call, which
    dominates runtime on anything but tiny inputs (recomputing RSI over
    ~85k 1-minute bars ~1000 times per backtest combo is what actually
    happened before these parameters were added).

    `end_idx` caps the scan at that bar (inclusive of it as a possible
    entry_idx), defaulting to the end of klines. A per-session-occurrence
    caller MUST pass the occurrence's own last bar here: leaving it open
    means every call scans from its start_idx all the way to the end of
    the *entire* klines array, not just its own session -- for N session
    occurrences that is an O(N * len(klines)) blowup (this is exactly what
    happened before end_idx was added: on a full year of 1-minute bars
    across ~1000 session occurrences, backtests that should take seconds
    took hours).
    """
    n = len(klines)
    stop = n if end_idx is None else min(n, end_idx + 1)
    if stop < start_idx + 2:
        return []

    if lows is None:
        lows = klines["low"].to_numpy(dtype=float)
    if highs is None:
        highs = klines["high"].to_numpy(dtype=float)
    if closes is None:
        closes = klines["close"].to_numpy(dtype=float)
    if rsi is None:
        rsi = compute_rsi(klines["close"], rsi_period)

    touch_threshold = val * (1.0 + val_proximity_pct)
    signals: list[dict] = []
    for i in range(start_idx, stop - 1):
        if lows[i] > touch_threshold:
            continue
        if np.isnan(rsi[i]) or rsi[i] >= rsi_threshold:
            continue
        j = i + 1
        if closes[j] > highs[i] and closes[j] > val:
            signals.append({
                "entry_idx":    j,
                "entry_price":  float(closes[j]),
                "touch_idx":    i,
                "rsi_at_touch": float(rsi[i]),
            })
    return signals

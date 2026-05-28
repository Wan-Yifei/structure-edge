"""SMC market structure detection: swing points, BOS, CHoCH.

All functions take a pandas DataFrame with columns open/high/low/close/time_key
and return plain lists of dicts — no matplotlib, no GUI.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def _session_key(time_key: str) -> str:
    """Return a composite session identifier: YYYY-MM-DD_<pre|regular|post>.

    Used by detect_bos_choch with max_session_gap=0 so that pre-market,
    regular, and post-market bars are treated as distinct sessions even when
    they share the same calendar date.

    US ET boundaries (minutes since midnight):
        pre     04:00–09:30  [240, 570)
        regular 09:30–16:00  [570, 960)
        post    16:00–20:00  [960, 1200)
    """
    date = time_key[:10]
    h = int(time_key[11:13])
    m = int(time_key[14:16])
    mins = h * 60 + m
    if 240 <= mins < 570:
        seg = "pre"
    elif 570 <= mins <= 960:   # include 16:00 (=960) — it's the regular close bar
        seg = "regular"
    else:
        seg = "post"
    return f"{date}_{seg}"


def determine_trend(bos_signals: list[dict], min_consecutive: int = 1) -> str | None:
    """Return current trend direction from a BOS/CHoCH signal list.

    Iterates signals chronologically. Each CHoCH resets the trend and starts
    a new consecutive count. Returns None when fewer than min_consecutive BOS
    confirmations have occurred since the last CHoCH (or overall if none).
    """
    if not bos_signals:
        return None

    current_trend: str | None = None
    consecutive: int = 0

    for sig in bos_signals:
        if sig["type"] == "CHoCH":
            current_trend = sig["direction"]
            consecutive = 1
        elif sig["type"] == "BOS" and sig["direction"] == current_trend:
            consecutive += 1

    # Fallback: no CHoCH seen — infer from a unanimous BOS cluster
    if current_trend is None and bos_signals:
        dirs = {s["direction"] for s in bos_signals}
        if len(dirs) == 1:
            current_trend = bos_signals[0]["direction"]
            consecutive = len(bos_signals)

    if current_trend is None:
        return None
    return current_trend if consecutive >= min_consecutive else None


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


def _local_trend_array(closes: np.ndarray, window: int = 20) -> np.ndarray:
    """Return per-bar local trend ('up'/'down') using a backward-looking window.

    At each bar i the window covers closes[max(0, i-window+1) : i+1].
    The first-third mean is compared to the last-third mean; no lookahead.
    """
    n = len(closes)
    result = np.empty(n, dtype=object)
    for i in range(n):
        start = max(0, i - window + 1)
        w = closes[start : i]   # exclude bar i — its own price must not bias the trend label
        m = len(w)
        if m < 4:
            result[i] = "up"
        else:
            third = m // 3
            early = float(w[:third].mean())
            late  = float(w[m - third :].mean())
            result[i] = "up" if late > early else "down"
    return result


def _is_displacement(
    opens: np.ndarray, closes: np.ndarray,
    highs: np.ndarray, lows: np.ndarray,
    j: int, lookback: int = 5,
    body_mult: float = 1.5, body_ratio_min: float = 0.5,
) -> bool:
    """True if bar j is a displacement candle (large body, small wick ratio)."""
    if j < 1:
        return False
    start = max(0, j - lookback)
    prior_bodies = np.abs(closes[start:j] - opens[start:j])
    mean_body = float(prior_bodies.mean()) if len(prior_bodies) else 0.0
    body = abs(closes[j] - opens[j])
    total = highs[j] - lows[j]
    if body < body_mult * mean_body:
        return False
    if total > 0 and body / total < body_ratio_min:
        return False
    return True


def _price_accepted(
    closes: np.ndarray, j: int, level: float,
    direction: str, n: int, check_bars: int = 2,
) -> bool:
    """True if the next check_bars bars all close on the breakout side of level."""
    end = min(j + check_bars + 1, n)
    for k in range(j + 1, end):
        if direction == "bull" and closes[k] <= level:
            return False
        if direction == "bear" and closes[k] >= level:
            return False
    return True


def detect_bos_choch(klines: pd.DataFrame, lookback: int = 2,
                     filter_choch: bool = True,
                     trend_window: int = 20,
                     max_span_bars: int | None = None,
                     max_session_gap: int | None = None) -> list[dict]:
    """Detect Break of Structure (BOS) and Change of Character (CHoCH).

    BOS: price breaks a swing level in the direction of the current trend.
    CHoCH: price breaks a swing level against the current trend (reversal signal).

    Args:
        max_session_gap: max overnight session boundaries allowed between the reference
            swing and the break bar.  0 = same trading day only (prevents overnight gaps
            from triggering intraday structure breaks).  None = no restriction.
            Requires a 'time_key' column (YYYY-MM-DD ...) in klines.

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

    closes   = klines["close"].values
    opens    = klines["open"].values
    highs_ar = klines["high"].values
    lows_ar  = klines["low"].values
    n        = len(klines)

    # Extract per-bar session keys for boundary filtering.
    # Uses sub-session granularity (pre/regular/post) so that extended-hours bars
    # don't corrupt intraday regular-session structure even on the same calendar date.
    _dates: list[str] | None = None
    if max_session_gap is not None and "time_key" in klines.columns:
        _dates = [_session_key(str(t)) for t in klines["time_key"].values]

    # Per-bar local trend: backward-looking window so each signal is classified
    # against the trend *at that moment*, not a global state from bar 0.
    local_t  = _local_trend_array(closes, trend_window)
    signals: list[dict] = []
    processed_highs:      set[int] = set()
    processed_lows:       set[int] = set()
    processed_break_bars: set[int] = set()  # one signal per break bar

    # Bar index where the current trend was last confirmed (by CHoCH or initialisation).
    # Set to the reference-swing index (from_idx) of the most recent CHoCH, NOT the
    # break bar: this keeps swing levels formed between the reference and the break bar
    # available as future reference points in the new trend context.
    trend_started_at: int = 0

    # SMC-derived trend direction, updated on each CHoCH emission.
    # When None (no CHoCH seen yet) we fall back to local_t for classification.
    # Using SMC-derived state rather than local_t avoids the lag problem where
    # bars shortly after a CHoCH still carry the old trend label, causing
    # bearish BOS signals to appear immediately after a bullish CHoCH.
    smc_trend: str | None = None

    for i in range(2, len(swings)):
        sw = swings[i]

        if sw["kind"] == "high":
            # Only reference levels that belong to the current trend window
            prev_highs = [s for s in swings[:i]
                          if s["kind"] == "high" and s["idx"] >= trend_started_at]
            if not prev_highs:
                continue
            prev_high = prev_highs[-1]
            if prev_high["idx"] in processed_highs:
                continue
            # Use wick high as break threshold — close-based swing price can sit below
            # the actual wick, causing a premature BOS before the true structural break.
            ref_high = float(highs_ar[prev_high["idx"]])
            ref_date = _dates[prev_high["idx"]] if _dates else None
            for j in range(sw["idx"] + 1, n):
                if max_span_bars is not None and j - prev_high["idx"] > max_span_bars:
                    break
                if ref_date is not None and _dates[j] != ref_date:
                    break  # crossed a session boundary — stop scanning
                if closes[j] > ref_high:
                    if j not in processed_break_bars:
                        # Bull break: BOS if in-trend, CHoCH if counter-trend.
                        # Prefer smc_trend (updated by prior CHoCH) over local_t
                        # so that bars immediately after a trend change are
                        # classified correctly without MA lag.
                        if smc_trend is not None:
                            sig_type = "BOS" if smc_trend == "bull" else "CHoCH"
                        else:
                            sig_type = "BOS" if local_t[j] == "up" else "CHoCH"
                        emit = True
                        if filter_choch and sig_type == "CHoCH":
                            emit = (
                                _is_displacement(opens, closes, highs_ar, lows_ar, j)
                                and _price_accepted(closes, j, ref_high, "bull", n)
                            )
                        if emit:
                            signals.append({
                                "type":      sig_type,
                                "direction": "bull",
                                "idx":       j,
                                "price":     ref_high,
                                "from_idx":  prev_high["idx"],
                            })
                            processed_break_bars.add(j)
                        if sig_type == "CHoCH":   # always reset context, even if filtered
                            trend_started_at = prev_high["idx"]
                            smc_trend = "bull"
                    processed_highs.add(prev_high["idx"])
                    break

        else:  # kind == "low"
            prev_lows = [s for s in swings[:i]
                         if s["kind"] == "low" and s["idx"] >= trend_started_at]
            if not prev_lows:
                continue
            prev_low = prev_lows[-1]
            if prev_low["idx"] in processed_lows:
                continue
            # Use wick low as break threshold — symmetric with bull case.
            ref_low = float(lows_ar[prev_low["idx"]])
            ref_date = _dates[prev_low["idx"]] if _dates else None
            for j in range(sw["idx"] + 1, n):
                if max_span_bars is not None and j - prev_low["idx"] > max_span_bars:
                    break
                if ref_date is not None and _dates[j] != ref_date:
                    break  # crossed a session boundary — stop scanning
                if closes[j] < ref_low:
                    if j not in processed_break_bars:
                        # Bear break: BOS if in-trend, CHoCH if counter-trend.
                        if smc_trend is not None:
                            sig_type = "BOS" if smc_trend == "bear" else "CHoCH"
                        else:
                            sig_type = "BOS" if local_t[j] == "down" else "CHoCH"
                        emit = True
                        if filter_choch and sig_type == "CHoCH":
                            emit = (
                                _is_displacement(opens, closes, highs_ar, lows_ar, j)
                                and _price_accepted(closes, j, ref_low, "bear", n)
                            )
                        if emit:
                            signals.append({
                                "type":      sig_type,
                                "direction": "bear",
                                "idx":       j,
                                "price":     ref_low,
                                "from_idx":  prev_low["idx"],
                            })
                            processed_break_bars.add(j)
                        if sig_type == "CHoCH":   # always reset context, even if filtered
                            trend_started_at = prev_low["idx"]
                            smc_trend = "bear"
                    processed_lows.add(prev_low["idx"])
                    break

    return _remove_bos_crossing_choch(signals)


def _remove_bos_crossing_choch(signals: list[dict]) -> list[dict]:
    """Discard BOS signals whose [from_idx, idx] span contains any CHoCH.

    A BOS that references a swing formed before a CHoCH is referencing stale
    structure — the CHoCH already invalidated that market context.
    """
    choch_idxs = [s["idx"] for s in signals if s["type"] == "CHoCH"]
    if not choch_idxs:
        return signals
    result = []
    for sig in signals:
        if sig["type"] != "BOS":
            result.append(sig)
            continue
        from_i  = sig.get("from_idx", sig["idx"] - 1)
        break_i = sig["idx"]
        if any(from_i < ci < break_i for ci in choch_idxs):
            continue   # this BOS crosses a CHoCH — invalid
        result.append(sig)
    return result

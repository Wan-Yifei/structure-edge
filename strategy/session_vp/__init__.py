"""Session Value-Area reversal strategy -- long-only, per-session volume profile.

Wait a configurable warmup window after each session (pre/regular/post/
overnight) starts, compute a frozen POC/VAH/VAL from the bars since session
start, then go long on a VAL-touch-then-reversal signal confirmed by an
oversold RSI reading. Target = POC, stop mirrored for an exact 1:1 R:R.

Public API:
    profile.compute_value_area(edges, bin_vols, va_pct) -> {"poc", "vah", "val"}
    reversal.detect_val_reversal(klines, val, rsi_period, rsi_threshold, start_idx) -> list[dict]
    reversal.compute_rsi(closes, period) -> np.ndarray

Wired into the backtest engine via backtest/session_vp_engine.py, not here --
this package only holds pure, backtest/GUI-agnostic detection logic (same
split as strategy/smc/).
"""

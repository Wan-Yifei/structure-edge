"""Unit tests for analysis/signal_scanner.py — SignalDetector.detect()."""

import json
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from backtest.engine import BacktestParams
from analysis.signal_scanner import SignalDetector


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_klines(n: int = 30) -> pd.DataFrame:
    """Minimal valid klines DataFrame with monotone uptrend."""
    prices = np.linspace(100.0, 115.0, n)
    times  = pd.date_range("2026-06-01 09:30", periods=n, freq="1h")
    return pd.DataFrame({
        "time_key": [t.strftime("%Y-%m-%d %H:%M:%S") for t in times],
        "open":     prices - 0.1,
        "high":     prices + 0.3,
        "low":      prices - 0.3,
        "close":    prices,
        "volume":   np.ones(n) * 1000.0,
    })


def _base_params(**kw) -> BacktestParams:
    base = BacktestParams(
        trend_tf="1h",
        entry_tf="15m",
        swing_lookback=2,
        htf_window_bars=20,
        fvg_min_width_pct=0.0,
        displacement_required=False,
        require_ltf_trend_bar=False,
        sl_buffer_pct=0.001,
        max_sl_pct=0.5,   # very permissive
        min_rr=0.1,       # very permissive
        allow_short=True,
    )
    return replace(base, **kw)


def _bull_bos() -> list[dict]:
    """Minimal BOS signal list representing a bull trend."""
    return [{"type": "CHoCH", "direction": "bull", "price": 105.0, "time": "2026-06-01 10:00:00"}]


def _bull_fvgs() -> list[dict]:
    """One unfilled bullish FVG between 104 and 106."""
    return [{"direction": "bull", "top": 106.0, "bottom": 104.0, "idx": 15, "filled": False}]


def _swings() -> list[dict]:
    """Swing points giving SL below FVG mid (105) and TP above it."""
    return [
        {"kind": "low",  "idx": 5,  "price": 101.0, "time": "2026-06-01 09:30:00"},
        {"kind": "high", "idx": 12, "price": 112.0, "time": "2026-06-01 10:00:00"},
    ]


# ── guard: too-short / None input ─────────────────────────────────────────────

def test_returns_empty_when_htf_is_none():
    result = SignalDetector.detect("US.AAPL", None, _base_params())
    assert result == []


def test_returns_empty_when_htf_too_short():
    params = replace(_base_params(), htf_window_bars=30)
    result = SignalDetector.detect("US.AAPL", _make_klines(n=5), params)
    assert result == []


# ── allow_short=False ─────────────────────────────────────────────────────────

def test_allow_short_false_suppresses_bear_signal():
    """detect() must return [] when trend is bear and allow_short=False."""
    bear_bos = [{"type": "CHoCH", "direction": "bear", "price": 108.0, "time": "2026-06-01 10:00:00"}]
    bear_fvg = [{"direction": "bear", "top": 108.0, "bottom": 106.0, "idx": 15, "filled": False}]
    bear_swings = [
        {"kind": "high", "idx": 5,  "price": 112.0, "time": "2026-06-01 09:30:00"},
        {"kind": "low",  "idx": 12, "price": 100.0, "time": "2026-06-01 10:00:00"},
    ]
    klines = _make_klines(n=30)
    params = replace(_base_params(), allow_short=False)
    with (
        patch("analysis.signal_scanner.detect_bos_choch", return_value=bear_bos),
        patch("analysis.signal_scanner.detect_fvg",        return_value=bear_fvg),
        patch("analysis.signal_scanner.find_swings",       return_value=bear_swings),
    ):
        result = SignalDetector.detect("US.TSLA", klines, params)
    assert result == []


# ── no FVGs ───────────────────────────────────────────────────────────────────

def test_returns_empty_when_no_fvg():
    klines = _make_klines(n=30)
    with (
        patch("analysis.signal_scanner.detect_bos_choch", return_value=_bull_bos()),
        patch("analysis.signal_scanner.detect_fvg",        return_value=[]),
        patch("analysis.signal_scanner.find_swings",       return_value=_swings()),
    ):
        result = SignalDetector.detect("US.AAPL", klines, _base_params())
    assert result == []


# ── happy path: signal dict keys & values ────────────────────────────────────

def _run_bull_detect(klines, **param_overrides):
    params = replace(_base_params(), **param_overrides) if param_overrides else _base_params()
    with (
        patch("analysis.signal_scanner.detect_bos_choch", return_value=_bull_bos()),
        patch("analysis.signal_scanner.detect_fvg",        return_value=_bull_fvgs()),
        patch("analysis.signal_scanner.find_swings",       return_value=_swings()),
    ):
        return SignalDetector.detect("US.AAPL", klines, params)


def test_signal_dict_has_required_keys():
    required = {
        "symbol", "direction", "signal_time",
        "trend_tf", "entry_tf",
        "entry_zone_top", "entry_zone_bottom",
        "sl_price", "tp_price", "rr_ratio",
        "strategy", "params_json", "algo_version",
        "source", "status", "created_at",
    }
    klines = _make_klines(n=30)
    # Close must be above FVG mid (105) for bull → last bar ~115 ✓
    result = _run_bull_detect(klines)
    assert len(result) == 1
    missing = required - result[0].keys()
    assert not missing, f"Signal missing keys: {missing}"


def test_signal_numeric_values_are_correct_types():
    klines = _make_klines(n=30)
    result = _run_bull_detect(klines)
    assert len(result) == 1
    sig = result[0]
    assert isinstance(sig["entry_zone_top"],    float)
    assert isinstance(sig["entry_zone_bottom"], float)
    assert isinstance(sig["sl_price"],          float)
    assert isinstance(sig["tp_price"],          float)
    assert isinstance(sig["rr_ratio"],          float)
    assert sig["entry_zone_top"] > sig["entry_zone_bottom"]
    assert sig["rr_ratio"] >= _base_params().min_rr


def test_signal_direction_matches_trend():
    klines = _make_klines(n=30)
    result = _run_bull_detect(klines)
    assert len(result) == 1
    assert result[0]["direction"] == "bull"


def test_signal_symbol_propagated():
    klines = _make_klines(n=30)
    params = _base_params()
    with (
        patch("analysis.signal_scanner.detect_bos_choch", return_value=_bull_bos()),
        patch("analysis.signal_scanner.detect_fvg",        return_value=_bull_fvgs()),
        patch("analysis.signal_scanner.find_swings",       return_value=_swings()),
    ):
        result = SignalDetector.detect("HK.00700", klines, params)
    assert len(result) == 1
    assert result[0]["symbol"] == "HK.00700"


def test_params_json_is_valid_and_contains_trend_tf():
    klines = _make_klines(n=30)
    result = _run_bull_detect(klines)
    assert len(result) == 1
    parsed = json.loads(result[0]["params_json"])
    assert isinstance(parsed, dict)
    assert parsed.get("trend_tf") == "1h"


def test_strategy_field_is_smc():
    klines = _make_klines(n=30)
    result = _run_bull_detect(klines)
    assert result[0]["strategy"] == "smc"


def test_status_defaults_to_open():
    klines = _make_klines(n=30)
    result = _run_bull_detect(klines)
    assert result[0]["status"] == "open"


# ── min_rr filter ─────────────────────────────────────────────────────────────

def test_min_rr_filter_excludes_low_rr_signals():
    """With min_rr=999, no signal should survive."""
    klines = _make_klines(n=30)
    result = _run_bull_detect(klines, min_rr=999.0)
    assert result == []


# ── require_ltf_trend_bar ─────────────────────────────────────────────────────

def test_require_ltf_trend_bar_rejects_bearish_close():
    """If last bar is bearish (close < open) and trend is bull, no signal."""
    klines = _make_klines(n=30)
    # Make last candle bearish: close < open
    klines = klines.copy()
    klines.loc[klines.index[-1], "open"]  = 116.0
    klines.loc[klines.index[-1], "close"] = 113.0  # bearish bar
    result = _run_bull_detect(klines, require_ltf_trend_bar=True)
    assert result == []

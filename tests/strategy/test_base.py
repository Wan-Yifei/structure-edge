"""Unit tests for strategy/base.py — BaseStrategy ABC and SMCStrategy adapter."""

import pytest
import pandas as pd
import numpy as np

from strategy.base import BaseStrategy, SMCStrategy


# ── Helpers ───────────────────────────────────────────────────────────────────

def _trending_klines(n: int, start: float, step: float) -> pd.DataFrame:
    closes  = [start + i * step for i in range(n)]
    highs   = [c + abs(step) * 0.6 for c in closes]
    lows    = [c - abs(step) * 0.6 for c in closes]
    times   = pd.date_range("2025-01-02 09:30", periods=n, freq="60min")
    return pd.DataFrame({
        "time_key": times.strftime("%Y-%m-%d %H:%M:%S"),
        "open":  closes, "high": highs, "low": lows,
        "close": closes, "volume": [10_000] * n,
    })


# ── BaseStrategy ABC ──────────────────────────────────────────────────────────

class TestBaseStrategyABC:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            BaseStrategy()

    def test_subclass_without_all_methods_fails(self):
        class Incomplete(BaseStrategy):
            def detect_zones(self, htf_df):
                return []
        with pytest.raises(TypeError):
            Incomplete()

    def test_concrete_subclass_ok(self):
        class Concrete(BaseStrategy):
            def detect_zones(self, htf_df):
                return []
            def generate_signals(self, htf_df, ltf_df):
                return []
        c = Concrete(config={"k": 1})
        assert c.config == {"k": 1}

    def test_default_config_is_empty_dict(self):
        class Concrete(BaseStrategy):
            def detect_zones(self, htf_df): return []
            def generate_signals(self, h, l): return []
        assert Concrete().config == {}


# ── SMCStrategy ───────────────────────────────────────────────────────────────

class TestSMCStrategy:
    def test_instantiates_with_config(self):
        s = SMCStrategy(config={"fvg_min_width_pct": 0.001, "min_rr": 2.0})
        assert s.config["min_rr"] == 2.0

    def test_detect_zones_returns_list(self):
        s = SMCStrategy(config={"fvg_min_width_pct": 0.001})
        htf = _trending_klines(80, 100.0, 0.2)
        zones = s.detect_zones(htf)
        assert isinstance(zones, list)

    def test_zone_schema(self):
        s = SMCStrategy(config={"fvg_min_width_pct": 0.001})
        htf = _trending_klines(120, 50.0, 0.15)
        zones = s.detect_zones(htf)
        for z in zones:
            assert "direction" in z
            assert "top"       in z
            assert "bottom"    in z
            assert "filled"    in z
            assert z["direction"] in ("bull", "bear")
            assert z["top"] >= z["bottom"]

    def test_generate_signals_returns_list(self):
        s = SMCStrategy(config={
            "fvg_min_width_pct": 0.001,
            "fvg_entry_depth_pct": 0.05,
            "min_rr": 1.0,
            "max_sl_pct": 0.05,
            "sl_buffer_pct": 0.001,
            "swing_lookback": 2,
            "bos_count": 1,
        })
        htf = _trending_klines(200, 50.0, 0.15)
        ltf = _trending_klines(800, 50.0, 0.04)
        signals = s.generate_signals(htf, ltf)
        assert isinstance(signals, list)

    def test_signal_schema(self):
        s = SMCStrategy(config={
            "fvg_min_width_pct": 0.001,
            "fvg_entry_depth_pct": 0.05,
            "min_rr": 1.0,
            "max_sl_pct": 0.05,
            "sl_buffer_pct": 0.001,
            "swing_lookback": 2,
            "bos_count": 1,
        })
        htf = _trending_klines(200, 50.0, 0.15)
        ltf = _trending_klines(800, 50.0, 0.04)
        signals = s.generate_signals(htf, ltf)
        for sig in signals:
            assert "direction"   in sig
            assert "entry_price" in sig
            assert "sl"          in sig
            assert "tp"          in sig
            assert "planned_rr"  in sig
            assert "entry_time"  in sig
            assert sig["direction"] in ("bull", "bear")
            assert sig["planned_rr"] >= 1.0

    def test_no_signals_on_flat_market(self):
        s = SMCStrategy(config={
            "fvg_min_width_pct": 0.001,
            "fvg_entry_depth_pct": 0.05,
            "min_rr": 2.0,
            "max_sl_pct": 0.01,
            "sl_buffer_pct": 0.001,
            "swing_lookback": 2,
            "bos_count": 1,
        })
        htf = _trending_klines(50, 100.0, 0.0)   # flat — no trend
        ltf = _trending_klines(200, 100.0, 0.0)
        signals = s.generate_signals(htf, ltf)
        assert signals == []

    def test_is_base_strategy_subclass(self):
        s = SMCStrategy()
        assert isinstance(s, BaseStrategy)

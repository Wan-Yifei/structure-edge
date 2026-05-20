"""Unit tests for strategy/smc — market structure, FVG, and confirmation."""

import pandas as pd
import numpy as np
import pytest

from strategy.smc.market_structure import find_swings, detect_bos_choch, determine_trend
from strategy.smc.fvg import (
    detect_fvg, fvg_entry_depth, is_displacement_candle,
    compute_volume_profile, fvg_overlaps_lvn,
)
from strategy.smc.confirmation import check_ltf_confirmation


# ── Helpers ───────────────────────────────────────────────────────────────────

def _klines(closes, highs=None, lows=None, opens=None, volumes=None):
    n = len(closes)
    closes = list(closes)
    highs  = highs  if highs  is not None else [c + 1.0 for c in closes]
    lows   = lows   if lows   is not None else [c - 1.0 for c in closes]
    opens  = opens  if opens  is not None else closes
    volumes = volumes if volumes is not None else [1000] * n
    return pd.DataFrame({
        "time_key": [f"2025-01-01 {i:02d}:00:00" for i in range(n)],
        "open":     opens,
        "high":     highs,
        "low":      lows,
        "close":    closes,
        "volume":   volumes,
    })


# ── find_swings ───────────────────────────────────────────────────────────────

class TestFindSwings:
    def test_detects_high_and_low(self):
        # clear peak at index 2, clear trough at index 4
        df = _klines([1, 2, 5, 2, 1, 2, 3], highs=[2,3,6,3,2,3,4], lows=[0,1,4,1,0,1,2])
        swings = find_swings(df, lookback=1)
        kinds = [s["kind"] for s in swings]
        assert "high" in kinds
        assert "low" in kinds

    def test_alternates_strictly(self):
        df = _klines([1, 3, 2, 4, 1, 3, 2])
        swings = find_swings(df, lookback=1)
        for a, b in zip(swings, swings[1:]):
            assert a["kind"] != b["kind"], "Consecutive swings must alternate"

    def test_empty_for_short_series(self):
        df = _klines([1, 2, 3])
        assert find_swings(df, lookback=2) == []

    def test_price_is_close(self):
        closes = [1, 3, 2, 1, 2, 3, 1]
        df = _klines(closes)
        swings = find_swings(df, lookback=1)
        for sw in swings:
            assert sw["price"] == closes[sw["idx"]]


# ── detect_bos_choch ──────────────────────────────────────────────────────────

class TestDetectBosChoch:
    def _trending_up(self):
        # Steadily rising closes: should produce bull BOS signals
        closes = [10, 11, 10, 12, 11, 13, 12, 14, 13, 15, 14, 16]
        return _klines(closes, highs=[c+0.5 for c in closes], lows=[c-0.5 for c in closes])

    def _trending_down(self):
        closes = [16, 15, 16, 14, 15, 13, 14, 12, 13, 11, 12, 10]
        return _klines(closes, highs=[c+0.5 for c in closes], lows=[c-0.5 for c in closes])

    def test_returns_list(self):
        assert isinstance(detect_bos_choch(self._trending_up()), list)

    def test_bull_signals_in_uptrend(self):
        sigs = detect_bos_choch(self._trending_up(), lookback=1)
        bulls = [s for s in sigs if s["direction"] == "bull"]
        assert len(bulls) > 0

    def test_bear_signals_in_downtrend(self):
        sigs = detect_bos_choch(self._trending_down(), lookback=1)
        bears = [s for s in sigs if s["direction"] == "bear"]
        assert len(bears) > 0

    def test_signal_fields_present(self):
        sigs = detect_bos_choch(self._trending_up(), lookback=1)
        if sigs:
            s = sigs[0]
            assert {"type", "direction", "idx", "price", "from_idx"} <= s.keys()
            assert s["type"] in ("BOS", "CHoCH")
            assert s["direction"] in ("bull", "bear")

    def test_empty_for_short_series(self):
        assert detect_bos_choch(_klines([1, 2, 3]), lookback=2) == []


# ── determine_trend ───────────────────────────────────────────────────────────

class TestDetermineTrend:
    def test_none_when_no_signals(self):
        assert determine_trend([]) is None

    def test_bull_after_choch(self):
        sigs = [{"type": "CHoCH", "direction": "bull"}]
        assert determine_trend(sigs, min_consecutive=1) == "bull"

    def test_bear_after_choch(self):
        sigs = [{"type": "CHoCH", "direction": "bear"}]
        assert determine_trend(sigs, min_consecutive=1) == "bear"

    def test_none_when_below_min_consecutive(self):
        sigs = [{"type": "CHoCH", "direction": "bull"}]
        assert determine_trend(sigs, min_consecutive=2) is None

    def test_consecutive_bos_counted(self):
        sigs = [
            {"type": "CHoCH", "direction": "bull"},
            {"type": "BOS",   "direction": "bull"},
        ]
        assert determine_trend(sigs, min_consecutive=2) == "bull"

    def test_choch_resets_direction(self):
        sigs = [
            {"type": "CHoCH", "direction": "bull"},
            {"type": "BOS",   "direction": "bull"},
            {"type": "CHoCH", "direction": "bear"},
        ]
        assert determine_trend(sigs, min_consecutive=1) == "bear"

    def test_unanimous_bos_fallback(self):
        sigs = [
            {"type": "BOS", "direction": "bear"},
            {"type": "BOS", "direction": "bear"},
        ]
        assert determine_trend(sigs, min_consecutive=2) == "bear"


# ── detect_fvg ────────────────────────────────────────────────────────────────

class TestDetectFvg:
    def _bull_fvg_df(self):
        # candle[0] high=10, candle[2] low=12 → gap 10–12
        return _klines(
            closes=[10, 11, 12],
            highs=[10, 11, 13],
            lows=[9,  10, 12],
        )

    def _bear_fvg_df(self):
        # candle[0] low=12, candle[2] high=10 → gap 10–12
        return _klines(
            closes=[12, 11, 10],
            highs=[13, 12, 10],
            lows=[12, 10, 9],
        )

    def test_detects_bull_fvg(self):
        gaps = detect_fvg(self._bull_fvg_df(), min_gap_pct=0.0)
        bulls = [g for g in gaps if g["direction"] == "bull"]
        assert len(bulls) == 1
        assert bulls[0]["bottom"] == pytest.approx(10.0)
        assert bulls[0]["top"]    == pytest.approx(12.0)

    def test_detects_bear_fvg(self):
        gaps = detect_fvg(self._bear_fvg_df(), min_gap_pct=0.0)
        bears = [g for g in gaps if g["direction"] == "bear"]
        assert len(bears) == 1

    def test_filled_flag(self):
        # Add a 4th candle whose close is inside the gap
        df = _klines(
            closes=[10, 11, 12, 11],
            highs=[10, 11, 13, 12],
            lows=[9,  10, 12, 10],
        )
        gaps = detect_fvg(df, min_gap_pct=0.0)
        bulls = [g for g in gaps if g["direction"] == "bull"]
        assert bulls[0]["filled"] is True

    def test_min_gap_filter(self):
        gaps = detect_fvg(self._bull_fvg_df(), min_gap_pct=0.99)
        assert gaps == []

    def test_fields_present(self):
        gaps = detect_fvg(self._bull_fvg_df(), min_gap_pct=0.0)
        if gaps:
            assert {"direction", "top", "bottom", "idx", "filled"} <= gaps[0].keys()


# ── fvg_entry_depth ───────────────────────────────────────────────────────────

class TestFvgEntryDepth:
    def _bull(self):
        return {"direction": "bull", "top": 100.0, "bottom": 90.0}

    def _bear(self):
        return {"direction": "bear", "top": 100.0, "bottom": 90.0}

    def test_bull_above_zone_returns_zero(self):
        assert fvg_entry_depth(self._bull(), 105.0) == 0.0

    def test_bull_at_top_edge(self):
        assert fvg_entry_depth(self._bull(), 100.0) == pytest.approx(0.0)

    def test_bull_at_midpoint(self):
        assert fvg_entry_depth(self._bull(), 95.0) == pytest.approx(0.5)

    def test_bull_below_bottom_clamped(self):
        assert fvg_entry_depth(self._bull(), 80.0) == pytest.approx(1.0)

    def test_bear_below_zone_returns_zero(self):
        assert fvg_entry_depth(self._bear(), 85.0) == 0.0

    def test_bear_at_midpoint(self):
        assert fvg_entry_depth(self._bear(), 95.0) == pytest.approx(0.5)

    def test_degenerate_zone_returns_zero(self):
        fvg = {"direction": "bull", "top": 10.0, "bottom": 10.0}
        assert fvg_entry_depth(fvg, 10.0) == 0.0


# ── is_displacement_candle ────────────────────────────────────────────────────

class TestIsDisplacementCandle:
    def _make(self, ranges, body_ratios):
        rows = []
        for rng, br in zip(ranges, body_ratios):
            lo = 100.0
            hi = lo + rng
            body = rng * br
            op = lo + (rng - body) / 2
            cl = op + body
            rows.append({"open": op, "high": hi, "low": lo, "close": cl,
                         "time_key": "2025-01-01 00:00:00", "volume": 1000})
        return pd.DataFrame(rows)

    def test_qualifies_when_large_and_directional(self):
        # B (index 1) has range 10, neighbors have range 1
        df = self._make([1, 10, 1], [0.5, 0.8, 0.5])
        assert is_displacement_candle(df, fvg_idx=2, atr_mult=1.5, body_ratio_min=0.5) is True

    def test_fails_when_range_too_small(self):
        df = self._make([1, 1.2, 1], [0.5, 0.8, 0.5])
        assert is_displacement_candle(df, fvg_idx=2, atr_mult=1.5, body_ratio_min=0.5) is False

    def test_fails_when_doji(self):
        df = self._make([1, 10, 1], [0.5, 0.1, 0.5])
        assert is_displacement_candle(df, fvg_idx=2, atr_mult=1.5, body_ratio_min=0.5) is False

    def test_edge_returns_true(self):
        # fvg_idx=0 → a < 0 → fallback True
        df = self._make([1, 2], [0.5, 0.5])
        assert is_displacement_candle(df, fvg_idx=0, atr_mult=1.5, body_ratio_min=0.5) is True


# ── compute_volume_profile / fvg_overlaps_lvn ─────────────────────────────────

class TestVolumeProfile:
    def _klines_with_vol(self, n=20):
        closes  = list(np.linspace(100, 110, n))
        highs   = [c + 0.5 for c in closes]
        lows    = [c - 0.5 for c in closes]
        volumes = [1000] * n
        return _klines(closes, highs=highs, lows=lows, volumes=volumes)

    def test_returns_arrays(self):
        edges, vols = compute_volume_profile(self._klines_with_vol())
        assert edges is not None and vols is not None
        assert len(edges) == 101
        assert len(vols)  == 100

    def test_degenerate_returns_none(self):
        df = _klines([10, 10, 10], highs=[10,10,10], lows=[10,10,10])
        edges, vols = compute_volume_profile(df)
        assert edges is None and vols is None

    def test_lvn_overlap_returns_bool(self):
        edges, vols = compute_volume_profile(self._klines_with_vol())
        fvg = {"direction": "bull", "top": 105.5, "bottom": 104.5}
        result = fvg_overlaps_lvn(fvg, edges, vols, lvn_threshold=0.3)
        assert isinstance(result, (bool, np.bool_))

    def test_lvn_false_when_no_profile(self):
        fvg = {"direction": "bull", "top": 105.0, "bottom": 104.0}
        assert fvg_overlaps_lvn(fvg, None, None) is False


# ── check_ltf_confirmation ────────────────────────────────────────────────────

class TestCheckLtfConfirmation:
    def test_requires_choch_then_bos(self):
        sigs = [
            {"type": "CHoCH", "direction": "bull", "idx": 5},
            {"type": "BOS",   "direction": "bull", "idx": 8},
        ]
        assert check_ltf_confirmation(sigs, "bull", after_idx=0) is True

    def test_fails_without_bos_after_choch(self):
        sigs = [{"type": "CHoCH", "direction": "bull", "idx": 5}]
        assert check_ltf_confirmation(sigs, "bull", after_idx=0) is False

    def test_fails_when_bos_before_choch(self):
        sigs = [
            {"type": "BOS",   "direction": "bull", "idx": 3},
            {"type": "CHoCH", "direction": "bull", "idx": 5},
        ]
        assert check_ltf_confirmation(sigs, "bull", after_idx=0) is False

    def test_respects_after_idx(self):
        sigs = [
            {"type": "CHoCH", "direction": "bull", "idx": 5},
            {"type": "BOS",   "direction": "bull", "idx": 8},
        ]
        assert check_ltf_confirmation(sigs, "bull", after_idx=6) is False

    def test_wrong_direction_ignored(self):
        sigs = [
            {"type": "CHoCH", "direction": "bear", "idx": 5},
            {"type": "BOS",   "direction": "bear", "idx": 8},
        ]
        assert check_ltf_confirmation(sigs, "bull", after_idx=0) is False

    def test_empty_signals(self):
        assert check_ltf_confirmation([], "bull", after_idx=0) is False

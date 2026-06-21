"""Unit tests for backtest/fvg_width_sweep.py — combo grid and scoring aggregation.

Tests for the underlying detection primitives (gap_width_pct, gaps_for_combo,
build_daily_lvn_profiles, gap_overlaps_lvn) live in tests/strategy/test_smc.py
since those functions live in strategy/smc/fvg.py.
"""

import numpy as np
import pandas as pd
import pytest

from backtest.fvg_width_sweep import build_combo_list, score_combo
from strategy.smc.fvg import gap_width_pct, gaps_for_combo


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


def _displacement_klines() -> pd.DataFrame:
    """5 calm baseline candles, then a weak (non-displacement) FVG at idx 5-7,
    then a strong (displacement) FVG at idx 8-10. Both gaps are geometrically
    valid; only the second has a middle candle that passes is_displacement_candle.
    """
    opens  = [100.3, 100.3, 100.3, 100.3, 100.3, 102.3, 103.3, 105.2, 106.3, 107.5, 113.2]
    closes = [100.6, 100.6, 100.6, 100.6, 100.6, 102.7, 103.6, 105.8, 106.7, 111.5, 113.8]
    highs  = [101,   101,   101,   101,   101,   103,   104,   106,   107,   112,   114]
    lows   = [100,   100,   100,   100,   100,   102,   103,   105,   106,   107,   113]
    return _klines(closes=closes, highs=highs, lows=lows, opens=opens)


# ── build_combo_list ──────────────────────────────────────────────────────────

class TestBuildComboList:
    def _grid(self):
        return {
            "min_gap_pct":          [0.001, 0.002],
            "require_displacement": [False, True],
            "atr_mult":             [1.0, 1.5],
            "body_ratio_min":       [0.5],
            "lookback":             [5],
        }

    def test_dedups_irrelevant_displacement_params(self):
        # 2 min_gap_pct x 1 (deduped, displacement off) + 2 min_gap_pct x 2 atr_mult (displacement on) = 6
        combos = build_combo_list(self._grid())
        assert len(combos) == 6

    def test_no_duplicate_combos(self):
        combos = build_combo_list(self._grid())
        seen = {tuple(sorted(c.items(), key=lambda kv: kv[0])) for c in combos}
        assert len(seen) == len(combos)

    def test_displacement_off_combos_drop_displacement_keys(self):
        combos = build_combo_list(self._grid())
        off_combos = [c for c in combos if not c.get("require_displacement", False)]
        assert len(off_combos) == 2
        for c in off_combos:
            assert "atr_mult" not in c
            assert "body_ratio_min" not in c
            assert "lookback" not in c

    def test_displacement_on_combos_keep_displacement_keys(self):
        combos = build_combo_list(self._grid())
        on_combos = [c for c in combos if c.get("require_displacement", False)]
        assert len(on_combos) == 4
        for c in on_combos:
            assert {"atr_mult", "body_ratio_min", "lookback"} <= c.keys()

    def test_lvn_dedup_is_independent_of_displacement_dedup(self):
        grid = {
            "min_gap_pct":          [0.001],
            "require_displacement": [False, True],
            "atr_mult":             [1.0],
            "body_ratio_min":       [0.5],
            "lookback":             [5],
            "require_lvn_overlap":  [False, True],
            "lvn_threshold":        [0.2, 0.3],
        }
        combos = build_combo_list(grid)
        # disp=False,lvn=False -> 1 ; disp=False,lvn=True -> 2
        # disp=True, lvn=False -> 1 ; disp=True, lvn=True  -> 2
        assert len(combos) == 6
        for c in combos:
            if not c.get("require_lvn_overlap", False):
                assert "lvn_threshold" not in c
            else:
                assert "lvn_threshold" in c


# ── score_combo ───────────────────────────────────────────────────────────────

class TestScoreCombo:
    def test_no_gaps_returns_zeros(self):
        flat = _klines(closes=[100, 100, 100, 100, 100])
        result = score_combo(flat, {"min_gap_pct": 0.0, "require_displacement": False})
        assert result == {
            "n_gaps": 0, "total_width_pct": 0.0,
            "mean_width_pct": 0.0, "median_width_pct": 0.0,
        }

    def test_aggregation_matches_manual_computation(self):
        klines = _displacement_klines()
        combo  = {"min_gap_pct": 0.0, "require_displacement": False}
        gaps   = gaps_for_combo(klines, combo)
        widths = [gap_width_pct(g) for g in gaps]

        result = score_combo(klines, combo)
        assert result["n_gaps"]           == len(widths)
        assert result["total_width_pct"]  == pytest.approx(sum(widths))
        assert result["mean_width_pct"]   == pytest.approx(sum(widths) / len(widths))
        assert result["median_width_pct"] == pytest.approx(float(np.median(widths)))

    def test_raw_gaps_cache_does_not_change_result(self):
        # Combos sharing min_gap_pct must score identically whether or not a
        # cache dict is reused across calls (cache is a pure perf optimization).
        klines = _displacement_klines()
        combo  = {"min_gap_pct": 0.0, "require_displacement": False}
        uncached = score_combo(klines, combo)

        cache = {}
        cached_first  = score_combo(klines, combo, raw_gaps_cache=cache)
        cached_second = score_combo(klines, combo, raw_gaps_cache=cache)
        assert cached_first == uncached
        assert cached_second == uncached

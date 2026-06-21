"""Unit tests for backtest/fvg_width_sweep.py — combo grid, scoring, displacement filter."""

import numpy as np
import pandas as pd
import pytest

from backtest.fvg_width_sweep import (
    build_combo_list, gap_width_pct, gaps_for_combo, score_combo,
    build_daily_lvn_profiles, gap_overlaps_lvn,
)
from strategy.smc.fvg import detect_fvg


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


def _klines_with_dates(closes, highs, lows, dates, volumes=None):
    n = len(closes)
    volumes = volumes if volumes is not None else [1000] * n
    return pd.DataFrame({
        "time_key": [f"{dates[i]} {9 + i:02d}:00:00" for i in range(n)],
        "open":     list(closes),
        "high":     highs,
        "low":      lows,
        "close":    closes,
        "volume":   volumes,
    })


def _lvn_klines() -> pd.DataFrame:
    """Day 1 (idx 0-2) builds a volume profile with a heavy-volume zone at
    100-101 and a light-volume zone (LVN) at 102-103. Day 2 (idx 3-8) has two
    bull FVGs: one at idx 5 sitting inside day 1's LVN zone, one at idx 8
    sitting inside day 1's heavy-volume zone.
    """
    dates   = ["2025-01-01"] * 3 + ["2025-01-02"] * 6
    closes  = [100.5, 100.5, 102.5,  101.5, 102.0, 102.5,  100.15, 100.4, 100.75]
    highs   = [101,   101,   103,    102.0, 102.5, 103.0,  100.3,  100.5, 100.9]
    lows    = [100,   100,   102,    101.0, 101.5, 102.5,  100.0,  100.3, 100.6]
    volumes = [1000,  1000,  10,     1000,  1000,  1000,   1000,   1000,  1000]
    return _klines_with_dates(closes, highs, lows, dates, volumes)


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


# ── gap_width_pct ─────────────────────────────────────────────────────────────

class TestGapWidthPct:
    def test_known_width(self):
        gap = {"top": 110.0, "bottom": 90.0}
        assert gap_width_pct(gap) == pytest.approx(0.2)

    def test_zero_midpoint_returns_zero(self):
        gap = {"top": 0.0, "bottom": 0.0}
        assert gap_width_pct(gap) == 0.0


# ── gaps_for_combo ────────────────────────────────────────────────────────────

class TestGapsForCombo:
    def test_no_displacement_filter_keeps_both_gaps(self):
        klines = _displacement_klines()
        gaps = gaps_for_combo(klines, {"min_gap_pct": 0.0, "require_displacement": False})
        idxs = {g["idx"] for g in gaps}
        assert 7  in idxs  # weak gap
        assert 10 in idxs  # strong gap

    def test_displacement_filter_drops_weak_gap_keeps_strong_gap(self):
        klines = _displacement_klines()
        combo = {
            "min_gap_pct": 0.0, "require_displacement": True,
            "atr_mult": 1.5, "body_ratio_min": 0.5, "lookback": 5,
        }
        gaps = gaps_for_combo(klines, combo)
        idxs = {g["idx"] for g in gaps}
        assert 7  not in idxs
        assert 10 in idxs


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


# ── LVN volume profile ────────────────────────────────────────────────────────

class TestBuildDailyLvnProfiles:
    def test_first_date_has_no_profile(self):
        profiles = build_daily_lvn_profiles(_lvn_klines())
        assert "2025-01-01" not in profiles

    def test_second_date_profile_built_from_first_day(self):
        profiles = build_daily_lvn_profiles(_lvn_klines())
        edges, bin_vols = profiles["2025-01-02"]
        assert edges is not None
        # day 1's range is 100-103; the profile must span exactly that range.
        assert edges[0]  == pytest.approx(100.0)
        assert edges[-1] == pytest.approx(103.0)


class TestGapOverlapsLvn:
    def _gap(self, klines, idx):
        gaps = detect_fvg(klines, min_gap_pct=0.0, require_displacement=False)
        return next(g for g in gaps if g["idx"] == idx)

    def test_gap_in_low_volume_zone_overlaps(self):
        klines = _lvn_klines()
        profiles = build_daily_lvn_profiles(klines)
        gap = self._gap(klines, 5)  # zone 102.0-102.5, day1's light-volume zone
        assert gap_overlaps_lvn(klines, gap, profiles, lvn_threshold=0.30)

    def test_gap_in_high_volume_zone_does_not_overlap(self):
        klines = _lvn_klines()
        profiles = build_daily_lvn_profiles(klines)
        gap = self._gap(klines, 8)  # zone 100.3-100.6, day1's heavy-volume zone
        assert not gap_overlaps_lvn(klines, gap, profiles, lvn_threshold=0.30)

    def test_missing_profile_returns_false(self):
        klines = _lvn_klines()
        gap = self._gap(klines, 5)
        assert gap_overlaps_lvn(klines, gap, None, lvn_threshold=0.30) is False
        assert gap_overlaps_lvn(klines, gap, {}, lvn_threshold=0.30) is False


class TestGapsForComboLvnFilter:
    def test_require_lvn_overlap_keeps_only_lvn_gaps(self):
        klines = _lvn_klines()
        profiles = build_daily_lvn_profiles(klines)
        combo = {"min_gap_pct": 0.0, "require_displacement": False,
                  "require_lvn_overlap": True, "lvn_threshold": 0.30}
        gaps = gaps_for_combo(klines, combo, lvn_profiles=profiles)
        idxs = {g["idx"] for g in gaps}
        assert 5 in idxs
        assert 8 not in idxs

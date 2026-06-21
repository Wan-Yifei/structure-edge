"""Unit tests for strategy/smc — market structure, FVG, and confirmation."""

import pandas as pd
import numpy as np
import pytest

from strategy.smc.market_structure import find_swings, detect_bos_choch, determine_trend
from strategy.smc.fvg import (
    detect_fvg, fvg_entry_depth, is_displacement_candle,
    compute_volume_profile, fvg_overlaps_lvn,
    gap_width_pct, gaps_for_combo, build_daily_lvn_profiles, gap_overlaps_lvn,
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

    def test_converging_initial_swings_do_not_misdetect_uptrend(self):
        """Lower high + higher low (converging) must NOT set initial trend to 'up'.

        With the old OR logic the higher low alone would flip trend='up', making
        the subsequent low-break a CHoCH bear instead of BOS bear.
        """
        # H1=98 > H2=97 (lower high), L1=92 < L2=95 (higher low) → converging
        # Price then falls to 89, breaking below L1=92.
        # Expected: BOS bear (trend was down), NOT CHoCH bear.
        closes = [100, 92, 98, 95, 97, 89]
        df = _klines(closes,
                     highs=[c + 0.5 for c in closes],
                     lows=[c  - 0.5 for c in closes])
        sigs = detect_bos_choch(df, lookback=1, filter_choch=False)
        bear_chochs = [s for s in sigs if s["direction"] == "bear" and s["type"] == "CHoCH"]
        assert bear_chochs == [], (
            f"Converging initial structure should not produce bearish CHoCH: {bear_chochs}"
        )

    def test_choch_opposite_colours_never_overlap(self):
        """A CHoCH of one direction must not span across a CHoCH of the other direction.

        Sequence: downtrend → bullish CHoCH (price breaks above swing high) →
        uptrend → bearish CHoCH (price breaks below swing low).
        The bearish CHoCH's from_idx must be AFTER the bullish CHoCH's idx.
        """
        # Hand-crafted price sequence that produces one bull CHoCH then one bear CHoCH:
        #   bars 0-5: downtrend establishing swing highs/lows
        #   bar 6: spike up through the swing high → bullish CHoCH
        #   bars 7-10: new uptrend swing, then retreat
        #   bar 11: spike down through the swing low formed after bar 6 → bearish CHoCH
        closes = [
            100, 98, 102, 96, 101, 94,   # downtrend (bars 0-5)
            105,                           # bar 6 — breaks above swing high → bull CHoCH
            104, 107, 103, 106,            # bars 7-10 — uptrend swings
            101,                           # bar 11 — breaks below post-CHoCH swing low → bear CHoCH
        ]
        df = _klines(closes,
                     highs=[c + 0.5 for c in closes],
                     lows=[c  - 0.5 for c in closes])
        sigs = detect_bos_choch(df, lookback=1, filter_choch=False)
        chochs = [s for s in sigs if s["type"] == "CHoCH"]
        if len(chochs) < 2:
            return  # not enough structure in this synthetic data — skip

        bull_chochs = [s for s in chochs if s["direction"] == "bull"]
        bear_chochs = [s for s in chochs if s["direction"] == "bear"]
        if not bull_chochs or not bear_chochs:
            return

        # For every pair of opposite-colour CHoCH signals, the later one's from_idx
        # must not precede the earlier one's idx (no overlap / nesting).
        for b in bull_chochs:
            for d in bear_chochs:
                earlier, later = (b, d) if b["idx"] < d["idx"] else (d, b)
                assert later["from_idx"] >= earlier["idx"], (
                    f"CHoCH overlap: {earlier['direction']} CHoCH ends at bar "
                    f"{earlier['idx']} but opposite CHoCH starts from bar "
                    f"{later['from_idx']}"
                )


# ── determine_trend ───────────────────────────────────────────────────────────

class TestDetermineTrend:
    def test_none_when_no_signals(self):
        assert determine_trend([]) is None

    def test_choch_alone_confirms(self):
        # CHoCH immediately counts as one confirmation (consecutive=1).
        # With min_consecutive=1 the trend is returned right away.
        sigs = [{"type": "CHoCH", "direction": "bull"}]
        assert determine_trend(sigs, min_consecutive=1) == "bull"

    def test_bull_confirmed_after_choch_plus_bos(self):
        sigs = [
            {"type": "CHoCH", "direction": "bull"},
            {"type": "BOS",   "direction": "bull"},
        ]
        assert determine_trend(sigs, min_consecutive=1) == "bull"

    def test_bear_confirmed_after_choch_plus_bos(self):
        sigs = [
            {"type": "CHoCH", "direction": "bear"},
            {"type": "BOS",   "direction": "bear"},
        ]
        assert determine_trend(sigs, min_consecutive=1) == "bear"

    def test_none_when_below_min_consecutive(self):
        # CHoCH alone → consecutive=1; needs 2 → still None.
        sigs = [{"type": "CHoCH", "direction": "bull"}]
        assert determine_trend(sigs, min_consecutive=2) is None

    def test_consecutive_bos_counted(self):
        # CHoCH(1) + BOS(2) + BOS(3) → consecutive=3 >= 2 → confirmed.
        sigs = [
            {"type": "CHoCH", "direction": "bull"},
            {"type": "BOS",   "direction": "bull"},
            {"type": "BOS",   "direction": "bull"},
        ]
        assert determine_trend(sigs, min_consecutive=2) == "bull"

    def test_choch_resets_direction(self):
        # Bear CHoCH immediately sets trend=bear (consecutive=1); bear BOS
        # increments to 2 → bear trend confirmed.  No reverse BOS veto.
        sigs = [
            {"type": "CHoCH", "direction": "bull"},
            {"type": "BOS",   "direction": "bull"},
            {"type": "CHoCH", "direction": "bear"},
            {"type": "BOS",   "direction": "bear"},
        ]
        assert determine_trend(sigs, min_consecutive=1) == "bear"

    def test_choch_alone_at_end_confirms(self):
        # Bear CHoCH at the end: consecutive=1 >= 1 and no reverse BOS → "bear".
        sigs = [
            {"type": "CHoCH", "direction": "bull"},
            {"type": "BOS",   "direction": "bull"},
            {"type": "CHoCH", "direction": "bear"},
        ]
        assert determine_trend(sigs, min_consecutive=1) == "bear"

    def test_reverse_bos_after_choch_vetoes_trend(self):
        # Classic false setup: CHoCH bear then BOS bull reclaims the level.
        # The reverse BOS must cancel the trend → None.
        sigs = [
            {"type": "CHoCH", "direction": "bear"},
            {"type": "BOS",   "direction": "bull"},
        ]
        assert determine_trend(sigs, min_consecutive=1) is None

    def test_reverse_bos_after_choch_vetoes_even_with_confirming_bos(self):
        # CHoCH bear → BOS bear (confirms) → BOS bull (reverse, veto) → None.
        sigs = [
            {"type": "CHoCH", "direction": "bear"},
            {"type": "BOS",   "direction": "bear"},
            {"type": "BOS",   "direction": "bull"},
        ]
        assert determine_trend(sigs, min_consecutive=1) is None

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
        # require_displacement=False: test pure gap geometry without displacement filter
        gaps = detect_fvg(self._bull_fvg_df(), min_gap_pct=0.0, require_displacement=False)
        bulls = [g for g in gaps if g["direction"] == "bull"]
        assert len(bulls) == 1
        assert bulls[0]["bottom"] == pytest.approx(10.0)
        assert bulls[0]["top"]    == pytest.approx(12.0)

    def test_detects_bear_fvg(self):
        gaps = detect_fvg(self._bear_fvg_df(), min_gap_pct=0.0, require_displacement=False)
        bears = [g for g in gaps if g["direction"] == "bear"]
        assert len(bears) == 1

    def test_filled_flag(self):
        # Add a 4th candle whose close is inside the gap
        df = _klines(
            closes=[10, 11, 12, 11],
            highs=[10, 11, 13, 12],
            lows=[9,  10, 12, 10],
        )
        gaps = detect_fvg(df, min_gap_pct=0.0, require_displacement=False)
        bulls = [g for g in gaps if g["direction"] == "bull"]
        assert bulls[0]["filled"] is True

    def test_min_gap_filter(self):
        gaps = detect_fvg(self._bull_fvg_df(), min_gap_pct=0.99, require_displacement=False)
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


# ── combo-based detection helpers (gap_width_pct / gaps_for_combo / LVN) ───────

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


class TestGapWidthPct:
    def test_known_width(self):
        gap = {"top": 110.0, "bottom": 90.0}
        assert gap_width_pct(gap) == pytest.approx(0.2)

    def test_zero_midpoint_returns_zero(self):
        gap = {"top": 0.0, "bottom": 0.0}
        assert gap_width_pct(gap) == 0.0


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

    def test_require_lvn_overlap_keeps_only_lvn_gaps(self):
        klines = _lvn_klines()
        profiles = build_daily_lvn_profiles(klines)
        combo = {"min_gap_pct": 0.0, "require_displacement": False,
                  "require_lvn_overlap": True, "lvn_threshold": 0.30}
        gaps = gaps_for_combo(klines, combo, lvn_profiles=profiles)
        idxs = {g["idx"] for g in gaps}
        assert 5 in idxs
        assert 8 not in idxs


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
        # B (index 5) has range 10; preceding 5 candles all have range 1 → mean=1
        df = self._make([1, 1, 1, 1, 1, 10, 1], [0.5, 0.5, 0.5, 0.5, 0.5, 0.8, 0.5])
        assert is_displacement_candle(df, fvg_idx=6, atr_mult=1.5, body_ratio_min=0.5,
                                      lookback=5) == True

    def test_fails_when_range_too_small(self):
        # B range 1.2 vs mean of preceding 5 = 1.0 → 1.2 < 1.5 → False
        df = self._make([1, 1, 1, 1, 1, 1.2, 1], [0.5, 0.5, 0.5, 0.5, 0.5, 0.8, 0.5])
        assert is_displacement_candle(df, fvg_idx=6, atr_mult=1.5, body_ratio_min=0.5,
                                      lookback=5) == False

    def test_fails_when_doji(self):
        # Large range but tiny body
        df = self._make([1, 1, 1, 1, 1, 10, 1], [0.5, 0.5, 0.5, 0.5, 0.5, 0.1, 0.5])
        assert is_displacement_candle(df, fvg_idx=6, atr_mult=1.5, body_ratio_min=0.5,
                                      lookback=5) == False

    def test_edge_returns_true(self):
        # fvg_idx=0 → mid < 1 → fallback True
        df = self._make([1, 2], [0.5, 0.5])
        assert is_displacement_candle(df, fvg_idx=0, atr_mult=1.5, body_ratio_min=0.5,
                                      lookback=5) == True


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

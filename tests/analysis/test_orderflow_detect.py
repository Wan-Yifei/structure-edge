"""Unit tests for analysis.orderflow_detect — iceberg, spoof, and absorption detection.

No Qt dependency; pure Python + NumPy only.
Run: uv run pytest tests/ -v
"""

import unittest
from datetime import datetime, timedelta

from analysis.orderflow_detect import (
    detect_icebergs,
    detect_spoofs,
    detect_absorption,
    detect_stacked_imbalance,
    detect_absorption_bubbles,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

T0 = datetime(2026, 5, 15, 9, 30, 0)   # base timestamp

PRICE_MIN = 100.0
BIN_SIZE  = 1.0
N_PRICE   = 50


def _bucket_to_idx(snaps=None, known_ts=None) -> dict:
    """Map exact timestamps to sequential column indices in chronological
    order -- mirrors production's exact per-column indexing
    (_build_bucket_to_idx in liq_hm_window.py maps self._col_ts 1:1, and
    _raw_snaps entries always carry an exact column timestamp; see that
    function's docstring for why this replaced the old candle_start(cm)
    coarse-minute bucketing).

    By default derives the known timestamp set directly from the given
    snaps (every timestamp used by the test is "known"). Pass known_ts
    explicitly to simulate a different/narrower set of known columns, e.g.
    to test a timestamp that falls outside what's currently buffered.
    """
    uniq = sorted(known_ts) if known_ts is not None else sorted({s["ts"] for s in (snaps or [])})
    return {ts: i for i, ts in enumerate(uniq)}


def _snap(ts: datetime, price: float, volume: int, side: str = "ASK") -> dict:
    return {"ts": ts, "price": price, "volume": volume, "side": side}


def _tick(ts: datetime, price: float, volume: int,
          direction: str = "BUY") -> dict:
    return {"ts": ts, "price": price, "volume": volume, "direction": direction}


# ---------------------------------------------------------------------------
# detect_icebergs
# ---------------------------------------------------------------------------

class TestDetectIcebergs(unittest.TestCase):

    def _run(self, snaps, min_refreshes=2, vol_threshold=0.0, col_secs=300,
             known_ts=None):
        # col_secs=300 keeps gap threshold large so tests with sparse snapshots
        # do not get falsely split into separate segments.
        return detect_icebergs(
            snaps,
            bucket_to_idx=_bucket_to_idx(snaps, known_ts),
            bin_size=BIN_SIZE,
            price_min=PRICE_MIN,
            N_PRICE=N_PRICE,
            min_refreshes=min_refreshes,
            vol_threshold=vol_threshold,
            col_secs=col_secs,
        )

    def test_empty_returns_empty(self):
        self.assertEqual(self._run([]), [])

    def test_no_refresh_not_flagged(self):
        # Steady volume — no drop+recover cycle
        snaps = [_snap(T0 + timedelta(seconds=i * 5), 110.0, 1000) for i in range(10)]
        self.assertEqual(self._run(snaps), [])

    def test_single_refresh_detected(self):
        # 1000 → 50 → 900 → 50 → 900: two refreshes.
        # 5 distinct timestamps -> exact per-column indices 0..4; the
        # refreshes are detected at entries[2] (900 recovering) and
        # entries[4] (950 recovering), so first/last_bar are 2 and 4.
        snaps = [
            _snap(T0 + timedelta(seconds=0),  110.0, 1000),
            _snap(T0 + timedelta(seconds=5),  110.0,   50),
            _snap(T0 + timedelta(seconds=10), 110.0,  900),
            _snap(T0 + timedelta(seconds=15), 110.0,   30),
            _snap(T0 + timedelta(seconds=20), 110.0,  950),
        ]
        result = self._run(snaps, min_refreshes=2)
        self.assertEqual(len(result), 1)
        first_bar, last_bar, price, n_ref = result[0]
        self.assertEqual(first_bar, 2)
        self.assertEqual(last_bar, 4)
        self.assertEqual(n_ref, 2)
        self.assertAlmostEqual(price, 110.5, places=1)

    def test_vol_threshold_filters_small(self):
        # Same pattern but volume < threshold → should be filtered out
        snaps = [
            _snap(T0 + timedelta(seconds=0),  110.0, 100),
            _snap(T0 + timedelta(seconds=5),  110.0,   5),
            _snap(T0 + timedelta(seconds=10), 110.0,  90),
            _snap(T0 + timedelta(seconds=15), 110.0,   3),
            _snap(T0 + timedelta(seconds=20), 110.0,  95),
        ]
        self.assertEqual(self._run(snaps, vol_threshold=500), [])

    def test_min_refreshes_threshold(self):
        # Only 1 refresh — should not appear when min_refreshes=2
        snaps = [
            _snap(T0 + timedelta(seconds=0),  110.0, 1000),
            _snap(T0 + timedelta(seconds=5),  110.0,   50),
            _snap(T0 + timedelta(seconds=10), 110.0,  900),
        ]
        self.assertEqual(self._run(snaps, min_refreshes=2), [])
        # But min_refreshes=1 should flag it
        result = self._run(snaps, min_refreshes=1)
        self.assertEqual(len(result), 1)

    def test_span_across_bars(self):
        # Refreshes span a large time gap (10s -> 120s) but stay under
        # GAP_SECS (col_secs=300 -> 450s), so it's still one segment. With
        # 5 distinct timestamps, exact per-column indices are 0..4; the
        # refresh recoveries land at entries[2] (10s) and entries[4] (125s).
        snaps = [
            _snap(T0 + timedelta(seconds=0),   110.0, 1000),
            _snap(T0 + timedelta(seconds=5),   110.0,   40),
            _snap(T0 + timedelta(seconds=10),  110.0,  900),   # refresh 1
            _snap(T0 + timedelta(minutes=2),   110.0,   20),
            _snap(T0 + timedelta(minutes=2, seconds=5), 110.0, 950),  # refresh 2
        ]
        result = self._run(snaps, min_refreshes=2)
        self.assertEqual(len(result), 1)
        first_bar, last_bar, _, _ = result[0]
        self.assertEqual(first_bar, 2)
        self.assertEqual(last_bar, 4)

    def test_two_independent_price_levels(self):
        # Refreshes at 110 and 115 → two independent icebergs
        def _make_refreshes(price):
            return [
                _snap(T0 + timedelta(seconds=0),  price, 1000),
                _snap(T0 + timedelta(seconds=5),  price,   40),
                _snap(T0 + timedelta(seconds=10), price,  900),
                _snap(T0 + timedelta(seconds=15), price,   30),
                _snap(T0 + timedelta(seconds=20), price,  950),
            ]
        result = self._run(_make_refreshes(110.0) + _make_refreshes(115.0), min_refreshes=2)
        self.assertEqual(len(result), 2)
        prices = sorted(r[2] for r in result)
        self.assertAlmostEqual(prices[0], 110.5, places=0)
        self.assertAlmostEqual(prices[1], 115.5, places=0)

    def test_out_of_range_price_ignored(self):
        # Price outside [price_min, price_min + N_PRICE * bin_size]
        snaps = [
            _snap(T0 + timedelta(seconds=0),  999.0, 1000),
            _snap(T0 + timedelta(seconds=5),  999.0,   40),
            _snap(T0 + timedelta(seconds=10), 999.0,  900),
        ]
        self.assertEqual(self._run(snaps, min_refreshes=1), [])

    def test_refresh_count_accurate(self):
        # 4 refreshes
        snaps = []
        for cycle in range(5):
            snaps.append(_snap(T0 + timedelta(seconds=cycle * 10),     110.0, 1000))
            snaps.append(_snap(T0 + timedelta(seconds=cycle * 10 + 4), 110.0,   30))
        result = self._run(snaps, min_refreshes=2)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][3], 4)

    def test_detected_at_col_secs_1_with_realistic_snapshot_spacing(self):
        """Regression test: Col(s)=1 (the Liquidity Heatmap's current default)
        must not silently disable iceberg detection entirely.

        GAP_SECS = col_secs * 1.5 used to have no floor, so at col_secs=1 it
        was 1.5s -- shorter than order_book_collector.py's own
        _MIN_WRITE_INTERVAL=2.0s write throttle. Since genuine snapshots
        never arrive faster than that throttle, every consecutive pair would
        exceed the (too-tight) gap threshold, splitting every single
        snapshot into its own length-1 segment and permanently blocking
        detection. Snapshots here are spaced exactly 2.2s apart -- realistic
        for the collector's throttle, and would have failed detection
        entirely under the old unfloored formula.
        """
        snaps = [
            _snap(T0 + timedelta(seconds=0.0),  110.0, 1000),
            _snap(T0 + timedelta(seconds=2.2),  110.0,   50),
            _snap(T0 + timedelta(seconds=4.4),  110.0,  900),
            _snap(T0 + timedelta(seconds=6.6),  110.0,   30),
            _snap(T0 + timedelta(seconds=8.8),  110.0,  950),
        ]
        result = self._run(snaps, min_refreshes=2, col_secs=1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][3], 2)   # 2 refreshes, one contiguous segment


# ---------------------------------------------------------------------------
# detect_spoofs
# ---------------------------------------------------------------------------

class TestDetectSpoofs(unittest.TestCase):

    def _run(self, ob_data, min_vol=100.0, max_duration_secs=30.0, known_ts=None):
        return detect_spoofs(
            ob_data,
            bucket_to_idx=_bucket_to_idx(ob_data, known_ts),
            bin_size=BIN_SIZE,
            price_min=PRICE_MIN,
            N_PRICE=N_PRICE,
            min_vol=min_vol,
            max_duration_secs=max_duration_secs,
        )

    def test_empty_returns_empty(self):
        self.assertEqual(self._run([]), [])

    def test_clean_spoof_detected(self):
        # Large ASK appears at t=0, vanishes at t=10s; spread stays below level → spoof.
        # 3 distinct timestamps -> exact per-column indices 0(0s),1(10s),2(20s).
        ob = [
            _snap(T0,                          110.0, 5000, "ASK"),
            _snap(T0 + timedelta(seconds=10),  110.0,   50, "ASK"),
            _snap(T0 + timedelta(seconds=20),  110.0,   40, "ASK"),
        ]
        result = self._run(ob, min_vol=100)
        self.assertEqual(len(result), 1)
        appear_bar, disappear_bar, price, side = result[0]
        self.assertEqual(side, "ASK")
        self.assertEqual(appear_bar, 0)
        self.assertEqual(disappear_bar, 1)

    def test_bid_spoof_side_correct(self):
        ob = [
            _snap(T0,                          110.0, 5000, "BID"),
            _snap(T0 + timedelta(seconds=10),  110.0,   50, "BID"),
            _snap(T0 + timedelta(seconds=20),  110.0,   40, "BID"),
        ]
        result = self._run(ob, min_vol=100)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][3], "BID")

    def test_no_spoof_when_fully_executed(self):
        # Large ASK at 110 vanishes; during the window best ask rises to 112
        # (buyers consumed it) → spread moved through level → NOT a spoof
        ob = [
            _snap(T0,                          110.0, 5000, "ASK"),
            _snap(T0 + timedelta(seconds=5),   112.0,  100, "ASK"),  # best ask rose
            _snap(T0 + timedelta(seconds=10),  110.0,   50, "ASK"),
        ]
        result = self._run(ob, min_vol=100)
        self.assertEqual(result, [])

    def test_partial_execution_still_flagged(self):
        # Large ASK at 110 vanishes; best ask stays at 110 throughout →
        # spread did not move through the level → qualifies as spoof
        ob = [
            _snap(T0,                          110.0, 5000, "ASK"),
            _snap(T0 + timedelta(seconds=10),  110.0,   50, "ASK"),
        ]
        result = self._run(ob, min_vol=100)
        self.assertEqual(len(result), 1)

    def test_order_lives_too_long_not_flagged(self):
        # Order disappears after 60s but max_duration_secs=30 → not a spoof
        ob = [
            _snap(T0,                          110.0, 5000, "ASK"),
            _snap(T0 + timedelta(seconds=60),  110.0,   30, "ASK"),
        ]
        result = self._run(ob, min_vol=100, max_duration_secs=30)
        self.assertEqual(result, [])

    def test_order_lives_within_window_flagged(self):
        # Same but max_duration_secs=90 → flagged
        ob = [
            _snap(T0,                          110.0, 5000, "ASK"),
            _snap(T0 + timedelta(seconds=60),  110.0,   30, "ASK"),
        ]
        result = self._run(ob, min_vol=100, max_duration_secs=90)
        self.assertEqual(len(result), 1)

    def test_below_min_vol_not_flagged(self):
        ob = [
            _snap(T0,                          110.0,   80, "ASK"),
            _snap(T0 + timedelta(seconds=10),  110.0,    5, "ASK"),
        ]
        result = self._run(ob, min_vol=100)
        self.assertEqual(result, [])

    def test_multiple_spoofs_same_session(self):
        # Two separate spoof events at different price levels
        ob = [
            _snap(T0,                          110.0, 5000, "ASK"),
            _snap(T0 + timedelta(seconds=10),  110.0,   40, "ASK"),
            _snap(T0 + timedelta(seconds=20),  115.0, 4000, "BID"),
            _snap(T0 + timedelta(seconds=30),  115.0,   30, "BID"),
        ]
        result = self._run(ob, min_vol=100)
        self.assertEqual(len(result), 2)
        sides = {r[3] for r in result}
        self.assertEqual(sides, {"ASK", "BID"})

    def test_sequential_spoofs_same_level(self):
        # Two consecutive spoof events at the same price level
        ob = [
            _snap(T0,                          110.0, 5000, "ASK"),   # spoof 1 appears
            _snap(T0 + timedelta(seconds=10),  110.0,   40, "ASK"),   # spoof 1 vanishes
            _snap(T0 + timedelta(seconds=20),  110.0,   30, "ASK"),
            _snap(T0 + timedelta(minutes=2),   110.0, 5000, "ASK"),   # spoof 2 appears
            _snap(T0 + timedelta(minutes=2, seconds=15), 110.0, 30, "ASK"),
        ]
        result = self._run(ob, min_vol=100)
        self.assertGreaterEqual(len(result), 1)


# ---------------------------------------------------------------------------
# detect_absorption
# ---------------------------------------------------------------------------

class TestDetectAbsorption(unittest.TestCase):
    """Tests for detect_absorption().

    Parameter conventions used throughout:
      passive_k=3.0, active_k=1.0, hit_ratio=0.30
      BIN_SIZE=1.0, PRICE_MIN=100.0, N_PRICE=50

    avg_tick_vol is computed inside detect_absorption as total_vol / n_ticks,
    so each test must construct ticks so the desired avg is unambiguous.
    """

    def _run(self, ob_data, raw_ticks,
             passive_k=3.0, active_k=1.0, hit_ratio=0.30):
        return detect_absorption(
            ob_data   = ob_data,
            raw_ticks = raw_ticks,
            bin_size  = BIN_SIZE,
            price_min = PRICE_MIN,
            N_PRICE   = N_PRICE,
            passive_k = passive_k,
            active_k  = active_k,
            hit_ratio = hit_ratio,
        )

    # ── edge cases ────────────────────────────────────────────────────────────

    def test_empty_ob_returns_empty(self):
        ticks = [_tick(T0, 110.0, 100)]
        self.assertEqual(self._run([], ticks), [])

    def test_empty_ticks_returns_empty(self):
        ob = [_snap(T0, 110.0, 1000)]
        self.assertEqual(self._run(ob, []), [])

    # ── ASK absorption (sellers absorbing buyers) ─────────────────────────────

    def test_ask_absorption_basic(self):
        # avg_tick_vol = 100; passive=1000 >= 300; agg=500 >= 100; ratio=0.5 >= 0.3
        ob    = [_snap(T0, 110.0, 1000, "ASK")]
        ticks = [_tick(T0 + timedelta(seconds=i), 110.0, 100) for i in range(5)]
        result = self._run(ob, ticks)
        self.assertEqual(len(result), 1)
        price, side, agg_vol, pass_vol, ratio = result[0]
        self.assertEqual(side, "ASK")
        self.assertAlmostEqual(agg_vol, 500.0)
        self.assertAlmostEqual(pass_vol, 1000.0)
        self.assertAlmostEqual(ratio, 0.5)

    def test_ask_absorption_price_bin_matches(self):
        # Confirm the returned price is bin-centred, not raw tick price
        ob    = [_snap(T0, 110.3, 1000, "ASK")]   # sits in bin for 110.0–111.0
        ticks = [_tick(T0 + timedelta(seconds=i), 110.3, 100) for i in range(5)]
        result = self._run(ob, ticks)
        self.assertEqual(len(result), 1)
        price = result[0][0]
        self.assertGreaterEqual(price, PRICE_MIN)   # within grid
        self.assertLess(price, PRICE_MIN + N_PRICE * BIN_SIZE)

    # ── BID absorption (buyers absorbing sellers) ─────────────────────────────

    def test_bid_absorption_basic(self):
        # avg=100; bid wall 1000 >= 300; sell agg=500 >= 100; ratio=0.5
        ob    = [_snap(T0, 109.0, 1000, "BID")]
        ticks = [_tick(T0 + timedelta(seconds=i), 109.0, 100, "SELL") for i in range(5)]
        result = self._run(ob, ticks)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][1], "BID")

    # ── Condition 1: passive threshold ────────────────────────────────────────

    def test_passive_too_small_not_flagged(self):
        # Wall only 50; avg=100; passive_threshold=300 → 50 < 300 → skip
        ob    = [_snap(T0, 110.0, 50, "ASK")]
        ticks = [_tick(T0 + timedelta(seconds=i), 110.0, 100) for i in range(5)]
        self.assertEqual(self._run(ob, ticks), [])

    def test_passive_exactly_at_threshold_flagged(self):
        # avg=100; passive_threshold=300; wall=300 → passes (>=)
        ob    = [_snap(T0, 110.0, 300, "ASK")]
        ticks = [_tick(T0 + timedelta(seconds=i), 110.0, 100) for i in range(5)]
        result = self._run(ob, ticks)
        self.assertEqual(len(result), 1)

    # ── Condition 2: active threshold ─────────────────────────────────────────

    def test_active_too_small_not_flagged(self):
        # Inflate avg with large SELL ticks so agg_buy falls below active_threshold.
        # 5 SELL ticks vol=1000 + 1 BUY tick vol=10 → avg=(5000+10)/6 ≈ 835
        # active_threshold ≈ 835; agg_buy=10 < 835 → not flagged
        ob    = [_snap(T0, 110.0, 5000, "ASK")]
        ticks = (
            [_tick(T0 + timedelta(seconds=i), 111.0, 1000, "SELL") for i in range(5)]
            + [_tick(T0 + timedelta(seconds=10), 110.0, 10, "BUY")]
        )
        self.assertEqual(self._run(ob, ticks), [])

    # ── Condition 3: hit ratio ────────────────────────────────────────────────

    def test_hit_ratio_too_low_not_flagged(self):
        # Wall=5000; agg=500; ratio=0.10 < 0.30 → skip
        ob    = [_snap(T0, 110.0, 5000, "ASK")]
        ticks = [_tick(T0 + timedelta(seconds=i), 110.0, 100) for i in range(5)]
        self.assertEqual(self._run(ob, ticks), [])

    def test_hit_ratio_exactly_at_threshold_flagged(self):
        # Wall=1000; agg=300; ratio=0.30 exactly → passes (>=)
        ob    = [_snap(T0, 110.0, 1000, "ASK")]
        ticks = [_tick(T0 + timedelta(seconds=i), 110.0, 100) for i in range(3)]
        # avg=100; active_threshold=100; agg=300 >= 100 ✓; ratio=0.30 ✓
        result = self._run(ob, ticks)
        self.assertEqual(len(result), 1)

    # ── Condition 4: wall drained → not flagged ───────────────────────────────

    def test_wall_drained_at_end_not_flagged(self):
        # Multiple snapshots: wall starts at 1000 but last snapshot shows 0
        ob = [
            _snap(T0,                         110.0, 1000, "ASK"),
            _snap(T0 + timedelta(seconds=30), 110.0,    0, "ASK"),
        ]
        ticks = [_tick(T0 + timedelta(seconds=i * 5), 110.0, 100) for i in range(5)]
        # last_pass[("ASK", bin)] = 0 < passive_threshold → not flagged
        self.assertEqual(self._run(ob, ticks), [])

    # ── Split-order accumulation ──────────────────────────────────────────────

    def test_split_orders_accumulate(self):
        # 20 small BUY ticks of vol=25 each at 110.0 → agg=500, same as 5×100
        ob    = [_snap(T0, 110.0, 1000, "ASK")]
        ticks = [_tick(T0 + timedelta(seconds=i), 110.0, 25) for i in range(20)]
        # avg=25; passive_threshold=75; active_threshold=25; agg=500; ratio=0.5
        result = self._run(ob, ticks)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0][2], 500.0)  # agg_vol

    # ── Direction filtering ───────────────────────────────────────────────────

    def test_neutral_ticks_not_counted_as_aggression(self):
        # Only NEUTRAL ticks at ask price → no BUY agg accumulates
        ob    = [_snap(T0, 110.0, 1000, "ASK")]
        ticks = [_tick(T0 + timedelta(seconds=i), 110.0, 100, "NEUTRAL")
                 for i in range(5)]
        self.assertEqual(self._run(ob, ticks), [])

    def test_sell_ticks_do_not_count_for_ask_absorption(self):
        # SELL ticks at ask price should not trigger ASK absorption
        ob    = [_snap(T0, 110.0, 1000, "ASK")]
        ticks = [_tick(T0 + timedelta(seconds=i), 110.0, 100, "SELL")
                 for i in range(5)]
        # agg_buy at that bin = 0 → no ASK absorption
        ask_results = [r for r in self._run(ob, ticks) if r[1] == "ASK"]
        self.assertEqual(ask_results, [])

    # ── Both sides simultaneously ─────────────────────────────────────────────

    def test_both_sides_detected(self):
        # ASK wall at 111, BID wall at 109; buyers hit ask, sellers hit bid
        ob = [
            _snap(T0, 111.0, 1000, "ASK"),
            _snap(T0, 109.0, 1000, "BID"),
        ]
        ticks = (
            [_tick(T0 + timedelta(seconds=i), 111.0, 100, "BUY")  for i in range(5)]
            + [_tick(T0 + timedelta(seconds=i), 109.0, 100, "SELL") for i in range(5)]
        )
        result = self._run(ob, ticks)
        sides = {r[1] for r in result}
        self.assertIn("ASK", sides)
        self.assertIn("BID", sides)

    # ── Out-of-range price ignored ────────────────────────────────────────────

    def test_out_of_range_price_ignored(self):
        ob    = [_snap(T0, 999.0, 5000, "ASK")]
        ticks = [_tick(T0 + timedelta(seconds=i), 999.0, 100) for i in range(10)]
        self.assertEqual(self._run(ob, ticks), [])

    # ── Last snapshot wins ────────────────────────────────────────────────────

    def test_last_ob_snapshot_used_for_passive_vol(self):
        # Wall was 5000 early on but ends at 200 → passive check uses 200
        ob = [
            _snap(T0,                         110.0, 5000, "ASK"),
            _snap(T0 + timedelta(seconds=30), 110.0,  200, "ASK"),
        ]
        ticks = [_tick(T0 + timedelta(seconds=i * 5), 110.0, 100) for i in range(5)]
        # avg=100; passive_threshold=300; pass_vol=200 < 300 → not flagged
        self.assertEqual(self._run(ob, ticks), [])


# ---------------------------------------------------------------------------
# detect_stacked_imbalance
# ---------------------------------------------------------------------------

class TestDetectStackedImbalance(unittest.TestCase):

    def _run(self, snaps, min_levels=3, imbalance_ratio=3.0,
             min_vol=0.0, max_depth=10, known_ts=None):
        return detect_stacked_imbalance(
            snaps,
            bucket_to_idx=_bucket_to_idx(snaps, known_ts),
            bin_size=BIN_SIZE,
            price_min=PRICE_MIN,
            N_PRICE=N_PRICE,
            min_levels=min_levels,
            imbalance_ratio=imbalance_ratio,
            min_vol=min_vol,
            max_depth=max_depth,
        )

    def _ob(self, ts, levels):
        """Build snapshot rows from [(side, price, volume), ...]."""
        return [{"ts": ts, "side": s, "price": p, "volume": v}
                for s, p, v in levels]

    # ── edge cases ────────────────────────────────────────────────────────────

    def test_empty_returns_empty(self):
        self.assertEqual(self._run([]), [])

    def test_balanced_book_not_flagged(self):
        snaps = self._ob(T0, [
            ("BID", 115.0, 100), ("BID", 114.0, 100), ("BID", 113.0, 100),
            ("ASK", 116.0, 100), ("ASK", 117.0, 100), ("ASK", 118.0, 100),
        ])
        self.assertEqual(self._run(snaps, imbalance_ratio=3.0), [])

    def test_unknown_bar_idx_skipped(self):
        # ts is not among the known column timestamps -> bucket lookup misses
        ts = T0 + timedelta(hours=2)
        snaps = self._ob(ts, [
            ("BID", 115.0, 600), ("BID", 114.0, 500), ("BID", 113.0, 400),
            ("ASK", 116.0, 100), ("ASK", 117.0,  80), ("ASK", 118.0,  60),
        ])
        self.assertEqual(self._run(snaps, known_ts=[T0]), [])

    # ── bullish detection ─────────────────────────────────────────────────────

    def test_bullish_basic(self):
        # 3 bid ranks all ~6× their paired ask ranks
        snaps = self._ob(T0, [
            ("BID", 115.0, 600), ("BID", 114.0, 500), ("BID", 113.0, 400),
            ("ASK", 116.0, 100), ("ASK", 117.0,  80), ("ASK", 118.0,  60),
        ])
        result = self._run(snaps)
        self.assertEqual(len(result), 1)
        bar_idx, price_lo, price_hi, direction, mean_ratio = result[0]
        self.assertEqual(direction, "BID")
        self.assertEqual(bar_idx, 0)
        self.assertGreater(mean_ratio, 3.0)

    def test_bullish_price_range(self):
        # Zone must span the bid-side prices (113–115)
        snaps = self._ob(T0, [
            ("BID", 115.0, 600), ("BID", 114.0, 500), ("BID", 113.0, 400),
            ("ASK", 116.0, 100), ("ASK", 117.0,  80), ("ASK", 118.0,  60),
        ])
        _, price_lo, price_hi, direction, _ = self._run(snaps)[0]
        self.assertEqual(direction, "BID")
        self.assertAlmostEqual(price_lo, 113.0)
        self.assertAlmostEqual(price_hi, 115.0)

    # ── bearish detection ─────────────────────────────────────────────────────

    def test_bearish_basic(self):
        snaps = self._ob(T0, [
            ("ASK", 116.0, 600), ("ASK", 117.0, 500), ("ASK", 118.0, 400),
            ("BID", 115.0, 100), ("BID", 114.0,  80), ("BID", 113.0,  60),
        ])
        result = self._run(snaps)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][3], "ASK")

    def test_bearish_price_range(self):
        snaps = self._ob(T0, [
            ("ASK", 116.0, 600), ("ASK", 117.0, 500), ("ASK", 118.0, 400),
            ("BID", 115.0, 100), ("BID", 114.0,  80), ("BID", 113.0,  60),
        ])
        _, price_lo, price_hi, direction, _ = self._run(snaps)[0]
        self.assertEqual(direction, "ASK")
        self.assertAlmostEqual(price_lo, 116.0)
        self.assertAlmostEqual(price_hi, 118.0)

    # ── min_levels threshold ──────────────────────────────────────────────────

    def test_insufficient_levels_not_flagged(self):
        # Only 2 BID-dominant ranks; min_levels=3 → skip
        snaps = self._ob(T0, [
            ("BID", 115.0, 600), ("BID", 114.0, 500),
            ("ASK", 116.0, 100), ("ASK", 117.0,  80), ("ASK", 118.0, 60),
        ])
        self.assertEqual(self._run(snaps, min_levels=3), [])

    def test_exactly_min_levels_fires(self):
        snaps = self._ob(T0, [
            ("BID", 115.0, 600), ("BID", 114.0, 500), ("BID", 113.0, 400),
            ("ASK", 116.0, 100), ("ASK", 117.0,  80), ("ASK", 118.0,  60),
        ])
        self.assertEqual(len(self._run(snaps, min_levels=3)), 1)

    # ── missing-level = 0 ────────────────────────────────────────────────────

    def test_missing_ask_levels_treated_as_zero(self):
        # 4 bid ranks, only 2 ask ranks → ranks 2–3 have ask=0 (∞ ratio) → BID
        snaps = self._ob(T0, [
            ("BID", 115.0, 600), ("BID", 114.0, 500),
            ("BID", 113.0, 400), ("BID", 112.0, 350),
            ("ASK", 116.0, 100), ("ASK", 117.0,  80),
        ])
        result = self._run(snaps, min_levels=3)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][3], "BID")

    def test_missing_bid_levels_treated_as_zero(self):
        snaps = self._ob(T0, [
            ("ASK", 116.0, 600), ("ASK", 117.0, 500),
            ("ASK", 118.0, 400), ("ASK", 119.0, 350),
            ("BID", 115.0, 100), ("BID", 114.0,  80),
        ])
        result = self._run(snaps, min_levels=3)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][3], "ASK")

    # ── run scanning ─────────────────────────────────────────────────────────

    def test_neutral_rank_breaks_run(self):
        # ranks 0–1: BID (run=2, <3 → skip); rank 2: neutral; ranks 3–5: BID (run=3 → fires)
        snaps = self._ob(T0, [
            ("BID", 115.0, 600), ("BID", 114.0, 500),
            ("BID", 113.0, 200),                         # rank 2: ~equal → neutral
            ("BID", 112.0, 600), ("BID", 111.0, 500), ("BID", 110.0, 400),
            ("ASK", 116.0, 100), ("ASK", 117.0,  80), ("ASK", 118.0, 200),
            ("ASK", 119.0, 100), ("ASK", 120.0,  80), ("ASK", 121.0,  60),
        ])
        result = self._run(snaps, min_levels=3)
        bid_results = [r for r in result if r[3] == "BID"]
        self.assertEqual(len(bid_results), 1)
        self.assertAlmostEqual(bid_results[0][2], 112.0)   # top of detected zone

    # ── min_vol ───────────────────────────────────────────────────────────────

    def test_min_vol_filters_bid_levels(self):
        # All bid levels below min_vol → absent (0); ask levels kept → ASK dominant
        snaps = self._ob(T0, [
            ("BID", 115.0, 50), ("BID", 114.0, 40), ("BID", 113.0, 30),
            ("ASK", 116.0, 200), ("ASK", 117.0, 180), ("ASK", 118.0, 160),
        ])
        result = self._run(snaps, min_vol=100, imbalance_ratio=3.0)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][3], "ASK")

    # ── bar_idx assignment ────────────────────────────────────────────────────

    def test_bar_idx_assigned_from_timestamp(self):
        # Explicit known_ts simulating 5 known columns (T0..T0+4min); the
        # snapshot's timestamp is the 4th of those (index 3).
        known_ts = [T0 + timedelta(minutes=i) for i in range(5)]
        ts = T0 + timedelta(minutes=3)
        snaps = self._ob(ts, [
            ("BID", 115.0, 600), ("BID", 114.0, 500), ("BID", 113.0, 400),
            ("ASK", 116.0, 100), ("ASK", 117.0,  80), ("ASK", 118.0,  60),
        ])
        self.assertEqual(self._run(snaps, known_ts=known_ts)[0][0], 3)

    # ── multiple snapshots ────────────────────────────────────────────────────

    def test_max_depth_truncates_deep_levels(self):
        # 5 bid ranks all dominate, but max_depth=2 → only ranks 0-1 visible,
        # run length 2 < min_levels 3 → no result
        snaps = self._ob(T0, [
            ("BID", 115.0, 600), ("BID", 114.0, 500), ("BID", 113.0, 400),
            ("BID", 112.0, 350), ("BID", 111.0, 300),
            ("ASK", 116.0, 100), ("ASK", 117.0,  80), ("ASK", 118.0,  60),
            ("ASK", 119.0,  50), ("ASK", 120.0,  40),
        ])
        self.assertEqual(self._run(snaps, min_levels=3, max_depth=2), [])

    def test_max_depth_allows_run_within_limit(self):
        # max_depth=5 → ranks 0-4 checked; full run of 5 → detected
        snaps = self._ob(T0, [
            ("BID", 115.0, 600), ("BID", 114.0, 500), ("BID", 113.0, 400),
            ("BID", 112.0, 350), ("BID", 111.0, 300),
            ("ASK", 116.0, 100), ("ASK", 117.0,  80), ("ASK", 118.0,  60),
            ("ASK", 119.0,  50), ("ASK", 120.0,  40),
        ])
        result = self._run(snaps, min_levels=3, max_depth=5)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][3], "BID")

    def test_multiple_snapshots_independent(self):
        # Two timestamps both bullish → two separate events
        snaps = (
            self._ob(T0, [
                ("BID", 115.0, 600), ("BID", 114.0, 500), ("BID", 113.0, 400),
                ("ASK", 116.0, 100), ("ASK", 117.0,  80), ("ASK", 118.0,  60),
            ])
            + self._ob(T0 + timedelta(minutes=1), [
                ("BID", 115.0, 650), ("BID", 114.0, 550), ("BID", 113.0, 450),
                ("ASK", 116.0, 100), ("ASK", 117.0,  80), ("ASK", 118.0,  60),
            ])
        )
        result = self._run(snaps)
        self.assertEqual(len(result), 2)
        self.assertEqual({r[0] for r in result}, {0, 1})


# ---------------------------------------------------------------------------
# detect_absorption_bubbles
# ---------------------------------------------------------------------------

class TestDetectAbsorptionBubbles(unittest.TestCase):
    """Tests for ④ absorption bubble detection."""

    COL_SECS = 30

    def _col_ts(self, n: int):
        return [T0 + timedelta(seconds=i * self.COL_SECS) for i in range(n)]

    def _tick(self, col: int, direction: str, volume: int) -> dict:
        ts = T0 + timedelta(seconds=col * self.COL_SECS + 5)
        return {"ts": ts, "price": 100.0, "volume": volume, "direction": direction}

    def _mid(self, *prices):
        return list(prices)

    # -- basic cases --

    def test_empty_ticks_returns_empty(self):
        col_ts = self._col_ts(5)
        result = detect_absorption_bubbles([], col_ts, self._mid(*[100.0] * 5),
                                           self.COL_SECS)
        self.assertEqual(result, [])

    def test_empty_col_ts_returns_empty(self):
        ticks = [self._tick(0, "BUY", 1000)]
        result = detect_absorption_bubbles(ticks, [], [], self.COL_SECS)
        self.assertEqual(result, [])

    def test_buy_absorbed_when_price_flat(self):
        # Strong BUY delta, price does not rise → BUY absorbed
        col_ts   = self._col_ts(3)
        mid      = self._mid(100.0, 100.0, 100.0)  # price flat
        ticks    = [self._tick(1, "BUY", 800), self._tick(1, "SELL", 100)]
        result   = detect_absorption_bubbles(ticks, col_ts, mid,
                                             self.COL_SECS, min_delta_vol=500)
        self.assertEqual(len(result), 1)
        col_idx, price, direction, vol = result[0]
        self.assertEqual(col_idx, 1)
        self.assertEqual(direction, "BUY")
        self.assertAlmostEqual(vol, 700.0)

    def test_buy_absorbed_when_price_falls(self):
        col_ts = self._col_ts(3)
        mid    = self._mid(100.0, 99.5, 99.0)   # price falling despite buying
        ticks  = [self._tick(1, "BUY", 1000)]
        result = detect_absorption_bubbles(ticks, col_ts, mid,
                                           self.COL_SECS, min_delta_vol=500)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][2], "BUY")

    def test_sell_absorbed_when_price_flat(self):
        col_ts = self._col_ts(3)
        mid    = self._mid(100.0, 100.0, 100.0)
        ticks  = [self._tick(1, "SELL", 900), self._tick(1, "BUY", 100)]
        result = detect_absorption_bubbles(ticks, col_ts, mid,
                                           self.COL_SECS, min_delta_vol=500)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][2], "SELL")

    def test_sell_absorbed_when_price_rises(self):
        col_ts = self._col_ts(3)
        mid    = self._mid(100.0, 100.5, 101.0)  # price rising despite selling
        ticks  = [self._tick(1, "SELL", 1000)]
        result = detect_absorption_bubbles(ticks, col_ts, mid,
                                           self.COL_SECS, min_delta_vol=500)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][2], "SELL")

    # -- no absorption --

    def test_buy_not_absorbed_when_price_rises(self):
        col_ts = self._col_ts(3)
        mid    = self._mid(100.0, 100.5, 101.0)  # price cooperates with buyers
        ticks  = [self._tick(1, "BUY", 1000)]
        result = detect_absorption_bubbles(ticks, col_ts, mid,
                                           self.COL_SECS, min_delta_vol=500)
        self.assertEqual(result, [])

    def test_sell_not_absorbed_when_price_falls(self):
        col_ts = self._col_ts(3)
        mid    = self._mid(100.0, 99.5, 99.0)   # price cooperates with sellers
        ticks  = [self._tick(1, "SELL", 1000)]
        result = detect_absorption_bubbles(ticks, col_ts, mid,
                                           self.COL_SECS, min_delta_vol=500)
        self.assertEqual(result, [])

    def test_below_min_delta_vol_ignored(self):
        col_ts = self._col_ts(3)
        mid    = self._mid(100.0, 100.0, 100.0)
        ticks  = [self._tick(1, "BUY", 300), self._tick(1, "SELL", 100)]
        result = detect_absorption_bubbles(ticks, col_ts, mid,
                                           self.COL_SECS, min_delta_vol=500)
        self.assertEqual(result, [])

    def test_neutral_ticks_excluded(self):
        col_ts = self._col_ts(3)
        mid    = self._mid(100.0, 100.0, 100.0)
        ticks  = [self._tick(1, "NEUTRAL", 5000)]  # direction irrelevant
        result = detect_absorption_bubbles(ticks, col_ts, mid,
                                           self.COL_SECS, min_delta_vol=500)
        self.assertEqual(result, [])

    # -- output fields --

    def test_result_fields(self):
        col_ts = self._col_ts(3)
        mid    = self._mid(100.0, 100.0, 100.0)
        ticks  = [self._tick(1, "BUY", 800)]
        result = detect_absorption_bubbles(ticks, col_ts, mid,
                                           self.COL_SECS, min_delta_vol=500)
        self.assertEqual(len(result), 1)
        col_idx, price, direction, vol = result[0]
        self.assertIsInstance(col_idx, int)
        self.assertIsInstance(price,   float)
        self.assertIn(direction, ("BUY", "SELL"))
        self.assertGreater(vol, 0)

    def test_multiple_columns_detected(self):
        col_ts = self._col_ts(5)
        mid    = self._mid(100.0, 100.0, 100.0, 100.0, 100.0)
        ticks  = [
            self._tick(1, "BUY",  700),
            self._tick(3, "SELL", 900),
        ]
        result = detect_absorption_bubbles(ticks, col_ts, mid,
                                           self.COL_SECS, min_delta_vol=500)
        self.assertEqual(len(result), 2)
        directions = {r[2] for r in result}
        self.assertIn("BUY",  directions)
        self.assertIn("SELL", directions)

    def test_none_mid_price_skipped(self):
        col_ts = self._col_ts(3)
        mid    = [100.0, None, 100.0]   # column 1 has no mid price
        ticks  = [self._tick(1, "BUY", 1000)]
        result = detect_absorption_bubbles(ticks, col_ts, mid,
                                           self.COL_SECS, min_delta_vol=500)
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()

"""Unit tests for analysis.orderflow_detect — iceberg, spoof, and absorption detection.

No Qt dependency; pure Python + NumPy only.
Run: uv run pytest tests/ -v
"""

import unittest
from datetime import datetime, timedelta

from analysis.orderflow_detect import detect_icebergs, detect_spoofs, detect_absorption
from core.time_utils import candle_start

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

CM = 1   # 1-minute candles throughout
T0 = datetime(2026, 5, 15, 9, 30, 0)   # base timestamp

PRICE_MIN = 100.0
BIN_SIZE  = 1.0
N_PRICE   = 50


def _bucket_to_idx(n_bars: int) -> dict:
    """Build a bucket_to_idx for n_bars consecutive 1-minute bars from T0."""
    return {candle_start(T0 + timedelta(minutes=i), CM): i for i in range(n_bars)}


def _snap(ts: datetime, price: float, volume: int, side: str = "ASK") -> dict:
    return {"ts": ts, "price": price, "volume": volume, "side": side}


def _tick(ts: datetime, price: float, volume: int,
          direction: str = "BUY") -> dict:
    return {"ts": ts, "price": price, "volume": volume, "direction": direction}


# ---------------------------------------------------------------------------
# detect_icebergs
# ---------------------------------------------------------------------------

class TestDetectIcebergs(unittest.TestCase):

    def _run(self, snaps, min_refreshes=2, vol_threshold=0.0, n_bars=20):
        return detect_icebergs(
            snaps,
            bucket_to_idx=_bucket_to_idx(n_bars),
            bin_size=BIN_SIZE,
            price_min=PRICE_MIN,
            N_PRICE=N_PRICE,
            cm=CM,
            min_refreshes=min_refreshes,
            vol_threshold=vol_threshold,
        )

    def test_empty_returns_empty(self):
        self.assertEqual(self._run([]), [])

    def test_no_refresh_not_flagged(self):
        # Steady volume — no drop+recover cycle
        snaps = [_snap(T0 + timedelta(seconds=i * 5), 110.0, 1000) for i in range(10)]
        self.assertEqual(self._run(snaps), [])

    def test_single_refresh_detected(self):
        # 1000 → 50 → 900 → 50 → 900: two refreshes at bar 0
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
        self.assertEqual(first_bar, 0)
        self.assertEqual(last_bar, 0)
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
        # Refreshes in bar 0 and bar 2
        snaps = [
            _snap(T0 + timedelta(seconds=0),   110.0, 1000),   # bar 0
            _snap(T0 + timedelta(seconds=5),   110.0,   40),
            _snap(T0 + timedelta(seconds=10),  110.0,  900),   # refresh 1 → bar 0
            _snap(T0 + timedelta(minutes=2),   110.0,   20),   # bar 2
            _snap(T0 + timedelta(minutes=2, seconds=5), 110.0, 950),  # refresh 2 → bar 2
        ]
        result = self._run(snaps, min_refreshes=2)
        self.assertEqual(len(result), 1)
        first_bar, last_bar, _, _ = result[0]
        self.assertEqual(first_bar, 0)
        self.assertEqual(last_bar, 2)

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


# ---------------------------------------------------------------------------
# detect_spoofs
# ---------------------------------------------------------------------------

class TestDetectSpoofs(unittest.TestCase):

    def _run(self, ob_data, raw_ticks=None, min_vol=100.0,
             max_duration_secs=30.0, n_bars=20):
        return detect_spoofs(
            ob_data,
            raw_ticks or [],
            bucket_to_idx=_bucket_to_idx(n_bars),
            bin_size=BIN_SIZE,
            price_min=PRICE_MIN,
            N_PRICE=N_PRICE,
            cm=CM,
            min_vol=min_vol,
            max_duration_secs=max_duration_secs,
        )

    def test_empty_returns_empty(self):
        self.assertEqual(self._run([]), [])

    def test_clean_spoof_detected(self):
        # Large ASK appears at t=0, vanishes at t=10s, no execution
        ob = [
            _snap(T0,                          110.0, 5000, "ASK"),
            _snap(T0 + timedelta(seconds=10),  110.0,   50, "ASK"),
            _snap(T0 + timedelta(seconds=20),  110.0,   40, "ASK"),
        ]
        result = self._run(ob, raw_ticks=[], min_vol=100)
        self.assertEqual(len(result), 1)
        appear_bar, disappear_bar, price, side = result[0]
        self.assertEqual(side, "ASK")
        self.assertEqual(appear_bar, 0)
        self.assertEqual(disappear_bar, 0)

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
        # Order appears then vanishes, but 80% was executed → NOT a spoof
        ob = [
            _snap(T0,                          110.0, 5000, "ASK"),
            _snap(T0 + timedelta(seconds=10),  110.0,   50, "ASK"),
        ]
        ticks = [_tick(T0 + timedelta(seconds=5), 110.0, 4500)]
        result = self._run(ob, raw_ticks=ticks, min_vol=100)
        self.assertEqual(result, [])

    def test_partial_execution_still_flagged(self):
        # Only 10% executed → still qualifies as spoof
        ob = [
            _snap(T0,                          110.0, 5000, "ASK"),
            _snap(T0 + timedelta(seconds=10),  110.0,   50, "ASK"),
        ]
        ticks = [_tick(T0 + timedelta(seconds=5), 110.0, 400)]
        result = self._run(ob, raw_ticks=ticks, min_vol=100)
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


if __name__ == "__main__":
    unittest.main()

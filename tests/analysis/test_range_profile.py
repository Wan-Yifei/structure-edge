"""Unit tests for range volume profile helper functions.

Covers _compute_profile_bins and _compute_poc_vah_val from trade_viewer_qt.
No Qt dependency — runs headless.

Run: uv run pytest tests/analysis/test_range_profile.py -v
"""

import unittest
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from analysis.trade_viewer_qt import _compute_profile_bins, _compute_poc_vah_val
from core.time_utils import candle_start

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CM = 5  # 5-minute candles
T0 = datetime(2026, 1, 15, 9, 30)  # base bar-end timestamp


def _klines(*rows) -> pd.DataFrame:
    """Build a minimal klines DataFrame.  Each row: (time_key, open, high, low, close, volume)."""
    return pd.DataFrame(
        rows, columns=["time_key", "open", "high", "low", "close", "volume"]
    )


def _bar(i: int, lo: float, hi: float, vol: float) -> tuple:
    """One kline row with bar-end time T0 + i*CM minutes."""
    ts = (T0 + timedelta(minutes=i * CM)).strftime("%Y-%m-%d %H:%M")
    mid = (lo + hi) / 2
    return (ts, mid, hi, lo, mid, vol)


def _ticks_for_bar(bar_end_str: str, price_vol: dict[float, int]) -> tuple[datetime, dict]:
    """Build a ticks bucket key + data for _compute_profile_bins.

    bar_end_str: time_key string of the bar (e.g. '2026-01-15 09:35').
    price_vol: {price: total_volume} — stored as {'buy': v} for simplicity.
    """
    bar_end = datetime.strptime(bar_end_str[:16], "%Y-%m-%d %H:%M")
    bk = candle_start(bar_end - timedelta(minutes=CM), CM)
    data = {p: {"buy": v, "sell": 0, "neutral": 0} for p, v in price_vol.items()}
    return bk, data


# ---------------------------------------------------------------------------
# _compute_profile_bins — OHLCV path
# ---------------------------------------------------------------------------

class TestComputeProfileBinsOHLCV(unittest.TestCase):

    def _run(self, kl, i0=0, i1=None, n_bins=60):
        if i1 is None:
            i1 = len(kl) - 1
        return _compute_profile_bins(kl, None, CM, i0, i1, n_bins=n_bins)

    def test_single_candle_returns_60_bins(self):
        kl = _klines(_bar(0, 100.0, 110.0, 1000.0))
        centers, volumes, used_ticks = self._run(kl, n_bins=60)
        self.assertEqual(len(centers), 60)
        self.assertEqual(len(volumes), 60)
        self.assertFalse(used_ticks)

    def test_single_candle_volume_conserved(self):
        kl = _klines(_bar(0, 100.0, 110.0, 1000.0))
        _, volumes, _ = self._run(kl)
        self.assertAlmostEqual(float(volumes.sum()), 1000.0, places=6)

    def test_single_candle_all_volume_inside_range(self):
        kl = _klines(_bar(0, 100.0, 110.0, 500.0))
        centers, volumes, _ = self._run(kl)
        outside = volumes[(centers < 100.0) | (centers > 110.0)].sum()
        self.assertAlmostEqual(outside, 0.0, places=6)

    def test_multiple_candles_volume_conserved(self):
        kl = _klines(
            _bar(0, 100.0, 105.0, 300.0),
            _bar(1, 103.0, 108.0, 200.0),
            _bar(2, 106.0, 112.0, 500.0),
        )
        _, volumes, _ = self._run(kl)
        self.assertAlmostEqual(float(volumes.sum()), 1000.0, places=5)

    def test_price_range_derived_from_klines(self):
        kl = _klines(_bar(0, 50.0, 60.0, 100.0))
        centers, _, _ = self._run(kl)
        self.assertGreaterEqual(float(centers.min()), 50.0)
        self.assertLessEqual(float(centers.max()), 60.0)

    def test_degenerate_hi_eq_lo_returns_empty(self):
        kl = pd.DataFrame(
            [("2026-01-15 09:35", 100.0, 100.0, 100.0, 100.0, 0.0)],
            columns=["time_key", "open", "high", "low", "close", "volume"],
        )
        centers, volumes, used_ticks = _compute_profile_bins(kl, None, CM, 0, 0)
        self.assertEqual(len(centers), 0)
        self.assertEqual(len(volumes), 0)
        self.assertFalse(used_ticks)

    def test_index_subset_uses_only_selected_bars(self):
        kl = _klines(
            _bar(0, 100.0, 101.0, 100.0),  # bar 0 — should be excluded
            _bar(1, 200.0, 210.0, 500.0),  # bar 1
            _bar(2, 205.0, 215.0, 500.0),  # bar 2
        )
        centers, volumes, _ = self._run(kl, i0=1, i1=2)
        # No volume should land at price ~100 (bar 0 excluded)
        below_150 = volumes[centers < 150.0].sum()
        self.assertAlmostEqual(below_150, 0.0, places=6)

    def test_custom_n_bins(self):
        kl = _klines(_bar(0, 100.0, 110.0, 1000.0))
        centers, volumes, _ = self._run(kl, n_bins=20)
        self.assertEqual(len(centers), 20)
        self.assertEqual(len(volumes), 20)

    def test_zero_volume_candle_contributes_nothing(self):
        kl = _klines(
            _bar(0, 100.0, 110.0, 0.0),
            _bar(1, 105.0, 115.0, 200.0),
        )
        _, volumes, _ = self._run(kl)
        self.assertAlmostEqual(float(volumes.sum()), 200.0, places=5)

    def test_nan_volume_treated_as_zero(self):
        kl = _klines(_bar(0, 100.0, 110.0, float("nan")))
        _, volumes, _ = self._run(kl)
        self.assertAlmostEqual(float(volumes.sum()), 0.0, places=6)

    def test_used_ticks_false_when_no_ticks(self):
        kl = _klines(_bar(0, 100.0, 110.0, 500.0))
        _, _, used_ticks = self._run(kl)
        self.assertFalse(used_ticks)

    def test_high_volume_candle_dominates_its_range(self):
        kl = _klines(
            _bar(0, 100.0, 102.0, 10.0),   # low vol, narrow range
            _bar(1, 108.0, 110.0, 10000.0), # high vol
        )
        centers, volumes, _ = self._run(kl)
        poc_price = float(centers[int(np.argmax(volumes))])
        self.assertGreater(poc_price, 105.0)  # POC in high-vol candle range


# ---------------------------------------------------------------------------
# _compute_profile_bins — tick data path
# ---------------------------------------------------------------------------

class TestComputeProfileBinsTicks(unittest.TestCase):

    def _make_klines_and_ticks(self, bar_defs):
        """bar_defs: list of (lo, hi, vol, {price: tick_vol})."""
        rows = []
        ticks = {}
        for i, (lo, hi, vol, pv) in enumerate(bar_defs):
            rows.append(_bar(i, lo, hi, vol))
            ts_str = (T0 + timedelta(minutes=i * CM)).strftime("%Y-%m-%d %H:%M")
            bk, data = _ticks_for_bar(ts_str, pv)
            ticks[bk] = data
        return _klines(*rows), ticks

    def test_used_ticks_true_when_tick_data_present(self):
        kl, ticks = self._make_klines_and_ticks([
            (100.0, 110.0, 500.0, {105.0: 500}),
        ])
        _, _, used_ticks = _compute_profile_bins(kl, ticks, CM, 0, 0)
        self.assertTrue(used_ticks)

    def test_single_price_level_lands_in_correct_bin(self):
        kl, ticks = self._make_klines_and_ticks([
            (100.0, 110.0, 0.0, {105.0: 200}),
        ])
        centers, volumes, used_ticks = _compute_profile_bins(kl, ticks, CM, 0, 0)
        self.assertTrue(used_ticks)
        poc_price = float(centers[int(np.argmax(volumes))])
        # POC bin center should be near 105
        self.assertAlmostEqual(poc_price, 105.0, delta=0.5)

    def test_tick_volume_totals_match(self):
        kl, ticks = self._make_klines_and_ticks([
            (100.0, 110.0, 0.0, {102.0: 100, 105.0: 200, 108.0: 300}),
        ])
        _, volumes, used_ticks = _compute_profile_bins(kl, ticks, CM, 0, 0)
        self.assertTrue(used_ticks)
        self.assertAlmostEqual(float(volumes.sum()), 600.0, places=5)

    def test_tick_prices_outside_kline_range_ignored(self):
        kl, ticks = self._make_klines_and_ticks([
            (100.0, 110.0, 0.0, {99.0: 999, 105.0: 50, 111.0: 888}),
        ])
        _, volumes, _ = _compute_profile_bins(kl, ticks, CM, 0, 0)
        # Only the 105.0 tick (within 100–110) should land in bins
        self.assertAlmostEqual(float(volumes.sum()), 50.0, places=5)

    def test_multi_bar_tick_aggregation(self):
        kl, ticks = self._make_klines_and_ticks([
            (100.0, 110.0, 0.0, {105.0: 100}),
            (100.0, 110.0, 0.0, {105.0: 200}),
            (100.0, 110.0, 0.0, {105.0: 300}),
        ])
        _, volumes, used_ticks = _compute_profile_bins(kl, ticks, CM, 0, 2)
        self.assertTrue(used_ticks)
        self.assertAlmostEqual(float(volumes.sum()), 600.0, places=5)

    def test_tick_takes_precedence_over_ohlcv(self):
        # Bar has OHLCV vol=9999 but tick data is present: should use tick
        kl, ticks = self._make_klines_and_ticks([
            (100.0, 110.0, 9999.0, {105.0: 50}),
        ])
        _, volumes, used_ticks = _compute_profile_bins(kl, ticks, CM, 0, 0)
        self.assertTrue(used_ticks)
        # Tick total is 50, not 9999
        self.assertAlmostEqual(float(volumes.sum()), 50.0, places=5)

    def test_missing_tick_bucket_falls_back_to_ohlcv(self):
        kl = _klines(_bar(0, 100.0, 110.0, 400.0))
        # ticks dict exists but has no entry for this bar
        _, volumes, used_ticks = _compute_profile_bins(kl, {}, CM, 0, 0)
        self.assertFalse(used_ticks)
        self.assertAlmostEqual(float(volumes.sum()), 400.0, places=5)

    def test_buy_sell_neutral_all_counted(self):
        kl = _klines(_bar(0, 100.0, 110.0, 0.0))
        ts_str = T0.strftime("%Y-%m-%d %H:%M")
        bk, _ = _ticks_for_bar(ts_str, {})
        ticks = {bk: {105.0: {"buy": 10, "sell": 20, "neutral": 30}}}
        _, volumes, used_ticks = _compute_profile_bins(kl, ticks, CM, 0, 0)
        self.assertTrue(used_ticks)
        self.assertAlmostEqual(float(volumes.sum()), 60.0, places=5)

    def test_zero_total_tick_entry_skipped(self):
        kl = _klines(_bar(0, 100.0, 110.0, 0.0))
        ts_str = T0.strftime("%Y-%m-%d %H:%M")
        bk, _ = _ticks_for_bar(ts_str, {})
        ticks = {bk: {105.0: {"buy": 0, "sell": 0, "neutral": 0}}}
        _, volumes, used_ticks = _compute_profile_bins(kl, ticks, CM, 0, 0)
        # No valid tick data → OHLCV fallback
        self.assertFalse(used_ticks)


# ---------------------------------------------------------------------------
# _compute_poc_vah_val
# ---------------------------------------------------------------------------

class TestComputePocVahVal(unittest.TestCase):

    def _uniform(self, n=60, lo=100.0, hi=160.0):
        centers = np.linspace(lo, hi, n)
        volumes = np.ones(n, dtype=float)
        return centers, volumes

    def test_poc_at_max_bin(self):
        centers = np.array([100.0, 102.0, 104.0, 106.0])
        volumes = np.array([10.0,  50.0,  20.0,   5.0])
        poc, _, _ = _compute_poc_vah_val(centers, volumes)
        self.assertAlmostEqual(poc, 102.0)

    def test_val_le_poc_le_vah(self):
        centers, volumes = self._uniform()
        volumes[30] = 1000.0  # spike in the middle
        poc, vah, val = _compute_poc_vah_val(centers, volumes)
        self.assertLessEqual(val, poc)
        self.assertLessEqual(poc, vah)

    def test_value_area_covers_at_least_70_pct(self):
        centers, volumes = self._uniform(n=20)
        volumes[10] = 500.0  # dominant bin
        poc, vah, val = _compute_poc_vah_val(centers, volumes)
        # Collect bins between val and vah
        mask = (centers >= val) & (centers <= vah)
        covered = float(volumes[mask].sum()) / float(volumes.sum())
        self.assertGreaterEqual(covered, 0.70)

    def test_single_bin_poc_equals_vah_val(self):
        centers = np.array([100.0])
        volumes = np.array([500.0])
        poc, vah, val = _compute_poc_vah_val(centers, volumes)
        self.assertAlmostEqual(poc, 100.0)
        self.assertAlmostEqual(vah, 100.0)
        self.assertAlmostEqual(val, 100.0)

    def test_empty_arrays_returns_zeros(self):
        poc, vah, val = _compute_poc_vah_val(np.array([]), np.array([]))
        self.assertEqual(poc, 0.0)
        self.assertEqual(vah, 0.0)
        self.assertEqual(val, 0.0)

    def test_zero_volume_array_returns_poc_price(self):
        centers = np.array([100.0, 105.0, 110.0])
        volumes = np.zeros(3)
        poc, vah, val = _compute_poc_vah_val(centers, volumes)
        # poc = centers[argmax(zeros)] = centers[0]
        self.assertEqual(poc, vah)
        self.assertEqual(poc, val)

    def test_uniform_distribution_value_area_spans_70pct(self):
        # With perfectly uniform volumes argmax returns index 0 (first element).
        # The meaningful invariant is that vah-val covers ~70% of the price range.
        centers, volumes = self._uniform(n=60, lo=100.0, hi=160.0)
        _, vah, val = _compute_poc_vah_val(centers, volumes)
        full_range = float(centers[-1] - centers[0])
        covered    = (vah - val) / full_range
        self.assertGreaterEqual(covered, 0.65)   # allow small bin-edge rounding

    def test_custom_va_pct_90(self):
        centers, volumes = self._uniform(n=20)
        volumes[0] = 1000.0  # dominant bin at low end
        _, vah90, val90 = _compute_poc_vah_val(centers, volumes, va_pct=0.90)
        _, vah70, val70 = _compute_poc_vah_val(centers, volumes, va_pct=0.70)
        # Wider value area at 90 % must span at least as much as at 70 %
        self.assertGreaterEqual(vah90 - val90, vah70 - val70)

    def test_two_equal_peaks_val_below_vah(self):
        centers = np.linspace(100.0, 110.0, 10)
        volumes = np.zeros(10)
        volumes[2] = 100.0
        volumes[8] = 100.0
        poc, vah, val = _compute_poc_vah_val(centers, volumes)
        self.assertLessEqual(val, vah)

    def test_all_volume_in_one_bin_covers_100_pct(self):
        centers = np.linspace(100.0, 160.0, 30)
        volumes = np.zeros(30)
        volumes[15] = 999.0
        poc, vah, val = _compute_poc_vah_val(centers, volumes)
        self.assertAlmostEqual(poc, vah)
        self.assertAlmostEqual(poc, val)


if __name__ == "__main__":
    unittest.main()

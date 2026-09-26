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


def _bins(kl, ticks=None, i0=0, i1=None, n_bins=60):
    """Profile the bar range [i0, i1] inclusive.

    _compute_profile_bins used to take that index range and slice internally.
    It now profiles whatever rows it is handed, so that the Session Volume
    Profile can pass a non-contiguous, session-filtered subset -- the slicing
    moved to the caller. It also grew a fourth return value, range-wide
    direction/size stats, which these tests do not exercise: they are about
    the binning.

    Keeping the inclusive-range convention in one place here rather than
    spelling .iloc[i0:i1 + 1] out at each call site, where the off-by-one is
    easy to get wrong in a way that still passes.
    """
    return _profile(kl, ticks, i0, i1, n_bins)[:3]


def _profile(kl, ticks=None, i0=0, i1=None, n_bins=60):
    """As _bins, but keeping the range-wide stats dict."""
    if i1 is None:
        i1 = len(kl) - 1
    return _compute_profile_bins(kl.iloc[i0:i1 + 1], ticks, CM, n_bins=n_bins)


def _bucket(i: int):
    """Tick-bucket key for bar i, the key _compute_profile_bins looks up."""
    return _ticks_for_bar(
        (T0 + timedelta(minutes=i * CM)).strftime("%Y-%m-%d %H:%M"), {})[0]


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
        return _bins(kl, None, i0, i1, n_bins)

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
        centers, volumes, used_ticks = _bins(kl, None, 0, 0)
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
        # Asserting "no volume below 150" would pass for free: the excluded bar
        # also sets the range, so there are no bins down there to look at. The
        # claim worth pinning is that the range and the total both come from
        # bars 1-2 alone.
        self.assertGreaterEqual(float(centers.min()), 200.0)
        self.assertLessEqual(float(centers.max()), 215.0)
        self.assertAlmostEqual(float(volumes.sum()), 1000.0, places=6,
                               msg="bar 0's 100 units must not be included")

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
        _, _, used_ticks = _bins(kl, ticks, 0, 0)
        self.assertTrue(used_ticks)

    def test_single_price_level_lands_in_correct_bin(self):
        kl, ticks = self._make_klines_and_ticks([
            (100.0, 110.0, 0.0, {105.0: 200}),
        ])
        centers, volumes, used_ticks = _bins(kl, ticks, 0, 0)
        self.assertTrue(used_ticks)
        poc_price = float(centers[int(np.argmax(volumes))])
        # POC bin center should be near 105
        self.assertAlmostEqual(poc_price, 105.0, delta=0.5)

    def test_tick_volume_totals_match(self):
        kl, ticks = self._make_klines_and_ticks([
            (100.0, 110.0, 0.0, {102.0: 100, 105.0: 200, 108.0: 300}),
        ])
        _, volumes, used_ticks = _bins(kl, ticks, 0, 0)
        self.assertTrue(used_ticks)
        self.assertAlmostEqual(float(volumes.sum()), 600.0, places=5)

    def test_tick_prices_outside_kline_range_ignored(self):
        kl, ticks = self._make_klines_and_ticks([
            (100.0, 110.0, 0.0, {99.0: 999, 105.0: 50, 111.0: 888}),
        ])
        _, volumes, _ = _bins(kl, ticks, 0, 0)
        # Only the 105.0 tick (within 100–110) should land in bins
        self.assertAlmostEqual(float(volumes.sum()), 50.0, places=5)

    def test_multi_bar_tick_aggregation(self):
        kl, ticks = self._make_klines_and_ticks([
            (100.0, 110.0, 0.0, {105.0: 100}),
            (100.0, 110.0, 0.0, {105.0: 200}),
            (100.0, 110.0, 0.0, {105.0: 300}),
        ])
        _, volumes, used_ticks = _bins(kl, ticks, 0, 2)
        self.assertTrue(used_ticks)
        self.assertAlmostEqual(float(volumes.sum()), 600.0, places=5)

    def test_tick_takes_precedence_over_ohlcv(self):
        # Bar has OHLCV vol=9999 but tick data is present: should use tick
        kl, ticks = self._make_klines_and_ticks([
            (100.0, 110.0, 9999.0, {105.0: 50}),
        ])
        _, volumes, used_ticks = _bins(kl, ticks, 0, 0)
        self.assertTrue(used_ticks)
        # Tick total is 50, not 9999
        self.assertAlmostEqual(float(volumes.sum()), 50.0, places=5)

    def test_missing_tick_bucket_falls_back_to_ohlcv(self):
        kl = _klines(_bar(0, 100.0, 110.0, 400.0))
        # ticks dict exists but has no entry for this bar
        _, volumes, used_ticks = _bins(kl, {}, 0, 0)
        self.assertFalse(used_ticks)
        self.assertAlmostEqual(float(volumes.sum()), 400.0, places=5)

    def test_buy_sell_neutral_all_counted(self):
        kl = _klines(_bar(0, 100.0, 110.0, 0.0))
        ts_str = T0.strftime("%Y-%m-%d %H:%M")
        bk, _ = _ticks_for_bar(ts_str, {})
        ticks = {bk: {105.0: {"buy": 10, "sell": 20, "neutral": 30}}}
        _, volumes, used_ticks = _bins(kl, ticks, 0, 0)
        self.assertTrue(used_ticks)
        self.assertAlmostEqual(float(volumes.sum()), 60.0, places=5)

    def test_zero_total_tick_entry_skipped(self):
        kl = _klines(_bar(0, 100.0, 110.0, 0.0))
        ts_str = T0.strftime("%Y-%m-%d %H:%M")
        bk, _ = _ticks_for_bar(ts_str, {})
        ticks = {bk: {105.0: {"buy": 0, "sell": 0, "neutral": 0}}}
        _, volumes, used_ticks = _bins(kl, ticks, 0, 0)
        # No valid tick data → OHLCV fallback
        self.assertFalse(used_ticks)


# ---------------------------------------------------------------------------
# _compute_profile_bins - range-wide stats
# ---------------------------------------------------------------------------

class TestComputeProfileBinsStats(unittest.TestCase):
    """The fourth return value, which feeds the Range Profile readout
    (Tot / Buy / Sell / Neu / Mid*).

    Two rules make it easy to misread. It counts tick-covered bars ONLY --
    OHLCV-fallback bars carry no direction or size information, so they add to
    the visual bins but not to these totals. And "medium" is a size breakdown
    *within* buy + sell, not a fourth additive bucket, so total stays a closed
    sum of buy + sell + neutral.
    """

    KEYS = ("total", "buy", "sell", "neutral", "medium")

    def _stats(self, kl, ticks=None, i0=0, i1=None):
        return _profile(kl, ticks, i0, i1)[3]

    # -- nothing to count --------------------------------------------------
    def test_no_ticks_leaves_every_field_zero(self):
        kl = _klines(_bar(0, 100.0, 110.0, 400.0))
        st = self._stats(kl, None)
        self.assertEqual(set(st), set(self.KEYS))
        for k in self.KEYS:
            self.assertEqual(st[k], 0.0, k)

    def test_ohlcv_volume_does_not_leak_into_the_totals(self):
        """400 units of OHLCV volume, no ticks: the bins get it, stats do not."""
        kl = _klines(_bar(0, 100.0, 110.0, 400.0))
        _, volumes, used, st = _profile(kl, {})
        self.assertFalse(used)
        self.assertAlmostEqual(float(volumes.sum()), 400.0, places=6)
        self.assertEqual(st["total"], 0.0)

    def test_degenerate_range_returns_zeroed_stats(self):
        kl = _klines(("2026-01-15 09:35", 100.0, 100.0, 100.0, 100.0, 50.0))
        st = self._stats(
            kl, {_bucket(0): {100.0: {"buy": 5, "sell": 5, "neutral": 0}}})
        for k in self.KEYS:
            self.assertEqual(st[k], 0.0, k)

    # -- tallying ----------------------------------------------------------
    def test_buy_sell_neutral_tallied_separately(self):
        kl = _klines(_bar(0, 100.0, 110.0, 0.0))
        st = self._stats(
            kl, {_bucket(0): {105.0: {"buy": 10, "sell": 20, "neutral": 30}}})
        self.assertEqual((st["buy"], st["sell"], st["neutral"]), (10, 20, 30))

    def test_total_is_a_closed_sum(self):
        kl = _klines(_bar(0, 100.0, 110.0, 0.0))
        st = self._stats(kl, {_bucket(0): {
            102.0: {"buy": 10, "sell": 20, "neutral": 30},
            108.0: {"buy": 1, "sell": 2, "neutral": 3},
        }})
        self.assertEqual(st["total"], st["buy"] + st["sell"] + st["neutral"])
        self.assertEqual(st["total"], 66)

    def test_aggregates_across_bars(self):
        kl = _klines(_bar(0, 100.0, 110.0, 0.0), _bar(1, 100.0, 110.0, 0.0))
        st = self._stats(kl, {
            _bucket(0): {105.0: {"buy": 10, "sell": 0, "neutral": 0}},
            _bucket(1): {105.0: {"buy": 5, "sell": 7, "neutral": 0}},
        }, i0=0, i1=1)
        self.assertEqual((st["buy"], st["sell"], st["total"]), (15, 7, 22))

    def test_stats_match_the_binned_volume_when_everything_is_counted(self):
        kl = _klines(_bar(0, 100.0, 110.0, 0.0))
        _, volumes, _, st = _profile(kl, {_bucket(0): {
            102.0: {"buy": 40, "sell": 0, "neutral": 0},
            107.0: {"buy": 0, "sell": 60, "neutral": 0},
        }})
        self.assertAlmostEqual(float(volumes.sum()), st["total"], places=6)

    def test_independent_of_bin_count(self):
        kl = _klines(_bar(0, 100.0, 110.0, 0.0))
        ticks = {_bucket(0): {105.0: {"buy": 10, "sell": 20, "neutral": 30}}}
        self.assertEqual(_profile(kl, ticks, n_bins=10)[3],
                         _profile(kl, ticks, n_bins=200)[3])

    # -- medium ------------------------------------------------------------
    def test_medium_sums_both_sides(self):
        kl = _klines(_bar(0, 100.0, 110.0, 0.0))
        st = self._stats(kl, {_bucket(0): {
            105.0: {"buy": 50, "sell": 50, "neutral": 0,
                    "buy_m": 20, "sell_m": 15},
        }})
        self.assertEqual(st["medium"], 35)

    def test_medium_is_not_added_into_total(self):
        """It is a slice of buy + sell, so counting it again would double it."""
        kl = _klines(_bar(0, 100.0, 110.0, 0.0))
        st = self._stats(kl, {_bucket(0): {
            105.0: {"buy": 50, "sell": 50, "neutral": 0,
                    "buy_m": 20, "sell_m": 15},
        }})
        self.assertEqual(st["total"], 100)
        self.assertLessEqual(st["medium"], st["buy"] + st["sell"])

    def test_medium_is_zero_when_the_feed_omits_the_size_keys(self):
        kl = _klines(_bar(0, 100.0, 110.0, 0.0))
        st = self._stats(
            kl, {_bucket(0): {105.0: {"buy": 10, "sell": 0, "neutral": 0}}})
        self.assertEqual(st["medium"], 0)

    # -- what gets excluded ------------------------------------------------
    def test_prices_outside_the_bar_range_are_not_counted(self):
        kl = _klines(_bar(0, 100.0, 110.0, 0.0))
        st = self._stats(kl, {_bucket(0): {
            99.0:  {"buy": 999, "sell": 0, "neutral": 0},
            105.0: {"buy": 50,  "sell": 0, "neutral": 0},
            111.0: {"buy": 888, "sell": 0, "neutral": 0},
        }})
        self.assertEqual(st["total"], 50)

    def test_zero_total_entries_are_skipped(self):
        kl = _klines(_bar(0, 100.0, 110.0, 400.0))
        st = self._stats(
            kl, {_bucket(0): {105.0: {"buy": 0, "sell": 0, "neutral": 0}}})
        self.assertEqual(st["total"], 0.0)

    def test_a_bar_without_ticks_is_excluded_from_a_mixed_range(self):
        """Bar 1 falls back to OHLCV: its 400 units reach the bins, not the
        totals. This is the rule that makes Tot smaller than the profile."""
        kl = _klines(_bar(0, 100.0, 110.0, 0.0), _bar(1, 100.0, 110.0, 400.0))
        _, volumes, used, st = _profile(
            kl, {_bucket(0): {105.0: {"buy": 60, "sell": 0, "neutral": 0}}},
            i0=0, i1=1)
        self.assertTrue(used)
        self.assertAlmostEqual(float(volumes.sum()), 460.0, places=6)
        self.assertEqual(st["total"], 60)

    def test_only_the_requested_bars_are_counted(self):
        kl = _klines(_bar(0, 100.0, 110.0, 0.0), _bar(1, 100.0, 110.0, 0.0))
        ticks = {
            _bucket(0): {105.0: {"buy": 11, "sell": 0, "neutral": 0}},
            _bucket(1): {105.0: {"buy": 22, "sell": 0, "neutral": 0}},
        }
        self.assertEqual(self._stats(kl, ticks, i0=1, i1=1)["total"], 22)


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

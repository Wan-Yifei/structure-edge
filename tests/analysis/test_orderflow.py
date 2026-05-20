"""
Unit tests for analysis.orderflow

Run: uv run pytest tests/ -v
"""

import pathlib
import unittest
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

from core.time_utils import candle_start
from core.chart import build_ohlcv_profile
from analysis.orderflow import TIMEFRAME_MAP

OUTPUT_DIR = pathlib.Path(__file__).parent.parent / "outputs"


# ── candle_start ─────────────────────────────────────────────────────────────

class TestCandleStart(unittest.TestCase):

    def test_mid_candle_15m(self):
        t = datetime(2026, 5, 15, 9, 37, 42, 500000)
        self.assertEqual(candle_start(t, 15), datetime(2026, 5, 15, 9, 30))

    def test_already_on_boundary(self):
        t = datetime(2026, 5, 15, 9, 30, 0)
        self.assertEqual(candle_start(t, 15), datetime(2026, 5, 15, 9, 30))

    def test_last_second_of_candle(self):
        t = datetime(2026, 5, 15, 9, 44, 59, 999999)
        self.assertEqual(candle_start(t, 15), datetime(2026, 5, 15, 9, 30))

    def test_first_second_of_next_candle(self):
        t = datetime(2026, 5, 15, 9, 45, 0)
        self.assertEqual(candle_start(t, 15), datetime(2026, 5, 15, 9, 45))

    def test_hour_boundary_1m(self):
        t = datetime(2026, 5, 15, 10, 0, 1)
        self.assertEqual(candle_start(t, 1), datetime(2026, 5, 15, 10, 0))

    def test_5m_alignment(self):
        t = datetime(2026, 5, 15, 9, 37, 0)
        self.assertEqual(candle_start(t, 5), datetime(2026, 5, 15, 9, 35))

    def test_30m_alignment(self):
        t = datetime(2026, 5, 15, 9, 47, 0)
        self.assertEqual(candle_start(t, 30), datetime(2026, 5, 15, 9, 30))

    def test_1h_alignment(self):
        t = datetime(2026, 5, 15, 10, 23, 0)
        self.assertEqual(candle_start(t, 60), datetime(2026, 5, 15, 10, 0))

    def test_1h_different_hours_give_different_buckets(self):
        t1 = datetime(2026, 5, 15, 9, 55, 0)
        t2 = datetime(2026, 5, 15, 10, 5, 0)
        self.assertNotEqual(candle_start(t1, 60), candle_start(t2, 60))

    def test_preserves_date(self):
        t = datetime(2026, 1, 1, 0, 7, 30)
        result = candle_start(t, 15)
        self.assertEqual(result.date(), t.date())
        self.assertEqual(result.minute, 0)


# ── TIMEFRAME_MAP ────────────────────────────────────────────────────────────

class TestTimeframeMap(unittest.TestCase):

    def test_all_timeframes_present(self):
        for tf in ("1m", "5m", "15m", "30m", "1h"):
            self.assertIn(tf, TIMEFRAME_MAP)

    def test_candle_minutes_match_label(self):
        expected = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60}
        for tf, mins in expected.items():
            _, actual_mins = TIMEFRAME_MAP[tf]
            self.assertEqual(actual_mins, mins, f"Mismatch for {tf}")


# ── Tick bucketing ───────────────────────────────────────────────────────────

class TestTickBucketing(unittest.TestCase):

    def _make_buckets(self):
        return defaultdict(lambda: defaultdict(lambda: {"buy": 0, "sell": 0, "neutral": 0}))

    def _ingest(self, buckets, time_str, price, volume, direction, candle_minutes=15):
        fmt = "%Y-%m-%d %H:%M:%S.%f" if "." in time_str else "%Y-%m-%d %H:%M:%S"
        t = datetime.strptime(time_str, fmt)
        bucket = candle_start(t, candle_minutes)
        key = {"BUY": "buy", "SELL": "sell"}.get(direction.upper(), "neutral")
        buckets[bucket][price][key] += volume

    def test_buy_sell_separated(self):
        b = self._make_buckets()
        self._ingest(b, "2026-05-15 09:31:00", 1400.0, 100, "BUY")
        self._ingest(b, "2026-05-15 09:32:00", 1400.0, 200, "SELL")
        bucket = datetime(2026, 5, 15, 9, 30)
        self.assertEqual(b[bucket][1400.0]["buy"], 100)
        self.assertEqual(b[bucket][1400.0]["sell"], 200)

    def test_same_candle_aggregated(self):
        b = self._make_buckets()
        for _ in range(3):
            self._ingest(b, "2026-05-15 09:33:00", 1400.0, 50, "BUY")
        self.assertEqual(b[datetime(2026, 5, 15, 9, 30)][1400.0]["buy"], 150)

    def test_span_two_candles(self):
        b = self._make_buckets()
        self._ingest(b, "2026-05-15 09:44:59", 1400.0, 100, "BUY")
        self._ingest(b, "2026-05-15 09:45:00", 1400.0, 200, "BUY")
        self.assertEqual(b[datetime(2026, 5, 15, 9, 30)][1400.0]["buy"], 100)
        self.assertEqual(b[datetime(2026, 5, 15, 9, 45)][1400.0]["buy"], 200)

    def test_neutral_direction(self):
        b = self._make_buckets()
        self._ingest(b, "2026-05-15 09:31:00", 1400.0, 300, "NEUTRAL")
        self.assertEqual(b[datetime(2026, 5, 15, 9, 30)][1400.0]["neutral"], 300)

    def test_multiple_price_levels(self):
        b = self._make_buckets()
        self._ingest(b, "2026-05-15 09:31:00", 1400.0, 100, "BUY")
        self._ingest(b, "2026-05-15 09:31:30", 1400.5, 50, "SELL")
        self.assertEqual(len(b[datetime(2026, 5, 15, 9, 30)]), 2)

    def test_1h_timeframe_bucketing(self):
        b = self._make_buckets()
        self._ingest(b, "2026-05-15 09:55:00", 1400.0, 100, "BUY", candle_minutes=60)
        self._ingest(b, "2026-05-15 10:05:00", 1400.0, 200, "BUY", candle_minutes=60)
        self.assertEqual(b[datetime(2026, 5, 15, 9, 0)][1400.0]["buy"], 100)
        self.assertEqual(b[datetime(2026, 5, 15, 10, 0)][1400.0]["buy"], 200)


# ── build_ohlcv_profile ──────────────────────────────────────────────────────

class TestOhlcvProfile(unittest.TestCase):

    def _make_klines(self, rows):
        return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"])

    def test_returns_centers_and_volumes(self):
        klines = self._make_klines([
            (100, 110, 95, 105, 1000),
            (105, 115, 100, 108, 1500),
        ])
        result = build_ohlcv_profile(klines, n_bins=10)
        self.assertIsNotNone(result)
        centers, volumes = result
        self.assertEqual(len(centers), 10)
        self.assertEqual(len(volumes), 10)

    def test_volume_within_range(self):
        klines = self._make_klines([(100, 110, 95, 105, 1000)])
        centers, volumes = build_ohlcv_profile(klines, n_bins=20)
        # bins outside [95, 110] should be zero
        outside = volumes[(centers < 95) | (centers > 110)]
        self.assertTrue((outside == 0).all())

    def test_total_volume_preserved(self):
        klines = self._make_klines([
            (100, 110, 95, 105, 1000),
            (105, 115, 100, 108, 2000),
        ])
        _, volumes = build_ohlcv_profile(klines, n_bins=30)
        # total distributed volume should equal sum of candle volumes
        self.assertAlmostEqual(volumes.sum(), 3000, delta=1)

    def test_flat_candle_returns_none(self):
        klines = self._make_klines([(100, 100, 100, 100, 500)])
        self.assertIsNone(build_ohlcv_profile(klines))

    def test_poc_is_highest_volume_bin(self):
        klines = self._make_klines([
            (100, 110, 95, 105, 100),
            (104, 106, 103, 105, 5000),   # concentrated volume in 103-106 range
        ])
        centers, volumes = build_ohlcv_profile(klines, n_bins=30)
        poc = centers[np.argmax(volumes)]
        self.assertGreaterEqual(poc, 100)
        self.assertLessEqual(poc, 110)


# ── Chart smoke test ─────────────────────────────────────────────────────────

class TestChartOutput(unittest.TestCase):

    def _make_klines(self):
        return pd.DataFrame({
            "time_key": ["2026-05-15 09:30:00", "2026-05-15 09:45:00",
                         "2026-05-15 10:00:00", "2026-05-15 10:15:00"],
            "open":  [1390.0, 1400.0, 1405.0, 1402.0],
            "high":  [1395.0, 1410.0, 1412.0, 1408.0],
            "low":   [1385.0, 1398.0, 1400.0, 1398.0],
            "close": [1400.0, 1405.0, 1402.0, 1406.0],
            "volume": [5000, 8000, 6000, 7000],
        })

    def test_ohlcv_profile_chart(self):
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        klines = self._make_klines()
        result = build_ohlcv_profile(klines, n_bins=20)
        self.assertIsNotNone(result)
        centers, volumes = result

        fig = Figure(facecolor="#1a1a2e")
        ax_c, ax_p = fig.subplots(1, 2, gridspec_kw={"width_ratios": [3, 1]})

        for i, (_, row) in enumerate(klines.iterrows()):
            o, h, l, c = row["open"], row["high"], row["low"], row["close"]
            color = "#26a69a" if c >= o else "#ef5350"
            ax_c.bar(i, abs(c - o), bottom=min(o, c), color=color, width=0.6)
            ax_c.plot([i, i], [l, h], color=color)

        h = (centers[1] - centers[0]) * 0.85
        ax_p.barh(centers, volumes, height=h, color="#26a69a", alpha=0.85)
        poc = centers[np.argmax(volumes)]
        ax_p.axhline(poc, color="#ffd700", linewidth=1.2, linestyle="--")

        out = OUTPUT_DIR / "test_ohlcv_profile.png"
        fig.savefig(out, dpi=72)
        self.assertTrue(out.exists())
        self.assertGreater(out.stat().st_size, 1000)

    def test_tick_profile_chart(self):
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        klines = self._make_klines()
        buckets = {
            datetime(2026, 5, 15, 10, 15): {
                1402.0: {"buy": 300, "sell": 150, "neutral": 50},
                1403.0: {"buy": 100, "sell": 400, "neutral": 0},
                1404.0: {"buy": 500, "sell": 200, "neutral": 80},
            }
        }
        latest = max(buckets.keys())
        pd_ = buckets[latest]
        prices = sorted(pd_.keys())
        y = np.array(prices)
        h = (y[1] - y[0]) * 0.8

        fig = Figure(facecolor="#1a1a2e")
        ax_c, ax_p = fig.subplots(1, 2, gridspec_kw={"width_ratios": [3, 1]})
        ax_p.barh(y, [pd_[p]["buy"] for p in prices], height=h, color="#26a69a", label="Buy")
        ax_p.barh(y, [-pd_[p]["sell"] for p in prices], height=h, color="#ef5350", label="Sell")
        ax_p.set_title(f"Order Profile\n{latest.strftime('%Y-%m-%d %H:%M')}", color="white")

        out = OUTPUT_DIR / "test_tick_profile.png"
        fig.savefig(out, dpi=72)
        self.assertTrue(out.exists())
        self.assertGreater(out.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()

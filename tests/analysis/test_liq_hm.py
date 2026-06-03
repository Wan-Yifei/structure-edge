"""Unit tests for liq_hm_window pure helper functions.

No Qt dependency — tests run headless.
Run: uv run pytest tests/ -v
"""

import unittest
import numpy as np

from analysis.liq_hm_window import (
    _hot_rgba,
    _single_rgba,
    _calc_col_mid,
    _calc_depth_label,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _snap(*rows):
    """Build a minimal OB snapshot list.  Each row is (side, price, volume)."""
    return [{"side": s, "price": p, "volume": v} for s, p, v in rows]


def _grid(shape=(10, 20), max_val=1000.0):
    rng = np.random.default_rng(42)
    return rng.uniform(0, max_val, shape)


# ---------------------------------------------------------------------------
# ① _hot_rgba / _single_rgba — gamma correction
# ---------------------------------------------------------------------------

class TestHotRgba(unittest.TestCase):

    def test_returns_none_for_empty_grid(self):
        self.assertIsNone(_hot_rgba(np.zeros((5, 5))))

    def test_gamma_one_is_baseline(self):
        g = _grid()
        r1a = _hot_rgba(g.copy(), gamma=1.0)
        r1b = _hot_rgba(g.copy())        # default gamma=1
        np.testing.assert_array_equal(r1a, r1b)

    def test_gamma_gt1_suppresses_dim_areas(self):
        g = _grid()
        r1 = _hot_rgba(g.copy(), gamma=1.0)
        r2 = _hot_rgba(g.copy(), gamma=2.0)
        # Higher gamma → lower mean alpha (sparse zones darkened)
        self.assertLess(float(r2[..., 3].mean()), float(r1[..., 3].mean()))

    def test_gamma_lt1_boosts_dim_areas(self):
        g = _grid()
        r1 = _hot_rgba(g.copy(), gamma=1.0)
        r3 = _hot_rgba(g.copy(), gamma=0.5)
        self.assertGreater(float(r3[..., 3].mean()), float(r1[..., 3].mean()))

    def test_dense_pixel_stays_bright_under_high_gamma(self):
        g = np.zeros((5, 5))
        g[2, 2] = 1_000_000.0    # single very dense cell
        r = _hot_rgba(g, gamma=5.0)
        # The maximum cell should still be visible (alpha > 0)
        self.assertGreater(int(r[2, 2, 3]), 0)

    def test_output_shape_matches_input(self):
        g = _grid((12, 30))
        r = _hot_rgba(g)
        self.assertEqual(r.shape, (12, 30, 4))

    def test_alpha_channel_in_valid_range(self):
        r = _hot_rgba(_grid(), gamma=3.0)
        self.assertTrue((r[..., 3] <= 255).all())
        self.assertTrue((r[..., 3] >= 0).all())


class TestSingleRgba(unittest.TestCase):

    def test_returns_none_for_empty_grid(self):
        self.assertIsNone(_single_rgba(np.zeros((5, 5)), "#26a69a"))

    def test_gamma_one_is_baseline(self):
        g = _grid()
        np.testing.assert_array_equal(
            _single_rgba(g.copy(), "#26a69a", gamma=1.0),
            _single_rgba(g.copy(), "#26a69a"),
        )

    def test_gamma_gt1_suppresses_dim_areas(self):
        g = _grid()
        r1 = _single_rgba(g.copy(), "#26a69a", gamma=1.0)
        r2 = _single_rgba(g.copy(), "#26a69a", gamma=2.0)
        self.assertLess(float(r2[..., 3].mean()), float(r1[..., 3].mean()))

    def test_gamma_lt1_boosts_dim_areas(self):
        g = _grid()
        r1 = _single_rgba(g.copy(), "#26a69a", gamma=1.0)
        r3 = _single_rgba(g.copy(), "#26a69a", gamma=0.5)
        self.assertGreater(float(r3[..., 3].mean()), float(r1[..., 3].mean()))

    def test_rgb_channels_match_requested_color(self):
        g = _grid()
        r = _single_rgba(g, "#ef5350")   # RED
        # Every non-transparent pixel should use the red hue
        mask = r[..., 3] > 0
        np.testing.assert_array_equal(r[mask, 0], np.full(mask.sum(), 0xEF))
        np.testing.assert_array_equal(r[mask, 1], np.full(mask.sum(), 0x53))
        np.testing.assert_array_equal(r[mask, 2], np.full(mask.sum(), 0x50))


# ---------------------------------------------------------------------------
# ② _calc_col_mid — mid-price per column
# ---------------------------------------------------------------------------

class TestCalcColMid(unittest.TestCase):

    def test_both_sides_returns_mid(self):
        snap = _snap(("BID", 100.4, 500), ("ASK", 100.6, 300))
        self.assertAlmostEqual(_calc_col_mid(snap), 100.5)

    def test_multiple_bids_asks_uses_best(self):
        snap = _snap(
            ("BID", 100.3, 200), ("BID", 100.4, 500),   # best bid = 100.4
            ("ASK", 100.6, 300), ("ASK", 100.8, 100),   # best ask = 100.6
        )
        self.assertAlmostEqual(_calc_col_mid(snap), 100.5)

    def test_only_bid_returns_bid(self):
        snap = _snap(("BID", 100.5, 500))
        self.assertAlmostEqual(_calc_col_mid(snap), 100.5)

    def test_only_ask_returns_ask(self):
        snap = _snap(("ASK", 100.7, 300))
        self.assertAlmostEqual(_calc_col_mid(snap), 100.7)

    def test_empty_snap_returns_none(self):
        self.assertIsNone(_calc_col_mid([]))

    def test_neutral_entries_ignored(self):
        # Entries with unknown side should not be counted
        snap = [{"side": "UNKNOWN", "price": 99.0, "volume": 1000},
                {"side": "BID", "price": 100.4, "volume": 500},
                {"side": "ASK", "price": 100.6, "volume": 300}]
        self.assertAlmostEqual(_calc_col_mid(snap), 100.5)


# ---------------------------------------------------------------------------
# ③ _calc_depth_label — depth-to-cursor annotation
# ---------------------------------------------------------------------------

class TestCalcDepthLabel(unittest.TestCase):

    def _snap_ladder(self):
        return _snap(
            ("BID", 100.5, 500), ("BID", 100.4, 300), ("BID", 100.3, 200),
            ("ASK", 100.6, 400), ("ASK", 100.7, 600), ("ASK", 100.8, 800),
        )

    def test_cursor_in_spread_returns_spread(self):
        result = _calc_depth_label(self._snap_ladder(), 100.5, 100.6, 100.55)
        self.assertEqual(result, "[spread]")

    def test_cursor_above_ask_sums_ask_levels(self):
        # target 100.7 → eat ask at 100.6 and 100.7 = 400+600 = 1000
        result = _calc_depth_label(self._snap_ladder(), 100.5, 100.6, 100.7)
        self.assertEqual(result, "eat↑ 1,000")

    def test_cursor_far_above_ask_sums_all_intercepted(self):
        # target 100.9 → 400+600+800 = 1800
        result = _calc_depth_label(self._snap_ladder(), 100.5, 100.6, 100.9)
        self.assertEqual(result, "eat↑ 1,800")

    def test_cursor_exactly_at_ask(self):
        # target == best_ask → single level consumed = 400
        result = _calc_depth_label(self._snap_ladder(), 100.5, 100.6, 100.6)
        self.assertEqual(result, "eat↑ 400")

    def test_cursor_below_bid_sums_bid_levels(self):
        # target 100.4 → eat bid at 100.5 and 100.4 = 500+300 = 800
        result = _calc_depth_label(self._snap_ladder(), 100.5, 100.6, 100.4)
        self.assertEqual(result, "eat↓ 800")

    def test_cursor_far_below_bid(self):
        # target 100.2 → 500+300+200 = 1000
        result = _calc_depth_label(self._snap_ladder(), 100.5, 100.6, 100.2)
        self.assertEqual(result, "eat↓ 1,000")

    def test_cursor_exactly_at_bid(self):
        result = _calc_depth_label(self._snap_ladder(), 100.5, 100.6, 100.5)
        self.assertEqual(result, "eat↓ 500")

    def test_no_ask_levels_in_range(self):
        snap = _snap(("BID", 100.5, 500))   # no ask levels
        result = _calc_depth_label(snap, 100.5, 100.6, 100.9)
        self.assertEqual(result, "eat↑ 0")


if __name__ == "__main__":
    unittest.main()

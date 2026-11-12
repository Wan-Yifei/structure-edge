"""What the heatmap grid holds, and what brightness it turns into.

Three defects sat on one path and each one quietly corrupted the picture
before the next stage could help:

1. A price bin is wider than a tick. The span is at least +/-1% of mid split
   into n_price bands, about 2.76 cents on SOXL against a 1-cent tick, so a
   bin normally holds two or three book levels. _paint_column assigned rather
   than added, so whichever level happened to be written last owned the cell.
   Measured over 23 RTH days: 90% of nonzero bins held more than one level,
   the grid displayed 39% of the shares actually resting in the band, and the
   rank correlation between displayed and real was 0.61. That last number is
   the one that matters -- it is an ordering error, so no later normalizer
   could undo it.

2. The MinVol spinbox is the heatmap's display filter, but it was also handed
   to detect_stacked_imbalance, where it means something else entirely. A
   level under the threshold is dropped from the rank list, so the ranks
   below it shift up and the two sides stop being compared at the same
   distance from the touch -- which is the only reason pairing by rank is
   meaningful. A moderate filter pushes the trigger rate down. Then, once a
   side holds fewer than max_depth levels, the missing ranks are zero-filled,
   score an infinite ratio, and the rate jumps far above baseline. Measured
   twice on SOXL RTH at a 5s stride, 7 days and 1 day: 16.8% -> 5.2% at
   MinVol 500 -> 37.2% at 2000, against 4.5% -> 1.7% -> 43.1% at 3000. Same
   shape, different breakpoint, because it depends on how deep the book is
   that day. Non-monotonic with a turning point that moves: no user could
   have predicted it from the control.

3. Gamma shipped at 5.0, which is what breaks the one property a log scale
   is for: equal brightness per doubling of size.

These tests are about the grid and the mapping, not about whether the
imbalance overlay is worth showing -- that is a separate question with its
own answer.
"""
from __future__ import annotations

import os
import pathlib
import sys
import types
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from analysis.liq_hm_window import (           # noqa: E402
    LiqHmWindow, _percentile_norm, _single_rgba,
)


# ── 1. the grid cell ───────────────────────────────────────────────────────

def _stub(min_vol: float = 0.0, bin_size: float = 0.03):
    """The slice of LiqHmWindow that _paint_column touches.

    bin_size defaults wider than a tick on purpose: that is the real
    geometry, and the whole point of these tests.
    """
    s = types.SimpleNamespace(
        _min_vol_spin=types.SimpleNamespace(value=lambda: min_vol),
        _price_min=10.00, _bin_size=bin_size,
        _best_bid=None, _best_ask=None,
        _bid_grid=np.zeros((4, 50)), _ask_grid=np.zeros((4, 50)),
    )
    s._paint_column = types.MethodType(LiqHmWindow._paint_column, s)
    return s


class TestABinHoldsTheSumOfItsLevels(unittest.TestCase):

    def test_two_levels_in_one_bin_are_added(self):
        s = _stub()
        # 10.00 and 10.01 both land in bin 0 at a 3-cent bin.
        s._paint_column(0, [{"side": "BID", "price": 10.00, "volume": 300},
                            {"side": "BID", "price": 10.01, "volume": 700}])
        self.assertEqual(s._bid_grid[0, 0], 1000)

    def test_three_levels_in_one_bin_are_added(self):
        s = _stub()
        s._paint_column(0, [{"side": "ASK", "price": 10.00, "volume": 100},
                            {"side": "ASK", "price": 10.01, "volume": 200},
                            {"side": "ASK", "price": 10.02, "volume": 400}])
        self.assertEqual(s._ask_grid[0, 0], 700)

    def test_the_last_level_written_no_longer_wins(self):
        """The defect, stated as its own test: under assignment this cell
        held 1 share, the smallest of the three, purely by write order."""
        s = _stub()
        s._paint_column(0, [{"side": "BID", "price": 10.00, "volume": 5000},
                            {"side": "BID", "price": 10.01, "volume": 900},
                            {"side": "BID", "price": 10.02, "volume": 1}])
        self.assertEqual(s._bid_grid[0, 0], 5901)
        self.assertNotEqual(s._bid_grid[0, 0], 1)

    def test_levels_in_different_bins_stay_apart(self):
        s = _stub()
        s._paint_column(0, [{"side": "BID", "price": 10.00, "volume": 300},
                            {"side": "BID", "price": 10.05, "volume": 700}])
        self.assertEqual(s._bid_grid[0, 0], 300)
        self.assertEqual(s._bid_grid[0, 1], 700)

    def test_bid_and_ask_do_not_mix(self):
        s = _stub()
        s._paint_column(0, [{"side": "BID", "price": 10.00, "volume": 300},
                            {"side": "ASK", "price": 10.01, "volume": 700}])
        self.assertEqual(s._bid_grid[0, 0], 300)
        self.assertEqual(s._ask_grid[0, 0], 700)

    def test_other_columns_are_untouched(self):
        s = _stub()
        s._paint_column(2, [{"side": "BID", "price": 10.00, "volume": 300}])
        self.assertEqual(s._bid_grid[2, 0], 300)
        self.assertEqual(s._bid_grid[0, 0], 0)
        self.assertEqual(s._bid_grid[3, 0], 0)


class TestRepaintingDoesNotAccumulate(unittest.TestCase):
    """The clear at the top of _paint_column is what makes += safe.

    The live rightmost column is repainted on every push. Without the clear
    an added cell would grow without bound; with it, a repaint is a replace.
    """

    def test_painting_the_same_column_twice_replaces_it(self):
        s = _stub()
        snap = [{"side": "BID", "price": 10.00, "volume": 300},
                {"side": "BID", "price": 10.01, "volume": 700}]
        s._paint_column(0, snap)
        s._paint_column(0, snap)
        self.assertEqual(s._bid_grid[0, 0], 1000)

    def test_a_level_that_left_the_book_is_cleared(self):
        s = _stub()
        s._paint_column(0, [{"side": "BID", "price": 10.00, "volume": 300},
                            {"side": "BID", "price": 10.90, "volume": 900}])
        s._paint_column(0, [{"side": "BID", "price": 10.00, "volume": 300}])
        self.assertEqual(s._bid_grid[0, 0], 300)
        self.assertEqual(s._bid_grid[0, 30], 0)


class TestTheDisplayFilterStillFilters(unittest.TestCase):
    """MinVol keeps its meaning for the heatmap itself; only the detector
    stopped being fed by it."""

    def test_a_level_below_min_vol_is_not_painted(self):
        s = _stub(min_vol=500)
        s._paint_column(0, [{"side": "BID", "price": 10.00, "volume": 100}])
        self.assertEqual(s._bid_grid[0, 0], 0)

    def test_min_vol_applies_per_level_not_per_bin(self):
        """Two 400-share levels sharing a bin sum to 800, but each is below a
        500 filter, so neither is painted. This is the existing behaviour,
        pinned so the += change is not read as weakening the filter."""
        s = _stub(min_vol=500)
        s._paint_column(0, [{"side": "BID", "price": 10.00, "volume": 400},
                            {"side": "BID", "price": 10.01, "volume": 400}])
        self.assertEqual(s._bid_grid[0, 0], 0)

    def test_levels_above_the_filter_still_add(self):
        s = _stub(min_vol=500)
        s._paint_column(0, [{"side": "BID", "price": 10.00, "volume": 600},
                            {"side": "BID", "price": 10.01, "volume": 900},
                            {"side": "BID", "price": 10.02, "volume": 100}])
        self.assertEqual(s._bid_grid[0, 0], 1500)

    def test_out_of_range_prices_are_dropped_not_clamped(self):
        s = _stub()
        s._paint_column(0, [{"side": "BID", "price": 9.00, "volume": 400},
                            {"side": "BID", "price": 99.0, "volume": 400},
                            {"side": "BID", "price": 10.00, "volume": 100}])
        self.assertEqual(s._bid_grid[0, 0], 100)
        self.assertEqual(s._bid_grid[0].sum(), 100)


# ── 2. brightness per doubling ─────────────────────────────────────────────

# What the 99th-percentile reference actually comes out at on a real SOXL
# grid once _paint_column accumulates: 17,486 shares, with the median bin at
# 2,110. Measuring visibility against anything far above this says more about
# the synthetic reference than about the screen.
REAL_P99 = 17_486.0
REAL_MEDIAN_BIN = 2_110.0


def _alphas(values: list[float], gamma: float,
            ref: float = 500_000.0) -> list[int]:
    """Alpha each volume renders at, through the real colour function.

    The ladder is padded with a larger level so the 99th-percentile reference
    lands on *ref* and the rungs sit below it: otherwise the top rung clips to
    full brightness and the measurement would be of the clip, not the curve.
    The default ref is deliberately far above the ladder, which is what makes
    the per-doubling steps readable across four decades; pass REAL_P99 when
    the question is whether something is visible on the actual screen.
    """
    grid = np.zeros((1, len(values) + 100))
    grid[0, :len(values)] = values
    grid[0, len(values):] = ref
    rgba = _single_rgba(grid, "#26c6da", gamma=gamma)
    return [int(rgba[0, i, 3]) for i in range(len(values))]


class TestEqualBrightnessPerDoubling(unittest.TestCase):
    """At gamma 1.0 the mapping is log-linear: a doubling of size is the
    same brightness step wherever it happens. Higher gamma spends that on the
    top of the range.

    Which one ships is a display preference and is NOT asserted here -- it
    was 5.0, briefly 1.0, and is 5.0 again by choice. What is asserted is
    that both ends of the slider still behave the way the comments in the
    config claim they do, so the choice stays an informed one."""

    LADDER = [100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0, 6400.0, 12800.0]

    def test_gamma_one_gives_the_same_step_at_every_magnitude(self):
        a = _alphas(self.LADDER, gamma=1.0)
        steps = [a[i + 1] - a[i] for i in range(len(a) - 1)]
        self.assertGreater(min(steps), 0, "a doubling must be visible at all")
        self.assertLessEqual(max(steps) - min(steps), 1,
                             f"steps should be flat, got {steps}")

    def test_gamma_five_favours_the_top_of_the_range(self):
        a = _alphas(self.LADDER, gamma=5.0)
        steps = [a[i + 1] - a[i] for i in range(len(a) - 1)]
        self.assertGreater(steps[-1], steps[0] * 4,
                           f"expected a steep top, got {steps}")

    def test_the_factory_gamma_is_one_the_slider_can_reach(self):
        """A config value outside the control's own range cannot be undone
        from the UI, which is the one way a preference turns into a trap."""
        import analysis.viewer_config as vcfg
        self.assertGreaterEqual(float(vcfg.liq("gamma")), 0.2)
        self.assertLessEqual(float(vcfg.liq("gamma")), 10.0)

    def test_the_toml_and_the_fallback_do_not_drift_apart(self):
        """DEFAULTS is what is used when the toml is missing or lacks the key,
        so it is a second factory setting. It is unreachable while the toml
        carries gamma, which is exactly why it would otherwise drift
        unnoticed -- a mutation flipping it alone used to survive the whole
        suite."""
        import analysis.viewer_config as vcfg
        self.assertEqual(float(vcfg.DEFAULTS["liq_hm"]["gamma"]),
                         float(vcfg.liq("gamma")))

    def test_the_median_level_is_visible_at_the_factory_gamma(self):
        """The property that has to hold whatever gamma ships: the typical
        bin cannot render as an empty cell. Measured against the reference a
        real grid produces, not a synthetic one -- at REAL_P99 the median bin
        sits high enough on the curve that even gamma 5 keeps it legible,
        which is the whole reason the preference is affordable."""
        import analysis.viewer_config as vcfg
        a = _alphas([REAL_MEDIAN_BIN], gamma=float(vcfg.liq("gamma")),
                    ref=REAL_P99)
        self.assertGreater(a[0], 40, f"median bin renders at alpha {a[0]}")

    def test_the_median_level_would_be_visible_at_either_end_too(self):
        """Stated for both settings so the previous test cannot pass merely
        because the shipped value happens to be the lenient one."""
        for g in (1.0, 5.0):
            a = _alphas([REAL_MEDIAN_BIN], gamma=g, ref=REAL_P99)
            self.assertGreater(a[0], 40, f"gamma {g}: alpha {a[0]}")


class TestThePercentileReference(unittest.TestCase):

    def test_one_outlier_does_not_set_the_ceiling(self):
        """Why the reference is a high percentile and not the max: a single
        huge resting order used to darken everything else."""
        g = np.zeros((1, 200))
        g[0, :199] = 1000.0
        g[0, 199]  = 10_000_000.0
        norm = _percentile_norm(np.log1p(g))
        self.assertGreater(float(norm[0, 0]), 0.9)

    def test_above_the_reference_clips_rather_than_dimming_the_rest(self):
        g = np.zeros((1, 200))
        g[0, :190] = 1000.0
        g[0, 190:] = 10_000_000.0
        norm = _percentile_norm(np.log1p(g))
        self.assertEqual(float(norm[0, 199]), 1.0)

    def test_an_empty_grid_is_handed_back_unchanged(self):
        g = np.zeros((1, 10))
        norm = _percentile_norm(np.log1p(g))
        self.assertEqual(float(norm.max()), 0.0)


# ── 3. the display filter no longer reaches the detector ───────────────────

def _book(levels: list[tuple[str, float, float]], ts: str = "T"):
    return [{"ts": ts, "side": s, "price": p, "volume": v}
            for s, p, v in levels]


def _marker_stub(min_vol: float):
    """The slice of LiqHmWindow that _redraw_orderflow_markers touches.

    Only the imbalance branch is switched on; the rest are off so the test
    pins one call and nothing else.
    """
    off = types.SimpleNamespace(isChecked=lambda: False)
    on  = types.SimpleNamespace(isChecked=lambda: True)
    s = types.SimpleNamespace(
        _bin_size=0.03, _price_min=10.0,
        _raw_snaps=_book([("BID", 10.00, 900), ("ASK", 10.01, 100)]),
        _min_vol_spin=types.SimpleNamespace(value=lambda: min_vol),
        _n_price_spin=types.SimpleNamespace(value=lambda: 100),
        _col_secs_spin=types.SimpleNamespace(value=lambda: 1),
        _simb_levels_spin=types.SimpleNamespace(value=lambda: 3),
        _simb_ratio_spin=types.SimpleNamespace(value=lambda: 3.0),
        _simb_depth_spin=types.SimpleNamespace(value=lambda: 10),
        _ice_cb=off, _spoof_cb=off, _absorb_cb=off, _simb_cb=on,
        _absorb_ticks=[], _best_bid=10.00, _best_ask=10.01,
        _clear_overlay_items=lambda: None,
        _build_bucket_to_idx=lambda: {"T": 0},
        _draw_simb_markers=lambda simbs: None,
        _draw_wall_labels=lambda: None,
    )
    return s


class TestMinVolDoesNotReachTheImbalanceDetector(unittest.TestCase):

    def _captured(self, min_vol: float) -> dict:
        import analysis.orderflow_detect as ofd
        seen: dict = {}
        real = ofd.detect_stacked_imbalance

        def spy(*a, **kw):
            seen.update(kw)
            return real(*a, **kw)

        ofd.detect_stacked_imbalance = spy
        try:
            LiqHmWindow._redraw_orderflow_markers(_marker_stub(min_vol))
        finally:
            ofd.detect_stacked_imbalance = real
        return seen

    def test_the_detector_is_given_zero_however_the_spinbox_is_set(self):
        self.assertEqual(self._captured(500)["min_vol"], 0.0)
        self.assertEqual(self._captured(5000)["min_vol"], 0.0)

    def test_the_other_imbalance_controls_are_still_passed_through(self):
        """The decoupling must be surgical: it is one argument, not the lot."""
        kw = self._captured(500)
        self.assertEqual(kw["min_levels"], 3)
        self.assertEqual(kw["imbalance_ratio"], 3.0)
        self.assertEqual(kw["max_depth"], 10)


class TestWhyThatCouplingMattered(unittest.TestCase):
    """The detector itself, shown doing the thing the wiring used to cause.

    A level under min_vol is dropped from the list. When that leaves a side
    with fewer than max_depth levels, the missing ranks are zero-filled, and
    a zero divides into an infinite ratio that counts as fully imbalanced.
    Both cases below are the same book.

    On a real SOXL book, which carries 60-70 levels a side, it takes MinVol
    around 1000 to shorten a side enough to reach the zero-fill; the toy book
    here has three levels a side, so a 500 filter already empties one. The
    toy reaches the same end state by the same route, just sooner.
    """

    BOOK = _book([("BID", 10.00, 900), ("BID",  9.99, 800), ("BID", 9.98, 700),
                  ("ASK", 10.01, 400), ("ASK", 10.02, 300), ("ASK", 10.03, 200)])

    def _run(self, min_vol: float):
        from analysis.orderflow_detect import detect_stacked_imbalance
        return detect_stacked_imbalance(
            self.BOOK, {"T": 0}, 0.03, 9.0, 100,
            min_levels=3, imbalance_ratio=3.0,
            min_vol=min_vol, max_depth=10)

    def test_at_real_sizes_this_book_is_not_imbalanced(self):
        """900/400, 800/300, 700/200 -- none of them reaches 3x."""
        self.assertEqual(self._run(0.0), [])

    def test_a_filter_above_the_ask_side_manufactures_a_full_imbalance(self):
        """Nothing about the book changed. All three ask levels fell under
        the threshold and were dropped, which left that side empty, and an
        empty side is zero-filled rank by rank."""
        out = self._run(500)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0][3], "BID")

    def test_a_filter_above_both_sides_empties_it_again(self):
        """And this is the non-monotonicity: more filtering, fewer markers,
        with a peak in between that no user could have predicted."""
        self.assertEqual(self._run(1000), [])


if __name__ == "__main__":
    unittest.main()

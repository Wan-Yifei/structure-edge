"""Which rows of the tick profile get the gold outline.

The panel used to call this an imbalance highlight and compute it as
dominant / max(minor, 1). It could not mean that: only 8.4% of raw price
levels carry both buy and sell volume, so the divisor was usually the
one-share floor rather than an opposing side, and 72.7% of the time the
three rows drawn were simply the three largest bars.

It now says that outright -- the n heaviest levels per side -- so these
cover the two things that claim can still get wrong: ranking the sides
against each other instead of against themselves, and marking rows where
that side never traded.

No Qt widget is constructed; top_n_levels is a module-level pure function.
"""
from __future__ import annotations

import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from analysis.trade_viewer_qt import top_n_levels    # noqa: E402


class TestTopNLevels(unittest.TestCase):

    def test_it_marks_the_heaviest(self):
        got = top_n_levels(np.array([10.0, 500.0, 30.0, 200.0]), 2)
        self.assertEqual(list(got), [False, True, False, True])

    def test_fewer_traded_levels_than_n_marks_only_those(self):
        """A quiet candle must not outline empty bars -- an outline on a row
        that side never traded reads as 'nothing here' being noteworthy."""
        got = top_n_levels(np.array([0.0, 7.0, 0.0, 0.0]), 3)
        self.assertEqual(list(got), [False, True, False, False])

    def test_a_side_that_never_traded_marks_nothing(self):
        got = top_n_levels(np.zeros(5), 3)
        self.assertFalse(got.any())

    def test_n_larger_than_the_panel_is_harmless(self):
        got = top_n_levels(np.array([4.0, 9.0]), 10)
        self.assertEqual(list(got), [True, True])

    def test_zero_n_marks_nothing(self):
        self.assertFalse(top_n_levels(np.array([4.0, 9.0]), 0).any())

    def test_ties_resolve_to_the_lower_index_every_time(self):
        """Equal volumes must not let the mark wander between redraws of the
        same data -- the panel repaints on every hover."""
        vols = np.array([5.0, 5.0, 5.0, 1.0])
        first = top_n_levels(vols, 2)
        self.assertEqual(list(first), [True, True, False, False])
        for _ in range(5):
            self.assertEqual(list(top_n_levels(vols, 2)), list(first))

    def test_the_mask_is_exactly_n_wide_when_enough_levels_traded(self):
        got = top_n_levels(np.arange(1.0, 21.0), 3)
        self.assertEqual(int(got.sum()), 3)


class TestSidesAreRankedSeparately(unittest.TestCase):
    """The reason the function takes one side at a time rather than both."""

    def test_a_one_sided_candle_still_marks_the_quiet_side(self):
        buys = np.array([9000.0, 8000.0, 7000.0, 10.0])
        sells = np.array([5.0, 0.0, 3.0, 0.0])
        bm = top_n_levels(buys, 2)
        sm = top_n_levels(sells, 2)
        self.assertEqual(list(bm), [True, True, False, False])
        # Pooling the two would have spent both marks on the buy side; per
        # side, the two levels that did sell still get one each.
        self.assertEqual(list(sm), [True, False, True, False])

    def test_the_two_masks_are_independent_not_exclusive(self):
        """A level can be among the heaviest on both sides, and must be able
        to carry both outlines -- the old code picked one side per level."""
        buys = np.array([100.0, 1.0])
        sells = np.array([100.0, 1.0])
        self.assertTrue(top_n_levels(buys, 1)[0])
        self.assertTrue(top_n_levels(sells, 1)[0])


if __name__ == "__main__":
    unittest.main()

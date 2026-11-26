"""The rewrite of _near_touch_cutoff must not move a single bit.

WHY THIS FILE EXISTS. The function had no test of its own, and it is about
to be rewritten for speed: the shipped version calls np.median twice per
iteration over a growing slice, which costs 3.4-3.9 ms per push against a
book of ~96 levels a side. The rewrite keeps the rule and changes only how
it is computed, so "same answer" is the whole specification -- and the only
way to state it is against the old code. The old code is therefore kept
below as `_reference`, and every test here asks whether the two agree.

That also means this file keeps working once the numpy version is gone from
the module: the oracle lives here, not there.

WHAT MAKES THE SAMPLING NON-RANDOM. Measured over 64,020 one-sided calls
drawn from db/order_book.db (US.QQQ and US.SOXL, six days, pre-market
through post-market), the branches are extremely lopsided:

    n > max_keep (60)                       87.3%   QQQ median 96 a side
    a break happens at all                  66.3%   p10 i=5, median 23, p90 72
    the cap actually clamps (keep > 60)     47.5%
    mad == 0 on some iteration              91.0%
    mad == 0 decided the break              42.9% of calls, 64.7% of breaks
    n <= min_keep (the early return)         0.01%  (6 calls in 64,020)
    an empty side                            0%

So a handful of random books would exercise the same path over and over and
would miss the early return altogether -- 40 random one-sided calls have
about a 0.4% chance of hitting it. REAL_BOOKS below is one book per stratum
instead, walked out of 11,983 one-sided calls; three strata (6-10 levels,
11-59 levels, exactly 59) do not occur in the recorded data at all and are
covered synthetically. The seeded differential fuzz at the bottom is what
carries the weight for the float path.

ONE THING THESE TESTS DO NOT SAY. Bit-equivalence is not correctness. The
cutoff drops real liquidity: CHANGELOG v0.84.0 measured 8.1% of the visible
book's volume falling outside the painted band, which is a consequence of
this function plus the +/-1% floor. These tests pin what it does, not that
what it does is right.

INPUT DOMAIN. Prices, so positive and finite. That assumption is load
bearing for one case: with a NaN among an odd number of gaps, np.median
forces NaN through _median_nancheck while a plain sort does not, so the two
implementations would be free to disagree. No such book has been produced
from the feed (60,063 NaN-seeded books were tried without reaching a
divergence, because NaN makes both sides of the `>` comparison False), but
feeds.order_book_merge.parse_side filters on `price <= 0`, which does not
reject NaN. Nothing here asserts behaviour on non-finite input.
"""

import math
import random

import numpy as np
import pytest

from analysis.liq_hm_window import (
    _NEAR_TOUCH_MAD_MULT as MULT,
    _NEAR_TOUCH_MAX_LEVELS as MAXK,
    _NEAR_TOUCH_MIN_LEVELS as MINK,
    _near_touch_cutoff,
)


def _reference(
    prices_from_touch: list[float],
    min_keep: int = MINK,
    max_keep: int = MAXK,
    mad_mult: float = MULT,
) -> list[float]:
    """The shipped implementation as of v0.104.3, kept as the oracle.

    Copied verbatim, numpy and all. Do not "tidy" it: its value is that it
    is the thing the rewrite has to match, so any edit here silently moves
    the target.
    """
    n = len(prices_from_touch)
    if n <= min_keep:
        return prices_from_touch
    gaps = np.abs(np.diff(prices_from_touch))
    keep = min_keep
    for i in range(min_keep - 1, len(gaps)):
        history = gaps[:i + 1]
        med = float(np.median(history))
        mad = float(np.median(np.abs(history - med)))
        threshold = med + mad_mult * mad if mad > 0 else max(med * 5.0, 0.05)
        if gaps[i] > threshold:
            break
        keep = i + 2
    return prices_from_touch[:min(keep, max_keep)]


def _same(prices, **kw):
    """Both implementations, compared element for element.

    `==` on floats is deliberate. The claim under test is that the rewrite
    computes the same numbers, not numbers that are close; a tolerance here
    would pass a version that had quietly changed the rule near a boundary.
    """
    got = _near_touch_cutoff(list(prices), **kw)
    want = _reference(list(prices), **kw)
    assert len(got) == len(want), (
        f"kept {len(got)} levels, reference kept {len(want)}")
    assert got == want, "same length, different prices"
    return got


REAL_BOOKS = [
    ("SOXL_BID_break_at_cap", [
        33.18, 33.16, 33.15, 33.14, 33.13, 33.12, 33.11, 33.1,
        33.09, 33.08, 33.07, 33.06, 33.05, 33.04, 33.03, 33.02,
        33.01, 33.0, 32.99, 32.97, 32.96, 32.95, 32.94, 32.93,
        32.92, 32.9, 32.89, 32.88, 32.87, 32.86, 32.85, 32.84,
        32.83, 32.82, 32.81, 32.8, 32.79, 32.78, 32.77, 32.76,
        32.75, 32.74, 32.73, 32.72, 32.71, 32.7, 32.69, 32.68,
        32.67, 32.66, 32.65, 32.64, 32.63, 32.62, 32.6, 32.59,
        32.58, 32.57, 32.56, 32.55, 32.25, 31.84,
    ]),
    ("QQQ_ASK_break_early", [
        636.63, 636.64, 636.65, 636.66, 636.7, 636.72, 636.8, 636.82,
        636.89, 636.9, 637.0, 637.08, 637.1, 637.13, 637.17, 637.2,
        637.25, 637.26, 637.3, 637.35, 637.4, 637.46, 637.48, 637.49,
        637.5, 637.6, 637.61, 637.67, 637.69, 637.72, 637.73, 637.75,
        637.8, 637.85, 637.88, 637.93, 637.96, 637.98, 637.99, 638.0,
        638.02, 638.03, 638.06, 638.1, 638.15, 638.18, 638.2, 638.25,
        638.33, 638.34, 638.36, 638.4, 638.5, 638.55, 638.62, 638.66,
        638.69, 638.7, 638.8, 638.85, 638.9, 638.97, 639.0, 639.1,
        639.18, 639.2, 639.24, 639.28, 639.3, 639.39, 639.4, 639.44,
        639.47, 639.5, 639.58, 639.65, 639.73, 639.74, 639.79, 639.8,
        639.81, 639.84, 639.89, 639.9, 639.94, 639.99, 640.0, 640.01,
        640.2, 640.3, 640.5, 640.77, 640.97, 641.0, 641.5, 641.8,
        642.0, 642.09, 642.36, 642.5, 642.58, 642.65, 643.0, 643.1,
        643.45, 643.5, 643.88, 644.0, 644.4, 644.47,
    ]),
    ("QQQ_BID_break_first", [
        636.58, 636.54, 636.53, 636.51, 636.5, 636.4, 636.39, 636.38,
        636.31, 636.28, 636.25, 636.2, 636.16, 636.14, 636.1, 636.06,
        636.05, 636.02, 636.01, 636.0, 635.98, 635.82, 635.8, 635.74,
        635.55, 635.54, 635.5, 635.4, 635.3, 635.29, 635.28, 635.16,
        635.12, 635.01, 635.0, 634.98, 634.94, 634.76, 634.6, 634.51,
        634.5, 634.46, 634.44, 634.4, 634.35, 634.34, 634.3, 634.26,
        634.2, 634.1, 634.04, 634.01, 634.0, 633.94, 633.93, 633.9,
        633.86, 633.82, 633.8, 633.76, 633.7, 633.58, 633.56, 633.51,
        633.5, 633.45, 633.4, 633.35, 633.26, 633.2, 633.19, 633.06,
        632.5, 632.0, 631.5, 631.37, 631.25, 631.23, 631.18, 630.92,
        630.6, 630.52, 630.35, 630.3, 630.2, 630.11, 630.0, 629.73,
        629.62, 629.5, 629.26, 629.01, 629.0, 628.85, 628.5, 628.4,
        628.28, 628.0, 627.8, 627.7, 627.51, 627.0, 626.64, 626.28,
        626.12, 626.0, 625.5, 625.3, 625.21, 625.08,
    ]),
    ("QQQ_BID_break_late", [
        636.58, 636.51, 636.5, 636.38, 636.31, 636.28, 636.25, 636.2,
        636.16, 636.14, 636.1, 636.06, 636.05, 636.02, 636.01, 636.0,
        635.98, 635.82, 635.8, 635.74, 635.55, 635.54, 635.5, 635.4,
        635.3, 635.29, 635.28, 635.16, 635.12, 635.01, 635.0, 634.98,
        634.94, 634.76, 634.6, 634.51, 634.5, 634.46, 634.44, 634.4,
        634.35, 634.34, 634.3, 634.26, 634.2, 634.1, 634.04, 634.01,
        634.0, 633.94, 633.93, 633.9, 633.86, 633.82, 633.8, 633.76,
        633.7, 633.58, 633.56, 633.51, 633.5, 633.45, 633.4, 633.35,
        633.26, 633.2, 633.19, 633.06, 633.01, 633.0, 632.9, 632.82,
        632.5, 632.0, 631.5, 631.37, 631.25, 631.23, 631.18, 630.92,
        630.6, 630.52, 630.3, 630.2, 630.11, 630.0, 629.0, 628.5,
        628.4, 628.0, 627.51, 625.5, 625.0, 624.51, 624.0, 623.0,
        622.0, 621.67, 621.64, 621.1, 621.0, 620.0, 619.0, 618.03,
        618.0, 616.62, 616.6, 615.0, 614.0, 613.0, 612.0,
    ]),
    ("QQQ_BID_cap_clamped", [
        636.5, 636.38, 636.31, 636.28, 636.25, 636.2, 636.14, 636.1,
        636.05, 636.02, 636.01, 636.0, 635.98, 635.8, 635.74, 635.55,
        635.54, 635.4, 635.3, 635.29, 635.12, 635.01, 635.0, 634.98,
        634.94, 634.76, 634.6, 634.51, 634.5, 634.46, 634.44, 634.4,
        634.35, 634.34, 634.3, 634.26, 634.2, 634.04, 634.01, 634.0,
        633.94, 633.93, 633.86, 633.82, 633.8, 633.76, 633.7, 633.58,
        633.56, 633.51, 633.5, 633.45, 633.4, 633.35, 633.26, 633.2,
        633.06, 633.01, 633.0, 632.9, 632.82,
    ]),
    ("QQQ_BID_decided_by_mad", [
        636.58, 636.51, 636.5, 636.38, 636.31, 636.28, 636.25, 636.2,
        636.16, 636.14, 636.1, 636.06, 636.05, 636.02, 636.01, 636.0,
        635.98, 635.82, 635.8, 635.74, 635.55, 635.54, 635.5, 635.4,
        635.3, 635.29, 635.28, 635.16, 635.12, 635.01, 635.0, 634.98,
        634.94, 634.76, 634.6, 634.51, 634.5, 634.46, 634.44, 634.4,
        634.35, 634.34, 634.3, 634.26, 634.2, 634.1, 634.04, 634.01,
        634.0, 633.94, 633.93, 633.9, 633.86, 633.82, 633.8, 633.76,
        633.7, 633.58, 633.56, 633.51, 633.5, 633.45, 633.4, 633.35,
        633.26, 633.2, 633.19, 633.06, 633.01, 633.0, 632.9, 632.82,
        632.5, 632.0, 631.5, 631.37, 631.25, 631.23, 631.18, 630.92,
        630.6, 630.52, 630.3, 630.2, 630.11, 630.0, 629.0, 628.5,
        628.4, 628.0, 627.51, 625.5, 625.0, 624.51, 624.0, 623.0,
        622.0, 621.67, 621.64, 621.1, 621.0, 620.0, 619.0, 618.03,
        618.0, 616.62, 616.6, 615.0, 614.0, 613.0, 612.0,
    ]),
    ("SOXL_BID_decided_by_mad0", [
        33.18, 33.16, 33.15, 33.14, 33.13, 33.12, 33.11, 33.1,
        33.09, 33.08, 33.07, 33.06, 33.05, 33.04, 33.03, 33.02,
        33.01, 33.0, 32.99, 32.97, 32.96, 32.95, 32.94, 32.93,
        32.92, 32.9, 32.89, 32.88, 32.87, 32.86, 32.85, 32.84,
        32.83, 32.82, 32.81, 32.8, 32.79, 32.78, 32.77, 32.76,
        32.75, 32.74, 32.73, 32.72, 32.71, 32.7, 32.69, 32.68,
        32.67, 32.66, 32.65, 32.64, 32.63, 32.62, 32.6, 32.59,
        32.58, 32.57, 32.56, 32.55, 32.25, 31.84,
    ]),
    ("SOXL_ASK_n_60", [
        33.19, 33.2, 33.21, 33.22, 33.23, 33.24, 33.25, 33.27,
        33.28, 33.29, 33.3, 33.31, 33.32, 33.33, 33.34, 33.35,
        33.36, 33.37, 33.38, 33.39, 33.4, 33.41, 33.42, 33.43,
        33.44, 33.45, 33.46, 33.47, 33.48, 33.49, 33.5, 33.52,
        33.53, 33.54, 33.55, 33.57, 33.58, 33.59, 33.6, 33.61,
        33.62, 33.63, 33.64, 33.65, 33.66, 33.67, 33.68, 33.69,
        33.7, 33.71, 33.72, 33.73, 33.74, 33.75, 33.76, 33.77,
        33.78, 33.79, 33.8, 33.81,
    ]),
    ("QQQ_BID_n_61", [
        636.5, 636.38, 636.31, 636.28, 636.25, 636.2, 636.14, 636.1,
        636.05, 636.02, 636.01, 636.0, 635.98, 635.8, 635.74, 635.55,
        635.54, 635.4, 635.3, 635.29, 635.12, 635.01, 635.0, 634.98,
        634.94, 634.76, 634.6, 634.51, 634.5, 634.46, 634.44, 634.4,
        634.35, 634.34, 634.3, 634.26, 634.2, 634.04, 634.01, 634.0,
        633.94, 633.93, 633.86, 633.82, 633.8, 633.76, 633.7, 633.58,
        633.56, 633.51, 633.5, 633.45, 633.4, 633.35, 633.26, 633.2,
        633.06, 633.01, 633.0, 632.9, 632.82,
    ]),
    ("SOXL_BID_n_big", [
        33.18, 33.16, 33.15, 33.14, 33.13, 33.12, 33.11, 33.1,
        33.09, 33.08, 33.07, 33.06, 33.05, 33.04, 33.03, 33.02,
        33.01, 33.0, 32.99, 32.97, 32.96, 32.95, 32.94, 32.93,
        32.92, 32.9, 32.89, 32.88, 32.87, 32.86, 32.85, 32.84,
        32.83, 32.82, 32.81, 32.8, 32.79, 32.78, 32.77, 32.76,
        32.75, 32.74, 32.73, 32.72, 32.71, 32.7, 32.69, 32.68,
        32.67, 32.66, 32.65, 32.64, 32.63, 32.62, 32.6, 32.59,
        32.58, 32.57, 32.56, 32.55, 32.25, 31.84,
    ]),
    ("QQQ_BID_n_huge", [
        636.58, 636.51, 636.5, 636.38, 636.31, 636.28, 636.25, 636.2,
        636.16, 636.14, 636.1, 636.06, 636.05, 636.02, 636.01, 636.0,
        635.98, 635.82, 635.8, 635.74, 635.55, 635.54, 635.5, 635.4,
        635.3, 635.29, 635.28, 635.16, 635.12, 635.01, 635.0, 634.98,
        634.94, 634.76, 634.6, 634.51, 634.5, 634.46, 634.44, 634.4,
        634.35, 634.34, 634.3, 634.26, 634.2, 634.1, 634.04, 634.01,
        634.0, 633.94, 633.93, 633.9, 633.86, 633.82, 633.8, 633.76,
        633.7, 633.58, 633.56, 633.51, 633.5, 633.45, 633.4, 633.35,
        633.26, 633.2, 633.19, 633.06, 633.01, 633.0, 632.9, 632.82,
        632.5, 632.0, 631.5, 631.37, 631.25, 631.23, 631.18, 630.92,
        630.6, 630.52, 630.3, 630.2, 630.11, 630.0, 629.0, 628.5,
        628.4, 628.0, 627.51, 625.5, 625.0, 624.51, 624.0, 623.0,
        622.0, 621.67, 621.64, 621.1, 621.0, 620.0, 619.0, 618.03,
        618.0, 616.62, 616.6, 615.0, 614.0, 613.0, 612.0,
    ]),
    ("QQQ_BID_n_small", [
        636.54,
    ]),
    ("SOXL_ASK_no_break", [
        33.19, 33.2, 33.21, 33.22, 33.23, 33.24, 33.25, 33.27,
        33.28, 33.29, 33.3, 33.31, 33.32, 33.33, 33.34, 33.35,
        33.36, 33.37, 33.38, 33.39, 33.4, 33.41, 33.42, 33.43,
        33.44, 33.45, 33.46, 33.47, 33.48, 33.49, 33.5, 33.52,
        33.53, 33.54, 33.55, 33.57, 33.58, 33.59, 33.6, 33.61,
        33.62, 33.63, 33.64, 33.65, 33.66, 33.67, 33.68, 33.69,
        33.7, 33.71, 33.72, 33.73, 33.74, 33.75, 33.76, 33.77,
        33.78, 33.79, 33.8, 33.81,
    ]),
]


class TestRealBooks:
    """One recorded book per stratum. Labels say which branch each one
    reaches; see the module docstring for how often each occurs."""

    @pytest.mark.parametrize("label,prices", REAL_BOOKS,
                             ids=[lbl for lbl, _ in REAL_BOOKS])
    def test_it_matches_the_reference(self, label, prices):
        _same(prices)

    @pytest.mark.parametrize("label,prices", REAL_BOOKS,
                             ids=[lbl for lbl, _ in REAL_BOOKS])
    def test_the_result_is_a_prefix_of_the_input(self, label, prices):
        """Not implied by matching the reference -- it is what the caller
        relies on. _maybe_init_price_range takes min/max of the two sides'
        results and assumes they are still levels from this book, nearest
        to the touch first."""
        kept = _near_touch_cutoff(list(prices))
        assert kept == list(prices)[:len(kept)]


class TestTheStrataThatRealDataDoesNotReach:
    """6-10 levels, 11-59 levels, and exactly 59 never occurred in the
    recorded books, so they are built here.

    Two of the three (anything at or below min_keep) return before the loop
    and touch no floating point at all; the third does run the loop, and the
    fuzz below is what covers it in bulk."""

    def test_an_empty_side(self):
        """A merged snapshot can legitimately be one-sided, and the caller
        guards for the result being empty -- so this path is reachable even
        though no recorded book shows it."""
        assert _same([]) == []

    def test_a_single_level(self):
        assert _same([141.0]) == [141.0]

    def test_exactly_min_keep(self):
        p = [141.0 - 0.01 * i for i in range(MINK)]
        assert _same(p) == p

    def test_one_past_min_keep(self):
        """The first length that enters the loop."""
        p = [141.0 - 0.01 * i for i in range(MINK + 1)]
        _same(p)

    @pytest.mark.parametrize("n", [6, 7, 10, 11, 30, 58, 59, 60, 61, 62])
    def test_a_uniform_ladder_of_each_awkward_length(self, n):
        """Every gap identical, which is the mad == 0 branch -- 93.3% of
        iterations in real data, because a penny grid makes the gap multiset
        a pile of 0.01s."""
        _same([141.0 - 0.01 * i for i in range(n)])

    def test_a_ladder_with_one_far_outlier(self):
        """The shape the cutoff exists for: a stub order sitting well away
        from the touch."""
        p = [141.0 - 0.01 * i for i in range(20)] + [120.0]
        kept = _same(p)
        assert 120.0 not in kept, "the outlier was kept"

    def test_gaps_that_widen_smoothly_are_not_cut(self):
        """A book that thins out gradually has no outlier to find."""
        p, x = [141.0], 141.0
        for i in range(1, 25):
            x -= 0.01 * (1 + i * 0.05)
            p.append(x)
        _same(p)


class TestTheKnobsStillDoSomething:
    """Passing non-default parameters has to reach the same code. A rewrite
    that hard-coded the module constants would pass everything above."""

    @pytest.mark.parametrize("min_keep", [2, 3, 5, 9])
    def test_min_keep(self, min_keep):
        _same([141.0 - 0.01 * i for i in range(30)] + [130.0],
              min_keep=min_keep)

    @pytest.mark.parametrize("max_keep", [5, 7, 12, 60, 200])
    def test_max_keep(self, max_keep):
        kept = _same([141.0 - 0.01 * i for i in range(80)],
                     max_keep=max_keep)
        assert len(kept) <= max_keep

    @pytest.mark.parametrize("mad_mult", [0.0, 1.0, 8.0, 50.0])
    def test_mad_mult(self, mad_mult):
        p, x = [141.0], 141.0
        for i in range(1, 40):
            x -= 0.01 if i % 7 else 0.08
            p.append(x)
        _same(p, mad_mult=mad_mult)


class TestDifferentialFuzz:
    """The bulk of the float coverage, and the part that keeps working as
    the module changes.

    Seeded, so a failure is reproducible from the id alone. The generator
    is deliberately tie-heavy: real books sit on a one-cent grid, so the gap
    multiset is dominated by repeats and the two implementations' median
    paths get exercised on exactly the inputs where a sort and a partition
    could disagree about which element is "the middle" one.
    """

    @staticmethod
    def _book(rng):
        n = rng.choice([rng.randint(0, 8), rng.randint(9, 70),
                        rng.randint(71, 130)])
        tick = rng.choice([0.01, 0.01, 0.01, 0.05, 0.25, 1.0])
        x = rng.uniform(5.0, 600.0)
        out = [x]
        for _ in range(max(0, n - 1)):
            step = tick * rng.choice([1, 1, 1, 1, 2, 3, 17, 400])
            x -= step
            if x <= 0:
                break
            out.append(x)
        return out

    @pytest.mark.parametrize("seed", range(40))
    def test_two_hundred_books_per_seed(self, seed):
        rng = random.Random(seed)
        for _ in range(200):
            book = self._book(rng)
            assert all(math.isfinite(p) and p > 0 for p in book)
            _same(book)

    @pytest.mark.parametrize("seed", range(10))
    def test_books_that_are_one_long_flat_ladder(self, seed):
        """mad == 0 the whole way down, so the fallback threshold decides
        every iteration -- 64.7% of real breaks come from that branch."""
        rng = random.Random(1000 + seed)
        tick = rng.choice([0.01, 0.05])
        n = rng.randint(2, 120)
        x = rng.uniform(10.0, 500.0)
        _same([x - tick * i for i in range(n)])

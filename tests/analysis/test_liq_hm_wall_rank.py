"""Which walls get a label: distance from the touch, with two exceptions.

Ranking moved off brightness. Brightness (with Min.Vol) still decides what is
a wall at all, but among walls the nearest to the touch win the scarce label
slots -- a big wall a dollar away does not help a decision being made at the
touch, and the heatmap still paints its band either way.

Two things complicate that, and both are here because the obvious version of
each was wrong:

  incumbency  a label that slips out of the nearest top_n keeps its place for
              a couple of ranks, or price oscillating between two levels
              flickers labels on and off. Granting EXTRA slots instead does
              nothing: the nearest top_n always fill the budget, so the
              straggler is never drawn.
  the giant   a level that dwarfs everything in view keeps a label wherever
              it is. It has to be placed FIRST, or a full slate of near walls
              crowds it out before it is reached.

These exercise the selection arithmetic directly, which is where both bugs
were; the drawing itself is a few lines below it.
"""

import pytest

from analysis.liq_hm_window import (          # noqa: E402
    _WALL_RANK_SLACK, _WALL_TOP1_RATIO,
)

BIN = 0.01
PMIN = 100.0


def pick(levels, mid, top_n=3, prev=frozenset()):
    """The selection as the window runs it: (distance, bin, peak) -> labels.

    `levels` is {bin_index: peak_size}; everything given is assumed to have
    passed Min.Vol and the brightness floor already.
    """
    cands = []
    for bi, v in levels.items():
        p = PMIN + (bi + 0.5) * BIN
        cands.append((abs(p - mid), bi, float(v)))
    cands.sort(key=lambda c: c[0])

    def _eff(item):
        rank, c = item
        p = PMIN + (c[1] + 0.5) * BIN
        return rank - (_WALL_RANK_SLACK if round(p, 4) in prev else 0)

    chosen = [c for _, c in sorted(enumerate(cands), key=_eff)[:top_n]]
    if len(cands) >= 2:
        by_size = sorted(cands, key=lambda c: -c[2])
        if by_size[0][2] >= _WALL_TOP1_RATIO * max(by_size[1][2], 1.0):
            if by_size[0] in chosen:
                chosen.remove(by_size[0])
            chosen.insert(0, by_size[0])

    out, placed = [], []
    for _, bi, v in chosen:
        if len(placed) >= top_n + 1:
            break
        if any(abs(bi - q) <= 1 for q in placed):
            continue
        placed.append(bi)
        out.append((round(PMIN + (bi + 0.5) * BIN, 4), v))
    return out


def prices(labels):
    return [p for p, _ in labels]


class TestDistanceWins:
    def test_the_nearest_walls_are_labelled(self):
        """A far giant does not displace near walls -- it joins them."""
        got = pick({0: 50_000, 40: 2_000, 44: 2_000, 48: 2_000}, mid=100.50)
        assert 100.485 in prices(got)        # bin 48, nearest
        assert 100.445 in prices(got)
        assert 100.405 in prices(got)

    def test_a_far_big_wall_loses_to_near_ones_when_not_dominant(self):
        """Three near walls of 2,000 against one of 5,000 -- under 3x."""
        got = pick({0: 5_000, 40: 2_000, 44: 2_000, 48: 2_000}, mid=100.50)
        assert 100.005 not in prices(got)

    def test_brightness_order_no_longer_decides(self):
        """The biggest is furthest; it must not come first on size alone."""
        got = pick({0: 9_000, 46: 4_000, 48: 4_000}, mid=100.50, top_n=2)
        assert prices(got)[:2] == [100.485, 100.465] or \
            set(prices(got)) >= {100.485, 100.465}


class TestTheGiant:
    def test_a_dominant_far_wall_keeps_its_label(self):
        """The 2026-10-14 116.00 case: 73k a dollar away, near book thin."""
        got = pick({0: 73_000, 40: 2_000, 44: 2_000, 48: 2_000}, mid=100.50)
        assert 100.005 in prices(got)
        assert len(got) == 4                 # top_n near, plus the giant

    def test_it_is_placed_even_with_a_full_slate(self):
        """Appending it last let a full slate crowd it out -- the bug."""
        got = pick({0: 73_000, 40: 2_000, 42: 2_000, 44: 2_000, 46: 2_000,
                    48: 2_000}, mid=100.50)
        assert prices(got)[0] == 100.005

    def test_just_under_the_ratio_does_not_qualify(self):
        v = int(2_000 * _WALL_TOP1_RATIO) - 1
        got = pick({0: v, 40: 2_000, 44: 2_000, 48: 2_000}, mid=100.50)
        assert 100.005 not in prices(got)

    def test_at_the_ratio_it_does(self):
        v = int(2_000 * _WALL_TOP1_RATIO)
        got = pick({0: v, 40: 2_000, 44: 2_000, 48: 2_000}, mid=100.50)
        assert 100.005 in prices(got)

    def test_a_lone_wall_needs_no_exception(self):
        """Nothing to dwarf; the distance rule already labels it."""
        got = pick({0: 73_000}, mid=100.50)
        assert prices(got) == [100.005]


class TestIncumbency:
    def test_a_label_that_slips_one_rank_survives(self):
        """Bins are spaced: adjacent ones collapse under one-wall-one-label."""
        near = {40: 2_000, 44: 2_000, 48: 2_000, 52: 2_000}
        # By distance alone bin 40 is rank 3 and falls outside top_n=3.
        assert 100.405 not in prices(pick(near, mid=100.50))
        # As an incumbent it keeps its place.
        held = prices(pick(near, mid=100.50, top_n=3, prev={100.405}))
        assert 100.405 in held

    def test_it_is_dropped_once_it_slips_past_the_slack(self):
        levels = {i: 2_000 for i in (30, 40, 44, 48, 52, 56)}
        prev = {100.305}          # rank 5, past top_n + slack
        held = prices(pick(levels, mid=100.50, top_n=3, prev=prev))
        assert 100.305 not in held

    def test_incumbency_does_not_add_slots(self):
        levels = {i: 2_000 for i in (40, 44, 48, 52)}
        prev = {round(PMIN + (i + 0.5) * BIN, 4) for i in (40, 44, 48, 52)}
        got = pick(levels, mid=100.50, top_n=3, prev=prev)
        assert len(got) == 3

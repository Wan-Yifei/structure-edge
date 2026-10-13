"""High-volume nodes, and why they are not the value area.

The value area is one band around the POC covering a fixed share of volume. On
a session that traded two separate shelves it stretches across the empty gap
between them, so its VAL can land at a price where nothing changed hands. A
node is local -- two shelves give two nodes, each with its own edges.

That distinction is the whole point of the function, so most of these tests are
about the two-shelf case.
"""

import numpy as np
import pytest

from strategy.session_vp.profile import compute_value_area, find_nodes, in_node


def profile(vols, lo=100.0, step=1.0):
    """Evenly spaced bins holding *vols*, returned as (edges, bin_vols)."""
    v = np.asarray(vols, dtype=float)
    return lo + step * np.arange(len(v) + 1), v


class TestOneShelf:
    def test_a_single_peak_is_one_node(self):
        e, v = profile([1, 2, 8, 10, 8, 2, 1])
        n = find_nodes(e, v)
        assert len(n) == 1
        lo, hi, share = n[0]
        assert lo < hi
        assert 0 < share <= 1

    def test_its_edges_sit_at_half_height(self):
        # peak 10 at index 3; bins >= 5 are indices 2,3,4
        e, v = profile([1, 2, 8, 10, 8, 2, 1])
        lo, hi, _ = find_nodes(e, v)[0]
        assert lo == pytest.approx(e[2])
        assert hi == pytest.approx(e[5])

    def test_two_peaks_in_one_shelf_merge(self):
        """A double top inside one cluster is one node, not two -- otherwise
        the spans overlap almost entirely and the nearest edge is arbitrary."""
        e, v = profile([1, 9, 7, 10, 9, 1])
        assert len(find_nodes(e, v)) == 1


class TestTwoShelves:
    """The case the value area cannot describe."""

    GAP = [10, 9, 1, 0, 0, 1, 9, 10]

    def test_two_nodes_are_found(self):
        e, v = profile(self.GAP)
        assert len(find_nodes(e, v)) == 2

    def test_they_do_not_overlap(self):
        e, v = profile(self.GAP)
        (lo1, hi1, _), (lo2, hi2, _) = find_nodes(e, v)
        assert hi1 <= lo2

    def test_the_gap_is_inside_the_value_area_but_in_no_node(self):
        e, v = profile(self.GAP)
        va = compute_value_area(e, v, va_pct=0.95)
        mid = (e[4] + e[5]) / 2          # the empty middle
        assert va["val"] < mid < va["vah"], "value area spans the gap"
        assert not in_node(mid, find_nodes(e, v)), "no node holds it"

    def test_a_shelf_edge_is_not_the_value_area_low(self):
        e, v = profile(self.GAP)
        va = compute_value_area(e, v, va_pct=0.70)
        upper_lo = find_nodes(e, v)[1][0]
        assert abs(upper_lo - va["val"]) > 1.0


class TestThresholds:
    def test_a_ripple_below_the_floor_is_not_a_node(self):
        e, v = profile([10, 9, 1, 2, 1, 0, 0])   # 2 is a peak but tiny
        assert len(find_nodes(e, v, min_peak_frac=0.35)) == 1

    def test_lowering_the_floor_admits_it(self):
        e, v = profile([10, 9, 1, 2, 1, 0, 0])
        assert len(find_nodes(e, v, min_peak_frac=0.15)) == 2

    def test_shares_are_fractions_of_total_volume(self):
        e, v = profile(self.__class__ and [10, 9, 1, 0, 1, 9, 10])
        for _, _, share in find_nodes(e, v):
            assert 0 < share < 1


class TestDegenerate:
    @pytest.mark.parametrize("vols", [[], [5], [5, 5]])
    def test_too_few_bins(self, vols):
        e, v = profile(vols) if vols else (None, np.array([]))
        assert find_nodes(e, v) == []

    def test_all_zero(self):
        e, v = profile([0, 0, 0, 0, 0])
        assert find_nodes(e, v) == []

    def test_none_inputs(self):
        assert find_nodes(None, None) == []

    def test_flat_profile_has_no_peak(self):
        """Every bin equal: nothing stands out, so nothing is a node."""
        e, v = profile([5, 5, 5, 5, 5])
        assert find_nodes(e, v) == []


class TestInNode:
    def test_edges_count_as_inside(self):
        e, v = profile([1, 2, 8, 10, 8, 2, 1])
        lo, hi, _ = find_nodes(e, v)[0]
        assert in_node(lo, find_nodes(e, v))
        assert in_node(hi, find_nodes(e, v))

    def test_outside_is_outside(self):
        e, v = profile([1, 2, 8, 10, 8, 2, 1])
        n = find_nodes(e, v)
        assert not in_node(e[0] - 5, n)
        assert not in_node(e[-1] + 5, n)

    def test_no_nodes_means_nothing_is_inside(self):
        assert not in_node(100.0, [])

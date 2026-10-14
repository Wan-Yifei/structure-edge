"""High-volume shelves, and the three things that made the old version wrong.

The value area is one band around the POC covering a fixed share of volume. On
a session that traded two separate shelves it stretches across the empty gap
between them, so its VAL can land at a price where nothing changed hands. A
node is local -- two shelves give two nodes, each with its own edges.

The first version of find_nodes read peaks straight off a histogram binned
into the session's range, which made every answer a function of an arbitrary
bin count: SOXL 2026-10-12 reported one shelf at 30 bins and two at 100. The
tests that matter most here are therefore the ones about scale -- a shelf the
detector only finds at one resolution is not a shelf.
"""

import numpy as np
import pytest

from strategy.session_vp.profile import (
    Node, bins_for_price, compute_value_area, find_nodes, in_node,
)


def profile(vols, lo=100.0, step=1.0):
    """Evenly spaced bins holding *vols*, returned as (edges, bin_vols)."""
    v = np.asarray(vols, dtype=float)
    return lo + step * np.arange(len(v) + 1), v


def two_humps(n_bins, lo=100.0, hi=110.0, centres=(102.0, 108.0), sd=0.8):
    """Two well-separated Gaussian shelves sampled at *n_bins* resolution.

    The density is a property of price, not of the binning, so any honest
    detector has to return the same shelves whatever n_bins is.
    """
    edges = np.linspace(lo, hi, n_bins + 1)
    mid = (edges[:-1] + edges[1:]) / 2
    v = sum(np.exp(-0.5 * ((mid - c) / sd) ** 2) for c in centres)
    return edges, v * (1000.0 / n_bins)


class TestScaleInvariance:
    """The bug the rewrite exists for."""

    @pytest.mark.parametrize("n_bins", [100, 200, 400, 800])
    def test_the_same_density_gives_the_same_shelves(self, n_bins):
        nodes = find_nodes(*two_humps(n_bins))
        assert len(nodes) == 2
        assert nodes[0].peak == pytest.approx(102.0, abs=0.15)
        assert nodes[1].peak == pytest.approx(108.0, abs=0.15)

    def test_edges_agree_across_resolutions(self):
        runs = [find_nodes(*two_humps(n)) for n in (100, 200, 400, 800)]
        for i in range(2):
            los = [r[i].lo for r in runs]
            his = [r[i].hi for r in runs]
            # a tenth of the 10-unit range would be a different answer; this
            # is tighter than one bin at the coarsest resolution
            assert max(los) - min(los) < 0.15, f"shelf {i} low edge drifts"
            assert max(his) - min(his) < 0.15, f"shelf {i} high edge drifts"


class TestOneShelf:
    def test_a_single_peak_is_one_node(self):
        e, v = profile([1, 2, 8, 10, 8, 2, 1])
        n = find_nodes(e, v)
        assert len(n) == 1
        assert n[0].lo < n[0].peak < n[0].hi
        assert 0 < n[0].share <= 1

    def test_its_edges_sit_at_half_height(self):
        # peak 10 at index 3; bins >= 5 are indices 2, 3, 4
        e, v = profile([1, 2, 8, 10, 8, 2, 1])
        node = find_nodes(e, v)[0]
        assert node.lo == pytest.approx(e[2])
        assert node.hi == pytest.approx(e[5])

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

    def test_shelves_at_the_profile_edge_are_still_shelves(self):
        """Both peaks here sit on the first and last bin. find_peaks ignores
        the endpoints, so without padding this case returns nothing at all."""
        e, v = profile(self.GAP)
        nodes = find_nodes(e, v)
        assert nodes[0].lo == pytest.approx(e[0])
        assert nodes[1].hi == pytest.approx(e[-1])

    def test_they_do_not_overlap(self):
        e, v = profile(self.GAP)
        a, b = find_nodes(e, v)
        assert a.hi <= b.lo

    def test_the_gap_is_inside_the_value_area_but_in_no_node(self):
        e, v = profile(self.GAP)
        va = compute_value_area(e, v, va_pct=0.95)
        mid = (e[4] + e[5]) / 2          # the empty middle
        assert va["val"] < mid < va["vah"], "value area spans the gap"
        assert not in_node(mid, find_nodes(e, v)), "no node holds it"

    def test_a_shelf_edge_is_not_the_value_area_low(self):
        e, v = profile(self.GAP)
        va = compute_value_area(e, v, va_pct=0.70)
        assert abs(find_nodes(e, v)[1].lo - va["val"]) > 1.0


class TestProminence:
    """Lowering the threshold must split shelves, never join them.

    Bounding a shelf at half height alone let a tall peak's skirt run through
    the dip and up into its neighbour; the spans then overlapped and merged,
    so a *lower* threshold produced one shelf spanning the lot.
    """

    def test_a_lower_threshold_never_returns_fewer_shelves(self):
        e, v = two_humps(400)
        counts = [len(find_nodes(e, v, prominence=p))
                  for p in (0.10, 0.25, 0.40, 0.60)]
        assert counts == sorted(counts, reverse=True)

    def test_a_shallow_dip_does_not_split_a_shelf(self):
        e, v = two_humps(400, centres=(104.6, 105.4), sd=0.8)
        assert len(find_nodes(e, v)) == 1

    def test_a_deep_dip_does(self):
        e, v = two_humps(400, centres=(102.0, 108.0), sd=0.8)
        assert len(find_nodes(e, v)) == 2


class TestVolumeFloor:
    def test_a_shelf_below_the_floor_is_dropped(self):
        e, v = two_humps(400, centres=(102.0, 108.0), sd=0.8)
        mid = (e[:-1] + e[1:]) / 2
        v = v + 0.0
        v[mid > 105] *= 0.04            # starve the upper shelf
        assert len(find_nodes(e, v, min_share=0.05)) == 1

    def test_lowering_the_floor_admits_it(self):
        e, v = two_humps(400, centres=(102.0, 108.0), sd=0.8)
        mid = (e[:-1] + e[1:]) / 2
        v[mid > 105] *= 0.04
        assert len(find_nodes(e, v, min_share=0.001)) == 2

    def test_shares_are_fractions_of_total_volume(self):
        e, v = two_humps(400)
        assert all(0 < n.share < 1 for n in find_nodes(e, v))


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
        """Every bin equal: nothing stands out, so nothing is a node. Without
        an explicit guard find_peaks calls the middle of the plateau a peak
        and the whole range comes back as one shelf."""
        e, v = profile([5, 5, 5, 5, 5])
        assert find_nodes(e, v) == []


class TestBinsForPrice:
    def test_bin_width_tracks_price_not_range(self):
        """A quiet session and a wild one must get the same bin width, which
        is the whole reason this helper exists."""
        quiet = bins_for_price(127.0, 129.0)
        wild = bins_for_price(127.0, 137.0)
        w_quiet = 2.0 / quiet
        w_wild = 10.0 / wild
        assert w_quiet == pytest.approx(w_wild, rel=0.05)

    def test_width_is_the_requested_share_of_price(self):
        n = bins_for_price(100.0, 110.0, bin_pct=0.02)
        assert 10.0 / n == pytest.approx(105.0 * 0.0002, rel=0.05)

    @pytest.mark.parametrize("lo,hi", [(0.0, 10.0), (10.0, 10.0), (10.0, 1.0)])
    def test_degenerate_ranges(self, lo, hi):
        assert bins_for_price(lo, hi) == 0


class TestInNode:
    def test_edges_count_as_inside(self):
        e, v = profile([1, 2, 8, 10, 8, 2, 1])
        n = find_nodes(e, v)
        assert in_node(n[0].lo, n)
        assert in_node(n[0].hi, n)

    def test_outside_is_outside(self):
        e, v = profile([1, 2, 8, 10, 8, 2, 1])
        n = find_nodes(e, v)
        assert not in_node(e[0] - 5, n)
        assert not in_node(e[-1] + 5, n)

    def test_no_nodes_means_nothing_is_inside(self):
        assert not in_node(100.0, [])


class TestNodeShape:
    def test_fields_are_named(self):
        e, v = profile([1, 2, 8, 10, 8, 2, 1])
        n = find_nodes(e, v)[0]
        assert isinstance(n, Node)
        assert (n.lo, n.hi, n.peak, n.share) == tuple(n)

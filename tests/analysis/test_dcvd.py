"""DCVD: the CVD slope normalised to [-1, +1].

CVD's slope over N bars is the net delta across them; the largest slope those
bars could have produced is every classified trade landing on one side. The
ratio is therefore bounded by construction, not by a fitted scale -- which is
the whole point, since it means 0.4 reads the same on a quiet day as a busy one.

NEUTRAL volume is excluded from both halves, matching the CVD delta. It is the
majority of SOXL's tape (57.5% measured), and including it would pin the median
bar near 0.08 and leave the range beyond +/-0.5 unused.
"""

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6.QtWidgets")

from PyQt6.QtWidgets import QApplication   # noqa: E402

CM = 1
T0 = datetime(2026, 10, 5, 10, 0)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app
    app.processEvents()


@pytest.fixture(scope="module")
def viewer(qapp):
    from analysis.trade_viewer_qt import TradeViewerQt

    v = TradeViewerQt()
    v.resize(1400, 900)
    v.show()
    for _ in range(30):
        qapp.processEvents()
    yield v
    v.close()
    qapp.processEvents()


def _bars(flows):
    """flows: list of (buy, sell, neutral) per bar -> (klines, buckets)."""
    from core.time_utils import candle_start

    rows, buckets = [], {}
    for i, (b, s, n) in enumerate(flows):
        bar_end = T0 + timedelta(minutes=CM * (i + 1))
        rows.append((bar_end.strftime("%Y-%m-%d %H:%M"),
                     100.0, 101.0, 99.0, 100.0, float(b + s + n)))
        bk = candle_start(bar_end - timedelta(minutes=CM), CM)
        if b or s or n:
            buckets[bk] = {100.0: {"buy": float(b), "sell": float(s),
                                   "neutral": float(n)}}
    kl = pd.DataFrame(rows, columns=["time_key", "open", "high", "low",
                                     "close", "volume"])
    return kl, buckets


def _run(v, flows):
    kl, buckets = _bars(flows)
    v._draw_dcvd(kl, buckets, CM)
    return np.asarray(v._dcvd_arr, dtype=float)


class TestBounds:
    def test_all_buying_pins_at_plus_one(self, viewer):
        assert _run(viewer, [(100, 0, 0)] * 20)[-1] == pytest.approx(1.0)

    def test_all_selling_pins_at_minus_one(self, viewer):
        assert _run(viewer, [(0, 100, 0)] * 20)[-1] == pytest.approx(-1.0)

    def test_balanced_flow_is_zero(self, viewer):
        assert _run(viewer, [(50, 50, 0)] * 20)[-1] == pytest.approx(0.0)

    def test_never_leaves_the_range(self, viewer):
        rng = np.random.default_rng(0)
        flows = [(int(rng.integers(0, 5000)), int(rng.integers(0, 5000)),
                  int(rng.integers(0, 9000))) for _ in range(300)]
        d = _run(viewer, flows)
        assert d.min() >= -1.0 and d.max() <= 1.0

    def test_a_single_huge_bar_cannot_push_it_out(self, viewer):
        """An unnormalised slope would spike; this cannot exceed 1 by design."""
        flows = [(10, 10, 0)] * 19 + [(10_000_000, 0, 0)]
        assert _run(viewer, flows)[-1] <= 1.0


class TestNormalisation:
    def test_scale_invariant(self, viewer):
        """Ten times the volume, same imbalance, same reading -- the property
        an unnormalised slope lacks."""
        small = _run(viewer, [(70, 30, 0)] * 20)[-1]
        large = _run(viewer, [(700, 300, 0)] * 20)[-1]
        assert small == pytest.approx(large)
        assert small == pytest.approx(0.4)

    def test_equals_the_slope_over_the_max_possible_slope(self, viewer):
        from analysis.trade_viewer_qt import DCVD_LOOKBACK

        flows = [(80, 20, 0), (60, 40, 0), (90, 10, 0)] * 10
        d = _run(viewer, flows)
        buy = np.array([f[0] for f in flows], dtype=float)
        sell = np.array([f[1] for f in flows], dtype=float)
        i = len(flows) - 1
        j = max(0, i + 1 - DCVD_LOOKBACK)
        slope = (buy[j:i + 1] - sell[j:i + 1]).sum()
        cap = (buy[j:i + 1] + sell[j:i + 1]).sum()
        assert d[-1] == pytest.approx(slope / cap)

    def test_neutral_is_excluded_from_both_halves(self, viewer):
        """Adding neutral volume must not move the reading."""
        without = _run(viewer, [(70, 30, 0)] * 20)[-1]
        with_n = _run(viewer, [(70, 30, 100_000)] * 20)[-1]
        assert without == pytest.approx(with_n)


class TestLookback:
    def test_only_the_last_N_bars_count(self, viewer):
        from analysis.trade_viewer_qt import DCVD_LOOKBACK

        flows = [(100, 0, 0)] * 50 + [(0, 100, 0)] * DCVD_LOOKBACK
        assert _run(viewer, flows)[-1] == pytest.approx(-1.0), (
            "older buying must have rolled out of the window")

    def test_turns_over_as_the_window_fills(self, viewer):
        from analysis.trade_viewer_qt import DCVD_LOOKBACK

        n = DCVD_LOOKBACK
        d = _run(viewer, [(100, 0, 0)] * n + [(0, 100, 0)] * n)
        assert d[n - 1] == pytest.approx(1.0)
        assert d[2 * n - 1] == pytest.approx(-1.0)
        assert d[n + n // 2 - 1] < d[n - 1], "must be falling through the flip"

    def test_short_history_uses_what_there_is(self, viewer):
        d = _run(viewer, [(100, 0, 0)] * 3)
        assert len(d) == 3 and d[-1] == pytest.approx(1.0)


class TestGaps:
    def test_bars_without_ticks_contribute_nothing(self, viewer):
        """A coverage gap must not read as balanced flow."""
        flows = [(100, 0, 0)] * 5 + [(0, 0, 0)] * 3
        assert _run(viewer, flows)[-1] == pytest.approx(1.0)

    def test_no_ticks_at_all_is_flat_zero(self, viewer):
        d = _run(viewer, [(0, 0, 0)] * 10)
        assert np.all(d == 0.0)

    def test_neutral_only_bars_are_not_directional(self, viewer):
        d = _run(viewer, [(0, 0, 5000)] * 10)
        assert np.all(d == 0.0)


class TestWiring:
    def test_series_is_published_for_the_crosshair(self, viewer):
        d = _run(viewer, [(60, 40, 0)] * 20)
        assert viewer._dcvd_arr is not None
        assert len(viewer._dcvd_arr) == 20 and d[-1] == pytest.approx(0.2)

    def test_clearing_drops_the_series(self, viewer):
        _run(viewer, [(60, 40, 0)] * 20)
        viewer._clear_dcvd_items()
        assert viewer._dcvd_arr is None and viewer._dcvd_items == []

    def test_subplot_axis_is_pinned_to_the_bounds(self, viewer):
        """A fitted y-range would undo the normalisation's whole purpose."""
        lo, hi = viewer._plot_dcvd.vb.viewRange()[1]
        assert lo <= -1.0 and hi >= 1.0 and hi - lo < 3.0

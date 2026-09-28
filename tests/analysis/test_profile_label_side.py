"""Profile-panel level labels sit on the right edge, not the left.

Both profile panels draw horizontal bars growing rightward from x=0, so the
left edge is the one x the bars all pass through -- a label pinned there sits
on top of the histogram. The X range carries 15% headroom past the longest
bar, which leaves the right edge clear.

These drive the real widgets offscreen and read the placed items back, because
the thing being asserted is geometry: an anchor change without the matching x
coordinate (or the reverse) still renders, just in the wrong place.
"""

import os
from datetime import datetime, timedelta

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6.QtWidgets")

import pyqtgraph as pg                     # noqa: E402
from PyQt6.QtWidgets import QApplication   # noqa: E402

T0 = datetime(2026, 9, 25, 9, 30)
HEADROOM = 1.15   # _rebuild_session_profile / _draw_tick_vp pin X to this


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app
    app.processEvents()


@pytest.fixture(scope="module")
def viewer(qapp):
    from analysis.trade_viewer_qt import TradeViewerQt

    v = TradeViewerQt()
    v.resize(1500, 900)
    v.show()
    for _ in range(30):
        qapp.processEvents()
    yield v
    v.close()
    qapp.processEvents()


def _klines(n=60):
    """Bars with a volume bulge in the middle so POC/VAH/VAL land apart."""
    rows = []
    for i in range(n):
        lo = 100.0 + (i % 10) * 0.5
        hi = lo + 2.0
        vol = 1000.0 + (500.0 if 3 <= (i % 10) <= 5 else 0.0)
        rows.append(((T0 + timedelta(minutes=5 * i)).strftime("%Y-%m-%d %H:%M"),
                     (lo + hi) / 2, hi, lo, (lo + hi) / 2, vol))
    return pd.DataFrame(
        rows, columns=["time_key", "open", "high", "low", "close", "volume"])


def _labels(pw):
    """Every non-empty TextItem in a plot widget, by its text."""
    out = {}
    for item in pw.getPlotItem().items:
        if isinstance(item, pg.TextItem):
            txt = item.textItem.toPlainText()
            if txt:
                out[txt.split()[0]] = item
    return out


@pytest.fixture()
def session_panel(viewer, qapp):
    viewer._klines = _klines()
    viewer._ticks = {}
    viewer._candle_mins = 5
    viewer._rebuild_session_profile()
    for _ in range(10):
        qapp.processEvents()
    return viewer._profile_widget


class TestSessionProfilePanel:
    def test_every_level_label_is_drawn(self, session_panel):
        got = set(_labels(session_panel))
        assert {"POC1", "VAH", "VAL", "Last"} <= got, got

    def test_all_of_them_sit_on_the_right_edge(self, session_panel):
        xhi = session_panel.getViewBox().viewRange()[0][1]
        for tag, item in _labels(session_panel).items():
            assert item.pos().x() == pytest.approx(xhi), tag

    def test_all_of_them_are_right_anchored(self, session_panel):
        """x alone is not enough: a left-anchored label placed at xhi grows
        outward past the edge and gets clipped."""
        for tag, item in _labels(session_panel).items():
            assert item.anchor.x() == 1.0, tag

    def test_the_right_edge_clears_the_longest_bar(self, session_panel):
        """The headroom is what makes this spot empty; without it the labels
        would land on the POC bar instead."""
        vb = session_panel.getViewBox()
        xlo, xhi = vb.viewRange()[0]
        assert xlo == 0.0
        longest = max(
            float(it.opts["x1"].max())
            for it in session_panel.getPlotItem().items
            if isinstance(it, pg.BarGraphItem)
        )
        assert xhi > longest
        assert xhi == pytest.approx(longest * HEADROOM)

    def test_labels_follow_a_zoom(self, session_panel, qapp):
        """They are pinned through sigRangeChanged, not placed once."""
        vb = session_panel.getViewBox()
        vb.setXRange(0, 500, padding=0)
        for _ in range(5):
            qapp.processEvents()
        xhi = vb.viewRange()[0][1]
        for tag, item in _labels(session_panel).items():
            assert item.pos().x() == pytest.approx(xhi), tag


class TestTickVpPanel:
    """The single-candle profile mode, which places its labels once against a
    fixed X range rather than through a pin callback."""

    @pytest.fixture()
    def panel(self, viewer, qapp):
        prices = [100.0 + i * 0.1 for i in range(20)]
        buys = [10.0 + (50.0 if 8 <= i <= 11 else 0.0) for i in range(20)]
        sells = [8.0] * 20
        neutrals = [2.0] * 20
        viewer._tick_profile_widget.clear()
        viewer._draw_tick_vp(prices, buys, sells, neutrals, 0.1, "test")
        for _ in range(10):
            qapp.processEvents()
        return viewer._tick_profile_widget

    def test_levels_are_drawn(self, panel):
        assert {"POC", "VAH", "VAL"} <= set(_labels(panel))

    def test_right_anchored_at_the_right_edge(self, panel):
        xhi = panel.getViewBox().viewRange()[0][1]
        for tag, item in _labels(panel).items():
            assert item.anchor.x() == 1.0, tag
            assert item.pos().x() == pytest.approx(xhi), tag

    def test_placed_past_the_longest_bar(self, panel):
        longest = max(
            float(it.opts["x1"].max())
            for it in panel.getPlotItem().items
            if isinstance(it, pg.BarGraphItem)
        )
        for tag, item in _labels(panel).items():
            assert item.pos().x() > longest, tag

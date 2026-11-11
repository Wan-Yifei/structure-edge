"""The crosshair tags stay inside the chart at the top and bottom edges.

A ViewBox clips its children to its own rect, and a TextItem is positioned in
data units but drawn at a fixed pixel size. So a tag whose price is within one
line height of an edge gets cut in half -- at every zoom, since the box does
not shrink with the range. Measured before the fix, on a 451 px chart: at 0.5%
from the top 93% of the gold tag was outside the view, at 99.5% from the
bottom the whole pair tag was.

What replaced it measures the box in pixels and converts with viewPixelSize,
rather than reserving a percentage of the view span -- a percentage is not the
height of the box, and the two drift apart as the window is resized. The old
OHLCV clamp reserved 2% of the span, about 9 px, for a box 60 px tall.

The geometry tests drive the real _on_mouse_move and read each tag's box back
out of the item, because that is the layer where this breaks: every value in
between can be right while the thing on screen is still cut.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6.QtWidgets")

from PyQt6.QtCore import QPointF            # noqa: E402
from PyQt6.QtWidgets import QApplication    # noqa: E402

ANCHOR = ("US.SOXL", "US.SOXS", 54.00, 57.00)


class _FakeVB:
    """Just the two readings _fit_tag_y takes off a ViewBox."""

    def __init__(self, ylo: float, yhi: float, px: float):
        self._r, self._px = (ylo, yhi), px

    def viewRange(self):
        return [(0.0, 1.0), self._r]

    def viewPixelSize(self):
        return (self._px, self._px)


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


@pytest.fixture()
def soxl(viewer):
    viewer._code_edit.setText("US.SOXL")
    viewer._pair_anchor = ANCHOR
    yield viewer
    viewer._pair_anchor = None


@pytest.fixture()
def plain(viewer):
    """A symbol with no inverse pair: one tag, centred on the line."""
    viewer._code_edit.setText("US.AAPL")
    viewer._pair_anchor = None
    yield viewer


@pytest.fixture()
def bars(viewer, monkeypatch):
    """Klines, so the OHLCV tooltip is drawn at all.

    _show_tick_profile is stubbed out: it rebuilds a whole panel off tick data
    this test has none of, and it is not what is being measured here.
    """
    t0 = datetime(2026, 11, 9, 9, 30)
    rows = [((t0 + timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M"),
             54.0, 54.5, 53.8, 54.2, 162_402 + i) for i in range(120)]
    viewer._klines = pd.DataFrame(
        rows, columns=["time_key", "open", "high", "low", "close", "volume"])
    viewer._ticks = {}
    viewer._candle_mins = 1
    monkeypatch.setattr(type(viewer), "_show_tick_profile",
                        lambda self, idx: None)
    yield viewer
    viewer._klines = None


# ── reading a tag's box back off the chart ───────────────────────────────────

def _hover(v, frac: float):
    """Move the cursor to frac of the way down the candle plot."""
    rect = v._plot_c.sceneBoundingRect()
    v._on_mouse_move(QPointF(rect.center().x(),
                             rect.top() + rect.height() * frac))


def _box(v, label, anchor_y: float):
    """(lo, hi) of a label's box in data units, from its pixel height."""
    h = label.boundingRect().height() * v._plot_c.vb.viewPixelSize()[1]
    y = label.pos().y()
    return y - h * (1.0 - anchor_y), y + h * anchor_y


def _view(v):
    return v._plot_c.vb.viewRange()[1]


def _assert_inside(v, label, anchor_y, what):
    ylo, yhi = _view(v)
    lo, hi = _box(v, label, anchor_y)
    h = hi - lo
    assert lo >= ylo - h * 1e-6, (
        f"{what} hangs {(ylo - lo) / h * 100:.0f}% of its height below the "
        f"view: box [{lo:.4f},{hi:.4f}] against view [{ylo:.4f},{yhi:.4f}]")
    assert hi <= yhi + h * 1e-6, (
        f"{what} hangs {(hi - yhi) / h * 100:.0f}% of its height above the "
        f"view: box [{lo:.4f},{hi:.4f}] against view [{ylo:.4f},{yhi:.4f}]")


# ── the arithmetic on its own ────────────────────────────────────────────────

class TestTheFitItself:
    """One pixel is 0.1 data units here, so a 10 px tag is 1.0 tall."""

    @staticmethod
    def _fit(y, above_px, below_px, ylo=0.0, yhi=100.0, px=0.1):
        from analysis.trade_viewer_qt import TradeViewerQt
        return TradeViewerQt._fit_tag_y(_FakeVB(ylo, yhi, px),
                                        y, above_px, below_px)

    def test_well_inside_is_left_alone(self):
        """The common case by far: the tag must sit ON the crosshair line."""
        assert self._fit(50.0, 10, 10) == 50.0

    def test_it_is_pushed_down_from_the_top(self):
        assert self._fit(99.5, 10, 0) == pytest.approx(99.0)

    def test_it_is_pushed_up_from_the_bottom(self):
        assert self._fit(0.5, 0, 10) == pytest.approx(1.0)

    def test_an_upward_box_is_not_pushed_off_the_bottom(self):
        assert self._fit(0.5, 10, 0) == 0.5

    def test_exactly_touching_an_edge_is_already_a_fit(self):
        assert self._fit(99.0, 10, 0) == 99.0

    def test_a_centred_box_reserves_half_each_way(self):
        assert self._fit(99.9, 5, 5) == pytest.approx(99.5)
        assert self._fit(0.1, 5, 5) == pytest.approx(0.5)

    def test_the_pixel_size_is_what_converts_the_height(self):
        """Same tag, a view ten times tighter: it reserves ten times less."""
        assert self._fit(9.95, 10, 0, ylo=0.0, yhi=10.0,
                         px=0.01) == pytest.approx(9.9)

    def test_a_block_taller_than_the_view_is_left_where_it_is(self):
        """No answer fits, and the clamp would otherwise invert and land the
        tag somewhere unrelated to the price it marks."""
        assert self._fit(50.0, 600, 600) == 50.0

    def test_a_block_exactly_as_tall_as_the_view_too(self):
        assert self._fit(50.0, 500, 500) == 50.0


# ── what the chart actually draws ────────────────────────────────────────────

class TestThePairTagsAtTheEdges:

    @pytest.mark.parametrize("frac", [0.001, 0.005, 0.02, 0.05])
    def test_neither_is_cut_at_the_top(self, soxl, frac):
        _hover(soxl, frac)
        _assert_inside(soxl, soxl._price_label, 1.0, "the gold tag")
        _assert_inside(soxl, soxl._pair_price_label, 0.0, "the pair tag")

    @pytest.mark.parametrize("frac", [0.999, 0.995, 0.98, 0.95])
    def test_neither_is_cut_at_the_bottom(self, soxl, frac):
        _hover(soxl, frac)
        _assert_inside(soxl, soxl._price_label, 1.0, "the gold tag")
        _assert_inside(soxl, soxl._pair_price_label, 0.0, "the pair tag")

    def test_they_move_together(self, soxl):
        """The trap in fitting them one at a time: at the bottom edge the gold
        tag needs no room and stays, the pair tag walks up -- straight through
        it. They straddle the line as one block or they overlap."""
        _hover(soxl, 0.999)
        assert (soxl._price_label.pos().y()
                == soxl._pair_price_label.pos().y())
        gold_lo, _ = _box(soxl, soxl._price_label, 1.0)
        _, pair_hi = _box(soxl, soxl._pair_price_label, 0.0)
        assert gold_lo >= pair_hi - 1e-9, "the two tags overlap"

    def test_in_the_middle_they_sit_on_the_line(self, soxl):
        _hover(soxl, 0.5)
        y = soxl._hline.pos().y()
        assert soxl._price_label.pos().y() == pytest.approx(y)
        assert soxl._pair_price_label.pos().y() == pytest.approx(y)

    def test_the_price_printed_is_the_cursor_not_the_moved_box(self, soxl):
        """The box steps inward; the number must not follow it, or the tag
        reports a price the crosshair is not on."""
        _hover(soxl, 0.999)
        shown = float(soxl._price_label.textItem.toPlainText().split()[1])
        assert shown == pytest.approx(soxl._hline.pos().y(), abs=0.005)
        assert soxl._price_label.pos().y() != pytest.approx(
            soxl._hline.pos().y()), "nothing was clamped; the test is idle"


class TestTheSingleTag:

    @pytest.mark.parametrize("frac", [0.001, 0.5, 0.999])
    def test_it_is_never_cut(self, plain, frac):
        _hover(plain, frac)
        assert not plain._pair_price_label.isVisible()
        _assert_inside(plain, plain._price_label, 0.5, "the price tag")

    def test_in_the_middle_it_sits_on_the_line(self, plain):
        _hover(plain, 0.5)
        assert plain._price_label.pos().y() == pytest.approx(
            plain._hline.pos().y())


class TestTheOhlcvTooltip:
    """Four lines, ~60 px. Its old clamp reserved 2% of the view span for it,
    which is 9 px on a 451 px chart -- so it was cut at the top too."""

    @pytest.mark.parametrize("frac", [0.001, 0.02, 0.05, 0.1, 0.2])
    def test_it_is_not_cut_at_the_top(self, bars, frac):
        _hover(bars, frac)
        assert bars._ohlcv_label.isVisible()
        _assert_inside(bars, bars._ohlcv_label, 1.0, "the OHLCV tooltip")

    def test_it_still_floats_above_the_cursor_in_the_middle(self, bars):
        _hover(bars, 0.6)
        ylo, yhi = _view(bars)
        y = bars._hline.pos().y()
        assert bars._ohlcv_label.pos().y() == pytest.approx(
            y + (yhi - ylo) * 0.14)

    def test_it_carries_the_hovered_bar(self, bars):
        _hover(bars, 0.5)
        assert "Vol " in bars._ohlcv_label.textItem.toPlainText()
        assert "O 54.00" in bars._ohlcv_label.textItem.toPlainText()

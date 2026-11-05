"""The gold outline reaches the side it was computed for.

top_n_levels is covered on its own; this covers the four lines that carry
its answer into the panel. They are the kind that a unit test of the mask
cannot see go wrong: swapping buy_pens and sell_pens, or letting the Net
bar take either side's mark, leaves every mask correct and every outline on
the wrong bar.

A real window is built, the same way tests/analysis/test_viewer_config.py
does it, and the BarGraphItems are read back out of the panel.
"""
from __future__ import annotations

import pytest

PRICES = [10.00, 10.01, 10.02]
# Deliberately disagreeing: the heaviest buy level is NOT the heaviest sell
# level, so a swap cannot pass.
BUY = {10.00: 100.0, 10.01: 900.0, 10.02: 50.0}      # top is 10.01
SELL = {10.00: 800.0, 10.01: 10.0, 10.02: 20.0}      # top is 10.00


def _pd():
    return {p: {"buy": BUY[p], "sell": SELL[p], "neutral": 0.0}
            for p in PRICES}


@pytest.fixture(scope="function")
def qapp():
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _gold_rows(viewer, want_sell_side: bool) -> list[float]:
    """Prices whose bar on one side carries a non-null pen.

    The two bars are told apart by geometry rather than by draw order: the
    sell bar is the one drawn to the left of zero.
    """
    import pyqtgraph as pg
    from PyQt6.QtCore import Qt

    out = []
    for item in viewer._tick_profile_widget.getPlotItem().items:
        if not isinstance(item, pg.BarGraphItem):
            continue
        opts = item.opts
        pens = opts.get("pens")
        if not pens:
            continue
        x0 = opts.get("x0")
        is_sell = x0 is not None and min(x0) < 0
        if is_sell != want_sell_side:
            continue
        for price, pen in zip(opts["y"], pens):
            # Compared against the enum member, not 0: in PyQt6 a PenStyle is
            # not an int, so `!= 0` is true for every pen including NoPen --
            # which is how this helper first reported every row as outlined.
            if pen is not None and pen.style() != Qt.PenStyle.NoPen:
                out.append(round(float(price), 2))
    return sorted(out)


def _label_rows(viewer) -> list[float]:
    """Prices that got the text label printed above their row.

    Labels are driven by the union of the two masks, not by either one, so
    they need their own look: a level that is top-buy but not top-sell still
    has to be named.
    """
    import pyqtgraph as pg

    out = []
    for item in viewer._tick_profile_widget.getPlotItem().items:
        if isinstance(item, pg.TextItem):
            # Anchored a sliver above the row centre; the offset is a
            # fraction of one bin, far under a cent on this data.
            out.append(round(float(item.pos().y()), 2))
    return sorted(out)


class TestGoldOutlineLandsOnTheRightSide:

    def _viewer(self, qapp, net: bool):
        from analysis.trade_viewer_qt import TradeViewerQt
        v = TradeViewerQt()
        for _ in range(10):
            qapp.processEvents()
        v._ind_checks["tick_vp"].setChecked(False)
        v._ind_checks["tick_net"].setChecked(net)
        v._tick_top_n.setValue(1)
        return v

    def test_each_side_is_outlined_on_its_own_heaviest_level(self, qapp):
        v = self._viewer(qapp, net=False)
        try:
            v._draw_tick_profile(_pd(), "hdr")
            qapp.processEvents()
            assert _gold_rows(v, want_sell_side=False) == [10.01]
            assert _gold_rows(v, want_sell_side=True) == [10.00]
        finally:
            v.close()
            qapp.processEvents()

    def test_a_level_topping_either_side_gets_named(self, qapp):
        """The label follows the union. Requiring a level to top BOTH sides
        would silently drop the name from most marked rows while leaving
        every outline exactly where it was."""
        v = self._viewer(qapp, net=False)
        try:
            v._draw_tick_profile(_pd(), "hdr")
            qapp.processEvents()
            # 10.01 tops buys only, 10.00 tops sells only -- neither tops
            # both, so an AND would print nothing at all here.
            assert _label_rows(v) == [10.00, 10.01]
        finally:
            v.close()
            qapp.processEvents()

    def test_raising_n_widens_both_sides(self, qapp):
        v = self._viewer(qapp, net=False)
        try:
            v._tick_top_n.setValue(2)
            v._draw_tick_profile(_pd(), "hdr")
            qapp.processEvents()
            assert _gold_rows(v, want_sell_side=False) == [10.00, 10.01]
            assert _gold_rows(v, want_sell_side=True) == [10.00, 10.02]
        finally:
            v.close()
            qapp.processEvents()

    def test_the_net_bar_takes_the_mark_of_the_side_its_sign_names(self, qapp):
        """Net draws one bar per level, so it has to pick a side. 10.01 is
        net buy and is the top buy level, so it is marked; 10.00 is net sell
        and is the top sell level, so it is marked too -- but each for its
        own reason, which is what a sloppy `either mask` would also produce.
        10.02 is net buy while being the second sell level, so only a
        sign-aware rule leaves it clean at n=1."""
        v = self._viewer(qapp, net=True)
        try:
            v._tick_top_n.setValue(2)
            v._draw_tick_profile(_pd(), "hdr")
            qapp.processEvents()
            marked = sorted(set(_gold_rows(v, want_sell_side=False)) |
                            set(_gold_rows(v, want_sell_side=True)))
            # buy_mark at n=2 is {10.00, 10.01}; sell_mark is {10.00, 10.02}.
            # Net sign: 10.00 sell, 10.01 buy, 10.02 buy.
            # So 10.00 takes sell_mark (in), 10.01 takes buy_mark (in),
            # 10.02 takes buy_mark (10.02 is not a top-2 buy) -> out.
            assert marked == [10.00, 10.01]
            # And the labels say the same thing. Driving them from the union
            # instead left 10.02 named -- in the sell colour, with no bar
            # outlined beside it, because the bar there is net BUY.
            assert _label_rows(v) == [10.00, 10.01]
        finally:
            v.close()
            qapp.processEvents()

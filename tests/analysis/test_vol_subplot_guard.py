"""The volume subplot is only drawn when it is showing.

That block had no guard at all, and `vol` is OFF by default in
config/viewer_qt.toml, so the default chart spent part of every frame
building 4321 brushes and a rolling mean for a subplot nobody could see.
Measured: a frame went from 150.3ms to 47.5ms once this and the CVD change
were both in.

The risk a guard like this introduces is stale pixels -- the subplot showing
a previous frame's bars when it is switched back on. These drive the real
_render and the real toggle slot to show it does not happen, because the only
thing that would reveal it is what the widget actually holds.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6.QtWidgets")

from PyQt6.QtWidgets import QApplication        # noqa: E402

import analysis.trade_viewer_qt as tv           # noqa: E402

COLS = ["time_key", "open", "high", "low", "close", "volume"]


def klines(n: int = 40, vol_base: float = 1000.0) -> pd.DataFrame:
    t0 = datetime(2026, 11, 11, 9, 30)
    rows = []
    for i in range(n):
        up = i % 2 == 0                    # alternate, so both brushes are used
        o, c = (54.0, 54.4) if up else (54.4, 54.0)
        rows.append(((t0 + timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M:%S"),
                     o, 54.5, 53.8, c, vol_base + i))
    return pd.DataFrame(rows, columns=COLS)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app
    app.processEvents()


@pytest.fixture(scope="module")
def viewer(qapp):
    v = tv.TradeViewerQt()
    v.resize(1200, 800)
    yield v
    v.close()
    qapp.processEvents()


@pytest.fixture()
def chart(viewer):
    was = viewer._ind("vol")
    viewer._ind_checks["vol"].setChecked(False)
    viewer._candle_mins = 1
    viewer._ticks = {}
    viewer._db_ticks = {}
    yield viewer
    viewer._ind_checks["vol"].setChecked(was)


def heights(v):
    # `or []` is ambiguous here: once the subplot is filled this is a numpy
    # array, and bool() of one raises.
    h = v._vol_item.opts.get("height")
    return [] if h is None else list(h)


class TestWhileItIsHidden:

    def test_nothing_is_drawn(self, chart):
        chart._klines = klines()
        chart._render(chart._klines, {})
        assert heights(chart) == []

    def test_a_second_frame_leaves_it_empty(self, chart):
        chart._klines = klines()
        chart._render(chart._klines, {})
        chart._klines = klines(50, vol_base=9000.0)
        chart._render(chart._klines, {})
        assert heights(chart) == [], "a hidden subplot was filled in"

    def test_the_row_is_collapsed(self, chart):
        chart._klines = klines()
        chart._render(chart._klines, {})
        assert not chart._plot_v.isVisible()


class TestSwitchingItOn:
    """_on_indicator_toggle shows the row and calls _redraw() in the same
    slot, with no event loop turn between them, so there is no frame where
    an empty or stale subplot is on screen."""

    def test_it_fills_from_the_current_frame(self, chart, qapp):
        chart._klines = klines()
        chart._render(chart._klines, {})
        chart._klines = klines(50, vol_base=9000.0)
        chart._render(chart._klines, {})

        chart._ind_checks["vol"].setChecked(True)
        qapp.processEvents()

        got = heights(chart)
        assert len(got) == 50, f"drew {len(got)} bars for a 50-bar frame"
        assert max(got) == pytest.approx(9049.0), \
            "the subplot came back holding an earlier frame's volumes"

    def test_the_row_becomes_visible(self, chart, qapp):
        chart._klines = klines()
        chart._render(chart._klines, {})
        chart._ind_checks["vol"].setChecked(True)
        qapp.processEvents()
        assert chart._plot_v.isVisible()

    def test_switching_it_off_again_collapses_the_row(self, chart, qapp):
        chart._klines = klines()
        chart._ind_checks["vol"].setChecked(True)
        qapp.processEvents()
        chart._ind_checks["vol"].setChecked(False)
        qapp.processEvents()
        assert not chart._plot_v.isVisible()


class TestTheBrushes:
    """Two module-level brushes instead of one mkBrush per bar. Named by
    COLOUR, not by side: "Red Up" is a checkbox, so which colour means bull
    changes while the process runs."""

    def test_only_shared_brushes_are_used(self, chart, qapp):
        chart._klines = klines()
        chart._ind_checks["vol"].setChecked(True)
        qapp.processEvents()
        brushes = chart._vol_item.opts.get("brushes") or []
        assert len(brushes) == 40
        assert len({id(b) for b in brushes}) == 2, \
            "a brush was built per bar again"

    def test_they_are_the_module_level_pair(self, chart, qapp):
        chart._klines = klines()
        chart._ind_checks["vol"].setChecked(True)
        qapp.processEvents()
        ids = {id(b) for b in chart._vol_item.opts["brushes"]}
        assert ids == {id(tv._VOL_GREEN_BRUSH), id(tv._VOL_RED_BRUSH)}

    def test_red_up_decides_which_side_is_red(self, chart, qapp):
        chart._klines = klines()
        chart._ind_checks["vol"].setChecked(True)
        was = chart._ind("red_up")
        try:
            for red_up in (True, False):
                chart._ind_checks["red_up"].setChecked(red_up)
                qapp.processEvents()
                # bar 0 closes up, bar 1 closes down
                b = chart._vol_item.opts["brushes"]
                bull = tv._VOL_RED_BRUSH if red_up else tv._VOL_GREEN_BRUSH
                assert id(b[0]) == id(bull), f"red_up={red_up}"
                assert id(b[1]) != id(bull), f"red_up={red_up}"
        finally:
            chart._ind_checks["red_up"].setChecked(was)

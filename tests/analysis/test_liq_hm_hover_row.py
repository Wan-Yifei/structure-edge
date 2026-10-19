"""Hovering a price row names it, in the pinned labels' own words.

Only a few rows can carry a standing label -- the rest of the picture is
bands with no name on them, and "how big is THAT one" had no way to be
asked. The readout answers it for whatever row the cursor is on, however
far from the touch and whatever the label budget.

The text comes from the same builder the pinned labels use. That is the
point of the test below about agreement: two spellings of one fact is how a
panel starts contradicting itself, and the pinned label and the hover sit
inches apart on screen.
"""

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6.QtWidgets")

from PyQt6.QtWidgets import QApplication   # noqa: E402

BIN = 0.01
PMIN = 97.00


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app
    app.processEvents()


@pytest.fixture(scope="module")
def win(qapp):
    from analysis.liq_hm_window import LiqHmWindow

    w = LiqHmWindow()
    yield w
    w.close()
    qapp.processEvents()


@pytest.fixture
def grid(win):
    """Ten columns, a bid that gets eaten at bin 5 and an ask at bin 40."""
    import datetime as dt

    n_bins, n_cols = 60, 10
    win._bin_size = BIN
    win._price_min = PMIN
    win._bid_grid = np.zeros((n_cols, n_bins))
    win._ask_grid = np.zeros((n_cols, n_bins))
    win._bid_grid[:, 5] = [8000] * 7 + [3000] * 3     # eaten down
    win._ask_grid[:, 40] = 2500
    t0 = dt.datetime(2026, 10, 18, 20, 0)
    win._col_ts = [t0 + dt.timedelta(seconds=10 * i) for i in range(n_cols)]
    win._plot_widget.getPlotItem().vb.setXRange(0, n_cols, padding=0)
    return win


def price_of(bi):
    return PMIN + (bi + 0.5) * BIN


class TestHoverNamesTheRow:
    def test_a_bid_row_reports_now_and_peak(self, grid):
        got = grid._hovered_rows(price_of(5))
        assert got, "the hovered bid row produced no line"
        assert got[0].startswith("B ")
        assert "3,000" in got[0] and "8,000" in got[0], got[0]

    def test_an_ask_row_reports_its_side(self, grid):
        got = grid._hovered_rows(price_of(40))
        assert got and got[0].startswith("A "), got
        assert "2,500" in got[0], got[0]

    def test_an_empty_row_says_nothing(self, grid):
        assert grid._hovered_rows(price_of(25)) == []

    def test_both_sides_when_both_held_size(self, grid):
        """A level that was a bid and later an ask is how a price gets
        traded through; reporting one would hide half of it."""
        grid._ask_grid[:, 5] = 1200
        got = grid._hovered_rows(price_of(5))
        assert len(got) == 2, got
        assert got[0].startswith("B ") and got[1].startswith("A ")
        grid._ask_grid[:, 5] = 0

    def test_off_grid_prices_do_not_raise(self, grid):
        assert grid._hovered_rows(PMIN - 5.0) == []
        assert grid._hovered_rows(PMIN + 500.0) == []

    def test_it_agrees_with_the_pinned_label(self, grid):
        """Same builder, so the two can never word it differently."""
        direct = grid._row_label(grid._bid_grid, 5, 0, 10, "B")
        assert grid._hovered_rows(price_of(5))[0] == direct

    def test_distance_and_the_label_budget_do_not_apply(self, grid):
        """The pinned labels are rationed by distance and count; this is
        not, which is the whole reason it exists."""
        grid._bid_grid[:, 0] = 400          # far below, small, unlabellable
        got = grid._hovered_rows(price_of(0))
        assert got and "400" in got[0], got
        grid._bid_grid[:, 0] = 0

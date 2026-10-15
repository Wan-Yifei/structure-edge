"""Why a wall shrank: bought, or withdrawn.

A snapshot shows only that a level lost size. The two causes are opposite
signals and look identical in the depth alone -- on 2026-10-14 a 73,072-share
bid at 116.00 vanished in 588 ms with not one share printing there, while two
hours earlier a level took 2,518 shares and held. Matching the loss against
what printed AT that price separates them.

The suffix is an estimate and the tests say where it declines to guess: wide
bins, no tick feed, an intact wall. NEUTRAL volume is excluded throughout --
off-exchange prints consume no displayed size, and counting them would read a
withdrawal as absorption, which is the one mistake that matters here.
"""

import os
from datetime import datetime, timedelta

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6.QtWidgets")

from PyQt6.QtWidgets import QApplication        # noqa: E402

T0 = datetime(2026, 10, 14, 11, 10, 0)
N_COLS = 40
N_BINS = 150
PMIN, BIN = 115.00, 0.01
WALL_BIN = int((116.00 - PMIN) / BIN)      # the 116.00 band


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app
    app.processEvents()


@pytest.fixture
def win(qapp):
    from analysis.liq_hm_window import LiqHmWindow
    w = LiqHmWindow()
    w._price_min = PMIN
    w._bin_size = BIN
    w._col_ts = [T0 + timedelta(seconds=i) for i in range(N_COLS)]
    w._bid_grid = np.zeros((N_COLS, N_BINS), dtype=float)
    w._ask_grid = np.zeros((N_COLS, N_BINS), dtype=float)
    w._absorb_ticks = []
    # A finished load that happened to find nothing, not "not loaded yet".
    w._absorb_last_ts = T0 + timedelta(seconds=N_COLS)
    w._absorb_cb.setChecked(True)
    yield w
    w.close()
    qapp.processEvents()


def ticks(spec, price=116.00, direction="SELL"):
    """spec: list of (col_offset, volume)."""
    return [{"ts": T0 + timedelta(seconds=c), "price": price,
             "volume": v, "direction": direction} for c, v in spec]


def fate(win, peak, now, *, half=0.005, side="B", peak_col=10):
    grid = win._bid_grid if side == "B" else win._ask_grid
    grid[:, WALL_BIN] = now
    grid[peak_col, WALL_BIN] = peak
    return win._wall_fate(grid, WALL_BIN, 0, N_COLS, peak, now, half, side)


class TestItSeparatesTheTwo:
    def test_pulled_when_nothing_printed(self, win):
        """The 116.00 case: 73,072 gone, nothing printed there."""
        assert fate(win, 73_072, 0) == "  pulled 100%"

    def test_eaten_when_the_loss_printed_at_that_price(self, win):
        win._absorb_ticks = ticks([(12, 40_000), (14, 33_072)])
        assert fate(win, 73_072, 0) == "  eaten 100%"

    def test_a_mix_reports_the_larger_share(self, win):
        win._absorb_ticks = ticks([(12, 20_000)])
        got = fate(win, 73_072, 0)
        assert got.startswith("  pulled")
        assert got == f"  pulled {(73_072 - 20_000) / 73_072:.0%}"

    def test_more_printed_than_was_lost_is_fully_eaten(self, win):
        """Refills mean prints can exceed the net loss; never negative."""
        win._absorb_ticks = ticks([(12, 500_000)])
        assert fate(win, 73_072, 0) == "  eaten 100%"


class TestWhatDoesNotCount:
    def test_neutral_prints_are_not_absorption(self, win):
        """Off-exchange consumes no displayed size."""
        win._absorb_ticks = ticks([(12, 73_072)], direction="NEUTRAL")
        assert fate(win, 73_072, 0) == "  pulled 100%"

    def test_buys_do_not_explain_a_bid_shrinking(self, win):
        win._absorb_ticks = ticks([(12, 73_072)], direction="BUY")
        assert fate(win, 73_072, 0) == "  pulled 100%"

    def test_sells_do_not_explain_an_ask_shrinking(self, win):
        win._absorb_ticks = ticks([(12, 5_000)], direction="SELL")
        assert fate(win, 5_000, 0, side="A") == "  pulled 100%"

    def test_buys_do_explain_an_ask_shrinking(self, win):
        win._absorb_ticks = ticks([(12, 5_000)], direction="BUY")
        assert fate(win, 5_000, 0, side="A") == "  eaten 100%"

    def test_prints_before_the_peak_are_not_counted(self, win):
        """The wall still held its peak then, so they hit something else."""
        win._absorb_ticks = ticks([(2, 73_072)])
        assert fate(win, 73_072, 0, peak_col=10) == "  pulled 100%"

    def test_prints_at_another_price_are_not_counted(self, win):
        win._absorb_ticks = ticks([(12, 73_072)], price=116.40)
        assert fate(win, 73_072, 0) == "  pulled 100%"


class TestItDeclinesToGuess:
    def test_a_wide_bin_gets_no_suffix(self, win):
        """The price label already reads '116.00±0.25' at this width."""
        win._absorb_ticks = ticks([(12, 10_000)])
        assert fate(win, 73_072, 0, half=0.25) == ""

    def test_aggressor_off_gets_no_suffix(self, win):
        """Ticks only load while Aggressor is checked."""
        win._absorb_cb.setChecked(False)
        assert fate(win, 73_072, 0) == ""

    def test_a_load_that_has_not_finished_gets_no_suffix(self, win):
        """Empty-because-pending must not read as empty-because-nothing-traded."""
        win._absorb_last_ts = None
        assert fate(win, 73_072, 0) == ""

    def test_an_intact_wall_gets_no_suffix(self, win):
        win._absorb_ticks = ticks([(12, 10_000)])
        assert fate(win, 73_072, 73_072) == ""

    def test_a_level_that_was_never_there_gets_no_suffix(self, win):
        grid = win._bid_grid
        grid[:, WALL_BIN] = 0.0
        assert win._wall_fate(grid, WALL_BIN, 0, N_COLS,
                              1000.0, 0.0, 0.005, "B") == ""

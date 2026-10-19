"""Why a wall shrank: bought, or withdrawn.

A snapshot shows only that a level lost size. The two causes are opposite
signals and look identical in the depth alone -- on 2026-10-14 a 73,072-share
bid at 116.00 vanished in 588 ms with not one share printing there, while two
hours earlier a level took 2,518 shares and held. Matching the loss against
what printed AT that price separates them.

The suffix is an estimate and the tests say where it declines to guess: wide
bins, no tick feed, an intact wall.

It names the LOST amount, because the share it reports is a share of that
loss and not of the order: "300 / 1,020  pulled 100%" was read, reasonably,
as "all of it went" while 300 sat there still resting. With the loss on
screen the denominator is unambiguous, and at the extremes the percentage is
dropped -- there is nothing left to apportion. NEUTRAL volume is excluded throughout --
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
        assert fate(win, 73_072, 0) == "  −73,072 pulled"

    def test_eaten_when_the_loss_printed_at_that_price(self, win):
        win._absorb_ticks = ticks([(12, 40_000), (14, 33_072)])
        assert fate(win, 73_072, 0) == "  −73,072 eaten"

    def test_a_mix_reports_both_shares(self, win):
        win._absorb_ticks = ticks([(12, 20_000)])
        got = fate(win, 73_072, 0)
        assert "pulled" in got and "eaten" in got
        f = (73_072 - 20_000) / 73_072
        assert got == f"  −73,072  {f:.0%} pulled / {1 - f:.0%} eaten"

    def test_more_printed_than_was_lost_is_fully_eaten(self, win):
        """Refills mean prints can exceed the net loss; never negative."""
        win._absorb_ticks = ticks([(12, 500_000)])
        assert fate(win, 73_072, 0) == "  −73,072 eaten"


class TestWhatDoesNotCount:
    def test_neutral_prints_are_not_absorption(self, win):
        """Off-exchange consumes no displayed size."""
        win._absorb_ticks = ticks([(12, 73_072)], direction="NEUTRAL")
        assert fate(win, 73_072, 0) == "  −73,072 pulled"

    def test_buys_do_not_explain_a_bid_shrinking(self, win):
        win._absorb_ticks = ticks([(12, 73_072)], direction="BUY")
        assert fate(win, 73_072, 0) == "  −73,072 pulled"

    def test_sells_do_not_explain_an_ask_shrinking(self, win):
        win._absorb_ticks = ticks([(12, 5_000)], direction="SELL")
        assert fate(win, 5_000, 0, side="A") == "  −5,000 pulled"

    def test_buys_do_explain_an_ask_shrinking(self, win):
        win._absorb_ticks = ticks([(12, 5_000)], direction="BUY")
        assert fate(win, 5_000, 0, side="A") == "  −5,000 eaten"

    def test_prints_before_the_peak_are_not_counted(self, win):
        """The wall still held its peak then, so they hit something else."""
        win._absorb_ticks = ticks([(2, 73_072)])
        assert fate(win, 73_072, 0, peak_col=10) == "  −73,072 pulled"

    def test_prints_at_another_price_are_not_counted(self, win):
        win._absorb_ticks = ticks([(12, 73_072)], price=116.40)
        assert fate(win, 73_072, 0) == "  −73,072 pulled"


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


class TestTheGateIsReachable:
    """The first version capped the band at one cent, which no instrument this
    was written for could reach: the heatmap spans at least +/-1% of mid and
    cuts it into N_PRICE bins, so SOXL's half-band is about two cents at the
    default resolution. The feature shipped unable to appear. These pin the
    arithmetic rather than the constant, so narrowing it again fails here."""

    @staticmethod
    def _half(mid, n_price):
        """Half-band at the WIDEST zoom, which is the narrowest band on offer:
        the window is at least +/-1% of mid, so it is never thinner than this.
        Mirrors the span/bin arithmetic in _rebuild_grid."""
        span = mid * 0.02
        return ((span * 2) / n_price) / 2

    @pytest.mark.parametrize("mid", [45.0, 103.0, 116.0, 130.0])
    def test_default_resolution_can_show_a_suffix(self, win, mid):
        from analysis.liq_hm_window import N_PRICE
        win._absorb_ticks = ticks([(12, 73_072)], price=116.40)
        half = self._half(mid, N_PRICE)
        assert fate(win, 73_072, 0, half=half) != "", (
            f"no suffix at mid={mid} with N_PRICE={N_PRICE} "
            f"(half={half:.4f}) -- the gate is unreachable again")

    def test_a_quarter_dollar_band_is_still_refused(self, win):
        win._absorb_ticks = ticks([(12, 10_000)])
        assert fate(win, 73_072, 0, half=0.25) == ""

    def test_the_cutoff_is_at_least_three_ticks(self):
        """Three cents is three ticks at US equity tick size; below that the
        default heatmap resolution stops clearing it."""
        from analysis.liq_hm_window import _WALL_FATE_MAX_HALF
        assert _WALL_FATE_MAX_HALF >= 0.03


class TestItNamesTheLoss:
    """The share is OF THE LOSS, so the loss has to be on screen.

    Reported from a live panel: "B 98.81±0.02  300 / 1,020  pulled 100%".
    The arithmetic was right -- 720 shares left and every one of them was
    withdrawn -- but a bare 100% beside a level still holding 300 reads as
    "all of it went". The number the percentage applies to was the only
    thing missing.
    """

    def test_a_partial_loss_names_what_left_not_what_remains(self, win):
        got = fate(win, 1_020, 300)
        assert "720" in got, got
        assert "1,020" not in got and "300" not in got, got

    def test_no_bare_percentage_without_its_denominator(self, win):
        """Any percentage printed must have the lost amount beside it."""
        for peak, now in ((1_020, 300), (73_072, 0), (5_000, 4_000)):
            got = fate(win, peak, now)
            if "%" in got:
                assert "\u2212" in got, got

    def test_a_whole_loss_drops_the_percentage(self, win):
        """100% of a loss apportions nothing; the word alone says it."""
        got = fate(win, 1_020, 0)
        assert got == "  \u22121,020 pulled", got


class TestTheMixedFormIsASplit:
    """Two shares that sum to 100 declare themselves a partition.

    One share alone had to be read against a denominator that was never on
    screen, which is how "pulled 100%" came to sit beside a level still
    holding 300.
    """

    def test_the_two_shares_sum_to_one_hundred(self, win):
        win._absorb_ticks = ticks([(12, 20_000)])
        got = fate(win, 73_072, 0)
        import re
        pcts = [int(x) for x in re.findall(r"(\d+)%", got)]
        assert len(pcts) == 2, got
        assert sum(pcts) == 100, got

    def test_both_words_appear_only_when_mixed(self, win):
        win._absorb_ticks = ticks([(12, 20_000)])
        assert "eaten" in fate(win, 73_072, 0)
        win._absorb_ticks = ticks([])
        pure = fate(win, 73_072, 0)
        assert "eaten" not in pure and "%" not in pure, pure

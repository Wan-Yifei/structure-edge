"""The first live push must allocate a grid the derived bin count fits.

This is the half of the bins change with a real failure behind it, and it
shipped with no coverage at all: three separate mutations -- dropping the
reallocation, sizing it from MAX_COLS_DEF, dropping the column clear --
each survived all 1317 tests under tests/analysis. The twenty tests written
alongside the change are pure arithmetic, and the arithmetic was never the
dangerous part.

What the mutations cost, measured on 50 real QQQ snapshots:

  - no reallocation: bin_size is computed for 240 bins while the grid is
    still 100 wide, so the near-touch levels map to bins 117-122 and
    _paint_column drops every one of them with a silent `continue`. 0% of
    the levels next to the touch survive; what is left is deep bids far
    from the book. No exception, and no rebuild trigger fires because the
    window itself is perfectly valid.
  - MAX_COLS_DEF instead of the spin box: the constant is 240 and the
    stored setting is 720, while _push_column decides when to roll from
    the setting. IndexError on column 241 -- about four minutes after the
    open.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from analysis.liq_hm_window import LiqHmWindow    # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


def _book(mid: float, half_spread: float = 0.01, levels: int = 5):
    rows = []
    for i in range(1, levels + 1):
        rows.append({"side": "BID", "price": round(mid - i * half_spread, 4),
                     "volume": 500.0})
        rows.append({"side": "ASK", "price": round(mid + i * half_spread, 4),
                     "volume": 500.0})
    return rows


@pytest.fixture
def win(qapp):
    w = LiqHmWindow()
    w._code = "US.TEST"
    w._live = True
    w._hist_secs = False
    w._timer.stop()
    w._refill_timer.stop()
    yield w
    w._timer.stop()
    w._refill_timer.stop()
    w.close()


# A book priced like QQQ: the +/-1% floor dominates a tight touch, so the
# window is wide in dollars and the derived count climbs well past 100.
EXPENSIVE = 650.00
# And one priced like SOXL, where the touch is wider than the floor and the
# configured value already clears the gate.
CHEAP = 43.00


class TestTheGridIsSizedForTheCountThatWasDerived:

    def test_the_grid_is_as_wide_as_bin_size_implies(self, win):
        """Geometry, not the count function's opinion of itself.

        Asking _bins_for_window what the answer should be and checking the
        grid against it pins the wrong pair: it survives any change to
        bin_size, which is the number _paint_column actually divides by.
        Mutating bin_size to width/(n+20) while the grid stayed at n
        survived all 1324 tests -- the window's top 8.3% silently stopped
        being paintable, and the book sits in the middle so nothing noticed.

        The invariant is the one _render draws with: the grid spans exactly
        the price window, bin by bin.
        """
        win._reset_grid()
        win._maybe_init_price_range(_book(EXPENSIVE))

        width = win._price_max - win._price_min
        n = win._bid_grid.shape[1]
        assert win._ask_grid.shape[1] == n

        assert round(width / win._bin_size) == n, (
            f"the window is {width / win._bin_size:.4f} bins wide by "
            f"bin_size but the grid holds {n} -- the difference is a strip "
            f"of the window that _paint_column drops with a silent continue")
        # And the far edge really is the last bin, not one past it.
        top = int((win._price_max - 1e-9 - win._price_min) / win._bin_size)
        assert top == n - 1, (top, n)

    def test_the_count_really_did_climb_for_this_book(self, win):
        """Without this the test above passes on a book that never needed
        more bins, which is to say against no change at all."""
        win._reset_grid()
        win._maybe_init_price_range(_book(EXPENSIVE))
        assert win._bid_grid.shape[1] > win._n_price_spin.value(), (
            "this fixture no longer exercises the derived path; pick a "
            "higher price or a tighter touch")

    def test_a_cheap_symbol_is_left_at_the_configured_count(self, win):
        win._reset_grid()
        win._maybe_init_price_range(_book(CHEAP, half_spread=0.12))
        assert win._bid_grid.shape[1] == win._n_price_spin.value()

    def test_the_row_count_comes_from_the_control_not_the_constant(self, win):
        """_push_column decides when to roll from the same spin box, so a
        grid with a different number of rows rolls in the wrong place --
        IndexError, roughly max_cols seconds after the first push."""
        from analysis.liq_hm_window import MAX_COLS_DEF

        # Without this the test only discriminates by accident: it can tell
        # the control from the constant only while they differ, and they
        # differ only because config/viewer_qt.toml holds 720 against a
        # DEFAULTS value of 240 == MAX_COLS_DEF. Edit the config to 240 and
        # this test goes blind with no sign. Same guard the bins half has.
        assert win._max_cols_spin.value() != MAX_COLS_DEF, (
            "the configured max_cols now equals MAX_COLS_DEF, so this test "
            "can no longer tell them apart")

        win._reset_grid()
        win._maybe_init_price_range(_book(EXPENSIVE))
        assert win._bid_grid.shape[0] == win._max_cols_spin.value()


class TestTheBookActuallyLandsInTheGrid:
    """The shape assertions above cannot tell a right-sized grid from a
    right-sized grid nobody paints into."""

    def test_levels_next_to_the_touch_are_painted(self, win):
        win._reset_grid()
        book = _book(EXPENSIVE)
        win._maybe_init_price_range(book)
        win._push_column(book, "T0")

        painted = int(np.count_nonzero(win._bid_grid) +
                      np.count_nonzero(win._ask_grid))
        assert painted > 0, (
            "the window is valid and the grid is allocated, yet not one "
            "level landed in it -- the signature of a bin count that does "
            "not match the grid's width")

    def test_what_a_mismatched_grid_would_have_cost(self, win):
        """The counterfactual, computed rather than asserted on: it says
        how much a regression would silently remove.

        The book sits at the centre of the window by construction, so a
        grid left at the configured width loses the levels around the
        touch -- the only part of the panel anyone reads.
        """
        win._reset_grid()
        book = _book(EXPENSIVE)
        win._maybe_init_price_range(book)

        narrow = win._n_price_spin.value()
        prices = [r["price"] for r in book]
        bins_ = [int((p - win._price_min) / win._bin_size) for p in prices]
        assert all(b < win._bid_grid.shape[1] for b in bins_), bins_
        lost = [b for b in bins_ if b >= narrow]
        assert lost, (
            "no level would be lost at the configured width, so this "
            "fixture does not reproduce the failure it describes")


class TestRollingPastTheColumnLimit:

    def test_pushing_past_max_cols_keeps_the_columns_consistent(self, win):
        """A grid whose row count disagrees with the roll threshold raises
        IndexError the moment it rolls -- about four minutes in, on the
        stored settings. Rolling the wrong axis, or dropping the wrong end
        of the timestamp list, raises nothing at all.
        """
        import numpy as np

        # blockSignals, and _live off: setValue fires _on_max_cols_changed,
        # which calls set_live(True) -> _start_push -> OpenQuoteContext(...).
        # This test opened a real socket to OpenD and left a prefill QThread
        # unretired, which is the abort pattern the project notes record.
        win._live = False
        win._max_cols_spin.blockSignals(True)
        win._max_cols_spin.setValue(60)
        win._max_cols_spin.blockSignals(False)

        win._reset_grid()
        book = _book(EXPENSIVE)
        win._maybe_init_price_range(book)
        for i in range(win._max_cols_spin.value() + 5):
            win._push_column(book, f"T{i}")

        cap = win._max_cols_spin.value()
        # Not just "it did not raise". Rolling along the wrong axis moves
        # every historical column's PRICE mapping instead of its time slot,
        # and popping the wrong end of _col_ts leaves the labels describing
        # columns that are no longer there -- both silent.
        assert len(win._col_ts) == cap, (len(win._col_ts), cap)
        # The exact surviving window, in order. NOT sorted(): these are
        # strings, so "T10" sorts before "T5" and a lexicographic check
        # calls a correct list broken. Spelling out which pushes should
        # have survived is also what catches popping the wrong end -- the
        # oldest must be gone, not the newest.
        pushed = cap + 5
        assert win._col_ts == [f"T{i}" for i in range(pushed - cap, pushed)]
        assert np.count_nonzero(win._bid_grid[-1]) > 0, (
            "the newest column is empty after rolling")
        # Every column was painted from the SAME book, so they must all be
        # identical. This is what catches rolling along the wrong axis:
        # np.roll(..., axis=1) shifts prices within each row instead of
        # moving rows, so each surviving column ends up offset by its own
        # age. The newest column alone proves nothing -- _paint_column
        # writes it after the roll either way.
        newest = win._bid_grid[-1]
        for age in (1, 2, cap // 2, cap - 1):
            assert np.array_equal(win._bid_grid[-1 - age], newest), (
                f"column {age} back differs from the newest although both "
                f"were painted from the same book -- the roll moved prices, "
                f"not time")
        assert all(m is not None for m in win._mid_prices), (
            "a rolled column lost its mid price")

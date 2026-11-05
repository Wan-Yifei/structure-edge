"""After a price-band rebuild, the window reloads instead of waiting.

A rebuild drops every painted column -- it has to, because the bins have
just been resized under them. What it used to do next was nothing: the live
feed repaints one column a second, which at the default window is twelve
minutes of a mostly blank panel. Reported on 2026-11-05, when the SOXL open
rebuilt twice inside four minutes (the band collapsed 7.36 -> 1.57 at 09:31,
and the touch left the band at 09:34).

Everything the grid lost is still in the collector's database -- it is the
same fill the window already does at startup, which is why restarting the
window looked like a fix and waiting did not.

These drive a real LiqHmWindow but never let the debounce fire, so nothing
here opens a database or a feed.
"""
from __future__ import annotations

import pathlib
import sys
from datetime import datetime

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from analysis.liq_hm_window import LiqHmWindow    # noqa: E402


@pytest.fixture(scope="function")
def qapp():
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _snap(mid: float) -> list[dict]:
    """A tight two-sided book around *mid*."""
    rows = []
    for i in range(1, 6):
        rows.append({"side": "BID", "price": round(mid - i * 0.01, 2),
                     "volume": 500.0})
        rows.append({"side": "ASK", "price": round(mid + i * 0.01, 2),
                     "volume": 500.0})
    return rows


@pytest.fixture
def win(qapp):
    w = LiqHmWindow()
    w._code = "US.TEST"
    w._live = True
    w._hist_secs = False
    w._reset_grid()
    # Establish a band, the way the first live push does.
    w._maybe_init_price_range(_snap(100.00))
    w._needs_init = False
    w._timer.start(1000)
    w._refill_timer.stop()
    yield w
    w._timer.stop()
    w._refill_timer.stop()
    w.close()
    qapp.processEvents()


class TestARebuildAsksForARefill:

    def test_a_touch_outside_the_band_schedules_one(self, win):
        """What 09:34 did: price left the band in one move."""
        win._maybe_init_price_range(_snap(130.00))
        assert win._needs_init
        assert win._refill_timer.isActive()

    def test_the_column_timer_stops_until_the_refill_lands(self, win):
        """A column appended between the wipe and the refill would carry a
        newer timestamp than every column the refill then paints to the
        right of it."""
        assert win._timer.isActive()
        win._maybe_init_price_range(_snap(130.00))
        assert not win._timer.isActive()

    def test_a_band_that_did_not_move_schedules_nothing(self, win):
        """The rebuild is the trigger, not every push."""
        win._maybe_init_price_range(_snap(100.01))
        assert not win._refill_timer.isActive()
        assert win._timer.isActive()

    def test_a_burst_of_rebuilds_still_asks_once(self, win):
        """Single-shot and restarted, so the open's repeated rebuilds cost
        one query rather than one each."""
        for mid in (130.0, 160.0, 190.0):
            win._maybe_init_price_range(_snap(mid))
        assert win._refill_timer.isActive()
        assert win._refill_timer.isSingleShot()


class TestWhenItMustNotAsk:

    def test_a_replay_is_left_alone(self, win):
        """A replay owns the grid; refilling from live data would wipe it."""
        win._hist_secs = 3600
        win._maybe_init_price_range(_snap(130.00))
        assert not win._refill_timer.isActive()

    def test_a_rebuild_inside_the_fill_does_not_reschedule(self, win):
        """_on_bulk_ready sizes the band from the fill it was handed. If that
        rebuild asked for another fill, each would trigger the next."""
        win._in_bulk_fill = True
        try:
            win._maybe_init_price_range(_snap(130.00))
        finally:
            win._in_bulk_fill = False
        assert not win._refill_timer.isActive()

    def test_a_window_that_is_not_live_is_left_alone(self, win):
        win._live = False
        win._maybe_init_price_range(_snap(130.00))
        assert not win._refill_timer.isActive()


class TestTheDebounceTarget:

    def test_it_puts_the_column_timer_back_when_it_cannot_fill(self, win):
        """Scheduling stopped the timer, so every path out has to restart it
        or the panel stops advancing altogether."""
        win._maybe_init_price_range(_snap(130.00))
        assert not win._timer.isActive()
        win._needs_init = False          # nothing to fill
        win._refill_after_rebuild()
        assert win._timer.isActive()

    def test_a_replay_started_meanwhile_is_not_overridden(self, win):
        """Asserting on the timer alone is not enough: the path that fills
        also leaves the timer stopped, so the two outcomes look alike from
        there. What separates them is whether a fill was launched at all."""
        win._maybe_init_price_range(_snap(130.00))
        win._hist_secs = 3600
        win._refill_after_rebuild()
        assert win._bulk_worker is None
        assert not win._timer.isActive()


class TestTheTimerIsNotLeftArmed:
    """A scheduled rebuild outlives the grid it was scheduled for."""

    def test_switching_symbols_disarms_it(self, win):
        win._maybe_init_price_range(_snap(130.00))
        assert win._refill_timer.isActive()
        win._reset_grid()
        assert not win._refill_timer.isActive()

    def test_closing_the_window_disarms_it(self, win):
        """It would otherwise open a database worker on a dead window."""
        win._maybe_init_price_range(_snap(130.00))
        win.close()
        assert not win._refill_timer.isActive()


class TestOnePrefillAtATime:

    def test_a_worker_whose_callback_has_not_run_yet_still_counts(self, win):
        """Between done.emit() and the queued _on_bulk_ready the thread is
        stopped but the slot is still taken. Waving a second worker through
        there loses the reference to it when that callback nulls the slot,
        and Qt aborts the process over the running thread it destroys."""
        class _Finished:
            def isRunning(self):
                return False

        win._bulk_worker = _Finished()
        assert win._start_prefill() is False

    def test_it_declines_without_a_symbol(self, win):
        win._code = ""
        assert win._start_prefill() is False


class TestClearColumns:

    def test_it_drops_the_bubble_cache_with_the_columns(self, win):
        """_absorb_ticks holds (tick, column index) pairs; an index means
        nothing once columns renumber from 0."""
        win._col_ts = [1, 2, 3]
        win._mid_prices = [1.0, 2.0, 3.0]
        win._raw_snaps = [{"ts": 1}]
        win._absorb_ticks = [{"col": 2}]
        win._snap_seq = 7
        win._bid_grid[:] = 1.0
        win._clear_columns()
        assert win._col_ts == []
        assert win._mid_prices == []
        assert win._raw_snaps == []
        assert win._absorb_ticks == []
        assert win._snap_seq == 0
        assert not np.any(win._bid_grid)

    def test_it_drops_the_high_water_mark_with_them(self, win):
        """_load_absorb_ticks goes incremental whenever _absorb_last_ts is
        set, asking only for ticks newer than it. The columns a refill paints
        are all older than that, so a surviving mark is a fill with no
        bubbles on it."""
        win._absorb_ticks = [{"ts": 1}]
        win._absorb_last_ts = datetime(2026, 11, 5, 9, 34)
        win._clear_columns()
        assert win._absorb_last_ts is None

    def test_it_retires_a_tick_load_still_in_flight(self, win):
        """That worker carries its own high-water mark, and landing on a
        cleared grid would set it again behind the refill's back."""
        class _Stub:
            """No `done` signal: _retire_worker's disconnect is best-effort."""

            def isRunning(self):
                return False

        win._absorb_worker = _Stub()
        win._clear_columns()
        assert win._absorb_worker is None

    def test_the_price_band_survives_it(self, win):
        lo, hi, bin_h = win._price_min, win._price_max, win._bin_size
        win._clear_columns()
        assert (win._price_min, win._price_max, win._bin_size) == (lo, hi, bin_h)

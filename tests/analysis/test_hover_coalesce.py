"""The tick panel is redrawn on a timer, not inside the mouse handler.

Rebuilding the right-hand panel costs about 5 ms a bar crossing. The scene
spaces accepted mouse moves >= 10 ms apart, and a sweep at ~1000 px/s covers
more than the default 9.97 px/bar in that time, so nearly every event crossed a
bar: the draw ran 63-100 times a second and the crosshair fell behind the cursor
until the hand stopped. The handler now stashes the bar it wants and a
single-shot timer draws it -- 62/s nominal, 32-44/s measured once the blocking
draw is counted.

Three properties are what make that safe, and each has a test here because
each has a failure mode that looks like the bug it replaced:

  * the timer must NOT be re-armed while it is already running. QTimer.start()
    on an active timer resets the countdown, so re-arming on every move during
    continuous motion means it never fires at all -- a panel frozen until the
    hand stops, which is worse than the stutter.
  * the queued bar, not the displayed one, is what a later move compares
    against. Sweeping out to a bar and back again otherwise leaves the stale
    queued bar to win and the panel settles where the cursor is not.
  * firing with nothing queued must draw nothing, and firing must clear the
    queue, or a later fire redraws a bar the cursor left long ago.

The last class covers an unrelated crash found in the same pass: `code` was
read inside three branches of _on_indicator_toggle and used in a fourth.

Driven through the real widgets offscreen, because what is asserted is the
interaction between a Qt slot, a Qt timer and the handler's own guard.
"""

import os
import time
from datetime import datetime, timedelta

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6.QtWidgets")

from PyQt6.QtCore import QPointF            # noqa: E402
from PyQt6.QtWidgets import QApplication    # noqa: E402

N_BARS = 120


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


@pytest.fixture()
def bars(viewer, monkeypatch):
    """Flat bars plus an X range wide enough that one view unit is one bar."""
    t0 = datetime(2026, 11, 9, 9, 30)
    rows = [((t0 + timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M"),
             54.0, 54.5, 53.8, 54.2, 162_402 + i) for i in range(N_BARS)]
    viewer._klines = pd.DataFrame(
        rows, columns=["time_key", "open", "high", "low", "close", "volume"])
    viewer._ticks = {}
    viewer._candle_mins = 1
    viewer._range_region = None
    viewer._last_hover_idx = None
    viewer._pending_profile_idx = None
    viewer._profile_timer.stop()
    viewer._plot_c.vb.setXRange(0, N_BARS - 1, padding=0)
    viewer._plot_c.vb.setYRange(53.0, 55.0, padding=0)
    yield viewer
    viewer._profile_timer.stop()
    viewer._klines = None


class _FakeTimer:
    """Records start() calls and reports active the way QTimer would.

    Counting the calls is the point: the guard being tested is "do not call
    start() while active", and a real QTimer cannot tell a re-arm from a first
    arm after the fact.
    """

    def __init__(self):
        self.starts = []
        self.active = False

    def isActive(self):
        return self.active

    def start(self, ms):
        self.starts.append(ms)
        self.active = True

    def stop(self):
        self.active = False


def _hover(v, bar: float, y: float = 54.0):
    """Move the cursor to a bar index in the candle plot's own coordinates."""
    pt = v._plot_c.vb.mapViewToScene(QPointF(float(bar), y))
    v._on_mouse_move(QPointF(pt.x(), pt.y()))


def _drawn(monkeypatch, v) -> list:
    """Collect the indices _show_tick_profile is actually asked to draw."""
    calls = []
    monkeypatch.setattr(type(v), "_show_tick_profile",
                        lambda self, idx: calls.append(idx))
    return calls


class TestTheDrawLeavesTheMouseHandler:

    def test_crossing_bars_draws_nothing_in_the_handler(self, bars,
                                                        monkeypatch):
        """The whole point: the 5 ms draw is off the mouse event's path."""
        calls = _drawn(monkeypatch, bars)
        for bar in range(10, 40):
            _hover(bars, bar)
        assert calls == [], f"drew {len(calls)} times inside the handler"

    def test_the_bar_the_cursor_reached_is_the_one_queued(self, bars,
                                                          monkeypatch):
        _drawn(monkeypatch, bars)
        for bar in range(10, 40):
            _hover(bars, bar)
        assert bars._pending_profile_idx == 39

    def test_the_timer_draws_it(self, bars, monkeypatch):
        calls = _drawn(monkeypatch, bars)
        for bar in range(10, 40):
            _hover(bars, bar)
        bars._on_profile_timer()
        assert calls == [39]


class TestTheTimerIsNotRearmed:
    """A re-armed timer during continuous motion never fires at all."""

    def test_a_sweep_arms_the_timer_exactly_once(self, bars, monkeypatch):
        _drawn(monkeypatch, bars)
        fake = _FakeTimer()
        monkeypatch.setattr(bars, "_profile_timer", fake)
        for bar in range(10, 60):
            _hover(bars, bar)
        assert len(fake.starts) == 1, (
            f"timer re-armed {len(fake.starts)} times during one sweep -- "
            "with a restartable timer the panel never updates while moving")

    def test_it_arms_again_once_it_has_fired(self, bars, monkeypatch):
        _drawn(monkeypatch, bars)
        fake = _FakeTimer()
        monkeypatch.setattr(bars, "_profile_timer", fake)
        _hover(bars, 10)
        fake.stop()                      # what firing leaves behind
        bars._pending_profile_idx = None
        _hover(bars, 20)
        assert len(fake.starts) == 2

    def test_the_real_timer_is_single_shot(self, bars):
        """A repeating timer would wake every 16 ms forever once the cursor
        stopped. It would not redraw -- the queue is empty by then -- so the
        cost is an invisible one: a timer that never goes quiet."""
        assert bars._profile_timer.isSingleShot()

    def test_the_interval_is_one_frame(self, bars, monkeypatch):
        from analysis.trade_viewer_qt import _PROFILE_COALESCE_MS

        fake = _FakeTimer()
        monkeypatch.setattr(bars, "_profile_timer", fake)
        _drawn(monkeypatch, bars)
        _hover(bars, 10)
        assert fake.starts == [_PROFILE_COALESCE_MS]
        # Long enough to collapse a 125 Hz stream, short enough not to be seen.
        assert 8 <= _PROFILE_COALESCE_MS <= 33


class TestTheWiringIsReal:
    """Every test above this point either calls _on_profile_timer by hand or
    stubs the draw out. Both halves of the real path need one test of their
    own, or the file stays green through a change that breaks the panel
    outright."""

    def test_the_timeout_signal_is_connected(self, bars, qapp, monkeypatch):
        """Delete the timeout.connect line and the panel never updates again;
        without this test nothing in the file notices."""
        calls = _drawn(monkeypatch, bars)
        _hover(bars, 12)
        assert bars._profile_timer.isActive(), "the timer was never armed"
        deadline = time.monotonic() + 2.0
        while not calls and time.monotonic() < deadline:
            qapp.processEvents()
        assert calls == [12], "the timer fired but drew nothing"

    def test_the_real_draw_records_the_bar_it_drew(self, bars):
        """_on_mouse_move compares against _last_hover_idx, which only the
        real _show_tick_profile sets -- the sweep-back test below hands itself
        that value, so the coupling is asserted here instead of assumed."""
        bars._last_hover_idx = None
        bars._show_tick_profile(7)
        assert bars._last_hover_idx == 7


class TestSweepingBackDoesNotStrandThePanel:
    """The queued bar is compared against, not only the displayed one."""

    def test_returning_to_the_displayed_bar_replaces_the_queue(self, bars,
                                                               monkeypatch):
        calls = _drawn(monkeypatch, bars)
        _hover(bars, 4)
        bars._on_profile_timer()                 # panel now shows bar 4
        bars._last_hover_idx = 4                 # what the real draw sets
        calls.clear()

        _hover(bars, 5)                          # queue 5
        _hover(bars, 4)                          # back to what is displayed
        bars._on_profile_timer()

        assert calls == [4], (
            "the panel settled on a bar the cursor had already left")

    def test_staying_on_the_queued_bar_queues_nothing_new(self, bars,
                                                          monkeypatch):
        _drawn(monkeypatch, bars)
        fake = _FakeTimer()
        monkeypatch.setattr(bars, "_profile_timer", fake)
        _hover(bars, 7)
        _hover(bars, 7)
        _hover(bars, 7)
        assert len(fake.starts) == 1
        assert bars._pending_profile_idx == 7


class TestTheTimerSlot:

    def test_firing_with_an_empty_queue_draws_nothing(self, bars, monkeypatch):
        calls = _drawn(monkeypatch, bars)
        bars._pending_profile_idx = None
        bars._on_profile_timer()
        assert calls == []

    def test_firing_clears_the_queue(self, bars, monkeypatch):
        """Otherwise the next fire redraws a bar the cursor left long ago."""
        _drawn(monkeypatch, bars)
        bars._pending_profile_idx = 11
        bars._on_profile_timer()
        assert bars._pending_profile_idx is None

    def test_firing_twice_draws_once(self, bars, monkeypatch):
        calls = _drawn(monkeypatch, bars)
        bars._pending_profile_idx = 11
        bars._on_profile_timer()
        bars._on_profile_timer()
        assert calls == [11]


class TestIndicatorToggleReadsTheCodeOnce:
    """`code` was assigned inside the KD, CVD and A/D branches and used by the
    DCVD one, so all three off with DCVD on raised UnboundLocalError out of a
    Qt slot -- which aborted the rest of the toggle, leaving the DCVD row
    hidden and skipping the trailing _redraw()."""

    def _set(self, v, **flags):
        for key, on in flags.items():
            cb = v._ind_checks.get(key)
            assert cb is not None, f"no checkbox named {key!r}"
            cb.setChecked(on)

    def test_dcvd_alone_does_not_raise(self, bars):
        self._set(bars, kd=False, cvd=False, adi=False, dcvd=True)
        bars._on_indicator_toggle()          # used to raise UnboundLocalError

    def test_dcvd_alone_still_reaches_the_row_toggle(self, bars):
        """The consequence worth pinning: the crash was before this line, so
        ticking DCVD did nothing visible at all."""
        self._set(bars, kd=False, cvd=False, adi=False, dcvd=True)
        bars._on_indicator_toggle()
        assert bars._plot_dcvd.isVisible()

    def test_the_title_carries_the_code(self, bars):
        self._set(bars, kd=False, cvd=False, adi=False, dcvd=True)
        bars._code_edit.setText("US.SOXL")
        bars._on_indicator_toggle()
        assert "US.SOXL" in bars._plot_dcvd.titleLabel.text

    def test_every_other_combination_still_works(self, bars):
        """The fix hoists a read out of three branches; the three it came from
        must still title their own plots."""
        bars._code_edit.setText("US.SOXL")
        self._set(bars, kd=True, cvd=True, adi=True, dcvd=True)
        bars._on_indicator_toggle()
        for plot in (bars._plot_kd, bars._plot_cvd,
                     bars._plot_adi, bars._plot_dcvd):
            assert "US.SOXL" in plot.titleLabel.text

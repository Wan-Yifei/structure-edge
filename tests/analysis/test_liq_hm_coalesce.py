"""A burst of pushes costs one render; one column tick costs one marker pass.

Both paths used to pay per event. _on_push rendered on every push, against a
feed measured at 6.8-7.2 stored snapshots/s per symbol that arrives in bursts
(median 14 ms between two of them, 304 ms at p95). _redraw_orderflow_markers
reruns iceberg, spoof, stacked-imbalance and absorption detection from scratch
over the whole _raw_snaps window, and one Col(s) tick asked for that twice --
once from _on_tick, then again from _on_absorb_ready when the tick's own
_load_absorb_ticks worker returned. py-spy on the live viewer put the
detectors reached through that function at 14.2% of the whole process.

The arming rule has its own class here because the obvious alternative is
wrong in a way no one would notice from the outside: restarting the timer on
every request -- an ordinary debounce -- starves when requests arrive faster
than the interval, which is exactly the case this runs in.

What stays synchronous is also pinned, because deferring a control's own
redraw is the one thing that would read as broken.
"""

import os
import time
from datetime import datetime, timedelta

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6.QtWidgets")

from PyQt6.QtWidgets import QApplication   # noqa: E402

T0 = datetime(2026, 9, 21, 18, 0, 0)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app
    # No processEvents() at teardown -- see test_absorption_marker.py


@pytest.fixture()
def win(qapp):
    from analysis.liq_hm_window import LiqHmWindow

    w = LiqHmWindow()
    w.set_code("US.SOXL")
    # The column timer would drive rolls of its own; every test here calls the
    # handlers directly so the pass count is exact.
    w._timer.stop()
    yield w
    w.close()
    qapp.processEvents()


def _snap(ts, mid=141.0):
    return ([{"ts": ts, "side": "BID", "price": mid - 0.01 * i, "volume": 100 + i}
             for i in range(3)]
            + [{"ts": ts, "side": "ASK", "price": mid + 0.01 * (i + 1), "volume": 200 + i}
               for i in range(3)])


def _rows(mid=141.0):
    """A push payload: no "ts", which _on_push supplies from the column."""
    return ([{"side": "BID", "price": mid - 0.01 * i, "volume": 100 + i}
             for i in range(3)]
            + [{"side": "ASK", "price": mid + 0.01 * (i + 1), "volume": 200 + i}
               for i in range(3)])


def _pump(qapp, ms: int) -> None:
    """Let Qt timers fire for *ms* without depending on PyQt6.QtTest."""
    deadline = time.monotonic() + ms / 1000.0
    while time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.005)
    qapp.processEvents()


class _Spy:
    """Count calls to the two functions the timers front, without running them.

    Patched on the INSTANCE, so the timers -- which were connected to the
    bound methods in __init__ -- still reach the real ones. Each test that
    needs the timer's own call counted reconnects it explicitly.
    """

    def __init__(self, win, monkeypatch):
        self.renders = 0
        self.markers = 0
        monkeypatch.setattr(win, "_render",
                            lambda: setattr(self, "renders", self.renders + 1))
        monkeypatch.setattr(win, "_redraw_orderflow_markers",
                            lambda: setattr(self, "markers", self.markers + 1))
        monkeypatch.setattr(win, "_load_absorb_ticks", lambda: None)
        # Re-point the timers at the spies. Without this they would fire into
        # the real methods and this would count the requests, not the passes.
        win._render_timer.timeout.disconnect()
        win._render_timer.timeout.connect(
            lambda: setattr(self, "renders", self.renders + 1))
        win._marker_timer.timeout.disconnect()
        win._marker_timer.timeout.connect(
            lambda: setattr(self, "markers", self.markers + 1))


class TestABurstOfPushesCostsOneRender:

    def test_a_push_does_not_render_inline(self, win, monkeypatch):
        win._on_snap_ready(_snap(T0))
        spy = _Spy(win, monkeypatch)
        win._on_push("US.SOXL", _rows(), time.time())
        assert spy.renders == 0, "the push rendered synchronously"
        assert win._render_timer.isActive(), "nothing was scheduled either"

    def test_the_whole_burst_collapses_to_one(self, win, monkeypatch, qapp):
        """Ten pushes a few ms apart, as the feed actually delivers them."""
        win._on_snap_ready(_snap(T0))
        spy = _Spy(win, monkeypatch)
        for i in range(10):
            win._on_push("US.SOXL", _rows(141.0 + 0.01 * i), time.time())
        assert spy.renders == 0
        _pump(qapp, 220)
        assert spy.renders == 1, (
            f"{spy.renders} renders for one burst of 10 pushes")

    def test_every_push_still_reaches_the_data(self, win, monkeypatch):
        """The premise. Coalescing the DRAWING must not coalesce the
        recording -- the detectors read _raw_snaps, and a push dropped from
        it is a book the feed reported and nothing can recover."""
        win._on_snap_ready(_snap(T0))
        spy = _Spy(win, monkeypatch)
        before = len(win._raw_snaps)
        for i in range(10):
            win._on_push("US.SOXL", _rows(141.0 + 0.01 * i), time.time())
        assert len(win._raw_snaps) == before + 10 * len(_rows())
        assert spy.renders == 0


class TestTheArmingRuleDoesNotStarve:
    """Arm-if-idle, not restart-on-every-call.

    A debounce resets its deadline on each request. Pushes arrive ~140 ms
    apart on average and ~14 ms apart inside a burst, against a 100 ms
    interval -- so a debounce would keep deferring and the grid would stop
    repainting precisely while the feed was busiest. These two tests fail on
    that implementation and pass on this one.
    """

    def test_a_second_request_does_not_push_the_deadline_back(self, win, qapp):
        """Compared against the INTERVAL, not against an earlier reading.

        Reading remainingTime() twice and requiring the second to be smaller
        looks equivalent and is fragile: the pump runs on 1-15.6 ms Windows
        granularity, and if it overruns the timer has already fired,
        remainingTime() answers -1, and the assertion becomes 100 <= -1. A
        fresh interval is what a debounce leaves behind, so that is what this
        rules out -- and -1 satisfies it too, correctly, since a debounce
        being 60 ms in could not have fired at all.
        """
        from analysis.liq_hm_window import _RENDER_COALESCE_MS

        win._schedule_render()
        _pump(qapp, 60)
        win._schedule_render()
        left = win._render_timer.remainingTime()
        assert left <= _RENDER_COALESCE_MS - 20, (
            f"{left}ms left of a {_RENDER_COALESCE_MS}ms interval after 60ms "
            "had already elapsed: the second request reset the deadline, "
            "which is a debounce and starves under a continuous feed")

    def test_a_continuous_stream_still_fires(self, win, monkeypatch, qapp):
        """Requests closer together than the interval, for longer than it.

        The counts are read INSIDE the stream, not after it. Read afterwards
        this passes on a debounce too: once the requests stop, the last
        deferred deadline finally expires and fires once. The failure a
        debounce actually causes is that nothing repaints WHILE the feed is
        busy, so that is what gets asserted.
        """
        spy = _Spy(win, monkeypatch)
        deadline = time.monotonic() + 0.45
        while time.monotonic() < deadline:
            win._schedule_render()
            win._schedule_orderflow_redraw()
            qapp.processEvents()
            time.sleep(0.01)
        during_renders, during_markers = spy.renders, spy.markers
        assert during_renders >= 1, (
            "450ms of continuous requests at a 100ms interval produced no "
            "render while they were still arriving -- the deadline is being "
            "deferred, not held")
        assert during_markers >= 1, (
            "same for the marker pass at its 250ms interval")

    def test_a_render_cancels_a_pending_one(self, win, monkeypatch, qapp):
        """_render() means the view is current, whoever asked for it."""
        win._schedule_render()
        assert win._render_timer.isActive()
        win._render()                      # the real one, via the class
        assert not win._render_timer.isActive(), (
            "a completed render left a scheduled one armed behind it")


class TestOneTickCostsOneMarkerPass:

    def test_the_tick_and_its_worker_share_one_pass(self, win, monkeypatch,
                                                    qapp):
        """_on_tick asks, then _on_absorb_ready asks again when the tick's own
        absorb-tick worker returns. That was two full detection passes over
        the same window."""
        win._on_snap_ready(_snap(T0))
        spy = _Spy(win, monkeypatch)
        win._push_ok = True
        win._last_push_ts = time.time()

        win._on_tick()
        win._on_absorb_ready([])           # the worker coming back

        assert spy.markers == 0, "one of them still ran inline"
        _pump(qapp, 400)
        assert spy.markers == 1, (
            f"{spy.markers} detection passes for one column tick")

    def test_the_poll_fallback_is_left_alone(self, win, monkeypatch):
        """_on_snap_ready only runs when the push feed is down, and its exact
        redraw-per-roll count is what test_liq_hm_marker_refresh.py guards
        against a bug reported twice. It must stay synchronous."""
        win._on_snap_ready(_snap(T0))
        spy = _Spy(win, monkeypatch)
        for i in range(3):
            win._on_snap_ready(_snap(T0 + timedelta(seconds=i + 1)))
        assert spy.markers == 3, "the fallback path was coalesced too"


class TestUserDrivenRedrawsStaySynchronous:
    """A control that answers a quarter second late reads as broken."""

    def test_controls_changed_redraws_now(self, win, monkeypatch):
        spy = _Spy(win, monkeypatch)
        win._on_controls_changed()
        assert spy.markers == 1

    def test_absorb_changed_redraws_now(self, win, monkeypatch):
        win._absorb_cb.setChecked(True)
        win._absorb_ticks = [{"ts": T0, "price": 141.0, "delta": 100}]
        spy = _Spy(win, monkeypatch)
        win._on_absorb_changed()
        assert spy.markers == 1

    def test_reset_view_redraws_now(self, win, monkeypatch):
        win._on_snap_ready(_snap(T0))
        spy = _Spy(win, monkeypatch)
        win._reset_view()
        assert spy.markers == 1


class TestTeardownDisarms:
    """A timer armed against a grid that no longer exists repaints from
    reallocated arrays, or runs the detectors over another code's snapshots."""

    def test_reset_grid_stops_both(self, win):
        win._schedule_render()
        win._schedule_orderflow_redraw()
        win._reset_grid()
        assert not win._render_timer.isActive()
        assert not win._marker_timer.isActive()

    def test_close_stops_both(self, win, qapp):
        win._schedule_render()
        win._schedule_orderflow_redraw()
        win.close()
        assert not win._render_timer.isActive()
        assert not win._marker_timer.isActive()

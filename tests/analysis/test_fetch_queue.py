"""A request made while a fetch is running is queued, not thrown away.

_trigger_fetch used to refuse outright when one was in flight -- a log line
and nothing else. The toolbar had moved and the chart had not. In Live that
self-heals, because the refresh timer fetches again within its interval; in
Historical nothing fetches at all (_stop_live stops that timer), so the
chart stayed on the old bars until the user acted again. The worst path is
Live -> Historical: _on_mode_changed stops the timer and then makes the
request that gets dropped.

What replaces it is one bit of state, deliberately not a queue: several
toolbar changes during a fetch produce one refetch, which reads the toolbar
when it fires and so sees where the user ended up rather than replaying
every step.

Three things have to hold for that to be safe, and each has a test here:
the refresh timer must never queue (an automatic source would chain fetches
end to end), a finished() from a fetcher that is no longer the current one
must be ignored (it would clear the flag belonging to the fetch that IS
running and let a second start beside it), and a close must stop the retry.

Nothing here opens a connection: _trigger_fetch is stubbed wherever it
would otherwise run, and the in-flight branch returns before anything
touches the context.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6.QtWidgets")

from PyQt6.QtCore import QObject, pyqtSignal      # noqa: E402
from PyQt6.QtWidgets import QApplication          # noqa: E402

import analysis.trade_viewer_qt as tv             # noqa: E402


class _Stub(QObject):
    """Stands in for a DataFetcher, to emit finished() with a real sender."""
    ready    = pyqtSignal(dict)
    error    = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, running: bool = False):
        super().__init__()
        self._running = running
        # isRunning() and isFinished() are asked at different moments, and
        # the gap between them is a real window -- see
        # test_a_thread_that_ends_mid_retire_is_not_held_forever.
        self._finished = not running

    def isRunning(self) -> bool:
        return self._running

    def isFinished(self) -> bool:
        return self._finished


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def viewer(qapp):
    v = tv.TradeViewerQt()
    v._ctx = object()          # past the "Not connected" guard, nothing more
    v._fetch_inflight = False
    v._refetch_pending = False
    v._closing = False
    v._fetcher = None
    yield v
    v._ctx = None
    v.close()
    qapp.processEvents()


@pytest.fixture
def fetches(viewer, monkeypatch):
    """Counts _trigger_fetch calls instead of letting them reach the feed."""
    calls: list[int] = []
    monkeypatch.setattr(type(viewer), "_trigger_fetch",
                        lambda self: calls.append(1))
    return calls


class TestARequestDuringAFetch:

    def test_it_is_queued(self, viewer):
        viewer._fetch_inflight = True
        viewer._trigger_fetch()
        assert viewer._refetch_pending

    def test_it_does_not_start_a_second_fetcher(self, viewer):
        viewer._fetch_inflight = True
        viewer._trigger_fetch()
        assert viewer._fetcher is None

    def test_several_of_them_are_one_request(self, viewer):
        """One bit, by design: the refetch reads the toolbar when it fires."""
        viewer._fetch_inflight = True
        for _ in range(4):
            viewer._trigger_fetch()
        assert viewer._refetch_pending is True


class TestTheRefreshTimerDoesNotQueue:
    """An automatic source must not be able to queue work: a fetch that
    outruns the interval would have every tick queue the next one, and they
    would run end to end against the history-kline rate limit."""

    def test_a_tick_during_a_fetch_is_dropped(self, viewer, fetches):
        viewer._fetch_inflight = True
        viewer._on_refresh_tick()
        assert fetches == []
        assert viewer._refetch_pending is False

    def test_a_tick_while_idle_fetches(self, viewer, fetches):
        viewer._on_refresh_tick()
        assert len(fetches) == 1

    def test_the_timer_goes_through_the_tick_slot(self, viewer, fetches):
        """Wiring the timer straight to _trigger_fetch is what reintroduces
        the chain, and the slot alone cannot show that it does not."""
        viewer._fetch_inflight = True
        viewer._refresh_timer.timeout.emit()
        assert fetches == []
        assert viewer._refetch_pending is False


class TestWhenTheFetchEnds:

    def test_the_queued_request_runs(self, viewer, fetches):
        viewer._fetch_inflight = True
        viewer._refetch_pending = True
        viewer._on_fetch_finished()
        assert viewer._fetch_inflight is False
        assert viewer._refetch_pending is False
        assert len(fetches) == 1

    def test_a_request_arriving_during_the_retry_survives(
            self, viewer, monkeypatch):
        """Which is why the flag is cleared before the call and not after:
        clearing afterwards wipes the request the retry itself queued."""
        monkeypatch.setattr(
            type(viewer), "_trigger_fetch",
            lambda self: setattr(self, "_refetch_pending", True))
        viewer._fetch_inflight = True
        viewer._refetch_pending = True

        viewer._on_fetch_finished()

        assert viewer._refetch_pending is True

    def test_nothing_queued_means_no_refetch(self, viewer, fetches):
        viewer._fetch_inflight = True
        viewer._on_fetch_finished()
        assert viewer._fetch_inflight is False
        assert fetches == []

    def test_a_close_stops_the_retry(self, viewer, fetches):
        viewer._fetch_inflight = True
        viewer._refetch_pending = True
        viewer._closing = True
        viewer._on_fetch_finished()
        assert fetches == []

    def test_closing_the_window_sets_that_flag(self, viewer):
        viewer.close()
        assert viewer._closing


class TestAStaleFinished:
    """The one that breaks 'one fetch at a time' if it is missed: a fetcher
    that has been abandoned or superseded still emits finished(), and acting
    on it clears the flag belonging to the fetch that is actually running."""

    def test_it_is_ignored(self, viewer, fetches):
        stale, current = _Stub(), _Stub()
        stale.finished.connect(viewer._on_fetch_finished)
        viewer._fetcher = current
        viewer._fetch_inflight = True
        viewer._refetch_pending = True

        stale.finished.emit()

        assert viewer._fetch_inflight is True, \
            "a stale finished() cleared the running fetch's flag"
        assert viewer._refetch_pending is True
        assert fetches == []

    def test_the_current_one_is_honoured(self, viewer, fetches):
        current = _Stub()
        current.finished.connect(viewer._on_fetch_finished)
        viewer._fetcher = current
        viewer._fetch_inflight = True
        viewer._refetch_pending = True

        current.finished.emit()

        assert viewer._fetch_inflight is False
        assert len(fetches) == 1


class TestAbandoningAFetcher:
    """What a reconnect does with a fetch it no longer wants. Nothing else
    will clear the flags afterwards -- the signals that would have are the
    ones being dropped."""

    def test_it_clears_both_flags(self, viewer):
        viewer._fetcher = object()
        viewer._fetch_inflight = True
        viewer._refetch_pending = True
        viewer._abandon_fetcher()
        assert viewer._fetcher is None
        assert viewer._fetch_inflight is False
        assert viewer._refetch_pending is False

    def test_a_fetcher_without_those_signals_does_not_raise(self, viewer):
        viewer._fetcher = object()      # disconnect() raises; it is swallowed
        viewer._abandon_fetcher()
        assert viewer._fetcher is None

    def test_a_thread_still_running_is_held_onto(self, viewer):
        """Letting the last reference go calls the C++ destructor on a live
        thread, and Qt aborts the process over it -- and a fetch still
        running is the whole reason this method exists."""
        worker = _Stub(running=True)
        viewer._fetcher = worker
        try:
            viewer._abandon_fetcher()
            assert worker in tv._PENDING_FETCHERS
            worker.finished.emit()          # the OS thread actually exits
            assert worker not in tv._PENDING_FETCHERS
        finally:
            if worker in tv._PENDING_FETCHERS:
                tv._PENDING_FETCHERS.remove(worker)

    def test_a_finished_thread_is_not_held(self, viewer):
        worker = _Stub(running=False)
        viewer._fetcher = worker
        viewer._abandon_fetcher()
        assert worker not in tv._PENDING_FETCHERS

    def test_a_thread_that_ends_mid_retire_is_not_held_forever(self, viewer):
        """isRunning() says yes, then the thread ends before finished can be
        connected -- so that signal has already been emitted and the release
        it was supposed to trigger never comes. The worker would sit in the
        list for the life of the process, holding its whole klines frame and
        tick dict with it. Asking isFinished() once more after connecting
        closes the window."""
        worker = _Stub(running=True)
        worker._finished = True            # isFinished() -> True
        viewer._fetcher = worker
        try:
            viewer._abandon_fetcher()
            assert worker not in tv._PENDING_FETCHERS, \
                "an already-finished worker was pinned forever"
            worker.finished.emit()         # the late signal must be harmless
            assert worker not in tv._PENDING_FETCHERS
        finally:
            if worker in tv._PENDING_FETCHERS:
                tv._PENDING_FETCHERS.remove(worker)

    def test_a_thread_that_is_genuinely_running_is_still_held(self, viewer):
        """The other direction: releasing unconditionally would undo the
        whole point and let Qt abort the process."""
        worker = _Stub(running=True)
        worker._finished = False
        viewer._fetcher = worker
        try:
            viewer._abandon_fetcher()
            assert worker in tv._PENDING_FETCHERS
        finally:
            if worker in tv._PENDING_FETCHERS:
                tv._PENDING_FETCHERS.remove(worker)


class TestTheRangeRadios:
    """One click emits twice -- toggled(False) on the button being left,
    toggled(True) on the one being taken. Both used to refetch; harmless
    while the second was dropped, a redundant round trip now it is queued."""

    def test_leaving_a_button_does_not_refetch(self, viewer, monkeypatch):
        calls: list[int] = []
        monkeypatch.setattr(type(viewer), "_on_range_changed",
                            lambda self: calls.append(1))
        viewer._on_range_radio_toggled(False)
        assert calls == []

    def test_taking_one_does(self, viewer, monkeypatch):
        calls: list[int] = []
        monkeypatch.setattr(type(viewer), "_on_range_changed",
                            lambda self: calls.append(1))
        viewer._on_range_radio_toggled(True)
        assert len(calls) == 1

    def test_a_real_click_fetches_once(self, viewer, monkeypatch):
        calls: list[int] = []
        monkeypatch.setattr(type(viewer), "_on_range_changed",
                            lambda self: calls.append(1))
        viewer._ind_checks["range_2d"].setChecked(True)
        assert len(calls) == 1


class TestLoadingATradeForReview:
    """Setting the combos fetches as a side effect, so whatever the toolbar
    holds at that moment is what goes out. The date has to be in place
    first."""

    def test_every_fetch_it_triggers_carries_the_trade_date(
            self, viewer, monkeypatch):
        seen: list[str] = []
        monkeypatch.setattr(type(viewer), "_trigger_fetch",
                            lambda self: seen.append(
                                self._date_edit.text().strip()))
        monkeypatch.setattr(tv, "_load_trade_from_db",
                            lambda tid: ({"symbol": "US.SOXL",
                                          "entry_time": "2026-10-14 10:05:00",
                                          "config_json": '{"entry_tf": "5m"}'},
                                         "backtest.duckdb"))
        viewer._date_edit.setText("2026-01-01")        # some older date
        viewer._trade_id_edit.setText("abc123")

        viewer._load_trade_review()

        assert seen, "loading a trade fetched nothing at all"
        assert set(seen) == {"2026-10-14"}, \
            f"a fetch went out for the wrong date: {seen}"

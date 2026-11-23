"""The tick handler's counters, not its row counts.

A shadowed import (`from datetime import ... time` over `import time`) made
`time.time()` raise inside the handler. The SDK swallows what is raised
there, so the symptom was invisible -- and it was NOT lost ticks: the rows
are committed before that line runs, verified by replaying a real burst
against both versions and getting 4 rows either way.

What it destroyed was every sign of life. state["ticks"] stayed 0,
last_tick_time stayed None, DATA RECEIVED never printed. The runbook says
"no DATA RECEIVED within a minute of the open -> restart", so the bug would
have had an operator restarting a healthy collector, and any zombie check
keyed on last_tick_time would have restarted it forever.

Which is why these tests assert on the COUNTERS. A row-count assertion
passes against the broken version and pins nothing -- the eighteen tests
that existed when this shipped were all at the store and reconcile layer,
and not one of them entered the handler.
"""
import pytest

from analysis.option_tick_collector import _make_handler
from feeds.option_tick_store import OptionTickStore

# A real burst off the 2026-11-20 SOXL tape: six prints inside 40 ms, two of
# them the same 10 contracts at the same price -- the shape that the first
# UNIQUE key collapsed.
BURST = [
    ("2026-11-20 15:14:37.983", 0.47, 1, "BUY", 900),
    ("2026-11-20 15:14:37.991", 0.47, 10, "BUY", 901),
    ("2026-11-20 15:14:37.998", 0.47, 10, "BUY", 902),
    ("2026-11-20 15:14:38.004", 0.48, 3, "BUY", 903),
    ("2026-11-20 15:14:38.019", 0.48, 1, "SELL", 904),
    ("2026-11-20 15:14:38.022", 0.48, 1, "", 905),      # no side -> 'N/A'
    ("2026-11-20 15:14:38.031", 0.48, 2, "NEUTRAL", 906),  # between the quotes
]


class FakeTickerBase:
    """Stands in for TickerHandlerBase: hands the frame straight through."""

    def on_recv_rsp(self, rsp_pb):
        return 0, rsp_pb


@pytest.fixture
def handler_and_state(tmp_path, monkeypatch):
    import pandas as pd
    import analysis.option_tick_collector as m

    monkeypatch.setattr(
        m, "_make_handler",
        m._make_handler)                       # keep the real one; see below
    store = OptionTickStore(tmp_path / "opt.db")
    state: dict = {"last_tick_time": None, "first_tick_done": False}

    # _make_handler imports TickerHandlerBase from moomoo at call time, so
    # the stand-in is injected there rather than by patching the module.
    import moomoo
    monkeypatch.setattr(moomoo, "TickerHandlerBase", FakeTickerBase)
    # _make_handler returns an INSTANCE -- see the test at the bottom of this
    # file for why that is load-bearing rather than a style choice.
    handler = m._make_handler(store, state)
    yield handler, state, store, pd
    store.close()


def _frame(pd, rows, code="US.SOXL261127P25000"):
    return pd.DataFrame([
        {"code": code, "time": ts, "price": px, "volume": v,
         "ticker_direction": d, "type": "AUTO_MATCH", "sequence": s}
        for ts, px, v, d, s in rows
    ])


class TestTheHandlerReportsThatItIsAlive:

    def test_counters_move_and_nothing_raises(self, handler_and_state):
        h, state, store, pd = handler_and_state
        h.on_recv_rsp(_frame(pd, BURST))

        # The load-bearing pair. Row counts pass against the broken version;
        # these do not.
        assert state.get("handler_errors") in (None, 0), state
        assert state["ticks"] == len(BURST), state

        assert state["offered"] == len(BURST)
        assert state["dropped"] == 0
        assert state["last_tick_time"] is not None, "no sign of life recorded"
        assert state["first_tick_done"] is True, "DATA RECEIVED would never print"

    def test_the_rows_really_landed(self, handler_and_state):
        h, state, store, pd = handler_and_state
        h.on_recv_rsp(_frame(pd, BURST))
        got = store._con.execute(
            "SELECT COUNT(*), SUM(volume) FROM option_ticks").fetchone()
        assert got == (len(BURST), sum(r[2] for r in BURST)), got

    def test_unknown_and_mid_market_stay_apart(self, handler_and_state):
        """'N/A' and NEUTRAL are different facts and must not merge.

        moomoo's TickerDirect carries both: NEUTRAL means the print landed
        between the bid and the ask, 'N/A' means the feed could not say.
        The collector used to write both as NEUTRAL, so a session could not
        report how much of its dropped flow was classified-as-mid versus
        simply unknown -- and the delta indicator drops that whole bucket.
        """
        h, state, store, pd = handler_and_state
        h.on_recv_rsp(_frame(pd, BURST))
        dirs = dict(store._con.execute(
            "SELECT direction, COUNT(*) FROM option_ticks GROUP BY 1").fetchall())
        assert dirs.get("N/A") == 1, dirs
        assert dirs.get("NEUTRAL") == 1, dirs
        assert dirs.get("BUY") == 4 and dirs.get("SELL") == 1, dirs

    def test_an_unrecognised_side_is_unknown_not_a_crash(self, handler_and_state):
        """Anything outside the enum is 'N/A' -- unknown is what it is."""
        h, state, store, pd = handler_and_state
        h.on_recv_rsp(_frame(pd, [
            ("2026-11-20 15:20:00.000", 0.5, 1, "WHATEVER", 950)]))
        dirs = dict(store._con.execute(
            "SELECT direction, COUNT(*) FROM option_ticks GROUP BY 1").fetchall())
        assert dirs == {"N/A": 1}, dirs

    def test_a_replayed_batch_counts_as_dropped_not_as_new(self, handler_and_state):
        h, state, store, pd = handler_and_state
        h.on_recv_rsp(_frame(pd, BURST))
        h.on_recv_rsp(_frame(pd, BURST))

        assert state["offered"] == 2 * len(BURST)
        assert state["ticks"] == len(BURST), "a replay was counted as new ticks"
        assert state["dropped"] == len(BURST)
        assert state.get("handler_errors") in (None, 0)

    def test_a_raising_batch_is_counted_not_swallowed_silently(
            self, handler_and_state):
        """The SDK eats exceptions here, so the handler counts its own."""
        h, state, store, pd = handler_and_state
        bad = pd.DataFrame([{"code": "X", "time": "t", "price": "not a number",
                             "volume": 1, "ticker_direction": "BUY"}])
        h.on_recv_rsp(bad)

        assert state.get("handler_errors") == 1, state
        assert state.get("ticks", 0) == 0

    def test_an_empty_push_is_not_an_error(self, handler_and_state):
        h, state, store, pd = handler_and_state
        h.on_recv_rsp(pd.DataFrame())
        assert state.get("handler_errors") in (None, 0)
        assert state.get("ticks", 0) == 0


class TestTheSleepCanBeInterrupted:
    """A long wait must not outlast the supervisor's patience.

    On Windows a signal handler runs only between bytecodes in the main
    thread, and Event.wait() blocks below that level, so a single wait(60)
    ignores CTRL_BREAK for up to a minute. The scheduler gives the
    collector 10 s before falling back to terminate(), which runs no
    cleanup -- so the idle branch's wait(60) AND the default 15 s sampling
    wait both outlasted it, and unsubscribe / reconcile / prune / checkpoint
    were skipped on every scheduled stop.

    Caught only by driving the scheduler's real stop path; it passed an
    earlier check that happened to use a 5 s interval.
    """

    def test_it_returns_as_soon_as_the_flag_is_set(self):
        import threading
        import time as t
        from analysis.option_tick_collector import wait_interruptible

        stop = threading.Event()
        threading.Timer(0.3, stop.set).start()
        t0 = t.monotonic()
        wait_interruptible(stop, 60)
        elapsed = t.monotonic() - t0

        assert elapsed < 2.0, (
            f"waited {elapsed:.1f}s of 60 after the flag was set at 0.3s -- "
            f"the supervisor gives up at 10s and skips the cleanup")
        assert stop.is_set()

    def test_it_still_waits_the_full_time_when_nothing_interrupts(self):
        import threading
        import time as t
        from analysis.option_tick_collector import wait_interruptible

        t0 = t.monotonic()
        wait_interruptible(threading.Event(), 1.5)
        elapsed = t.monotonic() - t0
        assert 1.3 <= elapsed <= 3.0, elapsed

    def test_an_already_set_flag_returns_at_once(self):
        import threading
        import time as t
        from analysis.option_tick_collector import wait_interruptible

        stop = threading.Event()
        stop.set()
        t0 = t.monotonic()
        wait_interruptible(stop, 30)
        assert t.monotonic() - t0 < 0.5

    def test_the_slices_are_short_enough_for_the_supervisor(self):
        """Under 10 s, which is what _kill_opt allows before terminate()."""
        import inspect
        from analysis.option_tick_collector import wait_interruptible
        default = inspect.signature(wait_interruptible).parameters["slice_s"].default
        assert default <= 5.0, default


class TestTheHandlerIsActuallyInstallable:
    """The gap every other test in this file left open.

    On 2026-11-23 the collector ran for 22 minutes of the open and stored
    zero ticks while reporting 0 dropped and 0 handler errors. _make_handler
    returned the CLASS; HandlerContext.set_handler decides with
    isinstance(handler, TickerHandlerBase), a class is not an instance of
    itself, so it fell through every branch and returned RET_ERROR -- which
    the caller discarded. subscribe() then succeeded, OpenD pushed, and the
    SDK had no handler registered for the proto.

    Every test above instantiated the handler itself and called on_recv_rsp
    directly, so all ten passed against a collector that could never receive
    anything. These two pin the wiring instead of the logic.
    """

    def test_what_main_hands_to_set_handler_passes_isinstance(self, tmp_path):
        """The real SDK base class, not the stand-in: isinstance against the
        genuine TickerHandlerBase is the exact predicate set_handler uses."""
        from moomoo import TickerHandlerBase
        import analysis.option_tick_collector as m

        store = OptionTickStore(tmp_path / "opt.db")
        try:
            h = m._make_handler(store, {})
            assert isinstance(h, TickerHandlerBase), (
                "set_handler() rejects this and returns RET_ERROR; the "
                "collector would subscribe, log healthy telemetry and store "
                f"nothing. got {type(h)!r}")
        finally:
            store.close()

    def test_the_sdk_accepts_it(self, tmp_path):
        """One level further down: hand it to the SDK's own dispatch table.

        isinstance passing and HandlerContext accepting are the same thing
        today, but the test above asserts our end of the contract and this
        one asserts theirs -- an SDK that adds a second condition would break
        the collector silently again, and only this test would say so.
        """
        from moomoo.common.handler_context import HandlerContext
        from moomoo import RET_OK
        import analysis.option_tick_collector as m

        store = OptionTickStore(tmp_path / "opt.db")
        try:
            hc = HandlerContext(lambda *a, **k: None)
            assert hc.set_handler(m._make_handler(store, {})) == RET_OK
        finally:
            store.close()

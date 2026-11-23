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
    ("2026-11-20 15:14:38.022", 0.48, 1, "", 905),      # no side -> NEUTRAL
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
    cls = m._make_handler(store, state)
    yield cls(), state, store, pd
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

    def test_an_unknown_side_becomes_neutral_not_a_crash(self, handler_and_state):
        h, state, store, pd = handler_and_state
        h.on_recv_rsp(_frame(pd, BURST))
        dirs = dict(store._con.execute(
            "SELECT direction, COUNT(*) FROM option_ticks GROUP BY 1").fetchall())
        assert dirs.get("NEUTRAL") == 1, dirs
        assert dirs.get("BUY") == 4 and dirs.get("SELL") == 1, dirs

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

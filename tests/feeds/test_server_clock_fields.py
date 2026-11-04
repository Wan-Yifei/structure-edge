"""The feed's own clock, carried from both pushes into both databases.

Why these columns exist at all: the order book is stamped with the
collecting machine's datetime.now(), while the tape carries the exchange's
clock as reported by the feed. Those are two different clocks, and the
offset between them measured -0.5 s on 2026-10-30, -0.68 s on 2026-11-02
and -1.05 s overnight into 2026-11-03 -- more than one whole column of the
heatmap's 1 s grid, so every join of a trade to a book is made against a
misalignment nobody corrects.

The moomoo feed already stamps both streams with the quote server's clock:
Qot_UpdateOrderBook carries svrRecvTimeBid/AskTimestamp, Qot_Common.Ticker
carries recvTime. The SDK reads neither -- it takes the *string* variant on
the book side and drops recvTime on the tape side when TickerHandlerBase
builds its DataFrame from a col_list naming neither field. These tests pin
the path that goes around that, so a join on one clock becomes possible.

The tests do NOT assert that the two streams' server clocks agree. That is
the open question the columns exist to answer, and it needs live data from
a regular-hours session; asserting it here would only pin this file's own
fixtures.
"""

import pathlib
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch


# ── the merger carries a server time per side ─────────────────────────────

class TestMergerCarriesServerTimes(unittest.TestCase):
    """A push can refresh one side only, so the two times are cached apart."""

    def setUp(self):
        from feeds.order_book_merge import OrderBookMerger
        self.m = OrderBookMerger()
        self.bid = [{"price": 100.0, "volume": 500}]
        self.ask = [{"price": 101.0, "volume": 300}]

    def test_both_sides_are_kept(self):
        self.m.merge("X", self.bid, self.ask, 1000.0,
                     svr_bid=111.5, svr_ask=112.5)
        self.assertEqual(self.m.svr_times("X"), (111.5, 112.5))

    def test_a_bid_only_push_leaves_the_ask_time_alone(self):
        """The reason these are two columns and not one.

        The ask side's levels survive a bid-only push, so the timestamp
        describing where those levels came from has to survive with them.
        """
        self.m.merge("X", self.bid, self.ask, 1000.0,
                     svr_bid=111.5, svr_ask=112.5)
        self.m.merge("X", [{"price": 100.5, "volume": 400}], [], 1000.1,
                     svr_bid=113.5, svr_ask=0.0)
        self.assertEqual(self.m.svr_times("X"), (113.5, 112.5))

    def test_zero_is_the_feeds_way_of_saying_unchanged(self):
        """Qot_GetOrderBook.proto reports zero for a side it did not update.

        Stored as-is it would read as 1970, and any join on it would place
        the book 56 years before the trade.

        Both sides, because they are two separate assignments: a mutation
        that broke only the ask side survived a version of this test that
        checked only the bid.
        """
        self.m.merge("X", self.bid, self.ask, 1000.0,
                     svr_bid=0.0, svr_ask=112.5)
        self.assertEqual(self.m.svr_times("X"), (None, 112.5))

        self.m.reset("X")
        self.m.merge("X", self.bid, self.ask, 1000.0,
                     svr_bid=111.5, svr_ask=0.0)
        self.assertEqual(self.m.svr_times("X"), (111.5, None))

    def test_a_caller_that_passes_nothing_behaves_as_before(self):
        """liq_hm_window.py calls merge() with four positional arguments."""
        merged = self.m.merge("X", self.bid, self.ask, 1000.0)
        self.assertIsNotNone(merged)
        bids, asks = merged
        self.assertEqual(bids, [(100.0, 500)])
        self.assertEqual(asks, [(101.0, 300)])
        self.assertEqual(self.m.svr_times("X"), (None, None))

    def test_an_unseen_code_has_no_times(self):
        self.assertEqual(self.m.svr_times("NEVER_PUSHED"), (None, None))

    def test_dropping_a_stale_side_drops_its_time_too(self):
        """A time left behind would describe levels that no longer exist."""
        from feeds.order_book_merge import OrderBookMerger
        m = OrderBookMerger(stale_secs=1.0)
        m.merge("X", self.bid, self.ask, 1000.0, svr_bid=111.5, svr_ask=112.5)
        m.merge("X", self.bid, [], 1002.0, svr_bid=113.5)
        self.assertEqual(m.svr_times("X"), (113.5, None))


# ── the order book store ──────────────────────────────────────────────────

class TestOrderBookStoreColumns(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = pathlib.Path(self._tmp.name) / "ob.db"
        self._open = []

    def tearDown(self):
        # Windows will not delete a file that still has an open handle.
        for s in self._open:
            s.close()
        self._tmp.cleanup()

    def _store(self):
        from feeds.order_book_store import OrderBookStore
        s = OrderBookStore(self.path)
        self._open.append(s)
        return s

    def test_the_times_round_trip(self):
        s = self._store()
        s.insert_snapshot("X", "2026-11-03 10:00:00", [(100.0, 5)], [(101.0, 3)],
                          svr_bid_ts=111.5, svr_ask_ts=112.5)
        con = sqlite3.connect(str(self.path))
        row = con.execute(
            "SELECT svr_bid_ts, svr_ask_ts FROM ob_snapshots").fetchone()
        con.close()
        self.assertEqual(row, (111.5, 112.5))

    def test_the_old_call_still_works_and_writes_null(self):
        """Nothing that wrote snapshots before these columns existed changes."""
        s = self._store()
        s.insert_snapshot("X", "2026-11-03 10:00:00", [(100.0, 5)], [(101.0, 3)])
        con = sqlite3.connect(str(self.path))
        row = con.execute(
            "SELECT svr_bid_ts, svr_ask_ts FROM ob_snapshots").fetchone()
        con.close()
        self.assertEqual(row, (None, None))

    def test_a_database_made_before_the_columns_gets_them(self):
        """The real order_book.db is one of these -- it already has rows."""
        con = sqlite3.connect(str(self.path))
        con.executescript(
            "CREATE TABLE ob_snapshots (code TEXT NOT NULL, ts TEXT NOT NULL,"
            " bids TEXT NOT NULL, asks TEXT NOT NULL);"
            "INSERT INTO ob_snapshots VALUES ('X','2026-11-02 10:00:00','[]','[]');")
        con.commit()
        con.close()

        self._store()   # opening it runs the migration

        con = sqlite3.connect(str(self.path))
        cols = {r[1] for r in con.execute("PRAGMA table_info(ob_snapshots)")}
        n = con.execute("SELECT COUNT(*) FROM ob_snapshots").fetchone()[0]
        con.close()
        self.assertIn("svr_bid_ts", cols)
        self.assertIn("svr_ask_ts", cols)
        self.assertEqual(n, 1, "the existing row must survive the migration")


# ── the tick store ────────────────────────────────────────────────────────

class TestTickStoreColumns(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = pathlib.Path(self._tmp.name) / "ticks.db"
        self._open = []

    def tearDown(self):
        for s in self._open:
            s.close()
        self._tmp.cleanup()

    def _store(self):
        from feeds.tick_store import TickStore
        s = TickStore(self.path)
        self._open.append(s)
        return s

    def _row(self, **kw):
        base = {"code": "X", "ts": "2026-11-03 10:00:00.100", "price": 1.0,
                "volume": 10, "direction": "BUY"}
        base.update(kw)
        return base

    def test_the_times_round_trip(self):
        s = self._store()
        s.insert_ticks([self._row(recv_ts=111.5, feed_ts=112.5)])
        con = sqlite3.connect(str(self.path))
        row = con.execute("SELECT recv_ts, feed_ts FROM ticks").fetchone()
        con.close()
        self.assertEqual(row, (111.5, 112.5))

    def test_a_row_without_them_writes_null(self):
        s = self._store()
        s.insert_ticks([self._row()])
        con = sqlite3.connect(str(self.path))
        row = con.execute("SELECT recv_ts, feed_ts FROM ticks").fetchone()
        con.close()
        self.assertEqual(row, (None, None))

    def test_the_columns_do_not_join_the_unique_key(self):
        """ticks is UNIQUE(code, ts, price, volume).

        If a differing recv_ts let a duplicate through, every restart would
        re-insert whatever the feed replayed.
        """
        s = self._store()
        s.insert_ticks([self._row(recv_ts=111.5)])
        n = s.insert_ticks([self._row(recv_ts=999.0)])
        self.assertEqual(n, 0)


# ── the tick collector reads what the SDK drops ───────────────────────────

class TestTickCollectorReadsTheProtobuf(unittest.TestCase):
    """TickerHandlerBase's DataFrame has no recvTime column, so the handler
    goes to rsp_pb for it. These pin that it lines records up correctly."""

    def setUp(self):
        import pandas as pd
        self.pd = pd
        mock = MagicMock()
        mock.RET_OK = 0

        class FakeBase:
            # The real one returns a DataFrame built from the same records,
            # in the same order, with recvTime and timestamp already gone.
            def on_recv_rsp(self, rsp_pb):
                return 0, rsp_pb.frame

        mock.TickerHandlerBase = FakeBase
        self._patcher = patch.dict(sys.modules, {"moomoo": mock})
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()

    def _rsp(self, records, frame_rows=None):
        """records: list of (price, volume, recvTime, timestamp)."""
        rsp = MagicMock()
        rsp.s2c.tickerList = [
            MagicMock(recvTime=r[2], timestamp=r[3]) for r in records
        ]
        rows = frame_rows if frame_rows is not None else [
            {"code": "X", "time": f"2026-11-03 10:00:0{i}.000",
             "price": r[0], "volume": r[1], "ticker_direction": "BUY",
             "sequence": i, "type": "AUTO_MATCH"}
            for i, r in enumerate(records)
        ]
        rsp.frame = self.pd.DataFrame(rows)
        return rsp

    def _captured(self, rsp):
        from analysis.tick_collector import _make_handler
        store = MagicMock()
        store.insert_ticks.return_value = 0
        h = _make_handler(store, {})()
        h.on_recv_rsp(rsp)
        if not store.insert_ticks.called:
            return []
        return store.insert_ticks.call_args[0][0]

    def test_each_trade_keeps_its_own_server_time(self):
        """Off by one here would attach every trade to its neighbour's clock."""
        rows = self._captured(self._rsp([
            (1.0, 10, 111.1, 211.1),
            (2.0, 20, 111.2, 211.2),
            (3.0, 30, 111.3, 211.3),
        ]))
        self.assertEqual([r["price"] for r in rows], [1.0, 2.0, 3.0])
        self.assertEqual([r["recv_ts"] for r in rows], [111.1, 111.2, 111.3])
        self.assertEqual([r["feed_ts"] for r in rows], [211.1, 211.2, 211.3])

    def test_a_length_mismatch_drops_the_times_rather_than_guessing(self):
        """If the frame and the protobuf ever disagree, position means
        nothing, and a wrong timestamp is worse than no timestamp."""
        rsp = self._rsp([(1.0, 10, 111.1, 211.1), (2.0, 20, 111.2, 211.2)])
        rsp.s2c.tickerList = rsp.s2c.tickerList[:1]     # frame still has 2
        rows = self._captured(rsp)
        self.assertEqual(len(rows), 2)
        self.assertEqual([r["recv_ts"] for r in rows], [None, None])

    def test_a_protobuf_without_the_field_costs_ticks_nothing(self):
        rsp = self._rsp([(1.0, 10, 0.0, 0.0)])
        type(rsp.s2c).tickerList = property(
            lambda self: (_ for _ in ()).throw(AttributeError("no s2c")))
        rows = self._captured(rsp)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["price"], 1.0)
        self.assertIsNone(rows[0]["recv_ts"])


# ── the book collector reads what the SDK skips ───────────────────────────

class TestBookCollectorReadsTheProtobuf(unittest.TestCase):
    """The SDK's parsed dict carries svrRecvTimeBid/Ask as STRINGS and never
    the Timestamp doubles beside them, so the handler reads the protobuf.

    The older tests in tests/analysis/test_order_book_collector.py pass a
    plain dict as rsp_pb, which exercises only the guard; without the class
    below, deleting the protobuf read entirely broke nothing.
    """

    def setUp(self):
        mock = MagicMock()
        mock.RET_OK = 0

        class FakeBase:
            # The real one returns the SDK's parsed dict while rsp_pb stays
            # the protobuf, so the two must not be the same object here.
            def on_recv_rsp(self, rsp_pb):
                return 0, rsp_pb.parsed

        mock.OrderBookHandlerBase = FakeBase
        self._patcher = patch.dict(sys.modules, {"moomoo": mock})
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()

    def _rsp(self, bid_ts=111.5, ask_ts=112.5, has=("bid", "ask"),
             bids=None, asks=None):
        rsp = MagicMock()
        rsp.parsed = {
            "code": "US.SOXL",
            "Bid": bids if bids is not None else [{"price": 100.0, "volume": 5}],
            "Ask": asks if asks is not None else [{"price": 101.0, "volume": 3}],
        }
        rsp.s2c.svrRecvTimeBidTimestamp = bid_ts
        rsp.s2c.svrRecvTimeAskTimestamp = ask_ts
        rsp.s2c.HasField = lambda f: (
            ("bid" in has) if "Bid" in f else ("ask" in has))
        return rsp

    def _insert_kwargs(self, rsp):
        from analysis.order_book_collector import _make_handler
        store = MagicMock()
        store.insert_snapshot.return_value = 2
        state = {"last_update_time": None, "first_update_done": False,
                 "session_count": 0}
        h = _make_handler(store, state)()
        h.on_recv_rsp(rsp)
        if not store.insert_snapshot.called:
            return None
        return store.insert_snapshot.call_args[1]

    def test_both_times_reach_the_store(self):
        kw = self._insert_kwargs(self._rsp())
        self.assertEqual(kw["svr_bid_ts"], 111.5)
        self.assertEqual(kw["svr_ask_ts"], 112.5)

    def test_an_unset_field_is_not_invented(self):
        kw = self._insert_kwargs(self._rsp(has=("bid",)))
        self.assertEqual(kw["svr_bid_ts"], 111.5)
        self.assertIsNone(kw["svr_ask_ts"])

    def test_a_protobuf_without_s2c_costs_the_snapshot_nothing(self):
        """An older OpenD must cost the timestamp, not the session."""

        class NoS2C:
            """Has no s2c at all -- a real object, not a mock, because a
            MagicMock answers every attribute and so cannot be missing one."""
            parsed = {"code": "US.SOXL",
                      "Bid": [{"price": 100.0, "volume": 5}],
                      "Ask": [{"price": 101.0, "volume": 3}]}

        kw = self._insert_kwargs(NoS2C())
        self.assertIsNotNone(kw, "the snapshot itself must still be written")
        self.assertIsNone(kw["svr_bid_ts"])
        self.assertIsNone(kw["svr_ask_ts"])

    def test_the_stored_time_follows_the_side_it_describes(self):
        """A bid-only push keeps the cached ask levels, so it must keep the
        ask's own timestamp rather than restamping them with this push."""
        from analysis.order_book_collector import _make_handler
        store = MagicMock()
        store.insert_snapshot.return_value = 2
        state = {"last_update_time": None, "first_update_done": False,
                 "session_count": 0}
        h = _make_handler(store, state)()
        h.on_recv_rsp(self._rsp(bid_ts=111.5, ask_ts=112.5))
        h.on_recv_rsp(self._rsp(bid_ts=113.5, ask_ts=0.0, has=("bid",),
                                asks=[]))
        kw = store.insert_snapshot.call_args[1]
        self.assertEqual(kw["svr_bid_ts"], 113.5)
        self.assertEqual(kw["svr_ask_ts"], 112.5)


if __name__ == "__main__":
    unittest.main()

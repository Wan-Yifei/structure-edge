"""Unit tests for analysis.order_book_collector.

Tests _parse_side() and _parse_args() without moomoo or network.
Tests _make_handler() with a fully mocked moomoo module.
Run: uv run pytest tests/ -v
"""

import sys
import unittest
from unittest.mock import MagicMock, patch, call
import pathlib
import tempfile
from datetime import datetime


# ── _parse_side ───────────────────────────────────────────────────────────────

class TestParseSide(unittest.TestCase):

    def setUp(self):
        from analysis.order_book_collector import _parse_side
        self._fn = _parse_side

    def test_empty_returns_empty(self):
        self.assertEqual(self._fn([]), [])

    def test_dict_format(self):
        items = [{"price": "100.5", "volume": "300"}, {"price": "99.0", "volume": "500"}]
        result = self._fn(items)
        self.assertEqual(result, [(100.5, 300), (99.0, 500)])

    def test_sequence_format(self):
        items = [[101.0, 200, "extra"], [102.5, 150]]
        result = self._fn(items)
        self.assertEqual(result, [(101.0, 200), (102.5, 150)])

    def test_mixed_dict_and_sequence(self):
        items = [{"price": 100.0, "volume": 100}, [101.0, 200]]
        result = self._fn(items)
        self.assertEqual(result, [(100.0, 100), (101.0, 200)])

    def test_malformed_dict_missing_key_skipped(self):
        items = [{"price": 100.0}, {"price": 101.0, "volume": 200}]
        result = self._fn(items)
        self.assertEqual(result, [(101.0, 200)])

    def test_malformed_sequence_too_short_skipped(self):
        items = [[100.0], [101.0, 200]]
        result = self._fn(items)
        self.assertEqual(result, [(101.0, 200)])

    def test_non_numeric_value_skipped(self):
        items = [{"price": "N/A", "volume": 100}, {"price": 100.0, "volume": 200}]
        result = self._fn(items)
        self.assertEqual(result, [(100.0, 200)])

    def test_none_item_skipped(self):
        items = [None, {"price": 100.0, "volume": 100}]
        result = self._fn(items)
        self.assertEqual(result, [(100.0, 100)])

    def test_float_coercion(self):
        items = [{"price": "100", "volume": "50"}]
        price, vol = self._fn(items)[0]
        self.assertIsInstance(price, float)
        self.assertIsInstance(vol, int)

    def test_large_list(self):
        items = [{"price": float(i), "volume": i * 10} for i in range(1, 21)]
        result = self._fn(items)
        self.assertEqual(len(result), 20)
        self.assertEqual(result[0], (1.0, 10))
        self.assertEqual(result[-1], (20.0, 200))


# ── _parse_args ───────────────────────────────────────────────────────────────

class TestParseArgs(unittest.TestCase):

    def setUp(self):
        from analysis.order_book_collector import _parse_args
        self._fn = _parse_args

    def test_defaults(self):
        args = self._fn([])
        self.assertEqual(args.host, "127.0.0.1")
        self.assertEqual(args.port, 11111)
        self.assertEqual(args.timeout, 5)
        self.assertIsNone(args.codes)

    def test_codes_flag(self):
        args = self._fn(["--codes", "US.AAPL", "US.TSLA"])
        self.assertEqual(args.codes, ["US.AAPL", "US.TSLA"])

    def test_db_flag(self):
        args = self._fn(["--db", "/tmp/test.db"])
        self.assertEqual(args.db, "/tmp/test.db")

    def test_timeout_flag(self):
        args = self._fn(["--timeout", "10"])
        self.assertEqual(args.timeout, 10)

    def test_host_port(self):
        args = self._fn(["--host", "192.168.1.1", "--port", "22222"])
        self.assertEqual(args.host, "192.168.1.1")
        self.assertEqual(args.port, 22222)

    def test_config_flag(self):
        args = self._fn(["--config", "myconfig.json"])
        self.assertEqual(args.config, "myconfig.json")


# ── _make_handler (mocked moomoo) ─────────────────────────────────────────────

class TestMakeHandler(unittest.TestCase):
    """Tests _make_handler() by injecting a fake moomoo module."""

    def setUp(self):
        # Build a minimal moomoo mock before importing the collector
        self._moomoo_mock = MagicMock()
        self._moomoo_mock.RET_OK = 0

        class FakeBase:
            def on_recv_rsp(self, rsp_pb):
                return 0, rsp_pb   # ret=RET_OK, data=rsp_pb (we pass data directly)

        self._moomoo_mock.OrderBookHandlerBase = FakeBase

        self._patcher = patch.dict(sys.modules, {"moomoo": self._moomoo_mock})
        self._patcher.start()

        from analysis.order_book_collector import _make_handler
        self._make_handler = _make_handler

    def tearDown(self):
        self._patcher.stop()

    def _make_store_mock(self):
        store = MagicMock()
        store.insert_snapshot.return_value = 4
        return store

    def _make_data(self, code="US.AAPL", bids=None, asks=None):
        return {
            "code": code,
            "Bid": bids if bids is not None else [{"price": 100.0, "volume": 500}],
            "Ask": asks if asks is not None else [{"price": 101.0, "volume": 300}],
        }

    def test_handler_calls_insert_on_valid_data(self):
        store = self._make_store_mock()
        state = {"last_update_time": None, "first_update_done": False, "session_count": 0}
        HandlerClass = self._make_handler(store, state)
        h = HandlerClass()
        h.on_recv_rsp(self._make_data())
        store.insert_snapshot.assert_called_once()

    def test_handler_passes_correct_code(self):
        store = self._make_store_mock()
        state = {"last_update_time": None, "first_update_done": False, "session_count": 0}
        HandlerClass = self._make_handler(store, state)
        h = HandlerClass()
        h.on_recv_rsp(self._make_data(code="US.NVDA"))
        args = store.insert_snapshot.call_args
        self.assertEqual(args[0][0], "US.NVDA")

    def test_handler_skips_empty_bids_and_asks(self):
        store = self._make_store_mock()
        state = {"last_update_time": None, "first_update_done": False, "session_count": 0}
        HandlerClass = self._make_handler(store, state)
        h = HandlerClass()
        h.on_recv_rsp(self._make_data(bids=[], asks=[]))
        store.insert_snapshot.assert_not_called()

    def test_handler_updates_state_after_insert(self):
        store = self._make_store_mock()
        state = {"last_update_time": None, "first_update_done": False, "session_count": 0}
        HandlerClass = self._make_handler(store, state)
        h = HandlerClass()
        h.on_recv_rsp(self._make_data())
        self.assertIsNotNone(state["last_update_time"])
        self.assertTrue(state["first_update_done"])
        self.assertEqual(state["session_count"], 4)   # insert returned 4

    def test_handler_accumulates_session_count(self):
        store = self._make_store_mock()
        store.insert_snapshot.return_value = 2
        state = {"last_update_time": None, "first_update_done": False, "session_count": 0}
        HandlerClass = self._make_handler(store, state)
        h = HandlerClass()
        h.on_recv_rsp(self._make_data())
        h.on_recv_rsp(self._make_data())
        self.assertEqual(state["session_count"], 4)

    def test_handler_ret_error_skips_insert(self):
        store = self._make_store_mock()
        state = {"last_update_time": None, "first_update_done": False, "session_count": 0}

        # Override base to return error code
        class ErrBase:
            def on_recv_rsp(self, rsp_pb):
                return 1, None   # ret != RET_OK

        self._moomoo_mock.OrderBookHandlerBase = ErrBase
        from analysis.order_book_collector import _make_handler
        HandlerClass = _make_handler(store, state)
        h = HandlerClass()
        h.on_recv_rsp(self._make_data())
        store.insert_snapshot.assert_not_called()

    def test_handler_parses_sequence_format_bids(self):
        store = self._make_store_mock()
        state = {"last_update_time": None, "first_update_done": False, "session_count": 0}
        HandlerClass = self._make_handler(store, state)
        h = HandlerClass()
        data = self._make_data(bids=[[100.0, 500], [99.5, 300]], asks=[])
        h.on_recv_rsp(data)
        args = store.insert_snapshot.call_args[0]
        self.assertEqual(args[2], [(100.0, 500), (99.5, 300)])   # bids positional arg


# ── _make_handler partial-push merge (bid/ask carry-forward) ──────────────────

class TestMakeHandlerPartialUpdateMerge(unittest.TestCase):
    """ORDER_BOOK pushes can be partial: Qot_GetOrderBook.proto documents
    svrRecvTimeBid/svrRecvTimeAsk as separate fields precisely because a push
    can carry a fresh update for one side while the other is stale/unchanged.
    Treating each push as a complete snapshot would make the untouched side
    vanish from the stored "latest" row until the next push that happens to
    include it -- these tests cover the merge-cache that keeps it carried
    forward instead.
    """

    def setUp(self):
        self._moomoo_mock = MagicMock()
        self._moomoo_mock.RET_OK = 0

        class FakeBase:
            def on_recv_rsp(self, rsp_pb):
                return 0, rsp_pb

        self._moomoo_mock.OrderBookHandlerBase = FakeBase
        self._patcher = patch.dict(sys.modules, {"moomoo": self._moomoo_mock})
        self._patcher.start()
        from analysis.order_book_collector import _make_handler
        self._make_handler = _make_handler

    def tearDown(self):
        self._patcher.stop()

    def _make_store_mock(self):
        store = MagicMock()
        store.insert_snapshot.return_value = 1
        return store

    def test_bid_only_push_carries_forward_last_known_asks(self):
        store = self._make_store_mock()
        state = {"last_update_time": None, "first_update_done": False, "session_count": 0}
        HandlerClass = self._make_handler(store, state)
        h = HandlerClass()

        with patch("analysis.order_book_collector.time.time", side_effect=[100.0, 110.0]):
            h.on_recv_rsp({"code": "US.SOXL",
                          "Bid": [{"price": 100.0, "volume": 500}],
                          "Ask": [{"price": 101.0, "volume": 300}]})
            # Partial push: only Bid changed, Ask omitted (empty) this time.
            h.on_recv_rsp({"code": "US.SOXL",
                          "Bid": [{"price": 100.1, "volume": 600}],
                          "Ask": []})

        self.assertEqual(store.insert_snapshot.call_count, 2)
        bids_arg, asks_arg = store.insert_snapshot.call_args_list[1][0][2:4]
        self.assertEqual(bids_arg, [(100.1, 600)])
        self.assertEqual(asks_arg, [(101.0, 300)])   # carried forward, not blanked

    def test_ask_only_push_carries_forward_last_known_bids(self):
        store = self._make_store_mock()
        state = {"last_update_time": None, "first_update_done": False, "session_count": 0}
        HandlerClass = self._make_handler(store, state)
        h = HandlerClass()

        with patch("analysis.order_book_collector.time.time", side_effect=[100.0, 110.0]):
            h.on_recv_rsp({"code": "US.SOXL",
                          "Bid": [{"price": 100.0, "volume": 500}],
                          "Ask": [{"price": 101.0, "volume": 300}]})
            h.on_recv_rsp({"code": "US.SOXL",
                          "Bid": [],
                          "Ask": [{"price": 101.2, "volume": 250}]})

        bids_arg, asks_arg = store.insert_snapshot.call_args_list[1][0][2:4]
        self.assertEqual(bids_arg, [(100.0, 500)])   # carried forward, not blanked
        self.assertEqual(asks_arg, [(101.2, 250)])

    def test_merge_cache_is_per_code(self):
        store = self._make_store_mock()
        state = {"last_update_time": None, "first_update_done": False, "session_count": 0}
        HandlerClass = self._make_handler(store, state)
        h = HandlerClass()

        with patch("analysis.order_book_collector.time.time", side_effect=[100.0, 110.0]):
            h.on_recv_rsp({"code": "US.SOXL",
                          "Bid": [{"price": 100.0, "volume": 500}], "Ask": []})
            # A different code's partial push must not pull in SOXL's cached bids.
            h.on_recv_rsp({"code": "US.AAPL",
                          "Bid": [], "Ask": [{"price": 200.0, "volume": 10}]})

        self.assertEqual(store.insert_snapshot.call_count, 2)
        aapl_bids, aapl_asks = store.insert_snapshot.call_args_list[1][0][2:4]
        self.assertEqual(aapl_bids, [])   # not contaminated by SOXL's cached bids
        self.assertEqual(aapl_asks, [(200.0, 10)])


if __name__ == "__main__":
    unittest.main()

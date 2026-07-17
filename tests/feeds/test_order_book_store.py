"""Unit tests for feeds.order_book_store.OrderBookStore.

Uses a temporary SQLite file — no network or moomoo dependency.
Run: uv run pytest tests/ -v
"""

import tempfile
import pathlib
import unittest
from datetime import datetime, date, timedelta

from feeds.order_book_store import OrderBookStore

T0 = datetime(2026, 5, 30, 9, 30, 0)
CODE = "US.AAPL"


def _make_store(tmp_dir) -> OrderBookStore:
    db = pathlib.Path(tmp_dir) / "test_ob.db"
    return OrderBookStore(db)


class TestInsert(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = _make_store(self._tmp.name)

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()

    def test_empty_bids_asks_returns_zero(self):
        self.assertEqual(self.store.insert_snapshot(CODE, T0, [], []), 0)

    def test_bids_only_inserted(self):
        n = self.store.insert_snapshot(CODE, T0, [(100.0, 500), (99.5, 300)], [])
        self.assertEqual(n, 2)
        self.assertEqual(self.store.row_count(CODE), 2)

    def test_asks_only_inserted(self):
        n = self.store.insert_snapshot(CODE, T0, [], [(101.0, 400), (101.5, 200)])
        self.assertEqual(n, 2)

    def test_both_sides_inserted(self):
        n = self.store.insert_snapshot(
            CODE, T0,
            [(100.0, 500), (99.5, 300)],
            [(101.0, 400), (101.5, 200)],
        )
        self.assertEqual(n, 4)
        self.assertEqual(self.store.row_count(CODE), 4)

    def test_row_count_all_codes(self):
        self.store.insert_snapshot(CODE, T0, [(100.0, 100)], [])
        self.store.insert_snapshot("US.TSLA", T0, [(200.0, 200)], [])
        self.assertEqual(self.store.row_count(), 2)

    def test_row_count_filtered_by_code(self):
        self.store.insert_snapshot(CODE, T0, [(100.0, 100)], [(101.0, 50)])
        self.store.insert_snapshot("US.TSLA", T0, [(200.0, 200)], [])
        self.assertEqual(self.store.row_count(CODE), 2)
        self.assertEqual(self.store.row_count("US.TSLA"), 1)

    def test_ts_string_accepted(self):
        n = self.store.insert_snapshot(CODE, "2026-05-30 09:30:00", [(100.0, 100)], [])
        self.assertEqual(n, 1)


class TestQuery(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = _make_store(self._tmp.name)
        # Insert snapshots at T0, T0+30s, T0+60s
        for i in range(3):
            ts = T0 + timedelta(seconds=i * 30)
            self.store.insert_snapshot(
                CODE, ts,
                [(100.0 - i * 0.5, 500)],
                [(101.0 + i * 0.5, 400)],
            )

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()

    def test_query_all_in_range(self):
        rows = self.store.query_snapshots(CODE, T0, T0 + timedelta(minutes=5))
        # 3 snapshots × 2 levels each = 6 rows
        self.assertEqual(len(rows), 6)

    def test_query_start_inclusive(self):
        rows = self.store.query_snapshots(CODE, T0, T0 + timedelta(seconds=1))
        self.assertEqual(len(rows), 2)   # only T0 snapshot

    def test_query_end_exclusive(self):
        # end = T0+30s → T0+30s snapshot NOT included
        rows = self.store.query_snapshots(CODE, T0, T0 + timedelta(seconds=30))
        self.assertEqual(len(rows), 2)   # only T0

    def test_query_outside_range_empty(self):
        rows = self.store.query_snapshots(
            CODE,
            T0 + timedelta(hours=1),
            T0 + timedelta(hours=2),
        )
        self.assertEqual(rows, [])

    def test_query_returns_correct_fields(self):
        rows = self.store.query_snapshots(CODE, T0, T0 + timedelta(seconds=1))
        self.assertIn("ts",     rows[0])
        self.assertIn("side",   rows[0])
        self.assertIn("price",  rows[0])
        self.assertIn("volume", rows[0])

    def test_query_ts_is_datetime(self):
        rows = self.store.query_snapshots(CODE, T0, T0 + timedelta(seconds=1))
        self.assertIsInstance(rows[0]["ts"], datetime)

    def test_query_results_ordered_by_ts(self):
        rows = self.store.query_snapshots(CODE, T0, T0 + timedelta(minutes=5))
        ts_list = [r["ts"] for r in rows]
        self.assertEqual(ts_list, sorted(ts_list))

    def test_query_side_values(self):
        rows = self.store.query_snapshots(CODE, T0, T0 + timedelta(seconds=1))
        sides = {r["side"] for r in rows}
        self.assertEqual(sides, {"BID", "ASK"})

    def test_multiple_codes_isolated(self):
        self.store.insert_snapshot("US.TSLA", T0, [(200.0, 100)], [])
        rows = self.store.query_snapshots(CODE, T0, T0 + timedelta(minutes=5))
        codes_in_result = {r.get("code") for r in rows}
        # query_snapshots doesn't return code field, but verify TSLA data not mixed in
        self.assertEqual(len(rows), 6)   # unchanged from setUp

    def test_query_date_matches_query_snapshots(self):
        day = T0.date()
        rows_date  = self.store.query_date(CODE, day)
        rows_range = self.store.query_snapshots(
            CODE,
            datetime(day.year, day.month, day.day),
            datetime(day.year, day.month, day.day, 23, 59, 59, 999999),
        )
        self.assertEqual(len(rows_date), len(rows_range))


class TestPrune(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = _make_store(self._tmp.name)

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()

    def _insert_n(self, code: str, n: int) -> None:
        for i in range(n):
            self.store.insert_snapshot(
                code, T0 + timedelta(seconds=i), [(100.0, 100)], [(101.0, 50)]
            )

    def test_keeps_most_recent_rows_per_code(self):
        self._insert_n(CODE, 10)   # 20 rows (2 levels/snapshot)
        deleted = self.store.prune(keep=6)
        self.assertEqual(deleted, 14)   # 20 - 6
        self.assertEqual(self.store.row_count(CODE), 6)

    def test_keeps_newest_not_oldest(self):
        # 3 snapshots = 6 rows; keep=2 *rows* -> only the newest snapshot's
        # 2 rows survive (T0+2s) -- T0 and T0+1s must both be gone.
        self._insert_n(CODE, 3)
        self.store.prune(keep=2)
        rows = self.store.query_snapshots(CODE, T0, T0 + timedelta(seconds=2))
        self.assertEqual(rows, [])   # T0 and T0+1s rows were pruned
        rows = self.store.query_snapshots(CODE, T0 + timedelta(seconds=2),
                                          T0 + timedelta(seconds=3))
        self.assertEqual(len(rows), 2)   # only the newest snapshot remains

    def test_under_keep_threshold_deletes_nothing(self):
        self._insert_n(CODE, 3)
        deleted = self.store.prune(keep=1000)
        self.assertEqual(deleted, 0)
        self.assertEqual(self.store.row_count(CODE), 6)

    def test_codes_param_scopes_to_subset(self):
        # Mirrors order_book_collector.py's per-code retention override: a
        # single call must only touch the codes passed in, leaving others
        # completely alone regardless of their own row count.
        self._insert_n(CODE, 10)
        self._insert_n("US.TSLA", 10)
        deleted = self.store.prune(keep=2, codes=[CODE])
        self.assertEqual(self.store.row_count(CODE), 2)
        self.assertEqual(self.store.row_count("US.TSLA"), 20)   # untouched

    def test_different_keep_per_code_via_two_calls(self):
        # The actual pattern _watchdog uses: a low default keep for most
        # codes, a higher override keep for specific ones (e.g. SOXL).
        # keep is a *row* count, not a snapshot count.
        self._insert_n(CODE, 10)
        self._insert_n("US.SOXL", 10)
        self.store.prune(keep=8, codes=["US.SOXL"])
        self.store.prune(keep=2, codes=[CODE])
        self.assertEqual(self.store.row_count("US.SOXL"), 8)
        self.assertEqual(self.store.row_count(CODE), 2)

    def test_no_codes_arg_defaults_to_all(self):
        self._insert_n(CODE, 10)
        self._insert_n("US.TSLA", 10)
        self.store.prune(keep=2)
        self.assertEqual(self.store.row_count(CODE), 2)
        self.assertEqual(self.store.row_count("US.TSLA"), 2)

    def test_empty_codes_list_deletes_nothing(self):
        self._insert_n(CODE, 10)
        deleted = self.store.prune(keep=1, codes=[])
        self.assertEqual(deleted, 0)
        self.assertEqual(self.store.row_count(CODE), 20)


class TestAvailableDates(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = _make_store(self._tmp.name)

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()

    def test_empty_db_returns_empty(self):
        self.assertEqual(self.store.available_dates(CODE), [])

    def test_single_date(self):
        self.store.insert_snapshot(CODE, T0, [(100.0, 100)], [])
        self.assertEqual(self.store.available_dates(CODE), [T0.date()])

    def test_multiple_dates_deduplicated(self):
        for day_offset in [0, 0, 1, 2]:   # two inserts on day 0
            ts = T0 + timedelta(days=day_offset)
            self.store.insert_snapshot(CODE, ts, [(100.0, 100)], [])
        dates = self.store.available_dates(CODE)
        self.assertEqual(len(dates), 3)
        self.assertEqual(dates, sorted(dates))

    def test_code_isolation(self):
        self.store.insert_snapshot(CODE, T0, [(100.0, 100)], [])
        self.store.insert_snapshot("US.TSLA", T0 + timedelta(days=1), [(200.0, 100)], [])
        self.assertEqual(self.store.available_dates(CODE), [T0.date()])
        self.assertEqual(len(self.store.available_dates("US.TSLA")), 1)


class TestContextManager(unittest.TestCase):

    def test_with_statement(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = pathlib.Path(tmp) / "ctx.db"
            with OrderBookStore(db) as store:
                n = store.insert_snapshot(CODE, T0, [(100.0, 100)], [])
                self.assertEqual(n, 1)
            # Connection should be closed; verify DB file exists
            self.assertTrue(db.exists())

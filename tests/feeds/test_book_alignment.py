"""The order book's two clocks, and which one a read hands back.

A snapshot is stored with both the receiving machine's wall clock (`ts`) and
the quote server's (`svr_bid_ts` / `svr_ask_ts`). The local one free-runs --
measured -39 ms/h on this host -- so filing books under it slides them against
the tape by most of a heatmap column a day. Reads therefore return the aligned
time: the server stamp minus a constant transport lead.

What these cover, in the order they matter:

  - a row whose local clock is wrong still comes back at the right time
  - bucketing happens on the aligned clock, not the stored one, because
    labelling a column with one clock and filling it from another is exactly
    the off-by-a-column this is meant to remove
  - rows written before the server stamps existed behave as they always did
  - the two clocks are never mixed in one comparison

Every test writes its own throwaway database. None of them touch db/.
"""
from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from feeds.order_book_store import (          # noqa: E402
    SERVER_LEAD_SECS, OrderBookStore, _aligned_ts, _newer_svr,
)

_ET = ZoneInfo("America/New_York")

BIDS = [(10.00, 100), (9.99, 200)]
ASKS = [(10.01, 150), (10.02, 250)]


def epoch(ts: datetime) -> float:
    """A naive ET datetime as the true unix epoch the feed would report."""
    return ts.replace(tzinfo=_ET).timestamp()


def svr_for(true_ts: datetime) -> float:
    """The server stamp a book standing at *true_ts* would arrive carrying."""
    return epoch(true_ts) + SERVER_LEAD_SECS


class _StoreCase(unittest.TestCase):
    """Opens stores and closes them again -- Windows will not delete a file
    with an open SQLite handle, and a leaked one fails teardown rather than
    the test that leaked it."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._open: list[OrderBookStore] = []

    def tearDown(self) -> None:
        for s in self._open:
            s.close()
        self._tmp.cleanup()

    def store(self, name: str = "ob.db") -> OrderBookStore:
        s = OrderBookStore(pathlib.Path(self._tmp.name) / name)
        self._open.append(s)
        return s


class TestWhichClockComesBack(_StoreCase):

    def test_a_row_with_server_stamps_is_filed_under_them(self):
        store = self.store()
        true_ts = datetime(2026, 11, 3, 10, 0, 0)
        # The local clock is a second fast; without alignment the row would
        # come back a second late.
        store.insert_snapshot("US.SOXL", true_ts + timedelta(seconds=1),
                              BIDS, ASKS,
                              svr_bid_ts=svr_for(true_ts),
                              svr_ask_ts=svr_for(true_ts))
        got = store.latest_snapshot("US.SOXL")
        self.assertTrue(got)
        self.assertAlmostEqual(
            (got[0]["ts"] - true_ts).total_seconds(), 0.0, places=3)

    def test_a_row_without_them_keeps_the_time_it_was_stored_with(self):
        store = self.store()
        ts = datetime(2026, 10, 1, 9, 30, 0, 500000)
        store.insert_snapshot("US.SOXL", ts, BIDS, ASKS)
        got = store.latest_snapshot("US.SOXL")
        self.assertEqual(got[0]["ts"], ts)

    def test_the_lead_is_subtracted_not_added(self):
        """A sign slip here moves every book half a column the wrong way and
        nothing else complains, so it gets its own test."""
        store = self.store()
        true_ts = datetime(2026, 11, 3, 10, 0, 0)
        raw = epoch(true_ts)
        store.insert_snapshot("US.SOXL", true_ts, BIDS, ASKS,
                              svr_bid_ts=raw, svr_ask_ts=raw)
        got = store.latest_snapshot("US.SOXL")
        self.assertAlmostEqual(
            (got[0]["ts"] - true_ts).total_seconds(),
            -SERVER_LEAD_SECS, places=3)

    def test_every_level_of_a_snapshot_carries_the_same_aligned_time(self):
        store = self.store()
        true_ts = datetime(2026, 11, 3, 10, 0, 0)
        store.insert_snapshot("US.SOXL", true_ts + timedelta(seconds=3),
                              BIDS, ASKS, svr_bid_ts=svr_for(true_ts),
                              svr_ask_ts=svr_for(true_ts))
        got = store.latest_snapshot("US.SOXL")
        self.assertEqual(len(got), 4)
        self.assertEqual(len({r["ts"] for r in got}), 1)


class TestOneSidedPushes(unittest.TestCase):
    """_newer_svr picks the stamp; a push can refresh one side only."""

    def test_the_later_side_wins(self):
        self.assertEqual(_newer_svr(100.0, 150.0), 150.0)
        self.assertEqual(_newer_svr(150.0, 100.0), 150.0)

    def test_a_missing_side_is_ignored_rather_than_poisoning_the_pair(self):
        self.assertEqual(_newer_svr(None, 150.0), 150.0)
        self.assertEqual(_newer_svr(150.0, None), 150.0)

    def test_a_zero_counts_as_absent(self):
        """Qot_GetOrderBook.proto uses a zero server time to say 'this side
        is unchanged', so a zero must never be read as 1970."""
        self.assertEqual(_newer_svr(0.0, 150.0), 150.0)
        self.assertIsNone(_newer_svr(0.0, 0.0))

    def test_neither_side_falls_back_to_the_stored_string(self):
        ts = datetime(2026, 10, 1, 9, 30, 0)
        self.assertEqual(_aligned_ts(ts.isoformat(sep=" "), None, None), ts)


class TestBucketingUsesTheAlignedClock(_StoreCase):
    """The column a snapshot lands in, not just the label it wears."""

    def test_rows_sharing_a_stored_bucket_split_when_aligned_apart(self):
        store = self.store()
        base = datetime(2026, 11, 3, 10, 0, 0)
        # Both stored inside the same wall-clock second; their server stamps
        # put them two seconds apart. Bucketing on the stored clock returns
        # one column, on the aligned clock two.
        store.insert_snapshot("US.SOXL", base, BIDS, ASKS,
                              svr_bid_ts=svr_for(base),
                              svr_ask_ts=svr_for(base))
        store.insert_snapshot("US.SOXL", base + timedelta(milliseconds=400),
                              BIDS, ASKS,
                              svr_bid_ts=svr_for(base + timedelta(seconds=2)),
                              svr_ask_ts=svr_for(base + timedelta(seconds=2)))
        got = store.bucketed_snapshots(
            "US.SOXL", base - timedelta(seconds=5),
            base + timedelta(seconds=10), 1)
        self.assertEqual(len(got), 2)

    def test_rows_stored_a_bucket_apart_merge_when_aligned_together(self):
        store = self.store()
        base = datetime(2026, 11, 3, 10, 0, 0)
        for stored_offset in (0, 3):
            store.insert_snapshot(
                "US.SOXL", base + timedelta(seconds=stored_offset),
                BIDS, ASKS,
                svr_bid_ts=svr_for(base), svr_ask_ts=svr_for(base))
        got = store.bucketed_snapshots(
            "US.SOXL", base - timedelta(seconds=5),
            base + timedelta(seconds=10), 1)
        self.assertEqual(len(got), 1)

    def test_results_come_back_in_aligned_order_not_insertion_order(self):
        store = self.store()
        base = datetime(2026, 11, 3, 10, 0, 0)
        # Written newest-aligned first, so rowid order is the reverse of the
        # order a caller painting left to right needs.
        for aligned_offset in (6, 2):
            t = base + timedelta(seconds=aligned_offset)
            store.insert_snapshot("US.SOXL", base, BIDS, ASKS,
                                  svr_bid_ts=svr_for(t), svr_ask_ts=svr_for(t))
        got = store.bucketed_snapshots(
            "US.SOXL", base - timedelta(seconds=5),
            base + timedelta(seconds=20), 1)
        times = [g[0]["ts"] for g in got]
        self.assertEqual(times, sorted(times))

    def test_the_range_is_applied_to_the_aligned_time(self):
        store = self.store()
        base = datetime(2026, 11, 3, 10, 0, 0)
        # Stored inside the window, aligned three seconds past its end.
        store.insert_snapshot("US.SOXL", base, BIDS, ASKS,
                              svr_bid_ts=svr_for(base + timedelta(seconds=3)),
                              svr_ask_ts=svr_for(base + timedelta(seconds=3)))
        got = store.bucketed_snapshots(
            "US.SOXL", base - timedelta(seconds=1),
            base + timedelta(seconds=1), 1)
        self.assertEqual(got, [])

    def test_a_legacy_row_still_buckets_where_it_always_did(self):
        store = self.store()
        base = datetime(2026, 10, 1, 10, 0, 0)
        for off in (0, 5):
            store.insert_snapshot("US.SOXL", base + timedelta(seconds=off),
                                  BIDS, ASKS)
        got = store.bucketed_snapshots(
            "US.SOXL", base, base + timedelta(seconds=10), 1)
        self.assertEqual([g[0]["ts"] for g in got],
                         [base, base + timedelta(seconds=5)])

    def test_legacy_and_stamped_rows_are_not_compared_in_different_epochs(self):
        """SQLite reads the stored string as UTC while a server stamp is a
        true epoch. Left unreconciled the two frames sit five hours apart, so
        a mixed query would scatter the legacy rows into unrelated buckets."""
        store = self.store()
        base = datetime(2026, 11, 3, 10, 0, 0)
        store.insert_snapshot("US.SOXL", base, BIDS, ASKS)          # legacy
        later = base + timedelta(seconds=30)
        store.insert_snapshot("US.SOXL", later, BIDS, ASKS,
                              svr_bid_ts=svr_for(later),
                              svr_ask_ts=svr_for(later))
        got = store.bucketed_snapshots(
            "US.SOXL", base - timedelta(seconds=10),
            base + timedelta(seconds=60), 10)
        times = [g[0]["ts"] for g in got]
        self.assertEqual(len(times), 2)
        self.assertEqual(times, sorted(times))
        self.assertLess((times[1] - times[0]).total_seconds(), 31)

    def test_a_legacy_and_a_stamped_row_in_one_second_make_one_column(self):
        """The sharp end of the frame question. A whole-hour error in the
        legacy branch is invisible while the two kinds of row sit in separate
        buckets anyway -- an ET offset is a multiple of every bucket size in
        use, so it shifts each key by a constant and regroups nothing. It
        only shows when a legacy row and a stamped one belong in the SAME
        bucket, which is what the 2026-11-03 cutover produced."""
        store = self.store()
        base = datetime(2026, 11, 3, 10, 0, 0)
        store.insert_snapshot("US.SOXL", base, BIDS, ASKS)          # legacy
        near = base + timedelta(milliseconds=300)
        store.insert_snapshot("US.SOXL", near, BIDS, ASKS,
                              svr_bid_ts=svr_for(near),
                              svr_ask_ts=svr_for(near))
        got = store.bucketed_snapshots(
            "US.SOXL", base - timedelta(seconds=5),
            base + timedelta(seconds=5), 1)
        self.assertEqual(len(got), 1)

    def test_a_row_the_clock_pushed_out_of_the_scan_is_still_found(self):
        """The indexed scan runs on the stored clock, so a row whose aligned
        time lands inside the window while its stored one sits outside is
        only reachable because the scan is widened first."""
        store = self.store()
        base = datetime(2026, 11, 3, 10, 0, 0)
        aligned = base + timedelta(seconds=1)
        store.insert_snapshot("US.SOXL", base - timedelta(seconds=2),
                              BIDS, ASKS,
                              svr_bid_ts=svr_for(aligned),
                              svr_ask_ts=svr_for(aligned))
        got = store.bucketed_snapshots(
            "US.SOXL", base, base + timedelta(seconds=5), 1)
        self.assertEqual(len(got), 1)
        self.assertAlmostEqual(
            (got[0][0]["ts"] - aligned).total_seconds(), 0.0, places=3)


class TestPointLookupUsesTheAlignedClock(_StoreCase):

    def test_at_or_before_compares_against_the_aligned_time(self):
        store = self.store()
        base = datetime(2026, 11, 3, 10, 0, 0)
        # Stored two seconds before the asked-for moment, aligned two after.
        store.insert_snapshot("US.SOXL", base - timedelta(seconds=2),
                              BIDS, ASKS,
                              svr_bid_ts=svr_for(base + timedelta(seconds=2)),
                              svr_ask_ts=svr_for(base + timedelta(seconds=2)))
        self.assertEqual(store.snapshot_at_or_before("US.SOXL", base), [])

    def test_it_returns_the_newest_qualifying_row(self):
        store = self.store()
        base = datetime(2026, 11, 3, 10, 0, 0)
        for off, price in ((-5, 9.0), (-1, 9.5), (4, 9.9)):
            t = base + timedelta(seconds=off)
            store.insert_snapshot("US.SOXL", base, [(price, 10)], ASKS,
                                  svr_bid_ts=svr_for(t), svr_ask_ts=svr_for(t))
        got = store.snapshot_at_or_before("US.SOXL", base)
        self.assertEqual(got[0]["price"], 9.5)

    def test_a_legacy_row_is_found_exactly_as_before(self):
        store = self.store()
        base = datetime(2026, 10, 1, 10, 0, 0)
        store.insert_snapshot("US.SOXL", base - timedelta(seconds=1),
                              BIDS, ASKS)
        got = store.snapshot_at_or_before("US.SOXL", base)
        self.assertEqual(got[0]["ts"], base - timedelta(seconds=1))


class TestMissingServerStampMonitor(_StoreCase):
    """svrRecvTime*Timestamp is documented Hong Kong only. If the feed stops
    sending it, alignment silently falls back to the drifting local clock --
    nothing raises, so something has to count."""

    def test_it_counts_rows_with_neither_stamp(self):
        store = self.store()
        base = datetime(2026, 11, 3, 10, 0, 0)
        store.insert_snapshot("US.SOXL", base, BIDS, ASKS,
                              svr_bid_ts=svr_for(base),
                              svr_ask_ts=svr_for(base))
        store.insert_snapshot("US.SOXL", base + timedelta(seconds=1),
                              BIDS, ASKS)
        self.assertEqual(store.missing_server_ts_rate("US.SOXL"), (1, 2))

    def test_one_stamp_is_enough_to_count_as_present(self):
        store = self.store()
        base = datetime(2026, 11, 3, 10, 0, 0)
        store.insert_snapshot("US.SOXL", base, BIDS, ASKS,
                              svr_bid_ts=svr_for(base))
        self.assertEqual(store.missing_server_ts_rate("US.SOXL"), (0, 1))

    def test_since_excludes_the_rows_written_before_the_columns_existed(self):
        store = self.store()
        old = datetime(2026, 10, 1, 10, 0, 0)
        new = datetime(2026, 11, 3, 10, 0, 0)
        store.insert_snapshot("US.SOXL", old, BIDS, ASKS)
        store.insert_snapshot("US.SOXL", new, BIDS, ASKS,
                              svr_bid_ts=svr_for(new), svr_ask_ts=svr_for(new))
        self.assertEqual(
            store.missing_server_ts_rate("US.SOXL", since=new), (0, 1))

    def test_an_unseen_code_reports_nothing_rather_than_dividing_by_zero(self):
        store = self.store()
        self.assertEqual(store.missing_server_ts_rate("US.NOPE"), (0, 0))


class TestRangeFramesStayStored(_StoreCase):
    """latest_ts/earliest_ts feed range arguments that the index is built on,
    so they must stay in the stored frame even though reads do not."""

    def test_latest_ts_reports_the_stored_time(self):
        store = self.store()
        true_ts = datetime(2026, 11, 3, 10, 0, 0)
        stored = true_ts + timedelta(seconds=1)
        store.insert_snapshot("US.SOXL", stored, BIDS, ASKS,
                              svr_bid_ts=svr_for(true_ts),
                              svr_ask_ts=svr_for(true_ts))
        self.assertEqual(store.latest_ts("US.SOXL"), stored)

    def test_a_prefill_window_built_from_it_still_finds_its_rows(self):
        """The heatmap's prefill asks latest_ts() for an anchor and hands the
        result straight back as a range bound. A mismatch between the two
        frames would return an empty grid, which is how this would show up."""
        store = self.store()
        base = datetime(2026, 11, 3, 10, 0, 0)
        for i in range(5):
            t = base + timedelta(seconds=i)
            store.insert_snapshot("US.SOXL", t, BIDS, ASKS,
                                  svr_bid_ts=svr_for(t), svr_ask_ts=svr_for(t))
        newest = store.latest_ts("US.SOXL")
        got = store.bucketed_snapshots(
            "US.SOXL", newest - timedelta(seconds=10),
            newest + timedelta(seconds=1), 1)
        self.assertEqual(len(got), 5)


if __name__ == "__main__":
    unittest.main()

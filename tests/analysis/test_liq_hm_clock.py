"""The heatmap's live path files its columns on the aligned clock.

The database path gets this for free -- OrderBookStore hands back aligned
timestamps -- but the window also subscribes to ORDER_BOOK itself, and those
pushes never touch the store. If only one of the two were corrected the grid
would hold columns on two different clocks, which is worse than holding them
all on the wrong one.

Two things are checked: the correction computed from a push, and that a
column with no timestamp of its own picks it up. The second matters because
the timer's fill columns have no push behind them -- they are what keeps the
axis advancing between pushes, and they used to take a bare datetime.now().

No Qt widget is constructed. _push_column is called against a stub carrying
the attributes it touches, so the real function runs.
"""
from __future__ import annotations

import pathlib
import sys
import types
import unittest
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from analysis.liq_hm_window import (           # noqa: E402
    LiqHmWindow, _clock_correction,
)
from feeds.order_book_store import SERVER_LEAD_SECS   # noqa: E402

_ET = ZoneInfo("America/New_York")


class TestClockCorrection(unittest.TestCase):

    def test_a_push_in_step_with_the_local_clock_still_owes_the_lead(self):
        """The server stamp leads the trade it describes even when the local
        clock is right, so a correct clock does not mean a zero correction."""
        now = 1_000_000.0
        self.assertAlmostEqual(_clock_correction(now, now, now),
                               -SERVER_LEAD_SECS, places=6)

    def test_a_slow_local_clock_gives_a_positive_correction(self):
        now = 1_000_000.0          # this machine thinks it is this
        svr = now + 1.5            # the server says it is later
        self.assertAlmostEqual(_clock_correction(svr, svr, now),
                               1.5 - SERVER_LEAD_SECS, places=6)

    def test_a_fast_local_clock_gives_a_negative_correction(self):
        now = 1_000_000.0
        svr = now - 1.5
        self.assertAlmostEqual(_clock_correction(svr, svr, now),
                               -1.5 - SERVER_LEAD_SECS, places=6)

    def test_the_later_side_wins_on_a_one_sided_push(self):
        now = 1_000_000.0
        self.assertAlmostEqual(_clock_correction(now, now + 2.0, now),
                               2.0 - SERVER_LEAD_SECS, places=6)
        self.assertAlmostEqual(_clock_correction(now + 2.0, now, now),
                               2.0 - SERVER_LEAD_SECS, places=6)

    def test_no_stamp_leaves_the_old_behaviour_alone(self):
        self.assertEqual(_clock_correction(None, None, 1_000_000.0), 0.0)

    def test_a_zero_stamp_counts_as_absent_not_as_1970(self):
        now = 1_000_000.0
        self.assertEqual(_clock_correction(0.0, 0.0, now), 0.0)
        self.assertAlmostEqual(_clock_correction(0.0, now, now),
                               -SERVER_LEAD_SECS, places=6)


def _stub(corr: float) -> types.SimpleNamespace:
    """The slice of LiqHmWindow that _push_column touches."""
    s = types.SimpleNamespace(
        _max_cols_spin=types.SimpleNamespace(value=lambda: 10),
        _min_vol_spin=types.SimpleNamespace(value=lambda: 0),
        _col_ts=[], _mid_prices=[], _raw_snaps=[], _latest_snap=None,
        _snap_seq=0, _clock_corr=corr,
        _price_min=9.0, _price_max=11.0, _bin_size=0.01,
        _best_bid=None, _best_ask=None,
        _bid_grid=np.zeros((10, 200)), _ask_grid=np.zeros((10, 200)),
    )
    s._paint_column = types.MethodType(LiqHmWindow._paint_column, s)
    return s


SNAP = [{"side": "BID", "price": 10.00, "volume": 100},
        {"side": "ASK", "price": 10.01, "volume": 100}]


class TestUnstampedColumnsUseTheCorrection(unittest.TestCase):

    def _stamp(self, corr: float) -> datetime:
        stub = _stub(corr)
        before = datetime.now(_ET).replace(tzinfo=None)
        LiqHmWindow._push_column(stub, SNAP)
        after = datetime.now(_ET).replace(tzinfo=None)
        self.assertEqual(len(stub._col_ts), 1)
        # The call is instantaneous next to the offsets under test, so the
        # stamp is pinned to a window of a few milliseconds plus corr.
        self.assertGreaterEqual(stub._col_ts[0],
                                before + timedelta(seconds=corr))
        self.assertLessEqual(stub._col_ts[0],
                             after + timedelta(seconds=corr))
        return stub._col_ts[0]

    def test_a_zero_correction_stamps_the_local_clock(self):
        got = self._stamp(0.0)
        now = datetime.now(_ET).replace(tzinfo=None)
        self.assertLess(abs((got - now).total_seconds()), 1.0)

    def test_a_negative_correction_moves_the_column_earlier(self):
        got = self._stamp(-1.5)
        now = datetime.now(_ET).replace(tzinfo=None)
        self.assertLess((got - now).total_seconds(), -1.0)

    def test_a_positive_correction_moves_the_column_later(self):
        got = self._stamp(2.5)
        now = datetime.now(_ET).replace(tzinfo=None)
        self.assertGreater((got - now).total_seconds(), 2.0)

    def test_an_explicit_timestamp_is_never_second_guessed(self):
        """Pre-fill and replay pass a timestamp that is already aligned --
        applying the correction again would double-count it."""
        stub = _stub(-5.0)
        ts = datetime(2026, 11, 3, 10, 0, 0)
        LiqHmWindow._push_column(stub, SNAP, ts=ts)
        self.assertEqual(stub._col_ts, [ts])

    def test_the_recorded_snapshot_carries_the_same_stamp_as_the_column(self):
        """_raw_snaps feeds the iceberg and spoof detectors, which bucket by
        time; a column on one clock and its rows on another would split a
        single book across two buckets."""
        stub = _stub(-1.5)
        LiqHmWindow._push_column(stub, SNAP)
        self.assertTrue(stub._raw_snaps)
        for row in stub._raw_snaps:
            self.assertEqual(row["ts"], stub._col_ts[0])


class TestPrefillSpansBothClockFrames(unittest.TestCase):
    """The prefill asks latest_ts() for an anchor and hands it straight back
    as a range bound -- but latest_ts() answers in the stored frame while the
    range is trimmed in the aligned one. The gap between them is the clock
    error, so a bound tight enough to work when the two frames agreed starts
    dropping the newest columns once the machine has drifted past it.
    """

    def test_a_drifted_clock_does_not_cost_the_newest_columns(self):
        from unittest import mock

        from analysis import liq_hm_window as hm
        from feeds.order_book_store import OrderBookStore, SERVER_LEAD_SECS

        import tempfile

        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "ob.db"
            base = datetime(2026, 11, 3, 10, 0, 0)
            drift = 3.0        # past the old one-second bound, inside the pad
            store = OrderBookStore(path)
            try:
                for i in range(5):
                    stored = base + timedelta(seconds=i)
                    svr = ((stored + timedelta(seconds=drift))
                           .replace(tzinfo=_ET).timestamp() + SERVER_LEAD_SECS)
                    store.insert_snapshot(
                        "US.SOXL", stored,
                        [(10.00, 100)], [(10.01, 100)],
                        svr_bid_ts=svr, svr_ask_ts=svr)
            finally:
                store.close()

            got: list = []
            with mock.patch.object(hm, "_DB_PATH", path):
                worker = hm._WindowPrefillWorker("US.SOXL", 10, 1)
                worker.done.connect(got.append)
                worker.run()          # run(), not start() -- no event loop
                worker.done.disconnect()

        self.assertEqual(len(got), 1)
        self.assertEqual(len(got[0]), 5)

    def test_the_pad_does_not_push_the_window_back_in_time(self):
        """The pad can return more columns than the window holds. Keeping the
        wrong end of that is invisible while the pad is small relative to the
        window -- and shows up as a heatmap stuck minutes in the past."""
        from unittest import mock

        from analysis import liq_hm_window as hm
        from feeds.order_book_store import OrderBookStore, SERVER_LEAD_SECS

        import tempfile

        cols, rows = 5, 20
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "ob.db"
            base = datetime(2026, 11, 3, 10, 0, 0)
            store = OrderBookStore(path)
            try:
                for i in range(rows):
                    stored = base + timedelta(seconds=i)
                    svr = (stored.replace(tzinfo=_ET).timestamp()
                           + SERVER_LEAD_SECS)
                    store.insert_snapshot(
                        "US.SOXL", stored,
                        [(10.00, 100)], [(10.01, 100)],
                        svr_bid_ts=svr, svr_ask_ts=svr)
            finally:
                store.close()

            got: list = []
            with mock.patch.object(hm, "_DB_PATH", path):
                worker = hm._WindowPrefillWorker("US.SOXL", cols, 1)
                worker.done.connect(got.append)
                worker.run()
                worker.done.disconnect()

        out = got[0]
        self.assertEqual(len(out), cols)
        newest = base + timedelta(seconds=rows - 1)
        self.assertAlmostEqual(
            (out[-1][0]["ts"] - newest).total_seconds(), 0.0, places=3)


if __name__ == "__main__":
    unittest.main()

"""bucketed_snapshots: one snapshot per time bucket, for drawing long ranges.

A heatmap column covers N seconds and shows the book at the end of that window,
so pulling every snapshot in a multi-hour range and discarding almost all of
them is pure cost -- the retained history is ~535k snapshots against ~3.8k
columns at a 30s bucket.
"""

from datetime import datetime, timedelta

import pytest

from feeds.order_book_store import OrderBookStore

C = "US.SOXL"
T0 = datetime(2026, 9, 28, 10, 0, 0)


@pytest.fixture()
def store(tmp_path):
    s = OrderBookStore(tmp_path / "ob.db")
    yield s
    s.close()


def _book(base: float, n: int = 3):
    return ([(base - 0.01 * i, 100 + i) for i in range(n)],
            [(base + 0.01 * (i + 1), 200 + i) for i in range(n)])


def _fill(s, n, step_secs=1, code=C, base=10.0):
    """n snapshots step_secs apart, each priced so the bucket is identifiable."""
    for i in range(n):
        s.insert_snapshot(code, T0 + timedelta(seconds=i * step_secs),
                          *_book(base + i))


def _tops(snaps):
    """Highest bid of each returned snapshot -- identifies which one it is."""
    return [max(r["price"] for r in g if r["side"] == "BID") for g in snaps]


class TestBucketing:
    def test_one_snapshot_per_bucket(self, store):
        _fill(store, 60)                      # 60 snapshots, 1s apart
        got = store.bucketed_snapshots(C, T0, T0 + timedelta(minutes=1), 10)
        assert len(got) == 6

    def test_takes_the_last_of_each_bucket(self, store):
        """Matches the live path, which repaints a column in place as new
        pushes land -- the column ends up showing the newest book."""
        _fill(store, 20)
        got = store.bucketed_snapshots(C, T0, T0 + timedelta(seconds=20), 10)
        # buckets hold snapshots 0-9 and 10-19; bases are 10..29
        assert _tops(got) == pytest.approx([19.0, 29.0])

    def test_oldest_first(self, store):
        _fill(store, 40)
        got = store.bucketed_snapshots(C, T0, T0 + timedelta(seconds=40), 5)
        stamps = [g[0]["ts"] for g in got]
        assert stamps == sorted(stamps)

    def test_each_entry_is_one_whole_book(self, store):
        _fill(store, 30)
        for g in store.bucketed_snapshots(C, T0, T0 + timedelta(seconds=30), 10):
            assert len(g) == 6, "3 bids + 3 asks"
            assert len({r["ts"] for r in g}) == 1

    def test_a_bucket_with_no_data_is_simply_absent(self, store):
        """Collection gaps must not fabricate empty columns."""
        _fill(store, 5)                                   # 10:00:00-10:00:04
        _fill(store, 5, code=C, base=100.0)               # same window, ignore
        store.insert_snapshot(C, T0 + timedelta(seconds=90), *_book(50.0))
        got = store.bucketed_snapshots(C, T0, T0 + timedelta(seconds=120), 10)
        assert len(got) == 2, "one bucket at the start, one at +90s"

    def test_bucket_larger_than_the_range(self, store):
        _fill(store, 10)
        got = store.bucketed_snapshots(C, T0, T0 + timedelta(seconds=10), 3600)
        assert len(got) == 1

    def test_bucket_of_one_returns_everything(self, store):
        _fill(store, 12)
        got = store.bucketed_snapshots(C, T0, T0 + timedelta(seconds=12), 1)
        assert len(got) == 12

    def test_zero_or_negative_bucket_is_clamped_not_an_error(self, store):
        """Division by the bucket is in SQL; a 0 would be a runtime error."""
        _fill(store, 5)
        for bad in (0, -1, -3600):
            assert len(store.bucketed_snapshots(
                C, T0, T0 + timedelta(seconds=5), bad)) == 5


class TestBounds:
    def test_start_inclusive_end_exclusive(self, store):
        _fill(store, 10)
        got = store.bucketed_snapshots(C, T0, T0 + timedelta(seconds=5), 1)
        stamps = [g[0]["ts"] for g in got]
        assert T0 in stamps
        assert T0 + timedelta(seconds=5) not in stamps

    def test_range_outside_the_data(self, store):
        _fill(store, 10)
        assert store.bucketed_snapshots(
            C, T0 + timedelta(days=1), T0 + timedelta(days=2), 60) == []

    def test_empty_table(self, store):
        assert store.bucketed_snapshots(C, T0, T0 + timedelta(hours=1), 60) == []

    def test_other_codes_are_excluded(self, store):
        _fill(store, 10)
        for i in range(10):
            store.insert_snapshot("US.SOXS", T0 + timedelta(seconds=i), *_book(3.0))
        got = store.bucketed_snapshots(C, T0, T0 + timedelta(seconds=10), 10)
        assert len(got) == 1
        assert len(got[0]) == 6, "SOXS levels must not join the group"


class TestAgainstTheUnsampledQuery:
    def test_every_returned_snapshot_is_a_real_one(self, store):
        """Sampling must pick rows that exist, never synthesise or blend."""
        _fill(store, 50)
        all_rows = store.query_snapshots(C, T0, T0 + timedelta(seconds=50))
        real = {(r["ts"], r["price"], r["volume"]) for r in all_rows}
        for g in store.bucketed_snapshots(C, T0, T0 + timedelta(seconds=50), 7):
            for r in g:
                assert (r["ts"], r["price"], r["volume"]) in real

    def test_the_newest_snapshot_in_range_is_always_included(self, store):
        """The rightmost column has to be current, whatever the bucket."""
        _fill(store, 37)
        newest = max(g[0]["ts"]
                     for g in store.last_n_snapshots(C, 1))
        for bucket in (1, 5, 10, 60):
            got = store.bucketed_snapshots(C, T0, T0 + timedelta(seconds=37), bucket)
            assert got[-1][0]["ts"] == newest, bucket

    def test_duplicate_timestamps_do_not_collapse_a_bucket(self, store):
        """Two books can share a ts; bucketing is by row, and picks the later."""
        store.insert_snapshot(C, T0, *_book(10.0))
        store.insert_snapshot(C, T0, *_book(20.0))
        got = store.bucketed_snapshots(C, T0, T0 + timedelta(seconds=1), 1)
        assert len(got) == 1
        assert _tops(got) == pytest.approx([20.0]), "the later row wins"

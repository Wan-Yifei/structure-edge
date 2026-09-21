"""Unit tests for the read methods OrderBookStore grew when the heatmap, the
DOM window and two scripts stopped hand-rolling their own SELECTs.

Each of those callers had its own copy of the row shape and its own bound
conventions; these pin the behaviour they relied on so consolidating them
cannot change it.
"""

from datetime import datetime, timedelta

import pytest

from feeds.order_book_store import OrderBookStore

C = "US.SOXL"
T0 = datetime(2026, 9, 21, 10, 0, 0)


@pytest.fixture()
def store(tmp_path):
    s = OrderBookStore(tmp_path / "ob.db")
    yield s
    s.close()


def _book(base: float, n: int = 3):
    bids = [(base - 0.01 * i, 100 + i) for i in range(n)]
    asks = [(base + 0.01 * (i + 1), 200 + i) for i in range(n)]
    return bids, asks


def _fill(s: OrderBookStore, n_snaps: int, step_secs: int = 1, code: str = C):
    """n_snaps snapshots, one per step_secs, each a 3+3 level book."""
    for i in range(n_snaps):
        bids, asks = _book(10.0 + i * 0.01)
        s.insert_snapshot(code, T0 + timedelta(seconds=i * step_secs), bids, asks)


# ── latest_snapshot ──────────────────────────────────────────────────────────

class TestLatestSnapshot:
    def test_returns_every_row_of_the_newest_book(self, store):
        _fill(store, 4)
        rows = store.latest_snapshot(C)
        assert len(rows) == 6, "3 bids + 3 asks"
        assert {r["ts"] for r in rows} == {T0 + timedelta(seconds=3)}

    def test_only_the_newest(self, store):
        _fill(store, 4)
        newest = max(r["ts"] for r in store.latest_snapshot(C))
        assert newest == T0 + timedelta(seconds=3)

    def test_row_shape(self, store):
        _fill(store, 1)
        r = store.latest_snapshot(C)[0]
        assert set(r) == {"ts", "side", "price", "volume"}
        assert isinstance(r["ts"], datetime)
        assert isinstance(r["price"], float) and isinstance(r["volume"], float)
        assert r["side"] in ("BID", "ASK")

    def test_unknown_code_is_empty_not_an_error(self, store):
        _fill(store, 2)
        assert store.latest_snapshot("US.NOPE") == []

    def test_empty_table(self, store):
        assert store.latest_snapshot(C) == []

    def test_other_codes_do_not_leak_in(self, store):
        _fill(store, 2)
        store.insert_snapshot("US.SOXS", T0 + timedelta(seconds=99), *_book(3.0))
        rows = store.latest_snapshot(C)
        assert rows and max(r["ts"] for r in rows) == T0 + timedelta(seconds=1)


# ── snapshot_at_or_before ────────────────────────────────────────────────────

class TestSnapshotAtOrBefore:
    def test_exact_hit(self, store):
        _fill(store, 5)
        rows = store.snapshot_at_or_before(C, T0 + timedelta(seconds=2))
        assert rows and {r["ts"] for r in rows} == {T0 + timedelta(seconds=2)}

    def test_falls_back_to_the_one_before(self, store):
        _fill(store, 5, step_secs=10)
        rows = store.snapshot_at_or_before(C, T0 + timedelta(seconds=25))
        assert {r["ts"] for r in rows} == {T0 + timedelta(seconds=20)}

    def test_before_all_data_is_empty(self, store):
        _fill(store, 3)
        assert store.snapshot_at_or_before(C, T0 - timedelta(hours=1)) == []

    def test_after_all_data_gives_the_last(self, store):
        _fill(store, 3)
        rows = store.snapshot_at_or_before(C, T0 + timedelta(days=1))
        assert {r["ts"] for r in rows} == {T0 + timedelta(seconds=2)}

    def test_returns_a_whole_book_not_one_row(self, store):
        _fill(store, 3)
        assert len(store.snapshot_at_or_before(C, T0 + timedelta(seconds=1))) == 6


# ── last_n_snapshots ─────────────────────────────────────────────────────────

class TestLastNSnapshots:
    def test_oldest_first(self, store):
        _fill(store, 5)
        snaps = store.last_n_snapshots(C, 3)
        stamps = [s[0]["ts"] for s in snaps]
        assert stamps == sorted(stamps), "must be oldest first"
        assert stamps == [T0 + timedelta(seconds=i) for i in (2, 3, 4)]

    def test_each_inner_list_is_one_whole_book(self, store):
        _fill(store, 4)
        for snap in store.last_n_snapshots(C, 4):
            assert len(snap) == 6
            assert len({r["ts"] for r in snap}) == 1, "one timestamp per group"

    def test_asking_for_more_than_exists(self, store):
        _fill(store, 2)
        assert len(store.last_n_snapshots(C, 50)) == 2

    def test_zero_and_empty(self, store):
        _fill(store, 3)
        assert store.last_n_snapshots(C, 0) == []
        assert store.last_n_snapshots("US.NOPE", 5) == []

    def test_other_codes_excluded(self, store):
        _fill(store, 3)
        for i in range(3):
            store.insert_snapshot("US.SOXS", T0 + timedelta(seconds=i), *_book(3.0))
        snaps = store.last_n_snapshots(C, 5)
        assert len(snaps) == 3
        assert all(len(s) == 6 for s in snaps), "SOXS rows must not join the groups"

    def test_matches_the_per_snapshot_loop_it_replaced(self, store):
        """The old caller ran 1 + n queries; this runs one and groups."""
        _fill(store, 6)
        got = store.last_n_snapshots(C, 4)
        stamps = sorted({r["ts"] for r in store.query_snapshots(
            C, T0, T0 + timedelta(hours=1), end_inclusive=True)})[-4:]
        assert [s[0]["ts"] for s in got] == stamps


# ── query_snapshots bounds ───────────────────────────────────────────────────

class TestQuerySnapshotBounds:
    def test_end_exclusive_by_default(self, store):
        _fill(store, 5)
        end = T0 + timedelta(seconds=3)
        stamps = {r["ts"] for r in store.query_snapshots(C, T0, end)}
        assert end not in stamps

    def test_end_inclusive_when_asked(self, store):
        _fill(store, 5)
        end = T0 + timedelta(seconds=3)
        stamps = {r["ts"] for r in store.query_snapshots(
            C, T0, end, end_inclusive=True)}
        assert end in stamps

    def test_start_is_always_inclusive(self, store):
        _fill(store, 3)
        stamps = {r["ts"] for r in store.query_snapshots(
            C, T0, T0 + timedelta(seconds=10))}
        assert T0 in stamps

    def test_sorted_oldest_first(self, store):
        _fill(store, 5)
        got = [r["ts"] for r in store.query_snapshots(
            C, T0, T0 + timedelta(hours=1))]
        assert got == sorted(got)


# ── latest_ts ────────────────────────────────────────────────────────────────

class TestLatestTs:
    def test_per_code(self, store):
        _fill(store, 3)
        store.insert_snapshot("US.SOXS", T0 + timedelta(seconds=99), *_book(3.0))
        assert store.latest_ts(C) == T0 + timedelta(seconds=2)
        assert store.latest_ts("US.SOXS") == T0 + timedelta(seconds=99)

    def test_across_all_codes(self, store):
        _fill(store, 3)
        store.insert_snapshot("US.SOXS", T0 + timedelta(seconds=99), *_book(3.0))
        assert store.latest_ts() == T0 + timedelta(seconds=99)

    def test_none_when_empty(self, store):
        assert store.latest_ts() is None
        assert store.latest_ts(C) is None


# ── duplicate timestamps ─────────────────────────────────────────────────────

class TestDuplicateTimestamps:
    """Two genuinely different books can land on the same timestamp.

    The feed bursts several books per ~300ms cadence tick, 12-19ms apart
    (measured), while datetime.now() on Windows only advances in ~15.6ms
    steps. An earlier version of this schema keyed on (code, ts) and used
    INSERT OR REPLACE, which silently dropped the first of each colliding
    pair -- 30 pushes became 9 stored snapshots. These pin the fix.
    """

    def test_both_snapshots_are_kept(self, store):
        store.insert_snapshot(C, T0, *_book(10.0))
        store.insert_snapshot(C, T0, *_book(20.0))
        assert store.snapshot_count(C) == 2

    def test_latest_returns_one_book_not_two_fused(self, store):
        store.insert_snapshot(C, T0, *_book(10.0))
        store.insert_snapshot(C, T0, *_book(20.0))
        rows = store.latest_snapshot(C)
        assert len(rows) == 6, "one 3+3 book, not both fused into 12 rows"
        assert max(r["price"] for r in rows) > 19, "the later insert wins"

    def test_at_or_before_returns_one_book(self, store):
        store.insert_snapshot(C, T0, *_book(10.0))
        store.insert_snapshot(C, T0, *_book(20.0))
        assert len(store.snapshot_at_or_before(C, T0)) == 6

    def test_last_n_groups_by_row_not_timestamp(self, store):
        store.insert_snapshot(C, T0, *_book(10.0))
        store.insert_snapshot(C, T0, *_book(20.0))
        store.insert_snapshot(C, T0 + timedelta(seconds=1), *_book(30.0))
        snaps = store.last_n_snapshots(C, 5)
        assert len(snaps) == 3, "three snapshots, even though two share a ts"
        assert all(len(g) == 6 for g in snaps), "none fused"

    def test_insertion_order_is_preserved(self, store):
        for base in (10.0, 20.0, 30.0):
            store.insert_snapshot(C, T0, *_book(base))
        snaps = store.last_n_snapshots(C, 3)
        tops = [max(r["price"] for r in g) for g in snaps]
        assert tops == sorted(tops), "oldest-first must follow insertion order"

    def test_prune_counts_rows_not_timestamps(self, store):
        for base in (10.0, 20.0, 30.0, 40.0):
            store.insert_snapshot(C, T0, *_book(base))   # all the same ts
        store.prune(keep=2)
        assert store.snapshot_count(C) == 2, "a ts-based prune would keep 0 or 4"
        assert all(len(g) == 6 for g in store.last_n_snapshots(C, 2))

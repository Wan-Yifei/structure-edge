"""The ladder archive, and the properties that make it worth keeping.

`db/order_book.db` is pruned by row count -- roughly 1.4 days at the current
rate -- so the per-level book is gone almost as soon as it arrives. This
module may therefore only ever ADD: a load that quietly rebuilt from the live
database would truncate the archive to the retention window, and the loss
would be silent.

`feeds/depth_bars.py` keeps the same book as aggregates. The reason this
exists alongside it is the per-price detail, so the tests check that a level
can be read back exactly -- an aggregate archive cannot answer whether a
level that vanished was bought or withdrawn, and that is the whole point.
"""

import json
import sqlite3

import pandas as pd
import pytest

from feeds import book_ladder


def make_db(path, rows):
    """rows: (ts, bids, asks) with bids/asks as [[price, size], ...]."""
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE ob_snapshots (code TEXT, ts TEXT, "
                "bids TEXT, asks TEXT)")
    con.executemany(
        "INSERT INTO ob_snapshots VALUES ('US.SOXL', ?, ?, ?)",
        [(ts, json.dumps(b), json.dumps(a)) for ts, b, a in rows])
    con.commit()
    con.close()
    return path


def book(mid, bid_sizes, ask_sizes, step=0.01):
    """A ladder either side of *mid*, one cent apart."""
    bids = [[round(mid - step * (i + 1), 2), s] for i, s in enumerate(bid_sizes)]
    asks = [[round(mid + step * (i + 1), 2), s] for i, s in enumerate(ask_sizes)]
    return bids, asks


@pytest.fixture
def db(tmp_path):
    b1, a1 = book(100.0, [100, 200, 300], [150, 250])
    b2, a2 = book(100.0, [111, 222, 333], [150, 250])
    b3, a3 = book(100.0, [11, 22], [150, 250])
    return make_db(tmp_path / "ob.db", [
        ("2026-10-13 10:00:00.100", b1, a1),
        ("2026-10-13 10:00:00.600", b2, a2),   # later in the same second
        ("2026-10-13 10:00:01.200", b3, a3),
    ])


class TestOneDay:
    def test_one_row_per_level_per_second(self, db):
        d = book_ladder.build_day("US.SOXL", "2026-10-13", db_path=db)
        assert d.ts.nunique() == 2
        first = d[d.ts == pd.Timestamp("2026-10-13 10:00:00")]
        assert len(first) == 5          # 3 bids + 2 asks

    def test_the_last_snapshot_of_a_second_wins(self, db):
        """Sizes must be the book as the second ENDED, not as it began."""
        d = book_ladder.build_day("US.SOXL", "2026-10-13", db_path=db)
        b, _ = book_ladder.book_at(d, "2026-10-13 10:00:00")
        assert b[99.99] == 111          # the 10:00:00.600 snapshot
        assert b[99.98] == 222

    def test_a_level_reads_back_exactly(self, db):
        """The reason this archive exists at all."""
        d = book_ladder.build_day("US.SOXL", "2026-10-13", db_path=db)
        lvl = d[(d.price == 99.99) & (d.side == "B")].set_index("ts")["size"]
        assert list(lvl) == [111, 11]

    def test_both_sides_are_kept(self, db):
        d = book_ladder.build_day("US.SOXL", "2026-10-13", db_path=db)
        b, a = book_ladder.book_at(d, "2026-10-13 10:00:01")
        assert sorted(b) == [99.98, 99.99]
        assert sorted(a) == [100.01, 100.02]

    def test_a_vanished_level_is_absent_not_zero(self, db):
        """Distinguishing 'gone' from 'size 0' is what the estimator reads."""
        d = book_ladder.build_day("US.SOXL", "2026-10-13", db_path=db)
        b, _ = book_ladder.book_at(d, "2026-10-13 10:00:01")
        assert 99.97 not in b


class TestDegenerate:
    def test_a_pruned_day_is_empty_not_an_error(self, db):
        d = book_ladder.build_day("US.SOXL", "2020-01-01", db_path=db)
        assert d.empty
        assert list(d.columns) == book_ladder.COLUMNS

    def test_one_sided_books_are_skipped(self, tmp_path):
        b, a = book(100.0, [100], [100])
        p = make_db(tmp_path / "ob.db", [
            ("2026-10-13 10:00:00.100", [], a),
            ("2026-10-13 10:00:01.100", b, []),
            ("2026-10-13 10:00:02.100", b, a),
        ])
        d = book_ladder.build_day("US.SOXL", "2026-10-13", db_path=p)
        assert d.ts.nunique() == 1

    def test_malformed_json_does_not_stop_the_day(self, tmp_path):
        b, a = book(100.0, [100], [100])
        p = tmp_path / "ob.db"
        con = sqlite3.connect(p)
        con.execute("CREATE TABLE ob_snapshots (code TEXT, ts TEXT, "
                    "bids TEXT, asks TEXT)")
        con.executemany("INSERT INTO ob_snapshots VALUES ('US.SOXL', ?, ?, ?)", [
            ("2026-10-13 10:00:00.100", "{not json", "[]"),
            ("2026-10-13 10:00:01.100", json.dumps(b), json.dumps(a)),
        ])
        con.commit()
        con.close()
        d = book_ladder.build_day("US.SOXL", "2026-10-13", db_path=p)
        assert d.ts.nunique() == 1

    def test_more_seconds_than_sqlite_takes_parameters(self, tmp_path):
        """A real day holds tens of thousands of seconds.

        The first version passed every rowid in one `IN (...)`, which SQLite
        refuses past its host-parameter cap -- it failed on the first real
        day with "too many SQL variables".
        """
        b, a = book(100.0, [100], [100])
        rows = [(f"2026-10-13 {10 + s // 3600:02d}:"
                 f"{(s // 60) % 60:02d}:{s % 60:02d}.100", b, a)
                for s in range(2000)]
        p = make_db(tmp_path / "ob.db", rows)
        d = book_ladder.build_day("US.SOXL", "2026-10-13", db_path=p)
        assert d.ts.nunique() == 2000


class TestArchiveOnlyGrows:
    """The property the whole module exists for."""

    def test_a_pruned_source_cannot_shrink_the_archive(self, tmp_path, db):
        book_ladder.append_day("US.SOXL", "2026-10-13", db_path=db,
                               cache_dir=tmp_path)
        n = len(book_ladder.load("US.SOXL", cache_dir=tmp_path))
        empty = make_db(tmp_path / "pruned.db", [])
        added = book_ladder.append_day("US.SOXL", "2026-10-13",
                                       db_path=empty, cache_dir=tmp_path)
        assert added == 0
        assert len(book_ladder.load("US.SOXL", cache_dir=tmp_path)) == n

    def test_rerunning_the_same_day_adds_nothing(self, tmp_path, db):
        first = book_ladder.append_day("US.SOXL", "2026-10-13", db_path=db,
                                       cache_dir=tmp_path)
        again = book_ladder.append_day("US.SOXL", "2026-10-13", db_path=db,
                                       cache_dir=tmp_path)
        assert first > 0 and again == 0

    def test_a_second_already_archived_is_not_replaced(self, tmp_path, db):
        """A later run over a partly pruned day must keep the good rows."""
        book_ladder.append_day("US.SOXL", "2026-10-13", db_path=db,
                               cache_dir=tmp_path)
        b, a = book(100.0, [7], [7])
        thin = make_db(tmp_path / "thin.db",
                       [("2026-10-13 10:00:00.900", b, a)])
        book_ladder.append_day("US.SOXL", "2026-10-13", db_path=thin,
                               cache_dir=tmp_path)
        got = book_ladder.load("US.SOXL", cache_dir=tmp_path)
        bk, _ = book_ladder.book_at(got, "2026-10-13 10:00:00")
        assert bk[99.99] == 111          # the original, not the 7

    def test_appending_another_day_keeps_the_first(self, tmp_path, db):
        book_ladder.append_day("US.SOXL", "2026-10-13", db_path=db,
                               cache_dir=tmp_path)
        b, a = book(101.0, [100], [100])
        db2 = make_db(tmp_path / "ob2.db",
                      [("2026-10-14 10:00:00.100", b, a)])
        book_ladder.append_day("US.SOXL", "2026-10-14", db_path=db2,
                               cache_dir=tmp_path)
        got = book_ladder.load("US.SOXL", cache_dir=tmp_path)
        assert sorted({t.date().isoformat() for t in got.ts}) == \
            ["2026-10-13", "2026-10-14"]

    def test_load_never_reaches_the_live_database(self, tmp_path, db):
        """load() takes no db_path, so it cannot be pointed at the source."""
        assert book_ladder.load("US.SOXL", cache_dir=tmp_path).empty
        book_ladder.append_day("US.SOXL", "2026-10-13", db_path=db,
                               cache_dir=tmp_path)
        assert not book_ladder.load("US.SOXL", cache_dir=tmp_path).empty


class TestLoadAndCoverage:
    def test_range_filters_apply(self, tmp_path, db):
        book_ladder.append_day("US.SOXL", "2026-10-13", db_path=db,
                               cache_dir=tmp_path)
        got = book_ladder.load("US.SOXL", start="2026-10-13 10:00:01",
                               cache_dir=tmp_path)
        assert got.ts.nunique() == 1

    def test_coverage_reports_the_day(self, tmp_path, db):
        book_ladder.append_day("US.SOXL", "2026-10-13", db_path=db,
                               cache_dir=tmp_path)
        cov = book_ladder.coverage("US.SOXL", cache_dir=tmp_path)
        assert len(cov) == 1
        assert cov.iloc[0].seconds == 2

    def test_coverage_of_nothing_is_empty(self, tmp_path):
        assert book_ladder.coverage("US.SOXL", cache_dir=tmp_path).empty

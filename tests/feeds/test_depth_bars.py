"""The depth archive, and the one property that makes it an archive.

`db/order_book.db` is pruned by row count -- 24 hours for SOXL -- so the raw
snapshots are gone a day after they arrive. Everything here turns on the
difference between that and `delta_bars`, which can rebuild from ticks.db
whenever it likes: this module may only ever ADD. A load that quietly rebuilt
from the live database would truncate months of archive to the retention
window, and the loss would be silent.
"""

import json
import sqlite3

import pandas as pd
import pytest

from feeds import depth_bars


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
    b, a = book(100.0, [100, 200, 300], [150, 250, 350])
    return make_db(tmp_path / "ob.db", [
        ("2026-10-13 10:00:00.100", b, a),
        ("2026-10-13 10:00:00.600", b, a),
        ("2026-10-13 10:00:01.200", b, a),
    ])


class TestOneSnapshot:
    def test_rows_are_one_per_second(self, db):
        d = depth_bars.build_day("US.SOXL", "2026-10-13", db_path=db)
        assert len(d) == 2
        assert list(d.n) == [2, 1]

    def test_mid_sits_between_the_touches(self, db):
        d = depth_bars.build_day("US.SOXL", "2026-10-13", db_path=db)
        assert d.mid.iloc[0] == pytest.approx(100.0)
        assert d.spread.iloc[0] == pytest.approx(0.02)

    def test_touch_size_is_the_nearest_level(self, db):
        d = depth_bars.build_day("US.SOXL", "2026-10-13", db_path=db)
        assert d.b_top.iloc[0] == pytest.approx(100)
        assert d.a_top.iloc[0] == pytest.approx(150)

    def test_bands_widen_monotonically(self, db):
        d = depth_bars.build_day("US.SOXL", "2026-10-13", db_path=db).iloc[0]
        for side in ("b", "a"):
            sizes = [d[f"{side}{n}"] for _, n in depth_bars.BANDS]
            assert sizes == sorted(sizes)

    def test_the_widest_band_holds_the_whole_ladder(self, db):
        d = depth_bars.build_day("US.SOXL", "2026-10-13", db_path=db).iloc[0]
        assert d.b10 == pytest.approx(600)
        assert d.a10 == pytest.approx(750)

    def test_the_wall_is_found_with_its_distance(self, db):
        d = depth_bars.build_day("US.SOXL", "2026-10-13", db_path=db).iloc[0]
        assert d.b_max == pytest.approx(300)
        assert d.b_max_d == pytest.approx(0.03, abs=1e-3)
        assert d.a_max == pytest.approx(350)


class TestDegenerate:
    def test_a_day_with_no_snapshots_is_empty_not_an_error(self, db):
        d = depth_bars.build_day("US.SOXL", "2020-01-01", db_path=db)
        assert d.empty
        assert list(d.columns) == list(depth_bars.COLUMNS)

    def test_one_sided_books_are_skipped(self, tmp_path):
        b, a = book(100.0, [100], [100])
        p = make_db(tmp_path / "ob.db", [
            ("2026-10-13 10:00:00.100", [], a),
            ("2026-10-13 10:00:01.100", b, []),
            ("2026-10-13 10:00:02.100", b, a),
        ])
        d = depth_bars.build_day("US.SOXL", "2026-10-13", db_path=p)
        assert len(d) == 1

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
        d = depth_bars.build_day("US.SOXL", "2026-10-13", db_path=p)
        assert len(d) == 1


class TestArchiveOnlyGrows:
    """The property the whole module exists for."""

    def test_appending_a_second_day_keeps_the_first(self, tmp_path, db):
        cache = tmp_path / "depth.parquet"
        depth_bars.append_day("US.SOXL", "2026-10-13", db_path=db,
                              cache_path=cache)
        b, a = book(101.0, [100], [100])
        db2 = make_db(tmp_path / "ob2.db",
                      [("2026-10-14 10:00:00.100", b, a)])
        depth_bars.append_day("US.SOXL", "2026-10-14", db_path=db2,
                              cache_path=cache)
        got = depth_bars.load("US.SOXL", cache_path=cache)
        assert sorted({d.date().isoformat() for d in got.index}) == \
            ["2026-10-13", "2026-10-14"]

    def test_a_pruned_source_cannot_shrink_the_archive(self, tmp_path, db):
        """Re-running over a day the collector has since emptied is a no-op.

        This is the failure that would hurt most and leave no trace.
        """
        cache = tmp_path / "depth.parquet"
        depth_bars.append_day("US.SOXL", "2026-10-13", db_path=db,
                              cache_path=cache)
        n_before = len(depth_bars.load("US.SOXL", cache_path=cache))
        empty = make_db(tmp_path / "pruned.db", [])
        added = depth_bars.append_day("US.SOXL", "2026-10-13",
                                      db_path=empty, cache_path=cache)
        assert added == 0
        assert len(depth_bars.load("US.SOXL", cache_path=cache)) == n_before

    def test_rerunning_the_same_day_adds_nothing(self, tmp_path, db):
        cache = tmp_path / "depth.parquet"
        first = depth_bars.append_day("US.SOXL", "2026-10-13", db_path=db,
                                      cache_path=cache)
        again = depth_bars.append_day("US.SOXL", "2026-10-13", db_path=db,
                                      cache_path=cache)
        assert first > 0 and again == 0

    def test_load_never_reaches_the_live_database(self, tmp_path, db):
        """load() takes no db_path, so it cannot be pointed at the source."""
        cache = tmp_path / "depth.parquet"
        assert depth_bars.load("US.SOXL", cache_path=cache).empty
        depth_bars.append_day("US.SOXL", "2026-10-13", db_path=db,
                              cache_path=cache)
        assert not depth_bars.load("US.SOXL", cache_path=cache).empty


class TestLoadAndCoverage:
    def test_range_filters_apply(self, tmp_path, db):
        cache = tmp_path / "depth.parquet"
        depth_bars.append_day("US.SOXL", "2026-10-13", db_path=db,
                              cache_path=cache)
        got = depth_bars.load("US.SOXL", start="2026-10-13 10:00:01",
                              cache_path=cache)
        assert len(got) == 1

    def test_coverage_reports_the_day(self, tmp_path, db):
        cache = tmp_path / "depth.parquet"
        depth_bars.append_day("US.SOXL", "2026-10-13", db_path=db,
                              cache_path=cache)
        cov = depth_bars.coverage("US.SOXL", cache_path=cache)
        assert len(cov) == 1
        assert cov.iloc[0].rows == 2

    def test_coverage_of_nothing_is_empty(self, tmp_path):
        cov = depth_bars.coverage("US.SOXL",
                                  cache_path=tmp_path / "missing.parquet")
        assert cov.empty

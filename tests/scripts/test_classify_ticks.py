"""The off-exchange classifier, end to end over real store objects.

It had been reading `t["feed_dir"]` from a tick row, a key `query_ticks` has
never returned. Nothing caught it: the stores had their own tests, the script
had none, and the scheduler ran it as a subprocess whose traceback went to a
log nobody reads. It fired on schedule for two nights, died 2 minutes in both
times, and the only visible symptom was the viewer's OffEx line being flat.

So these tests drive classify_day against a real ticks.db and order_book.db
rather than mocking the reads -- a mock would have agreed with the typo.
"""

import json
import sqlite3
from datetime import datetime, timedelta

import pytest

classify_ticks = pytest.importorskip("scripts.classify_ticks")

from feeds.tick_mid_store import TickMidStore   # noqa: E402
from feeds.tick_store import TickStore          # noqa: E402

C = "US.SOXL"
DAY = datetime(2026, 10, 8)
T0 = DAY.replace(hour=10)


@pytest.fixture()
def dbs(tmp_path, monkeypatch):
    """A tick store and a book, wired into the script's module globals."""
    ob = tmp_path / "order_book.db"
    con = sqlite3.connect(ob)
    con.execute("CREATE TABLE ob_snapshots (code TEXT, ts TEXT, "
                "bids TEXT, asks TEXT)")
    con.commit()
    con.close()
    monkeypatch.setattr(classify_ticks, "OB_DB", ob)
    monkeypatch.setattr(classify_ticks, "TICK_DB", tmp_path / "ticks.db")

    ticks = TickStore(tmp_path / "ticks.db")
    mids = TickMidStore(tmp_path / "tick_mid.db")
    yield ticks, mids, ob
    ticks.close()
    mids.close()


def _book(ob_path, rows):
    """rows: (seconds-after-T0, best_bid, best_ask)."""
    con = sqlite3.connect(ob_path)
    con.executemany(
        "INSERT INTO ob_snapshots VALUES (?,?,?,?)",
        [(C, (T0 + timedelta(seconds=s)).isoformat(sep=" "),
          json.dumps([[bb, 100], [round(bb - 0.01, 2), 200]]),
          json.dumps([[ba, 100], [round(ba + 0.01, 2), 200]]))
         for s, bb, ba in rows])
    con.commit()
    con.close()


def _ticks(store, rows):
    """rows: (seconds-after-T0, price, volume, direction)."""
    store.insert_ticks([{"code": C, "ts": T0 + timedelta(seconds=s),
                         "price": p, "volume": v, "direction": d}
                        for s, p, v, d in rows])


def _run(dbs):
    ticks, mids, _ = dbs
    return classify_ticks.classify_day(C, DAY, ticks, mids)


class TestItActuallySignsSomething:
    """The regression: a wrong key made every run abort before the first row."""

    def test_a_neutral_print_above_the_mid_is_a_buy(self, dbs):
        ticks, mids, ob = dbs
        _book(ob, [(0, 100.00, 100.02)])            # mid 100.01
        _ticks(ticks, [(1, 100.015, 300, "NEUTRAL")])

        r = _run(dbs)
        assert r["signed"] == 1, r
        assert list(mids.load(C, DAY, DAY + timedelta(days=1)).values()) == ["BUY"]

    def test_below_the_mid_is_a_sell(self, dbs):
        ticks, mids, ob = dbs
        _book(ob, [(0, 100.00, 100.02)])
        _ticks(ticks, [(1, 100.005, 300, "NEUTRAL")])

        assert _run(dbs)["signed"] == 1
        assert list(mids.load(C, DAY, DAY + timedelta(days=1)).values()) == ["SELL"]

    def test_the_counts_add_up(self, dbs):
        ticks, mids, ob = dbs
        # one snapshot per second: MAX_STALE_SECS is 1.0, so a single book at
        # t=0 would make everything after t=1 stale and test nothing.
        # A wide book so the midpoint lands on 100.25, which a float holds
        # exactly: (100.00 + 100.02) / 2 is 100.00999999999999, and a tick
        # written as 100.01 then reads as above the mid rather than on it.
        _book(ob, [(s, 100.00, 100.50) for s in range(6)])
        _ticks(ticks, [(1, 100.30, 10, "NEUTRAL"),     # signed buy
                       (2, 100.20, 10, "NEUTRAL"),     # signed sell
                       (3, 100.25, 10, "NEUTRAL"),     # exactly at the mid
                       (4, 100.40, 10, "BUY"),         # the feed already knows
                       (5, 100.10, 10, "SELL")])

        r = _run(dbs)
        assert r["ticks"] == 5
        assert r["neutral"] == 3
        assert r["signed"] == 2
        assert r["at_mid"] == 1
        assert r["stale"] == 0


class TestWhatItLeavesAlone:
    def test_a_tagged_print_is_never_touched(self, dbs):
        """Where the exchange names the aggressor, the midpoint adds nothing."""
        ticks, mids, ob = dbs
        _book(ob, [(0, 100.00, 100.02)])
        _ticks(ticks, [(1, 100.015, 300, "BUY"), (2, 100.005, 300, "SELL")])

        assert _run(dbs)["signed"] == 0
        assert mids.count(C) == 0

    def test_a_stale_book_is_skipped_not_guessed(self, dbs):
        ticks, mids, ob = dbs
        _book(ob, [(0, 100.00, 100.02)])
        late = classify_ticks.MAX_STALE_SECS + 1.0
        _ticks(ticks, [(late, 100.015, 300, "NEUTRAL")])

        r = _run(dbs)
        assert r["signed"] == 0 and r["stale"] == 1
        assert mids.count(C) == 0

    def test_a_crossed_book_gives_no_midpoint(self, dbs):
        ticks, mids, ob = dbs
        _book(ob, [(0, 100.02, 100.00)])            # bid above ask
        _ticks(ticks, [(1, 100.015, 300, "NEUTRAL")])

        r = _run(dbs)
        assert r["note"] == "no book", r
        assert mids.count(C) == 0

    def test_an_empty_book_day_reports_rather_than_raising(self, dbs):
        ticks, _, _ = dbs
        _ticks(ticks, [(1, 100.015, 300, "NEUTRAL")])
        r = _run(dbs)
        assert r["note"] == "no book" and r["signed"] == 0


class TestTheKeyItWritesOn:
    def test_the_stored_timestamp_string_is_reused_verbatim(self, dbs):
        """A re-serialised datetime pads microseconds and misses every row."""
        ticks, mids, ob = dbs
        _book(ob, [(0, 100.00, 100.02)])
        raw = "2026-10-08 10:00:00.38"      # inside the staleness window
        ticks._con.execute(
            "INSERT INTO ticks (code, ts, price, volume, direction) "
            "VALUES (?,?,?,?,?)", (C, raw, 100.015, 300, "NEUTRAL"))
        ticks._con.commit()

        assert _run(dbs)["signed"] == 1
        got = ticks.query_ticks(C, DAY, DAY + timedelta(days=1), with_mid=True)
        assert got[0]["mid_dir"] == "BUY", "the viewer must find it again"

    def test_a_rerun_updates_rather_than_duplicates(self, dbs):
        ticks, mids, ob = dbs
        _book(ob, [(0, 100.00, 100.02)])
        _ticks(ticks, [(1, 100.015, 300, "NEUTRAL")])
        _run(dbs)
        _run(dbs)
        assert mids.count(C) == 1

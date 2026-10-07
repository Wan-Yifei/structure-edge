"""Sides inferred for off-exchange prints, and how a tick read folds them in.

The feed leaves most volume NEUTRAL. Those prints are off-exchange flow rather
than noise, and signing them against the midpoint takes delta from about an
eighth of the tape to nearly all of it -- see doc/ORDER_FLOW_GUIDE.md appendix A.

They live in their own file because ticks.db is write-locked by the collector
for the whole session: 82 attempts over 40 seconds with a 30-second timeout,
all refused. These pin that the two stores stay independent.
"""

from datetime import datetime, timedelta

import pytest

from feeds.tick_mid_store import TickMidStore
from feeds.tick_store import TickStore

C = "US.SOXL"
T0 = datetime(2026, 10, 7, 10, 0)


@pytest.fixture()
def stores(tmp_path):
    # TickStore looks for the side file next to its own, so no patching needed.
    mid_path = tmp_path / "tick_mid.db"
    ticks = TickStore(tmp_path / "ticks.db")
    mids = TickMidStore(mid_path)
    yield ticks, mids, mid_path
    ticks.close()
    mids.close()


def _tick(i, price, vol, direction):
    return {"code": C, "ts": T0 + timedelta(seconds=i), "price": price,
            "volume": vol, "direction": direction}


class TestStore:
    def test_round_trip(self, stores):
        _, mids, _ = stores
        mids.put([(C, T0, 100.0, 50, "BUY")])
        got = mids.load(C, T0, T0 + timedelta(minutes=1))
        assert got == {(T0.isoformat(sep=" "), 100.0, 50): "BUY"}

    def test_replaces_on_reclassify(self, stores):
        _, mids, _ = stores
        mids.put([(C, T0, 100.0, 50, "BUY")])
        mids.put([(C, T0, 100.0, 50, "SELL")])
        assert mids.count(C) == 1
        assert list(mids.load(C, T0, T0 + timedelta(minutes=1)).values()) == ["SELL"]

    def test_keyed_on_all_four_fields(self, stores):
        """Same instant, different size: two different prints."""
        _, mids, _ = stores
        mids.put([(C, T0, 100.0, 50, "BUY"), (C, T0, 100.0, 60, "SELL")])
        assert mids.count(C) == 2

    def test_codes_are_separate(self, stores):
        _, mids, _ = stores
        mids.put([(C, T0, 100.0, 50, "BUY"), ("US.SOXS", T0, 30.0, 10, "SELL")])
        assert mids.count(C) == 1 and mids.count() == 2

    def test_range_bounds(self, stores):
        _, mids, _ = stores
        for i in range(5):
            mids.put([(C, T0 + timedelta(seconds=i), 100.0, 1, "BUY")])
        got = mids.load(C, T0 + timedelta(seconds=1), T0 + timedelta(seconds=3))
        assert len(got) == 2

    def test_empty_put(self, stores):
        _, mids, _ = stores
        assert mids.put([]) == 0


class TestTickReadAttachesThemWithoutRewriting:
    """The store reports the inferred side; it never substitutes it.

    Folding the two together was tried and measurably made delta track price
    worse -- +0.276 to +0.225 against the minute's return with a fresh book --
    and it hides the divergence between the hurried and unhurried sides, which
    is the only thing this flow turned out to be good for.
    """

    def test_not_loaded_by_default(self, stores):
        ticks, mids, _ = stores
        ticks.insert_ticks([_tick(0, 100.0, 50, "NEUTRAL")])
        mids.put([(C, T0, 100.0, 50, "BUY")])
        got = ticks.query_ticks(C, T0, T0 + timedelta(minutes=1))
        assert got[0]["direction"] == "NEUTRAL"
        assert got[0]["mid_dir"] is None, "not asked for, not loaded"

    def test_attached_when_asked(self, stores):
        ticks, mids, _ = stores
        ticks.insert_ticks([_tick(0, 100.0, 50, "NEUTRAL")])
        mids.put([(C, T0, 100.0, 50, "BUY")])
        got = ticks.query_ticks(C, T0, T0 + timedelta(minutes=1), with_mid=True)
        assert got[0]["mid_dir"] == "BUY"

    def test_direction_is_always_the_feed_s(self, stores):
        """Even where the side file has an opinion, and even where it differs."""
        ticks, mids, _ = stores
        ticks.insert_ticks([_tick(0, 100.0, 50, "NEUTRAL"),
                            _tick(1, 100.0, 60, "SELL")])
        mids.put([(C, T0, 100.0, 50, "BUY"),
                  (C, T0 + timedelta(seconds=1), 100.0, 60, "BUY")])
        got = ticks.query_ticks(C, T0, T0 + timedelta(minutes=1), with_mid=True)
        assert [g["direction"] for g in got] == ["NEUTRAL", "SELL"]

    def test_neutral_with_no_entry(self, stores):
        ticks, _, _ = stores
        ticks.insert_ticks([_tick(0, 100.0, 50, "NEUTRAL")])
        got = ticks.query_ticks(C, T0, T0 + timedelta(minutes=1), with_mid=True)
        assert got[0]["direction"] == "NEUTRAL" and got[0]["mid_dir"] is None

    def test_a_missing_side_file_is_not_fatal(self, tmp_path):
        """A tick read must survive the file being absent or rebuilt."""
        ts = TickStore(tmp_path / "ticks.db")
        ts.insert_ticks([_tick(0, 100.0, 50, "NEUTRAL")])
        got = ts.query_ticks(C, T0, T0 + timedelta(minutes=1), with_mid=True)
        assert got and got[0]["direction"] == "NEUTRAL"
        ts.close()

    def test_the_timestamp_key_is_the_stored_string(self, stores):
        """ticks.db keeps the feed's own string, and '...:34.38' re-serialised
        from a datetime becomes '...:34.380000'. Keying on the parsed value
        made every lookup miss while every unit test still passed."""
        ticks, mids, _ = stores
        raw = "2026-10-07 10:00:00.38"
        ticks._con.execute(
            "INSERT INTO ticks (code, ts, price, volume, direction) "
            "VALUES (?, ?, ?, ?, ?)", (C, raw, 100.0, 50, "NEUTRAL"))
        ticks._con.commit()
        mids.put([(C, raw, 100.0, 50, "BUY")])
        got = ticks.query_ticks(C, datetime(2026, 10, 7, 9),
                                datetime(2026, 10, 7, 11), with_mid=True)
        assert got[0]["ts_raw"] == raw
        assert got[0]["mid_dir"] == "BUY", "unpadded timestamp must still match"


class TestTickerType:
    """moomoo's own ticker_type, which the collector discarded for months.

    Its 32 values include OTC_SOLD, DERIVATIVELY_PRICED, BULK and CROSS_MARKET
    -- the standard conditions for a trade reported away from an exchange. The
    off-exchange work had to infer venue from sub-penny prices because this was
    being thrown away; it is recorded now so that inference can be checked, and
    eventually replaced, against what the feed actually says.
    """

    def test_stored_and_returned(self, stores):
        ticks, _, _ = stores
        r = _tick(0, 100.0, 50, "NEUTRAL")
        r["ttype"] = "ODD_LOT"
        ticks.insert_ticks([r])
        got = ticks.query_ticks(C, T0, T0 + timedelta(minutes=1))
        assert got[0]["ttype"] == "ODD_LOT"

    def test_absent_is_none_not_an_error(self, stores):
        """Every row written before 2026-10-07 has no type."""
        ticks, _, _ = stores
        ticks.insert_ticks([_tick(0, 100.0, 50, "BUY")])
        got = ticks.query_ticks(C, T0, T0 + timedelta(minutes=1))
        assert got[0]["ttype"] is None

    def test_does_not_affect_direction(self, stores):
        ticks, _, _ = stores
        r = _tick(0, 100.0, 50, "BUY")
        r["ttype"] = "OTC_SOLD"
        ticks.insert_ticks([r])
        got = ticks.query_ticks(C, T0, T0 + timedelta(minutes=1))
        assert got[0]["direction"] == "BUY"

    def test_added_to_a_database_that_predates_it(self, tmp_path):
        """The column goes onto an existing file without rewriting 48M rows."""
        import sqlite3
        path = tmp_path / "old.db"
        con = sqlite3.connect(path)
        con.executescript(
            "CREATE TABLE ticks (code TEXT NOT NULL, ts TEXT NOT NULL, "
            "price REAL NOT NULL, volume INTEGER NOT NULL, "
            "direction TEXT NOT NULL, UNIQUE(code, ts, price, volume));")
        con.execute("INSERT INTO ticks VALUES (?,?,?,?,?)",
                    (C, "2026-10-01 10:00:00", 100.0, 50, "BUY"))
        con.commit(); con.close()

        ts = TickStore(path)
        cols = {r[1] for r in ts._con.execute("PRAGMA table_info(ticks)")}
        assert "ttype" in cols
        got = ts.query_ticks(C, datetime(2026, 10, 1), datetime(2026, 10, 2))
        assert got[0]["direction"] == "BUY" and got[0]["ttype"] is None
        ts.close()

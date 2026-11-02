"""What insert_ticks reports, and what the UNIQUE key costs.

`db/ticks.db` holds between 55% and 96% of the volume the 1-minute klines
report for the same session, and the figure swings day to day with no cause
yet found. Two things could be eating ticks: the feed may not send them, or
this process may throw them away at the UNIQUE key. Nothing recorded let
anyone tell those apart -- insert_ticks returned len(data), so every caller
counting inserts was counting offers, and the collector's session_count with
it.

Now the return value is the real write count and the feed's own per-tick
`seq` is stored, so a break in the numbering places the loss upstream while
offered-minus-written places it here.

The duplicate case below is not hypothetical: repeated prints of the same
size at one price are the classic iceberg signature, and UNIQUE(code, ts,
price, volume) keeps one of them.
"""

import pathlib
import tempfile
import unittest
from datetime import datetime, timedelta

from feeds.tick_store import TickStore

T0 = datetime(2026, 1, 1, 9, 30, 0)


def _tick(ts, price, volume, seq=None, **kw):
    row = {"code": "US.X", "ts": ts, "price": price, "volume": volume,
           "direction": "BUY", "ttype": "AUTO_MATCH"}
    if seq is not None:
        row["seq"] = seq
    row.update(kw)
    return row


class TestInsertTicks(unittest.TestCase):

    def setUp(self):
        self.store = TickStore(
            pathlib.Path(tempfile.mkdtemp()) / "ticks.db")

    def _count(self):
        return self.store._con.execute("SELECT COUNT(*) FROM ticks").fetchone()[0]

    def test_returns_rows_written_not_rows_offered(self):
        rows = [_tick(T0 + timedelta(milliseconds=i), 1.0 + i, 10 + i, seq=i)
                for i in range(3)]
        rows.append(dict(rows[0]))          # the same print offered twice
        self.assertEqual(self.store.insert_ticks(rows), 3)
        self.assertEqual(self._count(), 3)

    def test_same_price_and_size_at_one_timestamp_collapses(self):
        """The iceberg signature, and the cost of the UNIQUE key. Pinned so
        the loss is a decision on the record rather than a surprise."""
        rows = [_tick(T0, 2.0, 50, seq=200 + i) for i in range(4)]
        self.assertEqual(self.store.insert_ticks(rows), 1)

    def test_a_differing_size_is_kept(self):
        """Only the whole key collapses -- one share apart is a separate row."""
        rows = [_tick(T0, 2.0, 50, seq=1), _tick(T0, 2.0, 51, seq=2)]
        self.assertEqual(self.store.insert_ticks(rows), 2)

    def test_seq_is_stored_and_survives(self):
        self.store.insert_ticks(
            [_tick(T0 + timedelta(milliseconds=i), 1.0 + i, 10, seq=500 + i)
             for i in range(3)])
        got = [r[0] for r in self.store._con.execute(
            "SELECT seq FROM ticks ORDER BY seq")]
        self.assertEqual(got, [500, 501, 502])

    def test_seq_may_be_absent(self):
        """Replays and older callers pass no seq; the row still lands."""
        self.assertEqual(self.store.insert_ticks([_tick(T0, 1.0, 10)]), 1)
        self.assertIsNone(self.store._con.execute(
            "SELECT seq FROM ticks").fetchone()[0])

    def test_seq_is_not_part_of_the_unique_key(self):
        """Deliberate: including it would let a reconnect replay a window
        back into the table."""
        rows = [_tick(T0, 2.0, 50, seq=1), _tick(T0, 2.0, 50, seq=2)]
        self.assertEqual(self.store.insert_ticks(rows), 1)

    def test_empty_is_zero(self):
        self.assertEqual(self.store.insert_ticks([]), 0)

    def test_an_old_database_gains_the_column(self):
        """A table created before seq existed is migrated in place."""
        path = pathlib.Path(tempfile.mkdtemp()) / "old.db"
        import sqlite3
        con = sqlite3.connect(str(path))
        con.execute("CREATE TABLE ticks (code TEXT NOT NULL, ts TEXT NOT NULL,"
                    " price REAL NOT NULL, volume INTEGER NOT NULL,"
                    " direction TEXT NOT NULL,"
                    " UNIQUE(code, ts, price, volume))")
        con.commit()
        con.close()
        store = TickStore(path)
        cols = {r[1] for r in store._con.execute("PRAGMA table_info(ticks)")}
        self.assertIn("seq", cols)
        self.assertIn("ttype", cols)
        self.assertEqual(store.insert_ticks([_tick(T0, 1.0, 10, seq=7)]), 1)


if __name__ == "__main__":
    unittest.main()

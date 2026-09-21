"""SQLite-backed order book snapshot storage (WAL mode — concurrent reader + writer safe).

Schema:
    ob_snapshots(code TEXT, ts TEXT, bids TEXT, asks TEXT)   PRIMARY KEY (code, ts)

One row per snapshot. bids/asks are JSON arrays of [price, volume] pairs,
ordered as the feed delivered them. ts is ISO-8601: '2026-05-30 09:30:00.123456'.

This replaced a row-per-price-level table, which cost 120 rows per snapshot
for a 60+60 book and repeated the 26-byte ts and the code on every one of
them. Measured on SOXL 2026-09-21: the feed pushes 6.7 distinct books a
second during regular hours, so that schema wanted 803 rows/s -- about
1.9 GB per code per session. That is what forced the collector's 2s write
throttle, which was discarding 93% of the book updates it received. One row
per snapshot makes the same feed 6.7 rows/s.

The old order_book_snapshots table is not read or written any more. It is
left in place rather than dropped automatically; see drop_legacy_table().
"""

from __future__ import annotations

import json
import sqlite3
import pathlib
from datetime import datetime, date

_DEFAULT_DB = pathlib.Path(__file__).parent.parent / "db" / "order_book.db"

_SETUP_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
CREATE TABLE IF NOT EXISTS ob_snapshots (
    code TEXT NOT NULL,
    ts   TEXT NOT NULL,
    bids TEXT NOT NULL,
    asks TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_obsnap_code_ts ON ob_snapshots (code, ts);
"""
# Deliberately NOT keyed on (code, ts). Two consecutive pushes can carry
# genuinely different books and still land on the same timestamp: the feed
# bursts several books per ~300ms cadence tick (12-19ms apart, measured) and
# datetime.now() on Windows only advances in ~15.6ms steps. A (code, ts)
# primary key made the second one REPLACE the first -- caught in testing as
# 30 pushes becoming 9 stored snapshots, which in production would have been
# silent loss of exactly the fast-moving books worth keeping.
#
# Rows are therefore identified by their implicit rowid, which also gives
# insertion order for free: readers below break ts ties with it rather than
# relying on MAX(ts) to name a single row.


# Snapshots retained per code by prune(). The feed delivers 6.7 distinct
# books a second at the regular-hours peak (measured, SOXL 2026-09-21), so
# this is roughly 50 minutes of history -- comfortably more than the heatmap's
# 240-column window asks for, at about 36 MB per code. The old row-based
# budget of 1000 came out to 8 snapshots.
_DEFAULT_KEEP_SNAPSHOTS = 20_000


def _ts_str(ts) -> str:
    if isinstance(ts, datetime):
        return ts.isoformat(sep=" ")
    return str(ts)


class OrderBookStore:
    """Read/write access to order_book.db."""

    def __init__(self, db_path: str | pathlib.Path = _DEFAULT_DB,
                 read_only: bool = False):
        self._path = pathlib.Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if read_only:
            self._con = sqlite3.connect(str(self._path), check_same_thread=False)
        else:
            self._con = sqlite3.connect(str(self._path), check_same_thread=False)
            self._con.executescript(_SETUP_SQL)

    # ── write ──────────────────────────────────────────────────────────────────

    def insert_snapshot(self, code: str, ts,
                        bids: list[tuple[float, int]],
                        asks: list[tuple[float, int]]) -> int:
        """Insert one full order book snapshot atomically.

        bids / asks: list of (price, volume) tuples.
        Returns total rows inserted.
        """
        b = [[float(p), int(v)] for p, v in bids]
        a = [[float(p), int(v)] for p, v in asks]
        if not b and not a:
            return 0
        self._con.execute(
            "INSERT INTO ob_snapshots (code, ts, bids, asks) "
            "VALUES (?, ?, ?, ?)",
            (code, _ts_str(ts), json.dumps(b, separators=(",", ":")),
             json.dumps(a, separators=(",", ":"))),
        )
        self._con.commit()
        # Levels, not rows. One row is written now, but callers use this as
        # "how much book did I capture" -- the collector's session counter and
        # its first-snapshot log both read it that way.
        return len(b) + len(a)

    # ── read ───────────────────────────────────────────────────────────────────

    # Every read goes through one of the methods below rather than a
    # hand-rolled SELECT. The heatmap, the DOM window and two scripts each
    # used to open their own connection and spell out the same
    # "SELECT ts, side, price, volume FROM order_book_snapshots" -- six copies
    # of the row shape, which means six places to fix for any schema change.

    def _rows(self, sql: str, params: list) -> list[dict]:
        """Run a query returning (ts, bids, asks) and flatten it to level dicts.

        The single place that knows how a stored snapshot maps to the dicts
        readers expect -- which is why the schema change reaches exactly one
        function instead of the six hand-rolled SELECTs there used to be.
        Readers still see one dict per price level, bids before asks, so
        nothing downstream had to change.
        """
        out: list[dict] = []
        for ts_s, bids_j, asks_j in self._con.execute(sql, params).fetchall():
            ts = datetime.fromisoformat(ts_s)
            for side, payload in (("BID", bids_j), ("ASK", asks_j)):
                for price, volume in json.loads(payload):
                    out.append({"ts": ts, "side": side,
                                "price": float(price), "volume": float(volume)})
        return out

    def _snapshots(self, sql: str, params: list) -> list[list[dict]]:
        """Like _rows, but one list per stored snapshot rather than one flat
        list. Grouping happens by row, never by timestamp -- see the DDL note
        on why two snapshots can share one."""
        out: list[list[dict]] = []
        for ts_s, bids_j, asks_j in self._con.execute(sql, params).fetchall():
            ts = datetime.fromisoformat(ts_s)
            group: list[dict] = []
            for side, payload in (("BID", bids_j), ("ASK", asks_j)):
                for price, volume in json.loads(payload):
                    group.append({"ts": ts, "side": side,
                                  "price": float(price), "volume": float(volume)})
            out.append(group)
        return out

    def query_snapshots(self, code: str, start: datetime, end: datetime,
                        end_inclusive: bool = False) -> list[dict]:
        """Rows for *code* between *start* and *end*, oldest first.

        end_inclusive exists because the callers disagreed: the viewer's
        date-window loads were written against a half-open end, the DOM
        window's against a closed one. Rather than quietly change either,
        both keep the bound they had.
        """
        op = "<=" if end_inclusive else "<"
        return self._rows(
            "SELECT ts, bids, asks FROM ob_snapshots "
            f"WHERE code = ? AND ts >= ? AND ts {op} ? ORDER BY ts",
            [code, _ts_str(start), _ts_str(end)],
        )

    def latest_snapshot(self, code: str) -> list[dict]:
        """Every row of the newest snapshot for *code* -- one full book."""
        # ORDER BY ... LIMIT 1, not ts = MAX(ts): duplicate timestamps are
        # possible (see the DDL note), and MAX(ts) would return both books
        # fused into one 240-level "snapshot".
        return self._rows(
            "SELECT ts, bids, asks FROM ob_snapshots "
            "WHERE code = ? ORDER BY ts DESC, rowid DESC LIMIT 1",
            [code],
        )

    def snapshot_at_or_before(self, code: str, ts: datetime) -> list[dict]:
        """The newest snapshot at or before *ts* -- the book as of that moment."""
        return self._rows(
            "SELECT ts, bids, asks FROM ob_snapshots "
            "WHERE code = ? AND ts <= ? ORDER BY ts DESC, rowid DESC LIMIT 1",
            [code, _ts_str(ts)],
        )

    def last_n_snapshots(self, code: str, n: int) -> list[list[dict]]:
        """The last *n* snapshots for *code*, oldest first, one list each.

        One query rather than the 1 + n the callers used to issue: pull every
        row whose ts is in the newest n timestamps, then group. At ~120 rows a
        snapshot that is a round trip per snapshot saved.
        """
        # One row IS one snapshot, so this groups by row rather than by ts.
        # Grouping by ts would fuse two same-timestamp books into one.
        return self._snapshots(
            "SELECT ts, bids, asks FROM ("
            "  SELECT rowid, ts, bids, asks FROM ob_snapshots "
            "  WHERE code = ? ORDER BY ts DESC, rowid DESC LIMIT ?)"
            " ORDER BY ts, rowid",
            [code, n],
        )

    def latest_ts(self, code: str | None = None) -> datetime | None:
        """Newest snapshot timestamp, for *code* or across every code."""
        if code:
            r = self._con.execute(
                "SELECT MAX(ts) FROM ob_snapshots WHERE code = ?",
                [code]).fetchone()
        else:
            r = self._con.execute(
                "SELECT MAX(ts) FROM ob_snapshots").fetchone()
        return datetime.fromisoformat(r[0]) if r and r[0] else None

    def query_date(self, code: str, day: date) -> list[dict]:
        start = datetime(day.year, day.month, day.day)
        end   = datetime(day.year, day.month, day.day, 23, 59, 59, 999999)
        return self.query_snapshots(code, start, end)

    def available_dates(self, code: str) -> list[date]:
        cur = self._con.execute(
            "SELECT DISTINCT date(ts) FROM ob_snapshots "
            "WHERE code = ? ORDER BY 1",
            [code],
        )
        return [date.fromisoformat(r[0]) for r in cur.fetchall()]

    def snapshot_count(self, code: str | None = None) -> int:
        """Number of stored snapshots, for *code* or across every code.

        Renamed from row_count with the schema: a row is a snapshot now, so
        the old name would have quietly changed meaning by a factor of ~120
        while still reading as correct at every call site.
        """
        if code:
            return self._con.execute(
                "SELECT COUNT(*) FROM ob_snapshots WHERE code = ?",
                [code],
            ).fetchone()[0]
        return self._con.execute(
            "SELECT COUNT(*) FROM ob_snapshots"
        ).fetchone()[0]

    def prune(self, keep: int = _DEFAULT_KEEP_SNAPSHOTS,
              codes: list[str] | None = None) -> int:
        """Keep only the most recent *keep* SNAPSHOTS per code.

        Snapshots, not rows -- the distinction used to matter and was wrong.
        Under the row-per-level table a "keep 1000 rows" budget bought 1000/120
        = 8 snapshots of a 60+60 book, i.e. under 20 seconds of history, and
        the retained depth silently shrank as the book got deeper. That is why
        the heatmap's pre-fill could only ever find a handful of columns.

        codes: restrict pruning to this subset (e.g. so a caller can run this
        twice with different *keep* values per code group, mirroring
        TickStore.prune_older_than's codes-list pattern). Defaults to every
        code currently in the table.

        Returns snapshots deleted.
        """
        if codes is None:
            codes = [r[0] for r in self._con.execute(
                "SELECT DISTINCT code FROM ob_snapshots"
            ).fetchall()]
        deleted = 0
        for code in codes:
            # rowid, not ts: two snapshots can share a timestamp, and a
            # ts-based NOT IN would either keep or delete both of them.
            cur = self._con.execute(
                "DELETE FROM ob_snapshots WHERE code = ? AND rowid NOT IN ("
                "  SELECT rowid FROM ob_snapshots WHERE code = ?"
                "  ORDER BY ts DESC, rowid DESC LIMIT ?"
                ")",
                [code, code, keep],
            )
            deleted += cur.rowcount
        if deleted:
            self._con.commit()
        return deleted

    def drop_legacy_table(self) -> bool:
        """Drop the pre-schema-change order_book_snapshots table, if present.

        Not called automatically: it is the user's data, and the file does not
        shrink until a VACUUM anyway. Returns whether a table was dropped.
        """
        row = self._con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' "
            "AND name='order_book_snapshots'").fetchone()
        if not row:
            return False
        self._con.execute("DROP TABLE order_book_snapshots")
        self._con.commit()
        return True

    # ── lifecycle ──────────────────────────────────────────────────────────────

    def close(self) -> None:
        self._con.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

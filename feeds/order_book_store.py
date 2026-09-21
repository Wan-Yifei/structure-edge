"""SQLite-backed order book snapshot storage (WAL mode — concurrent reader + writer safe).

Schema:
    order_book_snapshots(code TEXT, ts TEXT, side TEXT, price REAL, volume INTEGER)

side: 'BID' | 'ASK'
ts is stored as ISO-8601 string: '2026-05-30 09:30:00.123456'
Each call to insert_snapshot() stores one full order book state (all bid + ask levels).
"""

from __future__ import annotations

import sqlite3
import pathlib
from datetime import datetime, date

_DEFAULT_DB = pathlib.Path(__file__).parent.parent / "db" / "order_book.db"

_SETUP_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
CREATE TABLE IF NOT EXISTS order_book_snapshots (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    code    TEXT    NOT NULL,
    ts      TEXT    NOT NULL,
    side    TEXT    NOT NULL,
    price   REAL    NOT NULL,
    volume  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_obs_code_ts ON order_book_snapshots (code, ts);
"""


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
        ts_s = _ts_str(ts)
        data = (
            [(code, ts_s, "BID", float(p), int(v)) for p, v in bids]
            + [(code, ts_s, "ASK", float(p), int(v)) for p, v in asks]
        )
        if not data:
            return 0
        self._con.executemany(
            "INSERT INTO order_book_snapshots (code, ts, side, price, volume) "
            "VALUES (?, ?, ?, ?, ?)",
            data,
        )
        self._con.commit()
        return len(data)

    # ── read ───────────────────────────────────────────────────────────────────

    # Every read goes through one of the methods below rather than a
    # hand-rolled SELECT. The heatmap, the DOM window and two scripts each
    # used to open their own connection and spell out the same
    # "SELECT ts, side, price, volume FROM order_book_snapshots" -- six copies
    # of the row shape, which means six places to fix for any schema change.

    def _rows(self, sql: str, params: list) -> list[dict]:
        """Run a query returning (ts, side, price, volume) and shape the rows.

        The single place that knows how a stored snapshot maps to the dicts
        readers expect.
        """
        return [
            {"ts": datetime.fromisoformat(r[0]),
             "side": r[1], "price": float(r[2]), "volume": float(r[3])}
            for r in self._con.execute(sql, params).fetchall()
        ]

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
            "SELECT ts, side, price, volume FROM order_book_snapshots "
            f"WHERE code = ? AND ts >= ? AND ts {op} ? ORDER BY ts",
            [code, _ts_str(start), _ts_str(end)],
        )

    def latest_snapshot(self, code: str) -> list[dict]:
        """Every row of the newest snapshot for *code* -- one full book."""
        return self._rows(
            "SELECT ts, side, price, volume FROM order_book_snapshots "
            "WHERE code = ? AND ts = ("
            "  SELECT MAX(ts) FROM order_book_snapshots WHERE code = ?)",
            [code, code],
        )

    def snapshot_at_or_before(self, code: str, ts: datetime) -> list[dict]:
        """The newest snapshot at or before *ts* -- the book as of that moment."""
        return self._rows(
            "SELECT ts, side, price, volume FROM order_book_snapshots "
            "WHERE code = ? AND ts = ("
            "  SELECT MAX(ts) FROM order_book_snapshots "
            "  WHERE code = ? AND ts <= ?)",
            [code, code, _ts_str(ts)],
        )

    def last_n_snapshots(self, code: str, n: int) -> list[list[dict]]:
        """The last *n* snapshots for *code*, oldest first, one list each.

        One query rather than the 1 + n the callers used to issue: pull every
        row whose ts is in the newest n timestamps, then group. At ~120 rows a
        snapshot that is a round trip per snapshot saved.
        """
        rows = self._rows(
            "SELECT ts, side, price, volume FROM order_book_snapshots "
            "WHERE code = ? AND ts IN ("
            "  SELECT ts FROM ("
            "    SELECT DISTINCT ts FROM order_book_snapshots "
            "    WHERE code = ? ORDER BY ts DESC LIMIT ?))"
            "ORDER BY ts",
            [code, code, n],
        )
        out: list[list[dict]] = []
        for row in rows:
            if not out or out[-1][0]["ts"] != row["ts"]:
                out.append([row])
            else:
                out[-1].append(row)
        return out

    def latest_ts(self, code: str | None = None) -> datetime | None:
        """Newest snapshot timestamp, for *code* or across every code."""
        if code:
            r = self._con.execute(
                "SELECT MAX(ts) FROM order_book_snapshots WHERE code = ?",
                [code]).fetchone()
        else:
            r = self._con.execute(
                "SELECT MAX(ts) FROM order_book_snapshots").fetchone()
        return datetime.fromisoformat(r[0]) if r and r[0] else None

    def query_date(self, code: str, day: date) -> list[dict]:
        start = datetime(day.year, day.month, day.day)
        end   = datetime(day.year, day.month, day.day, 23, 59, 59, 999999)
        return self.query_snapshots(code, start, end)

    def available_dates(self, code: str) -> list[date]:
        cur = self._con.execute(
            "SELECT DISTINCT date(ts) FROM order_book_snapshots "
            "WHERE code = ? ORDER BY 1",
            [code],
        )
        return [date.fromisoformat(r[0]) for r in cur.fetchall()]

    def row_count(self, code: str | None = None) -> int:
        if code:
            return self._con.execute(
                "SELECT COUNT(*) FROM order_book_snapshots WHERE code = ?",
                [code],
            ).fetchone()[0]
        return self._con.execute(
            "SELECT COUNT(*) FROM order_book_snapshots"
        ).fetchone()[0]

    def prune(self, keep: int = 1000, codes: list[str] | None = None) -> int:
        """Delete old rows keeping only the most recent *keep* rows per code.

        codes: restrict pruning to this subset (e.g. so a caller can run this
        twice with different *keep* values per code group, mirroring
        TickStore.prune_older_than's codes-list pattern). Defaults to every
        code currently in the table.

        Returns total rows deleted.
        """
        if codes is None:
            codes = [r[0] for r in self._con.execute(
                "SELECT DISTINCT code FROM order_book_snapshots"
            ).fetchall()]
        deleted = 0
        for code in codes:
            cur = self._con.execute(
                "DELETE FROM order_book_snapshots WHERE code = ? AND id NOT IN ("
                "  SELECT id FROM order_book_snapshots WHERE code = ?"
                "  ORDER BY id DESC LIMIT ?"
                ")",
                [code, code, keep],
            )
            deleted += cur.rowcount
        if deleted:
            self._con.commit()
        return deleted

    # ── lifecycle ──────────────────────────────────────────────────────────────

    def close(self) -> None:
        self._con.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

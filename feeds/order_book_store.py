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

    def query_snapshots(self, code: str,
                        start: datetime, end: datetime) -> list[dict]:
        cur = self._con.execute(
            "SELECT ts, side, price, volume FROM order_book_snapshots "
            "WHERE code = ? AND ts >= ? AND ts < ? ORDER BY ts",
            [code, _ts_str(start), _ts_str(end)],
        )
        return [
            {"ts": datetime.fromisoformat(r[0]),
             "side": r[1], "price": r[2], "volume": r[3]}
            for r in cur.fetchall()
        ]

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

    # ── lifecycle ──────────────────────────────────────────────────────────────

    def close(self) -> None:
        self._con.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

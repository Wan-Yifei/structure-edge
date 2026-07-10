"""SQLite-backed tick storage (WAL mode — concurrent reader + writer safe).

Schema:
    ticks(code TEXT, ts TEXT, price REAL, volume INTEGER, direction TEXT)

ts is stored as ISO-8601 string: '2026-05-18 09:30:00.123456'
Direction values: 'BUY' | 'SELL' | 'NEUTRAL'
"""

from __future__ import annotations

import sqlite3
import pathlib
from datetime import datetime, date, timedelta
from typing import Iterable

_DEFAULT_DB = pathlib.Path(__file__).parent.parent / "db" / "ticks.db"

_SETUP_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
CREATE TABLE IF NOT EXISTS ticks (
    code      TEXT    NOT NULL,
    ts        TEXT    NOT NULL,
    price     REAL    NOT NULL,
    volume    INTEGER NOT NULL,
    direction TEXT    NOT NULL,
    UNIQUE(code, ts, price, volume)
);
CREATE INDEX IF NOT EXISTS idx_ticks_code_ts ON ticks (code, ts);
"""


def _ts_str(ts) -> str:
    if isinstance(ts, datetime):
        return ts.isoformat(sep=" ")
    return str(ts)


class TickStore:
    def __init__(self, db_path: str | pathlib.Path = _DEFAULT_DB,
                 read_only: bool = False):
        self._path = pathlib.Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if read_only:
            # Open without DDL so we never contend with the writer's RESERVED lock.
            self._con = sqlite3.connect(str(self._path), check_same_thread=False)
        else:
            self._con = sqlite3.connect(str(self._path), check_same_thread=False)
            self._con.executescript(_SETUP_SQL)

    # ── write ──────────────────────────────────────────────────────────────

    def insert_ticks(self, rows: Iterable[dict]) -> int:
        """Insert tick dicts.  Silently skips duplicates.  Returns rows inserted."""
        data = [
            (
                r["code"],
                _ts_str(r["ts"]),
                float(r["price"]),
                int(r["volume"]),
                str(r["direction"]).upper(),
            )
            for r in rows
        ]
        if not data:
            return 0
        self._con.executemany(
            "INSERT OR IGNORE INTO ticks VALUES (?, ?, ?, ?, ?)", data
        )
        self._con.commit()
        return len(data)

    # ── read ───────────────────────────────────────────────────────────────

    def query_ticks(self, code: str, start: datetime, end: datetime) -> list[dict]:
        cur = self._con.execute(
            "SELECT ts, price, volume, direction FROM ticks "
            "WHERE code = ? AND ts >= ? AND ts < ? ORDER BY ts",
            [code, _ts_str(start), _ts_str(end)],
        )
        return [
            {"ts": datetime.fromisoformat(r[0]),
             "price": r[1], "volume": r[2], "direction": r[3]}
            for r in cur.fetchall()
        ]

    def query_date(self, code: str, day: date) -> list[dict]:
        start = datetime(day.year, day.month, day.day)
        end   = datetime(day.year, day.month, day.day, 23, 59, 59, 999999)
        return self.query_ticks(code, start, end)

    def available_dates(self, code: str) -> list[date]:
        cur = self._con.execute(
            "SELECT DISTINCT date(ts) FROM ticks WHERE code = ? ORDER BY 1", [code]
        )
        return [date.fromisoformat(r[0]) for r in cur.fetchall()]

    def prune_older_than(self, codes: list[str], days: int = 30) -> int:
        """Delete tick rows older than *days* days, per code in *codes*.

        Callers pass the known target codes (rather than SELECT DISTINCT code)
        so each delete is a plain (code, ts) index range scan via
        idx_ticks_code_ts, not a full table scan on a possibly huge table.
        Returns total rows deleted.
        """
        cutoff = _ts_str(datetime.now() - timedelta(days=days))
        deleted = 0
        for code in codes:
            cur = self._con.execute(
                "DELETE FROM ticks WHERE code = ? AND ts < ?", [code, cutoff]
            )
            deleted += cur.rowcount
        if deleted:
            self._con.commit()
        return deleted

    def row_count(self, code: str | None = None) -> int:
        if code:
            return self._con.execute(
                "SELECT COUNT(*) FROM ticks WHERE code = ?", [code]
            ).fetchone()[0]
        return self._con.execute("SELECT COUNT(*) FROM ticks").fetchone()[0]

    # ── lifecycle ──────────────────────────────────────────────────────────

    def close(self):
        self._con.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

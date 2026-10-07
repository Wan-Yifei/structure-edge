"""Sides inferred for off-exchange prints, kept beside ticks.db rather than in it.

The feed leaves 57.5% of SOXL's volume NEUTRAL, and that volume is
off-exchange flow rather than noise -- 43.9% of those prints carry sub-penny
prices, which only wholesaler price improvement produces. Signed against the
prevailing midpoint it tracks the minute's return about as well as the
exchange's own tags (+0.253 against +0.276). See doc/ORDER_FLOW_GUIDE.md
appendix A.

**Why a separate file.** ticks.db is held by the tick collector whenever a
session is open, and measurably so: 82 write attempts over 40 seconds, with a
30-second busy timeout, every one refused. Derived data that can be rebuilt at
any time has no business contending for that lock, or sitting inside a 7.8GB
primary store. This file is disposable -- delete it and re-run the classifier.

Rows are keyed the same way ticks.db enforces uniqueness, so the join is exact
and needs no row ids, which a VACUUM would renumber.
"""
from __future__ import annotations

import pathlib
import sqlite3
from datetime import datetime
from typing import Iterable

_DEFAULT_DB = pathlib.Path(__file__).parent.parent / "db" / "tick_mid.db"

_SETUP_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA busy_timeout=30000;
CREATE TABLE IF NOT EXISTS tick_mid (
    code    TEXT    NOT NULL,
    ts      TEXT    NOT NULL,
    price   REAL    NOT NULL,
    volume  INTEGER NOT NULL,
    mid_dir TEXT    NOT NULL,
    PRIMARY KEY (code, ts, price, volume)
) WITHOUT ROWID;
"""


def _ts_str(ts) -> str:
    return ts.isoformat(sep=" ") if isinstance(ts, datetime) else str(ts)


class TickMidStore:
    def __init__(self, db_path: str | pathlib.Path | None = None,
                 read_only: bool = False) -> None:
        # Resolved here, not as a default argument: a default binds at import
        # and cannot be redirected afterwards, which silently sent every
        # caller at the real file regardless of the ticks.db it belonged to.
        self._path = pathlib.Path(db_path or _DEFAULT_DB)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._con = sqlite3.connect(str(self._path), check_same_thread=False)
        if read_only:
            self._con.execute("PRAGMA busy_timeout=30000")
        else:
            self._con.executescript(_SETUP_SQL)

    def put(self, rows: Iterable[tuple]) -> int:
        """rows: (code, ts, price, volume, mid_dir). Replaces on conflict."""
        data = [(c, _ts_str(t), float(p), int(v), d) for c, t, p, v, d in rows]
        if not data:
            return 0
        self._con.executemany(
            "INSERT OR REPLACE INTO tick_mid "
            "(code, ts, price, volume, mid_dir) VALUES (?, ?, ?, ?, ?)", data)
        self._con.commit()
        return len(data)

    def load(self, code: str, start: datetime, end: datetime) -> dict:
        """(ts_str, price, volume) -> side, for [start, end).

        A dict rather than a SQL join: ticks.db and this file are separate
        connections, and ATTACHing one to the other would put this file's lock
        in the path of every tick read.
        """
        cur = self._con.execute(
            "SELECT ts, price, volume, mid_dir FROM tick_mid "
            "WHERE code = ? AND ts >= ? AND ts < ?",
            [code, _ts_str(start), _ts_str(end)])
        return {(r[0], r[1], r[2]): r[3] for r in cur.fetchall()}

    def count(self, code: str | None = None) -> int:
        if code:
            return self._con.execute(
                "SELECT COUNT(*) FROM tick_mid WHERE code = ?",
                [code]).fetchone()[0]
        return self._con.execute("SELECT COUNT(*) FROM tick_mid").fetchone()[0]

    def close(self) -> None:
        self._con.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

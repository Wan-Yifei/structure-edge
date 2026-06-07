"""Signal persistence layer for the SMC signal scanner.

Uses SQLite in WAL mode so the viewer can read concurrently without
blocking the scanner writer.
"""

import pathlib
import sqlite3
import uuid
from datetime import datetime
from typing import Optional

_DEFAULT_PATH = pathlib.Path(__file__).parent / "signals.db"

_DDL = """
CREATE TABLE IF NOT EXISTS signals (
    signal_id         TEXT PRIMARY KEY,
    symbol            TEXT NOT NULL,
    direction         TEXT NOT NULL,
    signal_time       TEXT NOT NULL,
    trend_tf          TEXT NOT NULL,
    entry_tf          TEXT NOT NULL,
    entry_zone_top    REAL NOT NULL,
    entry_zone_bottom REAL NOT NULL,
    sl_price          REAL NOT NULL,
    tp_price          REAL NOT NULL,
    rr_ratio          REAL NOT NULL,
    bos_price         REAL,
    strategy          TEXT NOT NULL DEFAULT 'smc',
    params_json       TEXT NOT NULL,
    algo_version      TEXT,
    source            TEXT NOT NULL DEFAULT 'auto',
    status            TEXT NOT NULL DEFAULT 'open',
    closed_at         TEXT,
    created_at        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sig_symbol_time ON signals(symbol, signal_time DESC);
CREATE INDEX IF NOT EXISTS idx_sig_status      ON signals(status);
"""


class SignalsDB:
    """Read/write access to signals.db (SQLite WAL)."""

    def __init__(
        self,
        path: str | pathlib.Path | None = None,
        read_only: bool = False,
    ) -> None:
        self._path = pathlib.Path(path) if path else _DEFAULT_PATH
        self._read_only = read_only
        self._conn = sqlite3.connect(
            str(self._path),
            check_same_thread=False,
            uri=read_only,
        )
        self._conn.row_factory = sqlite3.Row
        if not read_only:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_DDL)
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "SignalsDB":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ── write ─────────────────────────────────────────────────────────────────

    def insert_signal(self, sig: dict) -> str:
        """Insert a new signal record. Returns the generated signal_id."""
        signal_id = sig.get("signal_id") or str(uuid.uuid4())
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        self._conn.execute(
            """
            INSERT OR IGNORE INTO signals (
                signal_id, symbol, direction, signal_time,
                trend_tf, entry_tf,
                entry_zone_top, entry_zone_bottom,
                sl_price, tp_price, rr_ratio, bos_price,
                strategy, params_json, algo_version,
                source, status, closed_at, created_at
            ) VALUES (
                ?,?,?,?,  ?,?,  ?,?,  ?,?,?,?,  ?,?,?,  ?,?,?,?
            )
            """,
            (
                signal_id,
                sig["symbol"],
                sig["direction"],
                sig["signal_time"],
                sig["trend_tf"],
                sig["entry_tf"],
                float(sig["entry_zone_top"]),
                float(sig["entry_zone_bottom"]),
                float(sig["sl_price"]),
                float(sig["tp_price"]),
                float(sig["rr_ratio"]),
                sig.get("bos_price"),
                sig.get("strategy", "smc"),
                sig["params_json"],
                sig.get("algo_version"),
                sig.get("source", "auto"),
                sig.get("status", "open"),
                sig.get("closed_at"),
                sig.get("created_at", now),
            ),
        )
        self._conn.commit()
        return signal_id

    def update_status(
        self,
        signal_id: str,
        status: str,
        closed_at: Optional[str] = None,
    ) -> None:
        """Update the status (and optionally closed_at) of an existing signal."""
        closed = closed_at or datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        self._conn.execute(
            "UPDATE signals SET status=?, closed_at=? WHERE signal_id=?",
            (status, closed, signal_id),
        )
        self._conn.commit()

    # ── read ──────────────────────────────────────────────────────────────────

    def query_signals(
        self,
        symbol: str,
        since_dt: str,
        status: Optional[str] = None,
    ) -> list[dict]:
        """Return signals for symbol created at or after since_dt.

        Args:
            symbol:   moomoo stock code, e.g. 'US.AAPL'
            since_dt: ISO datetime string lower bound (inclusive)
            status:   filter by status ('open'|'hit_tp'|'hit_sl'|'expired');
                      None returns all statuses
        """
        if status is not None:
            rows = self._conn.execute(
                "SELECT * FROM signals WHERE symbol=? AND signal_time>=? AND status=?"
                " ORDER BY signal_time DESC",
                (symbol, since_dt, status),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM signals WHERE symbol=? AND signal_time>=?"
                " ORDER BY signal_time DESC",
                (symbol, since_dt),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_open_signals(self, symbol: str) -> list[dict]:
        """Return all open signals for symbol (used for deduplication in scanner)."""
        rows = self._conn.execute(
            "SELECT * FROM signals WHERE symbol=? AND status='open'"
            " ORDER BY signal_time DESC",
            (symbol,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_all_open_signals(self) -> list[dict]:
        """Return all open signals across all symbols."""
        rows = self._conn.execute(
            "SELECT * FROM signals WHERE status='open' ORDER BY signal_time DESC"
        ).fetchall()
        return [dict(r) for r in rows]

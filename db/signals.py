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

CREATE TABLE IF NOT EXISTS fvg_watch_signals (
    signal_id         TEXT PRIMARY KEY,
    symbol            TEXT NOT NULL,
    tf                TEXT NOT NULL,
    direction         TEXT NOT NULL,
    zone_top          REAL NOT NULL,
    zone_bottom       REAL NOT NULL,
    formed_time       TEXT NOT NULL,
    filled            INTEGER NOT NULL DEFAULT 0,
    params_json       TEXT NOT NULL,
    status            TEXT NOT NULL DEFAULT 'open',
    created_at        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_fvgw_symbol_tf ON fvg_watch_signals(symbol, tf, formed_time DESC);
CREATE INDEX IF NOT EXISTS idx_fvgw_status    ON fvg_watch_signals(status);

-- One row PER RULE, continuously upserted -- not an event stream like the
-- two tables above. "indicator"/"params_json"/"field" together describe
-- "which indicator, what params, which output series to compare" so this
-- one schema covers RSI, MACD, KD, or anything else registered into
-- analysis/indicator_alert_watcher.py's INDICATOR_REGISTRY without a
-- migration -- no column here is RSI-specific.
CREATE TABLE IF NOT EXISTS indicator_alert_state (
    rule_id       TEXT PRIMARY KEY,
    symbol        TEXT NOT NULL,
    tf            TEXT NOT NULL,
    indicator     TEXT NOT NULL,
    params_json   TEXT NOT NULL,
    field         TEXT NOT NULL,
    condition     TEXT NOT NULL,
    threshold     REAL NOT NULL,
    value         REAL,
    is_positive   INTEGER NOT NULL DEFAULT 0,
    is_muted      INTEGER NOT NULL DEFAULT 0,
    last_bar_time TEXT,
    updated_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_indalert_symbol ON indicator_alert_state(symbol);
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

    def delete_signal(self, signal_id: str) -> None:
        """Physically remove a signal row (unlike update_status, this cannot be undone)."""
        self._conn.execute("DELETE FROM signals WHERE signal_id=?", (signal_id,))
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

    # ── fvg_watch_signals (lightweight "FVG formed" alerts) ─────────────────────

    def insert_fvg_watch(self, sig: dict) -> str:
        """Insert a new FVG-watch signal record. Returns the generated signal_id."""
        signal_id = sig.get("signal_id") or str(uuid.uuid4())
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        self._conn.execute(
            """
            INSERT OR IGNORE INTO fvg_watch_signals (
                signal_id, symbol, tf, direction,
                zone_top, zone_bottom, formed_time, filled,
                params_json, status, created_at
            ) VALUES (
                ?,?,?,?,  ?,?,?,?,  ?,?,?
            )
            """,
            (
                signal_id,
                sig["symbol"],
                sig["tf"],
                sig["direction"],
                float(sig["zone_top"]),
                float(sig["zone_bottom"]),
                sig["formed_time"],
                int(sig.get("filled", False)),
                sig["params_json"],
                sig.get("status", "open"),
                sig.get("created_at", now),
            ),
        )
        self._conn.commit()
        return signal_id

    def get_open_fvg_watch(self, symbol: str, tf: str) -> list[dict]:
        """Return all open FVG-watch signals for (symbol, tf) — used for dedup."""
        rows = self._conn.execute(
            "SELECT * FROM fvg_watch_signals WHERE symbol=? AND tf=? AND status='open'"
            " ORDER BY formed_time DESC",
            (symbol, tf),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_all_open_fvg_watch(self) -> list[dict]:
        """Return all open FVG-watch signals across every symbol/tf."""
        rows = self._conn.execute(
            "SELECT * FROM fvg_watch_signals WHERE status='open' ORDER BY formed_time DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_fvg_watch(self, signal_id: str) -> None:
        """Physically remove a FVG-watch signal row (cannot be undone)."""
        self._conn.execute("DELETE FROM fvg_watch_signals WHERE signal_id=?", (signal_id,))
        self._conn.commit()

    def query_fvg_watch(
        self,
        symbol: str,
        since_dt: str,
        status: Optional[str] = None,
    ) -> list[dict]:
        """Return FVG-watch signals for symbol formed at or after since_dt."""
        if status is not None:
            rows = self._conn.execute(
                "SELECT * FROM fvg_watch_signals WHERE symbol=? AND formed_time>=? AND status=?"
                " ORDER BY formed_time DESC",
                (symbol, since_dt, status),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM fvg_watch_signals WHERE symbol=? AND formed_time>=?"
                " ORDER BY formed_time DESC",
                (symbol, since_dt),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── indicator_alert_state (one row per rule, continuously upserted) ────────

    def upsert_indicator_alert_state(self, state: dict) -> None:
        """Insert or refresh a rule's current reading.

        `is_muted` is deliberately NOT taken from the caller: on conflict it
        resolves to 0 whenever the new reading is negative (auto-unmute), or
        to whatever is already in the table when the new reading is positive
        (so a user's mute click -- a separate UPDATE via set_indicator_alert_muted
        -- from the main thread can never be raced/overwritten by the scan
        worker's next upsert on the same rule).
        """
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        self._conn.execute(
            """
            INSERT INTO indicator_alert_state (
                rule_id, symbol, tf, indicator, params_json, field,
                condition, threshold, value, is_positive, is_muted,
                last_bar_time, updated_at
            ) VALUES (?,?,?,?,?,?,  ?,?,?,?,0,  ?,?)
            ON CONFLICT(rule_id) DO UPDATE SET
                value         = excluded.value,
                is_positive   = excluded.is_positive,
                is_muted      = CASE WHEN excluded.is_positive = 0
                                     THEN 0
                                     ELSE indicator_alert_state.is_muted END,
                last_bar_time = excluded.last_bar_time,
                updated_at    = excluded.updated_at
            """,
            (
                state["rule_id"], state["symbol"], state["tf"], state["indicator"],
                state["params_json"], state["field"], state["condition"], state["threshold"],
                state.get("value"), int(bool(state.get("is_positive"))),
                state.get("last_bar_time"), now,
            ),
        )
        self._conn.commit()

    def get_all_indicator_alert_state(self) -> list[dict]:
        """Return every configured rule's current state, newest-updated first."""
        rows = self._conn.execute(
            "SELECT * FROM indicator_alert_state ORDER BY updated_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_indicator_alert_state(self, rule_id: str) -> Optional[dict]:
        """Return one rule's current state (e.g. to check is_muted right after
        an upsert), or None if it doesn't exist yet."""
        row = self._conn.execute(
            "SELECT * FROM indicator_alert_state WHERE rule_id=?", (rule_id,)
        ).fetchone()
        return dict(row) if row else None

    def set_indicator_alert_muted(self, rule_id: str, muted: bool) -> None:
        """Explicit user action (the Mute button) -- independent of upsert_indicator_alert_state."""
        self._conn.execute(
            "UPDATE indicator_alert_state SET is_muted=? WHERE rule_id=?",
            (int(bool(muted)), rule_id),
        )
        self._conn.commit()

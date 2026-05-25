"""backtest.duckdb schema and writer.

Stores runs, trades, and aggregate stats so every backtest is fully auditable
by run_id / trade_id without re-running.

DuckDB concurrency: workers return BacktestResult objects via Future;
the main process writes serially inside as_completed(). No concurrent writers.

Tables
------
runs       — one row per unique parameter configuration
trades     — one row per trade
run_stats  — aggregate metrics per run (written after all trades)
"""

from __future__ import annotations

import json
import pathlib
import uuid
from datetime import datetime
from typing import Optional

import duckdb
import pandas as pd

from backtest.engine import BacktestResult, Trade

_DEFAULT_DB  = pathlib.Path(__file__).parent.parent / "db" / "backtest.duckdb"
_REVIEW_DB   = pathlib.Path(__file__).parent.parent / "db" / "review_trades.duckdb"

_REVIEW_DDL = """
CREATE TABLE IF NOT EXISTS review_trades (
    trade_id    VARCHAR PRIMARY KEY,
    symbol      VARCHAR NOT NULL,
    direction   VARCHAR NOT NULL,
    entry_time  VARCHAR NOT NULL,
    exit_time   VARCHAR,
    entry_price DOUBLE  NOT NULL,
    exit_price  DOUBLE,
    sl_price    DOUBLE,
    tp_price    DOUBLE,
    result      VARCHAR,
    r_multiple  DOUBLE,
    planned_rr  DOUBLE,
    fvg_top     DOUBLE,
    fvg_bottom  DOUBLE,
    config_json JSON,
    created_at  TIMESTAMP DEFAULT now()
);
"""

_DDL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id       VARCHAR PRIMARY KEY,
    config_hash  VARCHAR NOT NULL,
    config_json  JSON    NOT NULL,
    symbol       VARCHAR NOT NULL,
    trend_tf     VARCHAR NOT NULL,
    entry_tf     VARCHAR NOT NULL,
    start_date   VARCHAR NOT NULL,
    end_date     VARCHAR NOT NULL,
    algo_version VARCHAR,
    commit_hash  VARCHAR,
    status       VARCHAR NOT NULL DEFAULT 'pending',
    created_at   TIMESTAMP DEFAULT now(),
    finished_at  TIMESTAMP
);

CREATE TABLE IF NOT EXISTS trades (
    trade_id      VARCHAR PRIMARY KEY,
    run_id        VARCHAR NOT NULL REFERENCES runs(run_id),
    symbol        VARCHAR NOT NULL,
    direction     VARCHAR NOT NULL,
    entry_time    VARCHAR NOT NULL,
    entry_price   DOUBLE  NOT NULL,
    sl_price      DOUBLE  NOT NULL,
    tp_price      DOUBLE  NOT NULL,
    exit_time     VARCHAR,
    exit_price    DOUBLE,
    result        VARCHAR,
    r_multiple    DOUBLE,
    planned_rr    DOUBLE,
    entry_ltf_bar INTEGER,
    fvg_top       DOUBLE,
    fvg_bottom    DOUBLE
);

CREATE TABLE IF NOT EXISTS run_stats (
    run_id          VARCHAR PRIMARY KEY REFERENCES runs(run_id),
    n_trades        INTEGER,
    win_rate        DOUBLE,
    total_r         DOUBLE,
    avg_r           DOUBLE,
    profit_factor   DOUBLE,
    max_drawdown_r  DOUBLE,
    max_loss_r      DOUBLE,
    sharpe          DOUBLE,
    sortino         DOUBLE,
    computed_at     TIMESTAMP DEFAULT now()
);


-- Live / paper trades from moomoo execution (not backtest simulation).
-- account_type = 'LIVE' | 'PAPER' enforced at application level.
-- This table is the source of truth for all real-money and sim trading activity.
CREATE TABLE IF NOT EXISTS live_trades (
    trade_id        VARCHAR PRIMARY KEY,
    account_type    VARCHAR NOT NULL,       -- 'LIVE' | 'PAPER'
    account_id      VARCHAR,               -- moomoo account identifier
    symbol          VARCHAR NOT NULL,
    direction       VARCHAR NOT NULL,       -- 'LONG' | 'SHORT'

    -- moomoo order references (populated on execution)
    order_id        VARCHAR,               -- entry order ID from moomoo
    exit_order_id   VARCHAR,               -- exit order ID from moomoo

    -- Position sizing
    entry_time      TIMESTAMP,
    entry_price     DOUBLE,
    qty             DOUBLE,                -- number of shares / contracts

    -- Risk plan (set at signal time, before execution)
    sl_price        DOUBLE,
    tp_price        DOUBLE,
    planned_rr      DOUBLE,

    -- Exit
    exit_time       TIMESTAMP,
    exit_price      DOUBLE,

    -- Outcome
    result          VARCHAR,               -- 'win' | 'loss' | 'breakeven' | 'manual' | 'open'
    pnl_gross       DOUBLE,               -- gross P&L in account currency
    commission      DOUBLE,               -- total commission / fees
    pnl_net         DOUBLE,               -- pnl_gross - commission
    r_multiple      DOUBLE,               -- realised R (relative to planned SL distance)

    -- Strategy linkage (for comparing live vs backtest performance)
    strategy        VARCHAR,              -- strategy name, e.g. 'SMC_v1'
    run_id          VARCHAR,              -- optional: backtest run_id that generated the signal
    signal_params   JSON,                -- strategy params used at signal time

    -- Manual annotations
    notes           TEXT,
    tags            VARCHAR,             -- comma-separated, e.g. 'news_risk,missed_entry'

    -- Audit
    created_at      TIMESTAMP DEFAULT now(),
    updated_at      TIMESTAMP DEFAULT now()
);
"""


class BacktestDB:
    """Thin wrapper around backtest.duckdb for writing runs, trades, and stats.

    Pass read_only=True to open an existing database without acquiring a write
    lock — safe to use alongside a running backtest (run.py) or from multiple
    Jupyter notebooks at the same time.  DDL and migrations are skipped in
    read-only mode; the database must already exist.
    """

    def __init__(
        self,
        db_path: str | pathlib.Path = _DEFAULT_DB,
        read_only: bool = False,
    ) -> None:
        self.db_path  = pathlib.Path(db_path)
        self.read_only = read_only
        if not read_only:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(str(self.db_path), read_only=read_only)
        if not read_only:
            self._conn.execute(_DDL)
            # Migrations for existing databases
            for col_ddl in [
                "ALTER TABLE runs   ADD COLUMN IF NOT EXISTS algo_version  VARCHAR",
                "ALTER TABLE runs   ADD COLUMN IF NOT EXISTS commit_hash   VARCHAR",
                "ALTER TABLE trades ADD COLUMN IF NOT EXISTS entry_ltf_bar INTEGER",
                "ALTER TABLE trades ADD COLUMN IF NOT EXISTS fvg_top       DOUBLE",
                "ALTER TABLE trades ADD COLUMN IF NOT EXISTS fvg_bottom    DOUBLE",
            ]:
                try:
                    self._conn.execute(col_ddl)
                except Exception:
                    pass

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "BacktestDB":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ── runs ──────────────────────────────────────────────────────────────────

    def get_or_create_run(
        self,
        config_hash: str,
        config_json: dict,
        symbol: str,
        trend_tf: str,
        entry_tf: str,
        start_date: str,
        end_date: str,
        algo_version: str = "",
        commit_hash: str = "",
    ) -> tuple[str, bool]:
        """Return (run_id, needs_write).

        needs_write=False: run already completed (status='done'), skip writing.
        needs_write=True:  new run created, or crashed run reset to pending.
        """
        self._require_write()
        row = self._conn.execute(
            "SELECT run_id, status FROM runs WHERE config_hash = ? "
            "AND symbol = ? AND trend_tf = ? AND entry_tf = ? "
            "AND COALESCE(algo_version, '') = ?",
            [config_hash, symbol, trend_tf, entry_tf, algo_version],
        ).fetchone()

        if row:
            run_id, status = row
            if status == "done":
                return run_id, False
            # crashed run — reset to pending, caller will re-write trades
            self._conn.execute(
                "UPDATE runs SET status='pending', finished_at=NULL WHERE run_id=?",
                [run_id],
            )
            return run_id, True

        run_id = str(uuid.uuid4())
        self._conn.execute(
            "INSERT INTO runs (run_id, config_hash, config_json, symbol, "
            "trend_tf, entry_tf, start_date, end_date, algo_version, commit_hash) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            [run_id, config_hash, json.dumps(config_json), symbol,
             trend_tf, entry_tf, start_date, end_date,
             algo_version or None, commit_hash or None],
        )
        return run_id, True

    def _require_write(self) -> None:
        if self.read_only:
            raise RuntimeError("BacktestDB opened in read-only mode — write operations not allowed")

    def mark_running(self, run_id: str) -> None:
        self._require_write()
        self._conn.execute(
            "UPDATE runs SET status='running' WHERE run_id=?", [run_id]
        )

    def mark_done(self, run_id: str) -> None:
        self._require_write()
        self._conn.execute(
            "UPDATE runs SET status='done', finished_at=? WHERE run_id=?",
            [datetime.now(), run_id],
        )

    def mark_failed(self, run_id: str, error: str) -> None:
        self._require_write()
        self._conn.execute(
            "UPDATE runs SET status='failed' WHERE run_id=?", [run_id]
        )

    # ── trades ────────────────────────────────────────────────────────────────

    def write_trades(self, run_id: str, symbol: str, trades: list[Trade]) -> None:
        """Batch-insert all trades for one run in a single transaction."""
        self._require_write()
        if not trades:
            return
        rows = [
            (
                t.trade_id,
                run_id,
                symbol,
                t.direction,
                t.entry_time,
                t.entry_price,
                t.sl,
                t.tp,
                t.exit_time,
                t.exit_price,
                t.result,
                t.r_multiple,
                t.planned_rr,
                t.entry_ltf_bar,
                t.fvg_top    if t.fvg_top    else None,
                t.fvg_bottom if t.fvg_bottom else None,
            )
            for t in trades
        ]
        self._conn.execute("BEGIN")
        try:
            self._conn.execute("DELETE FROM trades WHERE run_id = ?", [run_id])
            self._conn.executemany(
                "INSERT OR IGNORE INTO trades VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows
            )
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    # ── run_stats ─────────────────────────────────────────────────────────────

    def write_stats(self, run_id: str, bt: BacktestResult) -> None:
        """Compute and insert aggregate stats for a completed run."""
        self._require_write()
        s = bt.summary_dict()
        self._conn.execute(
            "INSERT OR REPLACE INTO run_stats VALUES (?,?,?,?,?,?,?,?,?,?,now())",
            [
                run_id,
                s["n_trades"],
                s["win_rate"],
                s["total_r"],
                s["avg_r"],
                s["profit_factor"],
                s["max_drawdown_r"],
                s["max_loss_r"],
                s.get("sharpe",  0.0),
                s.get("sortino", 0.0),
            ],
        )

    # ── date-range reuse ──────────────────────────────────────────────────────

    def covered_segments(
        self, config_hash: str, symbol: str, trend_tf: str, entry_tf: str,
        algo_version: str = "",
    ) -> list[tuple[str, str]]:
        """Return (start_date, end_date) for all completed runs of this combo."""
        rows = self._conn.execute(
            "SELECT start_date, end_date FROM runs "
            "WHERE config_hash=? AND symbol=? AND trend_tf=? AND entry_tf=? "
            "AND COALESCE(algo_version, '') = ? AND status='done' ORDER BY start_date",
            [config_hash, symbol, trend_tf, entry_tf, algo_version],
        ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def load_trades_in_range(
        self, config_hash: str, symbol: str,
        trend_tf: str, entry_tf: str,
        start: str, end: str,
        algo_version: str = "",
    ) -> list["Trade"]:
        """Load Trade objects from DB for this combo within [start, end]."""
        from backtest.engine import Trade  # local import to avoid circular
        rows = self._conn.execute(
            """
            SELECT t.trade_id, t.direction, t.entry_price, t.sl_price, t.tp_price,
                   t.planned_rr, t.entry_time, t.exit_time, t.exit_price,
                   t.result, t.r_multiple, t.entry_ltf_bar,
                   t.fvg_top, t.fvg_bottom
            FROM trades t
            JOIN runs r ON t.run_id = r.run_id
            WHERE r.config_hash=? AND t.symbol=?
              AND r.trend_tf=? AND r.entry_tf=?
              AND COALESCE(r.algo_version, '') = ?
              AND r.status='done'
              AND LEFT(t.entry_time, 10) >= ? AND LEFT(t.entry_time, 10) <= ?
            ORDER BY t.entry_time
            """,
            [config_hash, symbol, trend_tf, entry_tf, algo_version, start, end],
        ).fetchall()
        trades = []
        for r in rows:
            t = Trade(
                trade_id      = r[0],
                direction     = r[1],
                entry_price   = r[2],
                sl            = r[3],
                tp            = r[4],
                planned_rr    = r[5],
                entry_time    = r[6],
                exit_time     = r[7] or "",
                exit_price    = r[8] or 0.0,
                result        = r[9] or "",
                r_multiple    = r[10] or 0.0,
                entry_ltf_bar = r[11] or 0,
                fvg_top       = r[12] or 0.0,
                fvg_bottom    = r[13] or 0.0,
            )
            trades.append(t)
        return trades

    # ── queries ───────────────────────────────────────────────────────────────

    def get_trades(self, run_id: str) -> pd.DataFrame:
        return self._conn.execute(
            "SELECT * FROM trades WHERE run_id = ? ORDER BY entry_time", [run_id]
        ).df()

    def get_run_stats(self, top_n: int = 20) -> pd.DataFrame:
        return self._conn.execute(
            "SELECT r.symbol, r.trend_tf, r.entry_tf, r.algo_version, r.commit_hash, r.config_hash, "
            "s.n_trades, s.win_rate, s.total_r, s.avg_r, "
            "s.profit_factor, s.max_drawdown_r, s.sharpe, s.sortino "
            "FROM run_stats s JOIN runs r ON s.run_id = r.run_id "
            "WHERE r.status = 'done' "
            "ORDER BY s.profit_factor DESC LIMIT ?",
            [top_n],
        ).df()

    def fetch_trade(self, trade_id: str) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT t.*, r.config_json FROM trades t "
            "JOIN runs r ON t.run_id = r.run_id "
            "WHERE t.trade_id = ?",
            [trade_id],
        ).fetchone()
        if row is None:
            return None
        cols = [d[0] for d in self._conn.description]
        return dict(zip(cols, row))

    # ── live_trades ───────────────────────────────────────────────────────────

    def insert_live_trade(self, trade: dict) -> str:
        """Insert a new live/paper trade. Returns the generated trade_id.

        Required keys: account_type ('LIVE'|'PAPER'), symbol, direction,
                       entry_time, entry_price, qty.
        All other keys are optional and default to NULL.
        """
        self._require_write()
        if trade.get("account_type") not in ("LIVE", "PAPER"):
            raise ValueError("account_type must be 'LIVE' or 'PAPER'")
        tid = trade.get("trade_id") or str(uuid.uuid4())
        self._conn.execute(
            """
            INSERT INTO live_trades (
                trade_id, account_type, account_id, symbol, direction,
                order_id, exit_order_id,
                entry_time, entry_price, qty,
                sl_price, tp_price, planned_rr,
                exit_time, exit_price,
                result, pnl_gross, commission, pnl_net, r_multiple,
                strategy, run_id, signal_params,
                notes, tags
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            [
                tid,
                trade["account_type"],
                trade.get("account_id"),
                trade["symbol"],
                trade["direction"],
                trade.get("order_id"),
                trade.get("exit_order_id"),
                trade.get("entry_time"),
                trade.get("entry_price"),
                trade.get("qty"),
                trade.get("sl_price"),
                trade.get("tp_price"),
                trade.get("planned_rr"),
                trade.get("exit_time"),
                trade.get("exit_price"),
                trade.get("result"),
                trade.get("pnl_gross"),
                trade.get("commission"),
                trade.get("pnl_net"),
                trade.get("r_multiple"),
                trade.get("strategy"),
                trade.get("run_id"),
                json.dumps(trade["signal_params"]) if trade.get("signal_params") else None,
                trade.get("notes"),
                trade.get("tags"),
            ],
        )
        return tid

    def update_live_trade(self, trade_id: str, updates: dict) -> None:
        """Patch specific fields on an existing live trade (e.g. fill in exit data)."""
        self._require_write()
        allowed = {
            "exit_order_id", "exit_time", "exit_price",
            "result", "pnl_gross", "commission", "pnl_net", "r_multiple",
            "sl_price", "tp_price", "order_id", "notes", "tags",
            "strategy", "run_id", "signal_params",
        }
        fields = {k: v for k, v in updates.items() if k in allowed}
        if not fields:
            return
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        set_clause += ", updated_at = now()"
        self._conn.execute(
            f"UPDATE live_trades SET {set_clause} WHERE trade_id = ?",
            list(fields.values()) + [trade_id],
        )

    def fetch_live_trade(self, trade_id: str) -> Optional[dict]:
        """Fetch a single live/paper trade by trade_id."""
        row = self._conn.execute(
            "SELECT * FROM live_trades WHERE trade_id = ?", [trade_id]
        ).fetchone()
        if row is None:
            return None
        cols = [d[0] for d in self._conn.description]
        return dict(zip(cols, row))

    def get_live_trades(
        self,
        account_type: Optional[str] = None,
        symbol: Optional[str] = None,
        result: Optional[str] = None,
        limit: int = 200,
    ) -> pd.DataFrame:
        """Query live/paper trades with optional filters.

        account_type: 'LIVE' | 'PAPER' | None (both)
        result:       'win' | 'loss' | 'open' | None (all)
        """
        clauses: list[str] = []
        params:  list      = []
        if account_type:
            clauses.append("account_type = ?")
            params.append(account_type)
        if symbol:
            clauses.append("symbol = ?")
            params.append(symbol)
        if result:
            clauses.append("result = ?")
            params.append(result)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        return self._conn.execute(
            f"SELECT * FROM live_trades {where} ORDER BY entry_time DESC LIMIT ?",
            params + [limit],
        ).df()

    def get_open_live_trades(self, account_type: Optional[str] = None) -> pd.DataFrame:
        """Return trades with result = 'open' (position still active)."""
        if account_type:
            return self._conn.execute(
                "SELECT * FROM live_trades WHERE result = 'open' AND account_type = ? "
                "ORDER BY entry_time DESC",
                [account_type],
            ).df()
        return self._conn.execute(
            "SELECT * FROM live_trades WHERE result = 'open' ORDER BY entry_time DESC"
        ).df()


# ── ReviewTradesDB ────────────────────────────────────────────────────────────

class ReviewTradesDB:
    """Separate DuckDB file for audit/review trades.

    Kept apart from backtest.duckdb so audit.py and trade_viewer can read/write
    even while a long grid run holds an exclusive lock on backtest.duckdb.
    """

    def __init__(
        self,
        db_path: str | pathlib.Path = _REVIEW_DB,
        read_only: bool = False,
    ) -> None:
        self.db_path  = pathlib.Path(db_path)
        self.read_only = read_only
        if not read_only:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = duckdb.connect(str(self.db_path))
            self._conn.execute(_REVIEW_DDL)
            for col_ddl in [
                "ALTER TABLE review_trades ADD COLUMN IF NOT EXISTS fvg_top    DOUBLE",
                "ALTER TABLE review_trades ADD COLUMN IF NOT EXISTS fvg_bottom DOUBLE",
            ]:
                try:
                    self._conn.execute(col_ddl)
                except Exception:
                    pass
        else:
            if not self.db_path.exists():
                raise FileNotFoundError(f"Review trades DB not found: {self.db_path}")
            self._conn = duckdb.connect(str(self.db_path), read_only=True)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "ReviewTradesDB":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def insert_trades(self, symbol: str, params_dict: dict, trades: list[Trade]) -> None:
        """Write audit-generated trades so trade_viewer can look them up by ID."""
        config_json = json.dumps(params_dict)
        self._conn.executemany(
            """
            INSERT INTO review_trades
                (trade_id, symbol, direction, entry_time, exit_time,
                 entry_price, exit_price, sl_price, tp_price,
                 result, r_multiple, planned_rr, fvg_top, fvg_bottom, config_json)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT (trade_id) DO UPDATE SET
                fvg_top    = EXCLUDED.fvg_top,
                fvg_bottom = EXCLUDED.fvg_bottom
            """,
            [
                (t.trade_id, symbol, t.direction,
                 str(t.entry_time), str(t.exit_time) if t.exit_time else None,
                 t.entry_price, t.exit_price,
                 t.sl, t.tp,
                 t.result, t.r_multiple, t.planned_rr,
                 t.fvg_top    if t.fvg_top    else None,
                 t.fvg_bottom if t.fvg_bottom else None,
                 config_json)
                for t in trades
            ],
        )

    def fetch_trade(self, trade_id: str) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT * FROM review_trades WHERE trade_id = ?", [trade_id]
        ).fetchone()
        if row is None:
            return None
        cols = [d[0] for d in self._conn.description]
        return dict(zip(cols, row))

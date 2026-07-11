"""db/sim_trades.duckdb schema and writer -- simulated trades from the K-line
Replay Trainer (analysis/replay_trainer.py).

Separate DuckDB file, same reasoning as backtest.db.ReviewTradesDB being split
from backtest.duckdb: a long-running grid backtest can hold backtest.duckdb
locked, and this file has nothing to do with that workflow anyway.
"""

from __future__ import annotations

import pathlib

import duckdb

_DEFAULT_DB = pathlib.Path(__file__).parent / "sim_trades.duckdb"

_DDL = """
CREATE TABLE IF NOT EXISTS sim_trades (
    trade_id              VARCHAR   PRIMARY KEY,
    symbol                VARCHAR   NOT NULL,
    direction              VARCHAR   NOT NULL,  -- "bull" | "bear"
    entry_time             VARCHAR   NOT NULL,
    exit_time              VARCHAR   NOT NULL,
    entry_price            DOUBLE    NOT NULL,
    exit_price              DOUBLE    NOT NULL,
    sl_price                 DOUBLE,             -- NULL for chandelier-mode trades
    tp_price                  DOUBLE,             -- NULL for chandelier-mode trades
    exit_cause                 VARCHAR   NOT NULL,  -- "sl" | "tp" | "chandelier" | "timeout"
    result                      VARCHAR   NOT NULL,  -- "win" | "loss"
    r_multiple                   DOUBLE    NOT NULL,
    shares                        INTEGER   NOT NULL,
    pnl_usd                        DOUBLE    NOT NULL,
    chandelier_period                INTEGER,          -- NULL for fixed SL/TP trades
    chandelier_multiplier             DOUBLE,           -- NULL for fixed SL/TP trades
    created_at                         TIMESTAMP DEFAULT now()
);
"""


class SimTradesDB:
    """DuckDB-backed store for settled Replay Trainer trades."""

    def __init__(self, db_path: str | pathlib.Path = _DEFAULT_DB, read_only: bool = False) -> None:
        self.db_path   = pathlib.Path(db_path)
        self.read_only = read_only
        if not read_only:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = duckdb.connect(str(self.db_path))
            self._conn.execute(_DDL)
        else:
            if not self.db_path.exists():
                raise FileNotFoundError(f"Sim trades DB not found: {self.db_path}")
            self._conn = duckdb.connect(str(self.db_path), read_only=True)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "SimTradesDB":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def insert_trade(self, trade: dict) -> None:
        """Insert one settled trade. `trade` keys must match the sim_trades
        columns (see _DDL) except `created_at`, which defaults to now()."""
        self._conn.execute(
            """
            INSERT INTO sim_trades
                (trade_id, symbol, direction, entry_time, exit_time,
                 entry_price, exit_price, sl_price, tp_price, exit_cause,
                 result, r_multiple, shares, pnl_usd,
                 chandelier_period, chandelier_multiplier)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT (trade_id) DO NOTHING
            """,
            [
                trade["trade_id"], trade["symbol"], trade["direction"],
                trade["entry_time"], trade["exit_time"],
                trade["entry_price"], trade["exit_price"],
                trade.get("sl_price"), trade.get("tp_price"), trade["exit_cause"],
                trade["result"], trade["r_multiple"], trade["shares"], trade["pnl_usd"],
                trade.get("chandelier_period"), trade.get("chandelier_multiplier"),
            ],
        )

    def fetch_recent(self, limit: int = 50) -> list[dict]:
        cur = self._conn.execute(
            "SELECT * FROM sim_trades ORDER BY created_at DESC LIMIT ?", [limit]
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def session_stats(self) -> dict:
        """Aggregate stats over ALL stored trades (not scoped to one process's
        session -- the DB itself IS the cross-session history)."""
        row = self._conn.execute(
            """
            SELECT
                COUNT(*)                                       AS n_trades,
                SUM(CASE WHEN result = 'win' THEN 1 ELSE 0 END) AS n_wins,
                COALESCE(SUM(r_multiple), 0.0)                  AS total_r,
                COALESCE(SUM(pnl_usd), 0.0)                     AS total_pnl_usd
            FROM sim_trades
            """
        ).fetchone()
        n_trades, n_wins, total_r, total_pnl_usd = row
        win_rate = (n_wins / n_trades) if n_trades else 0.0
        return {
            "n_trades": n_trades, "win_rate": win_rate,
            "total_r": total_r, "total_pnl_usd": total_pnl_usd,
        }

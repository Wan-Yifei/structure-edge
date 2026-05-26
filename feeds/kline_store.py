"""DuckDB-backed OHLCV cache for backtesting.

Stored at db/backtest_klines.duckdb — completely separate from the live
db/ticks.db to avoid any interference with the live orderflow chart.
"""

from __future__ import annotations

import pathlib
import duckdb
import pandas as pd

_DEFAULT_DB = pathlib.Path(__file__).parent.parent / "db" / "backtest_klines.duckdb"

_DDL = """
CREATE TABLE IF NOT EXISTS klines (
    code     TEXT   NOT NULL,
    ktype    TEXT   NOT NULL,
    time_key TEXT   NOT NULL,
    open     DOUBLE NOT NULL,
    high     DOUBLE NOT NULL,
    low      DOUBLE NOT NULL,
    close    DOUBLE NOT NULL,
    volume   BIGINT NOT NULL,
    PRIMARY KEY (code, ktype, time_key)
);
"""

_COLS = ["time_key", "open", "high", "low", "close", "volume"]


class KlineStore:
    def __init__(self, db_path: str | pathlib.Path = _DEFAULT_DB):
        """Open (or create) the kline cache DuckDB and initialise the schema."""
        self._path = pathlib.Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._con = duckdb.connect(str(self._path))
        self._con.execute(_DDL)

    def save(self, code: str, ktype: str, df: pd.DataFrame) -> int:
        """Insert-or-ignore OHLCV rows. Returns number of rows in df."""
        if df.empty:
            return 0
        insert_df = df[_COLS].copy()
        insert_df.insert(0, "ktype", ktype)
        insert_df.insert(0, "code", code)
        # DuckDB resolves 'insert_df' from the local Python scope
        self._con.execute(
            "INSERT OR IGNORE INTO klines SELECT * FROM insert_df"
        )
        return len(df)

    def load(
        self,
        code: str,
        ktype: str,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """Load klines sorted by time_key. start/end are 'YYYY-MM-DD' strings."""
        conds = ["code = ?", "ktype = ?"]
        params: list = [code, ktype]
        if start:
            conds.append("time_key >= ?")
            params.append(start)
        if end:
            conds.append("time_key <= ?")
            params.append(end + " 23:59:59")
        sql = (
            f"SELECT {', '.join(_COLS)} FROM klines "
            f"WHERE {' AND '.join(conds)} ORDER BY time_key"
        )
        return self._con.execute(sql, params).df()

    def has_data(self, code: str, ktype: str) -> bool:
        n = self._con.execute(
            "SELECT COUNT(*) FROM klines WHERE code = ? AND ktype = ?",
            [code, ktype],
        ).fetchone()[0]
        return n > 0

    def date_range(self, code: str, ktype: str) -> tuple[str, str] | None:
        """Return (min_time_key, max_time_key) or None if no data."""
        row = self._con.execute(
            "SELECT MIN(time_key), MAX(time_key) FROM klines WHERE code=? AND ktype=?",
            [code, ktype],
        ).fetchone()
        if row is None or row[0] is None:
            return None
        return row[0], row[1]

    def close(self):
        """Close the DuckDB connection."""
        self._con.close()

    def __enter__(self):
        """Return self for use as a context manager."""
        return self

    def __exit__(self, *_):
        """Close the connection on block exit."""
        self.close()

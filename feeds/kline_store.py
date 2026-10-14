"""DuckDB-backed OHLCV cache for backtesting.

Stored at db/backtest_klines.duckdb — completely separate from the live
db/ticks.db to avoid any interference with the live orderflow chart.
"""

from __future__ import annotations

import pathlib
from datetime import date

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
CREATE TABLE IF NOT EXISTS kline_coverage (
    code  TEXT NOT NULL,
    ktype TEXT NOT NULL,
    day   DATE NOT NULL,
    PRIMARY KEY (code, ktype, day)
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
        # DuckDB 1.5.2: aggregate functions (COUNT/MIN/MAX) on an empty filtered
        # result set throw an InternalException inside .execute() itself.
        # SELECT 1 ... LIMIT 1 avoids aggregates entirely — returns 0 rows (not NULL).
        df = self._con.execute(
            "SELECT 1 FROM klines WHERE code = ? AND ktype = ? LIMIT 1",
            [code, ktype],
        ).df()
        return not df.empty

    def date_range(self, code: str, ktype: str) -> tuple[str, str] | None:
        """Return (min_time_key, max_time_key) or None if no data."""
        # DuckDB 1.5.2: MIN/MAX on empty filtered result sets crash inside .execute().
        # Use ORDER BY + LIMIT 1 instead — returns 0 rows when no match, not NULL.
        min_df = self._con.execute(
            "SELECT time_key FROM klines WHERE code=? AND ktype=? ORDER BY time_key ASC LIMIT 1",
            [code, ktype],
        ).df()
        if min_df.empty:
            return None
        max_df = self._con.execute(
            "SELECT time_key FROM klines WHERE code=? AND ktype=? ORDER BY time_key DESC LIMIT 1",
            [code, ktype],
        ).df()
        return str(min_df.iloc[0]["time_key"]), str(max_df.iloc[0]["time_key"])


    # ── coverage ───────────────────────────────────────────────────────────────

    # Which days have been fetched, as opposed to which days have bars. The
    # two are not the same: a market holiday inside a range that was fetched
    # successfully has no bars and needs none, while a day that was never
    # requested also has no bars and does. Recording the request is the only
    # way to tell them apart, and it needs no holiday calendar -- if a fetch
    # completed over a span, every empty day inside it was a closed market.
    #
    # Before this existed, coverage was inferred from min/max of the bars
    # themselves, so a cache with holes in the middle reported a hit and
    # load() quietly returned short. Six weekdays went missing that way
    # between May and October 2026 and nothing ever refetched them: once the
    # endpoints are wide, every later request matches.

    def mark_covered(self, code: str, ktype: str, start: str, end: str,
                     today: "date | None" = None) -> int:
        """Record that [start, end] was fetched. Days are 'YYYY-MM-DD'.

        Days from `today` onward are NOT recorded, for the same reason
        missing_days() skips them: the session is still running, so the bars
        on disk are a prefix of the day rather than the day. Marking one
        froze it -- a fetch that ran at 12:02 on 2026-10-13 stored 152 of the
        session's 390 bars and recorded the day as covered, so every later
        call trusted the cache and the afternoon never arrived. missing_days
        already treats a today with bars as present, which is what keeps
        repeated intraday calls off the API without this.
        """
        today = today or date.today()
        days = [d.date() for d in pd.date_range(start[:10], end[:10], freq="D")
                if d.date() < today]
        if not days:
            return 0
        rows = pd.DataFrame({"code": code, "ktype": ktype, "day": days})
        self._con.execute(
            "INSERT OR IGNORE INTO kline_coverage SELECT * FROM rows")
        return len(rows)

    def clear_coverage(self, code: str, ktype: str,
                       start: str | None = None,
                       end: str | None = None) -> None:
        """Forget coverage so the range is fetched again."""
        conds, params = ["code = ?", "ktype = ?"], [code, ktype]
        if start:
            conds.append("day >= ?")
            params.append(start[:10])
        if end:
            conds.append("day <= ?")
            params.append(end[:10])
        self._con.execute(
            f"DELETE FROM kline_coverage WHERE {' AND '.join(conds)}", params)

    def covered_days(self, code: str, ktype: str) -> set:
        """Every day recorded as fetched for this series."""
        df = self._con.execute(
            "SELECT day FROM kline_coverage WHERE code = ? AND ktype = ?",
            [code, ktype],
        ).df()
        return set() if df.empty else {d.date() if hasattr(d, "date") else d
                                       for d in df["day"]}

    def days_with_bars(self, code: str, ktype: str) -> set:
        """Every day that actually holds at least one bar."""
        df = self._con.execute(
            "SELECT DISTINCT substr(time_key, 1, 10) d FROM klines "
            "WHERE code = ? AND ktype = ?",
            [code, ktype],
        ).df()
        return set() if df.empty else {date.fromisoformat(x) for x in df["d"]}

    def seed_coverage(self, code: str, ktype: str,
                      today: "date | None" = None) -> int:
        """Seed coverage for a cache that predates the table.

        Only days that hold bars are marked. A weekday with none stays
        unmarked and is refetched on the next request, which is how the
        existing holes repair themselves; a holiday is refetched once, comes
        back empty, and is then marked for good.

        Today is left out for the same reason mark_covered leaves it out: the
        bars on disk are a prefix of a running session, and seeding from them
        would freeze that prefix as the whole day. Without this, a cache
        seeded for the first time on a trading afternoon inherits exactly the
        truncated day the mark_covered guard was added to prevent.
        """
        if self.covered_days(code, ktype):
            return 0
        today = today or date.today()
        days = [d for d in sorted(self.days_with_bars(code, ktype))
                if d < today]
        if not days:
            return 0
        rows = pd.DataFrame({"code": code, "ktype": ktype, "day": days})
        self._con.execute(
            "INSERT OR IGNORE INTO kline_coverage SELECT * FROM rows")
        return len(rows)

    def missing_days(self, code: str, ktype: str, start: str, end: str,
                     today: "date | None" = None) -> list:
        """Weekdays in [start, end] that have never been fetched.

        Saturdays are skipped -- there is never a session. Sundays are not,
        because the week opens at 20:00 on Sunday and those bars are stored
        under the Sunday date.

        Days from `today` onward are left out: the session is still running,
        so nothing can be called complete. Today is treated as present if it
        already holds a bar, which keeps repeated intraday calls off the API
        without ever freezing a partial day into the coverage table.
        """
        today = today or date.today()
        self.seed_coverage(code, ktype, today=today)
        covered = self.covered_days(code, ktype)
        with_bars = None
        missing = []
        for ts in pd.date_range(start[:10], end[:10], freq="D"):
            d = ts.date()
            if ts.weekday() == 5 or d > today:
                continue
            if d in covered:
                continue
            if d == today:
                if with_bars is None:
                    with_bars = self.days_with_bars(code, ktype)
                if d in with_bars:
                    continue
            missing.append(d)
        return missing

    def close(self):
        """Close the DuckDB connection."""
        self._con.close()

    def __enter__(self):
        """Return self for use as a context manager."""
        return self

    def __exit__(self, *_):
        """Close the connection on block exit."""
        self.close()

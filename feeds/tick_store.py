"""SQLite-backed tick storage (WAL mode — concurrent reader + writer safe).

Schema:
    ticks(code, ts, price, volume, direction, mid_dir, ttype)

ts is stored as ISO-8601 string: '2026-05-18 09:30:00.123456'
Direction values: 'BUY' | 'SELL' | 'NEUTRAL'

mid_dir is a side inferred for prints the feed left NEUTRAL, by comparing the
trade price with the prevailing midpoint. NEUTRAL is the majority of the tape
-- 57.5% on SOXL -- and it is off-exchange flow rather than noise: 43.9% of
those prints carry sub-penny prices, which only wholesaler price improvement
produces. Signed this way it tracks the minute's return about as well as the
exchange's own tags (+0.253 against +0.276), so leaving it out costs most of
the signal. See doc/ORDER_FLOW_GUIDE.md appendix A.

Those sides live in db/tick_mid.db, not here -- see feeds/tick_mid_store.py for
why. The mid_dir column below is a leftover from the first attempt at keeping
them in this table and is never written; it could not be dropped because that
needs the write lock the collector holds.

ttype is moomoo's own ticker_type: AUTO_MATCH, ODD_LOT, OTC_SOLD,
DERIVATIVELY_PRICED, BULK, CROSS_MARKET and 26 others. The collector discarded
it for months, which is why the off-exchange work had to infer venue from
sub-penny prices instead of reading it. NULL on every row written before
2026-10-07.
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
-- WAL allows one writer at a time, and the tick collector holds that slot
-- whenever the market is open. Without a timeout any other writer -- the
-- offline classifier, say -- fails outright on the first contended statement
-- instead of waiting the few milliseconds the collector needs.
PRAGMA busy_timeout=30000;
CREATE TABLE IF NOT EXISTS ticks (
    code      TEXT    NOT NULL,
    ts        TEXT    NOT NULL,
    price     REAL    NOT NULL,
    volume    INTEGER NOT NULL,
    direction TEXT    NOT NULL,
    ttype     TEXT,
    -- The feed's own per-tick counter. NOT part of the UNIQUE key on
    -- purpose: including it would stop the de-duplication that protects
    -- against a reconnect replaying a window. Its job is diagnostic --
    -- a break in the numbering says the gap is upstream of this process,
    -- and the offered-minus-inserted count that insert_ticks now returns
    -- says how much the UNIQUE key itself dropped. Those two together are
    -- what separates "the feed did not send it" from "we threw it away",
    -- which is the question behind ticks.db holding only 55-96% of the
    -- session volume the 1-minute klines report.
    seq       INTEGER,
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
            self._con.execute("PRAGMA busy_timeout=30000")
        else:
            self._con = sqlite3.connect(str(self._path), check_same_thread=False)
            self._con.executescript(_SETUP_SQL)
            self._ensure_columns()

    def _ensure_columns(self) -> None:
        """Add columns to a database created before they existed.

        ALTER TABLE ADD COLUMN is metadata-only in SQLite, so this is instant
        even on the 48M-row table. It needs the write lock, which the collector
        holds all session -- hence the try: a reader-side failure here must not
        stop anything, the column simply stays absent until the collector next
        restarts and adds it itself.
        """
        try:
            cols = {r[1] for r in self._con.execute("PRAGMA table_info(ticks)")}
            for name, decl in (("ttype", "TEXT"), ("seq", "INTEGER"),
                               ("recv_ts", "REAL"), ("feed_ts", "REAL")):
                if name not in cols:
                    self._con.execute(f"ALTER TABLE ticks ADD COLUMN {name} {decl}")
            self._con.commit()
        except Exception as exc:
            print(f"[TickStore] could not add columns: {exc}", flush=True)

    # ── write ──────────────────────────────────────────────────────────────

    def insert_ticks(self, rows: Iterable[dict]) -> int:
        """Insert tick dicts, skipping duplicates. Returns rows ACTUALLY written.

        It used to return len(data), so every caller counting inserts was
        really counting offers -- the collector's session_count among them.
        executemany's rowcount is the real figure under INSERT OR IGNORE
        (verified: five rows with two duplicates gives 3), and the gap
        between the two is the only direct measure of what the UNIQUE key
        discards.
        """
        data = [
            (
                r["code"],
                _ts_str(r["ts"]),
                float(r["price"]),
                int(r["volume"]),
                str(r["direction"]).upper(),
                (str(r["ttype"]) if r.get("ttype") else None),
                (int(r["seq"]) if r.get("seq") is not None else None),
                # Both discarded by the SDK before the collector ever saw
                # them: get_rt_ticker and TickerHandlerBase build their
                # DataFrame from a col_list that names neither.
                #
                # recv_ts is Ticker.recvTime, which the proto documents as
                # the time THIS machine received the push -- not a server
                # clock, despite the name, and not the one the order book's
                # svrRecvTime*Timestamp carries. The two were measured on
                # opposite sides of the local clock error: when Windows
                # stepped the clock on 2026-11-03 03:54, recv_ts moved with
                # it (+1.587) while the book's server stamp moved against it
                # (-1.563). Its use is measuring local receive lag, not
                # joining the two feeds.
                #
                # feed_ts is Ticker.timestamp, the trade's own time as a
                # double rather than the string in `ts` -- measured identical
                # to `ts` to the millisecond, so it carries no new
                # information and is kept only as a cross-check.
                (float(r["recv_ts"]) if r.get("recv_ts") else None),
                (float(r["feed_ts"]) if r.get("feed_ts") else None),
            )
            for r in rows
        ]
        if not data:
            return 0
        # Columns named, not positional: a bare VALUES (?,?,?,?,?) silently
        # became wrong the moment mid_dir was added.
        cur = self._con.executemany(
            "INSERT OR IGNORE INTO ticks "
            "(code, ts, price, volume, direction, ttype, seq, "
            "recv_ts, feed_ts) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", data
        )
        self._con.commit()
        return cur.rowcount

    # ── read ───────────────────────────────────────────────────────────────

    def query_ticks(self, code: str, start: datetime, end: datetime,
                    with_mid: bool = False) -> list[dict]:
        """Tick rows for *code* in [start, end).

        with_mid attaches "mid_dir", the side inferred for off-exchange prints
        from the prevailing midpoint. It never rewrites "direction": folding
        the two together measurably made delta track price *worse* (+0.276 to
        +0.225 against the minute's return, measured with a fresh book), and
        merging them also hides the one thing the off-exchange side is good
        for, which is disagreeing with the exchange side. Callers that want it
        keep it separate. See doc/ORDER_FLOW_GUIDE.md appendix A.
        """
        mids: dict = {}
        if with_mid:
            try:
                from feeds.tick_mid_store import TickMidStore
                # Beside this ticks.db, not the package default: the sides
                # describe the trades in *this* file, so a store pointed
                # elsewhere must not pick up the production side file.
                side = self._path.parent / "tick_mid.db"
                with TickMidStore(side, read_only=True) as ms:
                    mids = ms.load(code, start, end)
            except Exception as exc:
                # No side file yet, or it is being rebuilt. Fall back to the
                # feed's own tags rather than failing the whole load.
                print(f"[TickStore] mid-dir unavailable: {exc}", flush=True)

        cur = self._con.execute(
            "SELECT ts, price, volume, direction, ttype FROM ticks "
            "WHERE code = ? AND ts >= ? AND ts < ? ORDER BY ts",
            [code, _ts_str(start), _ts_str(end)],
        )
        out = []
        for ts_s, price, vol, d, ttype in cur.fetchall():
            # ts_raw is the string exactly as stored. Parsing and
            # re-serialising normalises it ('...:34.38' becomes
            # '...:34.380000'), so anything keyed on the timestamp -- the
            # side file above -- must use this, not str(ts).
            out.append({"ts": datetime.fromisoformat(ts_s), "ts_raw": ts_s,
                        "price": price, "volume": vol, "direction": d,
                        "ttype": ttype,
                        "mid_dir": mids.get((ts_s, price, vol)) if mids else None})
        return out

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

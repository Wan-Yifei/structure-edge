"""SQLite storage for the option tape and the greeks needed to sign it.

Why a separate file from db/ticks.db: that one is ~16 GB and the equity tick
collector holds its single writer slot for the whole session. A second
long-lived writer on the same file would queue behind it on every batch.

Three tables, because the data arrives at three completely different rates:

    option_contracts   once per contract per session -- what the code means
    option_ticks       the tape, every print
    option_greeks      sampled on a timer, written ONLY when something moved

The point of collecting all this is a signed delta flow: for each print,
side x size x delta-at-that-moment. The delta has to be the one in force
when the trade happened, not the one at the close -- an ATM contract a few
days from expiry carries gamma near 0.4, so a 1% move in a $31 underlying
shifts its delta by about 0.12. Multiplying the morning's prints by the
closing delta is not an approximation, it is a different number.

Hence option_greeks, and hence `spot` and `upd` on every row:

    spot  the underlying at OUR sample time, so a delta can be re-derived or
          checked later rather than trusted
    upd   the broker's own update_time for that contract. A single snapshot
          of 326 SOXL contracts came back carrying 143 distinct update_time
          values spanning three days -- a quiet strike's greeks can be hours
          stale while a busy one is current. Without this column there is no
          way to tell "unchanged" from "never refreshed".
"""

from __future__ import annotations

import pathlib
import sqlite3
from datetime import datetime, timedelta
from typing import Iterable

_DEFAULT_DB = pathlib.Path(__file__).parent.parent / "db" / "option_ticks.db"

_SETUP_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA busy_timeout=30000;

CREATE TABLE IF NOT EXISTS option_contracts (
    code       TEXT PRIMARY KEY,
    underlying TEXT NOT NULL,
    expiry     TEXT NOT NULL,          -- YYYY-MM-DD
    strike     REAL NOT NULL,
    opt_type   TEXT NOT NULL,          -- CALL | PUT
    first_seen TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_oc_underlying
    ON option_contracts (underlying, expiry);

CREATE TABLE IF NOT EXISTS option_ticks (
    code      TEXT    NOT NULL,
    ts        TEXT    NOT NULL,        -- exchange time, as the feed states it
    price     REAL    NOT NULL,
    volume    INTEGER NOT NULL,
    direction TEXT    NOT NULL,        -- BUY | SELL | NEUTRAL | N/A
                                       -- N/A is TickerDirect.NONE: field absent, non-int,
                                       -- or the feed's own unknown. Rows written before
                                       -- 2026-11-23 have N/A folded into NEUTRAL.
    ttype     TEXT,                    -- moomoo ticker_type: AUTO_MATCH, ...
    seq       INTEGER NOT NULL,        -- feed counter, -1 when absent
    recv_ts   REAL,
    feed_ts   REAL,
    -- seq IS in the key, unlike the equity ticks table.
    --
    -- Without it, a split fill is silently eaten: one order filling as
    -- 21 + 21 at the same millisecond and price arrives as two prints that
    -- differ ONLY by seq. Measured on three real SOXL option tapes from
    -- 2026-11-20: 8-15% of prints and 4-6% of contracts dropped, and the
    -- error does NOT cancel -- net signed contracts came out +4.7%, +1.8%
    -- and -4.4% on the three. That is a direct, permanent corruption of
    -- the exact quantity this table exists to total up.
    --
    -- The reason seq was left out -- that a reconnect replaying a window
    -- must de-duplicate -- does not require it: seq was verified unique
    -- and monotonic within each contract (1000/1000), and two independent
    -- fetches of the same window returned byte-identical (time, sequence)
    -- pairs. Replay still collapses.
    --
    -- -1 rather than NULL when the feed omits it: NULLs never collide in a
    -- SQLite UNIQUE index, so a NULL seq would disable de-duplication
    -- entirely for those rows. The sentinel degrades to the old key
    -- instead, which is the right fallback when there is nothing better.
    UNIQUE(code, ts, price, volume, seq)
);
CREATE INDEX IF NOT EXISTS idx_ot_code_ts ON option_ticks (code, ts);
CREATE INDEX IF NOT EXISTS idx_ot_ts      ON option_ticks (ts);

CREATE TABLE IF NOT EXISTS option_greeks (
    code  TEXT NOT NULL,
    ts    TEXT NOT NULL,               -- OUR sample time, not the feed's
    delta REAL,
    gamma REAL,
    iv    REAL,
    oi    INTEGER,
    spot  REAL,                        -- underlying at sample time
    upd   TEXT,                        -- the feed's own update_time
    -- 1 when the broker returned no greeks at all for this contract and
    -- the columns above are NULL rather than measured.
    --
    -- The feed zero-FILLS: 19 of 326 SOXL contracts on the 2026-11-27 chain
    -- came back delta = gamma = iv = 0 with update_time frozen at 09:30:00,
    -- and three of those traded that day -- a $70 put with 213 contracts
    -- done and a true delta near -0.97, reported as 0.0. Stored as a plain
    -- zero, the indicator multiplies real prints by nothing and loses them
    -- silently. A real zero and a missing value have to be different rows.
    nogreeks INTEGER NOT NULL DEFAULT 0,
    UNIQUE(code, ts)
);
CREATE INDEX IF NOT EXISTS idx_og_code_ts ON option_greeks (code, ts);

-- What fraction of each contract's day we actually caught.
--
-- The option tape cannot be re-fetched, so "did we miss anything?" has to
-- be answerable while the answer still matters. Comparing our own summed
-- volume against the broker's end-of-day figure for the same contract is
-- the only check available, and it has to be recorded rather than logged:
-- an indicator built later from a session that was only 60% collected
-- needs to be able to find that out.
CREATE TABLE IF NOT EXISTS option_coverage (
    session    TEXT    NOT NULL,       -- YYYY-MM-DD, the trading day
    code       TEXT    NOT NULL,
    feed_vol   INTEGER,                -- broker's daily volume for the contract
    our_vol    INTEGER NOT NULL,       -- what we stored
    our_ticks  INTEGER NOT NULL,
    checked_at TEXT    NOT NULL,
    UNIQUE(session, code)
);
CREATE INDEX IF NOT EXISTS idx_ocov_session ON option_coverage (session);

-- Which contracts were actually subscribed on a given day.
--
-- Reconciliation needs this, and option_contracts cannot provide it: that
-- table is cumulative and never pruned, so by the end of the first week it
-- holds every contract ever selected across rolling expiries. Reconciling
-- against all of them compares today's tape with contracts today's run
-- never subscribed to -- the broker reports volume for them, we stored
-- none, and every one lands in `caught_nothing`.
--
-- caught_nothing is the only detector for a dropped subscription. A few
-- hundred false positives a day and nobody reads it again, which costs
-- more than the bug it was built to find.
CREATE TABLE IF NOT EXISTS option_subscriptions (
    session    TEXT NOT NULL,
    code       TEXT NOT NULL,
    subscribed TEXT NOT NULL,
    UNIQUE(session, code)
);
CREATE INDEX IF NOT EXISTS idx_osub_session ON option_subscriptions (session);
"""


def _seq_or_sentinel(seq) -> int:
    """The feed's counter, or -1 when it did not send one.

    NULLs never collide in a SQLite UNIQUE index, so storing NULL here
    would silently turn de-duplication OFF for those rows -- a reconnect
    replaying a window would double-count every one. -1 instead makes the
    key degrade to (code, ts, price, volume), the old behaviour, which is
    the right fallback when there is no better discriminator. Verified
    present on every historical print pulled; unverified on live pushes,
    which is why the fallback exists at all.
    """
    try:
        return int(seq)
    except (TypeError, ValueError):
        return -1


def _ts_str(ts) -> str:
    if isinstance(ts, datetime):
        return ts.isoformat(sep=" ")
    return str(ts)


class OptionTickStore:
    def __init__(self, db_path: str | pathlib.Path = _DEFAULT_DB,
                 read_only: bool = False):
        self._path = pathlib.Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._con = sqlite3.connect(str(self._path), check_same_thread=False)
        if read_only:
            self._con.execute("PRAGMA busy_timeout=30000")
        else:
            self._con.executescript(_SETUP_SQL)

    # ── write ──────────────────────────────────────────────────────────────

    def upsert_contracts(self, rows: Iterable[dict]) -> int:
        """Record what each subscribed contract code means. Idempotent."""
        data = [(r["code"], r["underlying"], r["expiry"], float(r["strike"]),
                 r["opt_type"], _ts_str(r.get("first_seen") or datetime.now()))
                for r in rows]
        if not data:
            return 0
        cur = self._con.executemany(
            "INSERT OR IGNORE INTO option_contracts "
            "(code, underlying, expiry, strike, opt_type, first_seen) "
            "VALUES (?,?,?,?,?,?)", data)
        self._con.commit()
        return cur.rowcount

    def insert_ticks(self, rows: Iterable[dict]) -> int:
        """Insert prints, skipping duplicates. Returns rows ACTUALLY written.

        Written, not offered: the difference is what the UNIQUE key dropped,
        and a caller that reports offers as inserts cannot tell a quiet tape
        from a replayed one.
        """
        data = [(r["code"], _ts_str(r["ts"]), float(r["price"]),
                 int(r["volume"]), r["direction"], r.get("ttype"),
                 _seq_or_sentinel(r.get("seq")),
                 r.get("recv_ts"), r.get("feed_ts"))
                for r in rows]
        if not data:
            return 0
        cur = self._con.executemany(
            "INSERT OR IGNORE INTO option_ticks "
            "(code, ts, price, volume, direction, ttype, seq, recv_ts, feed_ts) "
            "VALUES (?,?,?,?,?,?,?,?,?)", data)
        self._con.commit()
        return cur.rowcount

    def insert_greeks(self, rows: Iterable[dict]) -> int:
        data = [(r["code"], _ts_str(r["ts"]), r.get("delta"), r.get("gamma"),
                 r.get("iv"), r.get("oi"), r.get("spot"), r.get("upd"),
                 1 if r.get("nogreeks") else 0)
                for r in rows]
        if not data:
            return 0
        cur = self._con.executemany(
            "INSERT OR IGNORE INTO option_greeks "
            "(code, ts, delta, gamma, iv, oi, spot, upd, nogreeks) "
            "VALUES (?,?,?,?,?,?,?,?,?)", data)
        self._con.commit()
        return cur.rowcount

    def record_subscriptions(self, session: str, codes: Iterable[str]) -> int:
        """Note which contracts were subscribed for this trading day."""
        now = _ts_str(datetime.now())
        data = [(session, c, now) for c in codes]
        if not data:
            return 0
        cur = self._con.executemany(
            "INSERT OR IGNORE INTO option_subscriptions (session, code, subscribed) "
            "VALUES (?,?,?)", data)
        self._con.commit()
        return cur.rowcount

    def subscribed_on(self, session: str) -> list[str]:
        """Codes subscribed that day, for reconciliation to score against."""
        return [r[0] for r in self._con.execute(
            "SELECT code FROM option_subscriptions WHERE session = ? ORDER BY code",
            (session,))]

    def our_volume_for(self, session: str, codes: list[str]) -> dict[str, tuple[int, int]]:
        """{code: (contracts, prints)} stored for that trading day.

        Grouped on the calendar date of the exchange timestamp. For options
        that is the same grouping as the 20:00-to-20:00 trading day the rest
        of the project uses, because no option prints fall anywhere near
        20:00 -- the tape runs 09:30 to about 16:15.
        """
        out: dict[str, tuple[int, int]] = {}
        for i in range(0, len(codes), 500):
            chunk = codes[i:i + 500]
            q = ("SELECT code, SUM(volume), COUNT(*) FROM option_ticks "
                 f"WHERE substr(ts,1,10) = ? AND code IN ({','.join('?' * len(chunk))}) "
                 "GROUP BY code")
            for c, v, n in self._con.execute(q, [session] + chunk):
                out[c] = (int(v or 0), int(n))
        return out

    def record_coverage(self, session: str, rows: Iterable[dict]) -> int:
        data = [(session, r["code"], r.get("feed_vol"), int(r["our_vol"]),
                 int(r["our_ticks"]), _ts_str(datetime.now()))
                for r in rows]
        if not data:
            return 0
        cur = self._con.executemany(
            "INSERT OR REPLACE INTO option_coverage "
            "(session, code, feed_vol, our_vol, our_ticks, checked_at) "
            "VALUES (?,?,?,?,?,?)", data)
        self._con.commit()
        return cur.rowcount

    def coverage_summary(self, session: str) -> dict:
        row = self._con.execute(
            "SELECT COUNT(*), SUM(feed_vol), SUM(our_vol), SUM(our_ticks), "
            "       SUM(CASE WHEN feed_vol IS NULL THEN 1 ELSE 0 END) "
            "FROM option_coverage WHERE session = ?", (session,)).fetchone()
        n, feed, ours, ticks, unknown = row
        return {"contracts": n or 0, "feed_vol": feed or 0, "our_vol": ours or 0,
                "our_ticks": ticks or 0, "feed_vol_unknown": unknown or 0,
                "ratio": (ours / feed) if feed else None}

    # ── housekeeping ───────────────────────────────────────────────────────

    def prune_before(self, days: int) -> tuple[int, int]:
        """Drop ticks and greeks older than *days*. Returns (ticks, greeks).

        Contracts are left alone: the table is tiny and a row explains codes
        that still appear in older rows elsewhere.
        """
        cutoff = (datetime.now() - timedelta(days=days)).isoformat(sep=" ")
        t = self._con.execute(
            "DELETE FROM option_ticks WHERE ts < ?", (cutoff,)).rowcount
        g = self._con.execute(
            "DELETE FROM option_greeks WHERE ts < ?", (cutoff,)).rowcount
        self._con.commit()
        return t, g

    def checkpoint(self) -> None:
        try:
            self._con.execute("PRAGMA wal_checkpoint(PASSIVE)")
        except sqlite3.Error:
            pass

    # ── read ───────────────────────────────────────────────────────────────

    def session_counts(self, since: str) -> tuple[int, int]:
        t = self._con.execute(
            "SELECT COUNT(*) FROM option_ticks WHERE ts >= ?", (since,)).fetchone()[0]
        g = self._con.execute(
            "SELECT COUNT(*) FROM option_greeks WHERE ts >= ?", (since,)).fetchone()[0]
        return t, g

    def contracts_for(self, underlying: str, expiry: str | None = None) -> list[dict]:
        sql = "SELECT code, underlying, expiry, strike, opt_type FROM option_contracts WHERE underlying = ?"
        args: list = [underlying]
        if expiry:
            sql += " AND expiry = ?"
            args.append(expiry)
        cols = ("code", "underlying", "expiry", "strike", "opt_type")
        return [dict(zip(cols, r)) for r in self._con.execute(sql, args)]

    def close(self) -> None:
        try:
            self._con.close()
        except sqlite3.Error:
            pass

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
import functools
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Iterable

_DEFAULT_DB = pathlib.Path(__file__).parent.parent / "db" / "option_ticks.db"

#: Key under which underlyings_on reports subscriptions that join to no
#: option_contracts row. Named rather than "", which every caller then has
#: to remember is not a symbol.
ORPHAN_KEY = "<no contract row>"

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
    -- Our first stored print for this contract that session. feed_vol is
    -- the WHOLE session's volume, so without this the ratio cannot be told
    -- apart from "we were late". On 2026-11-23 it was both: the day read
    -- 279,295/355,460 = 78.6% over all 261 contracts, or 279,295/334,294 =
    -- 83.5% over the 198 we captured anything from, and the collector went
    -- live late besides. This column is NULL for that session -- it was
    -- added afterwards -- so the lateness is not recoverable from here.
    first_ts   TEXT,
    UNIQUE(session, code)
);
CREATE INDEX IF NOT EXISTS idx_ocov_session ON option_coverage (session);
-- Added after the first live session; existing rows get NULL.
-- sqlite has no IF NOT EXISTS for columns, so the collector
-- calls _ensure_columns() rather than failing on a second run.

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

-- Every contract the selection CONSIDERED, not just the ones it took.
--
-- The subscribed set answers "did we store what we subscribed to". It
-- cannot answer "was the selection any good", because a contract the
-- ranking passed over leaves no trace anywhere: it is not subscribed, so
-- it has no ticks, no greeks and no coverage row, and the reconcile's
-- denominator is the subscribed set -- so a worse selection reads as
-- BETTER coverage. On 2026-11-24 that gap was 62.4% of the candidate
-- band's volume reported as 100.0%.
--
-- The scores cannot be reconstructed afterwards either. They come from a
-- chain snapshot taken at 09:15 and the chain is not re-fetchable, so a
-- selection not written down here is a selection nobody can ever grade.
--
-- rank is 1-based over the scored candidates, descending by score;
-- `picked` is whether it made the quota cut. One row per contract per
-- session, around 800 for SOXL and 3,200 for QQQ at the configured
-- pre-filter widths.
CREATE TABLE IF NOT EXISTS option_candidates (
    session    TEXT    NOT NULL,
    underlying TEXT    NOT NULL,
    code       TEXT    NOT NULL,
    rank       INTEGER NOT NULL,
    score      REAL,
    expiry     TEXT,
    strike     REAL,
    opt_type   TEXT,
    picked     INTEGER NOT NULL,
    recorded   TEXT    NOT NULL,
    UNIQUE(session, code)
);
CREATE INDEX IF NOT EXISTS idx_ocand_session
    ON option_candidates (session, underlying);

-- What each candidate actually did, read once after the close.
--
-- Its partner is option_candidates: that table is the INTENT (the score, the
-- rank, whether it made the cut, written at 09:15), this one is the OUTCOME
-- (the volume it really traded, read at 16:50). Join them on (session, code)
-- and every session yields a complete "score against realised volume"
-- dataset -- which is what lets a different ranking be graded offline,
-- without touching a live selection or spending one extra API call. The
-- reconcile's band pass already snapshots every candidate; before this table
-- it kept the total and threw the per-contract figures away.
--
-- Deliberately NOT columns on option_candidates. That row is written with
-- INSERT OR REPLACE at selection time, so a collector re-run after the close
-- -- which happens, 2026-11-23 re-picked mid-session -- would blank these.
-- Two writers, two times, two meanings.
--
-- feed_vol NULL means UNKNOWN, 0 means it did not trade. The snapshot can
-- fail, and a figure stamped another day is dropped by the same wrong-day
-- rule the reconcile uses, so the two have to stay distinguishable: folding
-- them together turns "we could not read it" into "nothing happened there".
--
-- delta_close and oi_close are read at the same moment, from the same
-- response, and are named for it: the indicator weights a print by the delta
-- at the TIME OF THE PRINT, and the close is only a proxy for that. Good
-- enough to RANK contracts against each other, not to state what share of
-- the signal was captured.
--
-- SCOPE, which a later reader will otherwise over-claim: the candidate band
-- is already cut twice before it reaches here -- by the +/-pct pre-filter
-- and by _MAX_CHAINS, which truncates the expiry list before the candidates
-- are built at all. So this grades RANKINGS within that band. It cannot
-- grade the band width, and it cannot grade the expiry cap.
CREATE TABLE IF NOT EXISTS option_band_volume (
    session    TEXT NOT NULL,
    code       TEXT NOT NULL,
    feed_vol   INTEGER,
    delta_close REAL,
    oi_close   INTEGER,
    recorded   TEXT NOT NULL,
    UNIQUE(session, code)
);
CREATE INDEX IF NOT EXISTS idx_obvol_session ON option_band_volume (session);

-- How many contracts OpenD actually listed as subscribed, against how many
-- were asked for. One row per underlying per session.
--
-- Written down rather than left in the log because the next morning's
-- parameter advice needs it, and because nothing else in the database can
-- answer it. The obvious proxies do not work: option_greeks has a row for
-- every REQUESTED code whether or not the subscription landed, since
-- sample_greeks reads a market snapshot and a snapshot needs no
-- subscription at all (886 never-subscribed contracts were snapshotted on
-- 2026-11-24 to prove it). Counting greeks therefore always equals the
-- request and measures nothing -- it is the object under test answering for
-- itself.
--
-- The distinction it buys: `asked 60, listed 48` is an account-wide quota
-- that was over-committed, and the fix is to lower a per-code share.
-- `asked 60, listed 60` with contracts in caught_nothing is a dropped
-- subscription, and lowering the quota would do nothing at all. Without
-- this row the two look identical from the database.
--
-- listed NULL means the subscription read itself failed, which is not the
-- same as zero contracts listed.
CREATE TABLE IF NOT EXISTS option_subscription_check (
    session    TEXT    NOT NULL,
    underlying TEXT    NOT NULL,
    asked      INTEGER NOT NULL,
    listed     INTEGER,
    recorded   TEXT    NOT NULL,
    UNIQUE(session, underlying)
);
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


def _locked(fn):
    """Hold the store's lock for the whole method body.

    Not per statement: the unit that must not interleave is
    `executemany(...)` followed by its `commit()`, and locking each call
    separately leaves exactly the window that produced two
    "cannot commit - no transaction is active" errors on 2026-11-23.
    """
    @functools.wraps(fn)
    def inner(self, *args, **kwargs):
        with self._lock:
            return fn(self, *args, **kwargs)
    return inner


class OptionTickStore:
    def __init__(self, db_path: str | pathlib.Path = _DEFAULT_DB,
                 read_only: bool = False):
        self._path = pathlib.Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False is what makes the lock necessary, not
        # what makes it safe: it only silences sqlite3's own guard.
        self._lock = threading.RLock()
        self._con = sqlite3.connect(str(self._path), check_same_thread=False)
        if read_only:
            self._con.execute("PRAGMA busy_timeout=30000")
        else:
            self._con.executescript(_SETUP_SQL)
            # CREATE TABLE IF NOT EXISTS does nothing to a table that
            # already exists, so a column added after the first live file
            # never reaches the database it was added to describe. Run on
            # every writable open rather than from one call site -- there
            # are three, and the one that matters most (--reconcile-only)
            # is the one most likely to be forgotten.
            self._ensure_columns()

    # ── write ──────────────────────────────────────────────────────────────

    @_locked
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

    @_locked
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

    @_locked
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

    @_locked
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

    @_locked
    def record_candidates(self, session: str, underlying: str,
                          rows: Iterable[dict]) -> int:
        """Write the whole scored candidate list for one day's selection.

        INSERT OR REPLACE, not OR IGNORE: a collector restarted mid-session
        re-picks, and the second selection is the one that produced the
        session's data. OR IGNORE would keep the first list and quietly
        describe a selection that was replaced -- which is what happened on
        2026-11-23, where 09:15 and 09:39 each picked a set.

        Each row needs code/rank/picked; score, expiry, strike and opt_type
        are recorded when present.
        """
        now = _ts_str(datetime.now())
        data = [(session, underlying, r["code"], int(r["rank"]),
                 (None if r.get("score") is None else float(r["score"])),
                 r.get("expiry"),
                 (None if r.get("strike") is None else float(r["strike"])),
                 r.get("opt_type"), int(bool(r.get("picked"))), now)
                for r in rows]
        if not data:
            return 0
        cur = self._con.executemany(
            "INSERT OR REPLACE INTO option_candidates "
            "(session, underlying, code, rank, score, expiry, strike, "
            " opt_type, picked, recorded) VALUES (?,?,?,?,?,?,?,?,?,?)", data)
        self._con.commit()
        return cur.rowcount

    @_locked
    def record_band_volume(self, session: str, rows: Iterable[dict]) -> int:
        """Write what each candidate actually traded. See option_band_volume.

        `feed_vol` absent or None is stored as NULL and means unknown; 0 is
        stored as 0 and means it did not trade. The caller must not
        substitute one for the other.
        """
        now = _ts_str(datetime.now())
        data = []
        for r in rows:
            vol = r.get("feed_vol")
            delta = r.get("delta_close")
            oi = r.get("oi_close")
            data.append((
                session, r["code"],
                (None if vol is None else int(vol)),
                (None if delta is None else float(delta)),
                (None if oi is None else int(oi)),
                now))
        if not data:
            return 0
        cur = self._con.executemany(
            "INSERT OR REPLACE INTO option_band_volume "
            "(session, code, feed_vol, delta_close, oi_close, recorded) "
            "VALUES (?,?,?,?,?,?)", data)
        self._con.commit()
        return cur.rowcount

    @_locked
    def record_subscription_check(self, session: str, underlying: str,
                                  asked: int, listed: int | None) -> None:
        """Note what was asked for and what OpenD listed. See the table.

        `listed` None is stored as NULL and means the read failed; 0 means
        the read succeeded and found nothing. Do not substitute one for the
        other -- an empty subscription list is the loudest possible signal
        and a failed read is no signal at all.
        """
        self._con.execute(
            "INSERT OR REPLACE INTO option_subscription_check "
            "(session, underlying, asked, listed, recorded) VALUES (?,?,?,?,?)",
            (session, underlying, int(asked),
             (None if listed is None else int(listed)), _ts_str(datetime.now())))
        self._con.commit()

    @_locked
    def subscription_check_on(self, session: str,
                              underlying: str) -> dict | None:
        """That session's asked/listed pair, or None if it was never written
        (a session collected before this existed, or one that never got
        past selection)."""
        row = self._con.execute(
            "SELECT asked, listed FROM option_subscription_check "
            "WHERE session = ? AND underlying = ?",
            (session, underlying)).fetchone()
        if row is None:
            return None
        return {"asked": row[0], "listed": row[1]}

    @_locked
    def band_volume_on(self, session: str) -> dict[str, dict]:
        """That session's realised band volumes, by code."""
        cur = self._con.execute(
            "SELECT code, feed_vol, delta_close, oi_close "
            "FROM option_band_volume WHERE session = ?", (session,))
        return {r[0]: {"feed_vol": r[1], "delta_close": r[2],
                       "oi_close": r[3]} for r in cur.fetchall()}

    @_locked
    def candidates_on(self, session: str,
                      underlying: str) -> list[dict]:
        """That day's scored candidate list, best-ranked first.

        `underlying` is REQUIRED for the same reason subscribed_on requires
        it: two underlyings share this table, and a reconcile that scored a
        band against both symbols' candidates would be comparing nothing in
        particular.
        """
        cur = self._con.execute(
            "SELECT code, rank, score, expiry, strike, opt_type, picked "
            "FROM option_candidates WHERE session = ? AND underlying = ? "
            "ORDER BY rank", (session, underlying))
        return [{"code": r[0], "rank": r[1], "score": r[2], "expiry": r[3],
                 "strike": r[4], "opt_type": r[5], "picked": bool(r[6])}
                for r in cur.fetchall()]

    @_locked
    def subscribed_on(self, session: str, underlying: str) -> list[str]:
        """Codes subscribed that day, for reconciliation to score against.

        `underlying` is REQUIRED. Leaving it optional would have kept
        exactly the hole this fixes: the collector's bug was a missing
        argument, and a default would hand the same trap to the next
        caller. Use `all_subscribed_on` to ask about a whole day on purpose.

        Without the filter, the answer is every contract from every symbol
        collected that day, so a reconcile run under one symbol's name
        scores another's contracts -- all of which read as uncollected and
        fill `caught_nothing`, the only POST-HOC detector for a dropped
        subscription. (log_telemetry compares OpenD's own subscription list
        against the requested codes while the session runs, so it is not
        the only detector; it is the only one that works after the fact.)

        The underlying comes from option_contracts rather than from a column
        here, and the join is total rather than lucky: upsert_contracts runs
        on the same list immediately before record_subscriptions, so a
        subscribed code without a contract row cannot be produced. A code
        belongs to one underlying for life, and option_contracts is INSERT
        OR IGNORE on the code, so the value never changes under a session.
        """
        return [r[0] for r in self._con.execute(
            "SELECT s.code FROM option_subscriptions s "
            "JOIN option_contracts c ON c.code = s.code "
            "WHERE s.session = ? AND c.underlying = ? ORDER BY s.code",
            (session, underlying))]

    @_locked
    def all_subscribed_on(self, session: str) -> list[str]:
        """Every contract subscribed that day, whatever the symbol.

        Separate from `subscribed_on` so that asking for a whole day is a
        deliberate act with its own name, rather than something you get by
        forgetting an argument.
        """
        return [r[0] for r in self._con.execute(
            "SELECT code FROM option_subscriptions WHERE session = ? "
            "ORDER BY code", (session,))]

    @_locked
    def underlyings_on(self, session: str) -> dict[str, int]:
        """Which symbols were collected that day, and how many contracts each.

        A subscription with no contract row is counted separately under
        ORPHAN_KEY, because subscribed_on's INNER JOIN drops it and "fewer
        rows" is the same shape as "this symbol was quiet".
        """
        out = {r[0]: r[1] for r in self._con.execute(
            "SELECT c.underlying, COUNT(*) FROM option_subscriptions s "
            "JOIN option_contracts c ON c.code = s.code "
            "WHERE s.session = ? GROUP BY c.underlying", (session,))}
        if (n := self.orphans_on(session)):
            # A named key, not "". An empty string reads as a symbol whose
            # name happens to be blank: every caller then has to remember to
            # skip it, and one of this class's own tests had to write
            # `if k` to get a sane total.
            out[ORPHAN_KEY] = n
        return out

    @_locked
    def orphans_on(self, session: str) -> int:
        """Subscriptions that day with no option_contracts row.

        Its own method because the reconcile path has to be able to ask
        WITHOUT going through the empty-set branch. subscribed_on uses an
        INNER JOIN, so an orphan is dropped from the answer, and "fifty
        fewer codes" has the same shape as "this symbol was quiet".

        The collector cannot produce one -- upsert_contracts runs on the
        same list immediately before record_subscriptions -- but that is a
        convention held up by a single call site, not a foreign key.
        """
        row = self._con.execute(
            "SELECT COUNT(*) FROM option_subscriptions s "
            "LEFT JOIN option_contracts c ON c.code = s.code "
            "WHERE s.session = ? AND c.code IS NULL", (session,)).fetchone()
        return int(row[0]) if row else 0

    @_locked
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

    @_locked
    def _ensure_columns(self) -> None:
        """Add columns that post-date an existing database file.

        _SETUP_SQL only runs CREATE TABLE IF NOT EXISTS, so a column added
        later never reaches a database that already exists -- and this one
        does exist, on the machine that produced the session the column is
        there to describe.
        """
        have = {r[1] for r in self._con.execute(
            "PRAGMA table_info(option_coverage)")}
        if "first_ts" not in have:
            self._con.execute(
                "ALTER TABLE option_coverage ADD COLUMN first_ts TEXT")
            self._con.commit()

    @_locked
    def first_ts_for(self, session: str, codes: list[str]) -> dict[str, str]:
        """Earliest stored print per contract for one session.

        Reconcile divides by the feed's WHOLE-SESSION volume, so without
        this a late start cannot be told from data loss.

        Both published figures for 2026-11-23 are the same numerator over
        different denominators, and neither is the "correct" one:

            279,295 / 355,460 = 78.6%   all 261 subscribed contracts
            279,295 / 334,294 = 83.5%   the 198 we captured anything from

        The 21,166-share gap is exactly the feed volume of the 61 contracts
        we caught nothing from. Quote whichever, with its denominator.

        What CANNOT be read off option_coverage for that session is when the
        tape began: first_ts is NULL on all 261 rows, because the column
        postdates the session it was added to describe. The 09:39:29 figure
        comes from option_ticks, where it is also approximate -- MIN(ts) is
        09:30:05.962 and the prints before 09:39:29 are 141 of them for 570
        contracts, 0.2% of the day.
        """
        found: dict[str, str] = {}
        for i in range(0, len(codes), 500):
            chunk = codes[i:i + 500]
            marks = ",".join("?" * len(chunk))
            rows = self._con.execute(
                "SELECT code, MIN(ts) FROM option_ticks "
                f"WHERE substr(ts,1,10)=? AND code IN ({marks}) GROUP BY code",
                (session, *chunk)).fetchall()
            for code, ts in rows:
                if ts:
                    found[code] = ts
        return found

    @_locked
    def record_coverage(self, session: str, rows: Iterable[dict]) -> int:
        now = _ts_str(datetime.now())
        data = [(session, r["code"], r.get("feed_vol"), int(r["our_vol"]),
                 int(r["our_ticks"]), now, r.get("first_ts"))
                for r in rows]
        if not data:
            return 0
        cur = self._con.executemany(
            "INSERT OR REPLACE INTO option_coverage "
            "(session, code, feed_vol, our_vol, our_ticks, checked_at, first_ts) "
            "VALUES (?,?,?,?,?,?,?)", data)
        self._con.commit()
        return cur.rowcount

    @_locked
    def coverage_summary(self, session: str, underlying: str | None) -> dict:
        """Coverage totals for one session, for ONE underlying or for all.

        `underlying` is REQUIRED, with None meaning "every symbol that day",
        because the version that took only a session answered a question
        nobody was asking: with two collectors writing one table, the
        reconcile for QQQ read SOXL's rows as well and printed 62.5% where
        the truth was 25% -- the value depending on which process wrote
        first. A caller must now say which it wants.

        option_coverage has no underlying column of its own; the symbol
        lives on option_contracts, so this joins the same way subscribed_on
        does. A coverage row whose contract row is missing is therefore
        dropped from a filtered answer and kept in an unfiltered one.
        """
        where = "WHERE v.session = ?"
        params: tuple = (session,)
        if underlying is not None:
            where += " AND c.underlying = ?"
            params = (session, underlying)
        row = self._con.execute(
            "SELECT COUNT(*), SUM(v.feed_vol), SUM(v.our_vol), "
            "       SUM(v.our_ticks), "
            "       SUM(CASE WHEN v.feed_vol IS NULL THEN 1 ELSE 0 END) "
            "FROM option_coverage v "
            + ("JOIN option_contracts c ON c.code = v.code "
               if underlying is not None else "")
            + where, params).fetchone()
        n, feed, ours, ticks, unknown = row
        return {"contracts": n or 0, "feed_vol": feed or 0, "our_vol": ours or 0,
                "our_ticks": ticks or 0, "feed_vol_unknown": unknown or 0,
                "ratio": (ours / feed) if feed else None}

    # ── housekeeping ───────────────────────────────────────────────────────

    @_locked
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

    @_locked
    def checkpoint(self) -> None:
        try:
            self._con.execute("PRAGMA wal_checkpoint(PASSIVE)")
        except sqlite3.Error:
            pass

    # ── read ───────────────────────────────────────────────────────────────

    @_locked
    def session_counts(self, since: str) -> tuple[int, int]:
        t = self._con.execute(
            "SELECT COUNT(*) FROM option_ticks WHERE ts >= ?", (since,)).fetchone()[0]
        g = self._con.execute(
            "SELECT COUNT(*) FROM option_greeks WHERE ts >= ?", (since,)).fetchone()[0]
        return t, g

    @_locked
    def contracts_for(self, underlying: str, expiry: str | None = None) -> list[dict]:
        sql = "SELECT code, underlying, expiry, strike, opt_type FROM option_contracts WHERE underlying = ?"
        args: list = [underlying]
        if expiry:
            sql += " AND expiry = ?"
            args.append(expiry)
        cols = ("code", "underlying", "expiry", "strike", "opt_type")
        return [dict(zip(cols, r)) for r in self._con.execute(sql, args)]

    @_locked
    def close(self) -> None:
        try:
            self._con.close()
        except sqlite3.Error:
            pass

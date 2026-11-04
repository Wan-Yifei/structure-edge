"""SQLite-backed order book snapshot storage (WAL mode — concurrent reader + writer safe).

Schema:
    ob_snapshots(code TEXT, ts TEXT, bids TEXT, asks TEXT,
                 svr_bid_ts REAL, svr_ask_ts REAL)

One row per snapshot. bids/asks are JSON arrays of [price, volume] pairs,
ordered as the feed delivered them. ts is ISO-8601: '2026-05-30 09:30:00.123456'.

Two clocks are stored, and they are not interchangeable. `ts` is this
machine's wall clock when the push was handled; svr_bid_ts / svr_ask_ts are
the quote server's, as unix epoch seconds, per side because a push can
refresh one side only. Writes record both; reads hand back one, the aligned
time -- see SERVER_LEAD_SECS and _aligned_ts for which and why. Range
arguments and the values returned by latest_ts() / earliest_ts() stay in the
stored frame, because that is what the index is built on.

This replaced a row-per-price-level table, which cost 120 rows per snapshot
for a 60+60 book and repeated the 26-byte ts and the code on every one of
them. Measured on SOXL 2026-09-21: the feed pushes 6.7 distinct books a
second during regular hours, so that schema wanted 803 rows/s -- about
1.9 GB per code per session. That is what forced the collector's 2s write
throttle, which was discarding 93% of the book updates it received. One row
per snapshot makes the same feed 6.7 rows/s.

The old order_book_snapshots table is not read or written any more. It is
left in place rather than dropped automatically; see drop_legacy_table().
"""

from __future__ import annotations

import json
import sqlite3
import pathlib
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo

_DEFAULT_DB = pathlib.Path(__file__).parent.parent / "db" / "order_book.db"

_ET = ZoneInfo("America/New_York")

_SETUP_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
CREATE TABLE IF NOT EXISTS ob_snapshots (
    code TEXT NOT NULL,
    ts   TEXT NOT NULL,
    bids TEXT NOT NULL,
    asks TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_obsnap_code_ts ON ob_snapshots (code, ts);
"""
# Deliberately NOT keyed on (code, ts). Two consecutive pushes can carry
# genuinely different books and still land on the same timestamp: the feed
# bursts several books per ~300ms cadence tick (12-19ms apart, measured) and
# datetime.now() on Windows only advances in ~15.6ms steps. A (code, ts)
# primary key made the second one REPLACE the first -- caught in testing as
# 30 pushes becoming 9 stored snapshots, which in production would have been
# silent loss of exactly the fast-moving books worth keeping.
#
# Rows are therefore identified by their implicit rowid, which also gives
# insertion order for free: readers below break ts ties with it rather than
# relying on MAX(ts) to name a single row.


# Snapshots retained per code by prune(). The feed delivers 6.7 distinct
# books a second at the regular-hours peak (measured, SOXL 2026-09-21), so
# this is roughly 50 minutes of history -- comfortably more than the heatmap's
# 240-column window asks for, at about 36 MB per code. The old row-based
# budget of 1000 came out to 8 snapshots.
_DEFAULT_KEEP_SNAPSHOTS = 20_000


# Seconds to subtract from a snapshot's server receive stamp to put it on the
# same clock as the tape.
#
# `ts` is this machine's wall clock, and on an undisciplined Windows host that
# free-runs: measured 2026-11-01..11-03 at -39 +/- 1 ms/h, with Windows
# stepping it back into line only every day or two. Filing books under it
# therefore slides against the tape by most of a column of the heatmap's 1 s
# grid per day, in one direction, until the next step.
#
# The server stamps do not have that problem -- they are taken on the quote
# server, so the only thing between them and the tape's exchange stamps is
# transport, and transport measured constant. On US.SOXL 2026-11-03
# 09:35-13:35 ET, five 10-minute windows put the best alignment at
# +0.20..+0.25 s, sd 0.02, flat across the four hours; the same windows scored
# against `ts` slid from +0.05 to -0.05 over the same stretch.
#
# Two limits, both worth re-checking before leaning harder on this:
#   - only 4.5 hours of one session has been measured, so "constant across
#     days" is untested rather than established
#   - Qot_UpdateOrderBook.proto documents svrRecvTime*Timestamp as Hong Kong
#     only. US.SOXL carries it on every push and it behaves correctly, but
#     that is outside what the protocol promises -- missing_server_ts_rate()
#     exists so a silent withdrawal does not go unnoticed.
SERVER_LEAD_SECS = 0.22

# How far a row's aligned timestamp can sit from its stored one. Only the
# clock error contributes, and the largest correction Windows has had to make
# on this host is 1.65 s, so this is a generous bound rather than a tuned one.
# It is used to widen an index-friendly range scan on the stored `ts` before
# filtering exactly on the aligned value.
_ALIGN_MARGIN = timedelta(seconds=5)

# Candidate rows snapshot_at_or_before() pulls before picking by aligned time.
# The feed pushes 6.7 distinct books a second at the regular-hours peak, so
# this is about 30 s of book -- six times _ALIGN_MARGIN, which is what it has
# to cover.
_AT_OR_BEFORE_SCAN = 200


def _ts_str(ts) -> str:
    if isinstance(ts, datetime):
        return ts.isoformat(sep=" ")
    return str(ts)


def _newer_svr(a: float | None, b: float | None) -> float | None:
    """The later of a row's two server stamps; None when neither is set.

    A push that refreshes one side only leaves the other side's cached stamp
    behind, so the later of the two is the age of the freshest information in
    the row. In practice they agree to the millisecond -- the median of
    svr_bid_ts - svr_ask_ts was 0.000 for every hour of 2026-11-03 -- so this
    matters only for genuinely one-sided pushes.
    """
    vals = [v for v in (a, b) if v]
    return max(vals) if vals else None


def _aligned_ts(ts_s: str, svr_bid: float | None,
                svr_ask: float | None) -> datetime:
    """The time a snapshot should be filed under, as a naive ET datetime.

    The server stamp minus SERVER_LEAD_SECS when the row has one; otherwise
    the stored `ts` unchanged, which is every row written before 2026-11-03
    00:46. Those keep exactly the behaviour they had -- this cannot repair
    them, since what they would need is the clock error at the moment they
    were written.
    """
    svr = _newer_svr(svr_bid, svr_ask)
    if svr is None:
        return datetime.fromisoformat(ts_s)
    return datetime.fromtimestamp(svr - SERVER_LEAD_SECS,
                                  _ET).replace(tzinfo=None)


def _et_offset_secs(when: datetime) -> int:
    """Seconds to add to a naive-ET `strftime('%s', ts)` to get a true epoch.

    SQLite has no notion of a timezone and reads a stored timestamp as UTC, so
    its epoch for an ET string is wrong by the ET offset. Server stamps are
    already true epochs, which is why a bucket key mixing the two has to put
    them in the same frame first.

    Supplied per query rather than per row: it only applies to rows with no
    server stamp, i.e. ones written before 2026-11-03, and a single query
    spanning a DST change AND reaching back that far would need two. Such a
    query would misplace the pre-cutover rows by an hour; it stops being
    possible once the retention window has rolled past 2026-11-03.
    """
    off = _ET.utcoffset(when)
    return -int(off.total_seconds()) if off else 0


class OrderBookStore:
    """Read/write access to order_book.db."""

    def __init__(self, db_path: str | pathlib.Path = _DEFAULT_DB,
                 read_only: bool = False):
        self._path = pathlib.Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if read_only:
            self._con = sqlite3.connect(str(self._path), check_same_thread=False)
        else:
            self._con = sqlite3.connect(str(self._path), check_same_thread=False)
            self._con.executescript(_SETUP_SQL)
            self._ensure_columns()

    def _ensure_columns(self) -> None:
        """Add columns to a database created before they existed.

        Same shape as TickStore._ensure_columns: ALTER TABLE ADD COLUMN is
        metadata-only in SQLite, so it is instant whatever the table size,
        and a failure here must not stop the collector -- the column simply
        stays absent until a restart adds it.
        """
        try:
            cols = {r[1] for r in self._con.execute(
                "PRAGMA table_info(ob_snapshots)")}
            for name, decl in (("svr_bid_ts", "REAL"), ("svr_ask_ts", "REAL")):
                if name not in cols:
                    self._con.execute(
                        f"ALTER TABLE ob_snapshots ADD COLUMN {name} {decl}")
            self._con.commit()
        except Exception as exc:
            print(f"[OrderBookStore] could not add columns: {exc}", flush=True)

    # ── write ──────────────────────────────────────────────────────────────────

    def insert_snapshot(self, code: str, ts,
                        bids: list[tuple[float, int]],
                        asks: list[tuple[float, int]],
                        svr_bid_ts: float | None = None,
                        svr_ask_ts: float | None = None) -> int:
        """Insert one full order book snapshot atomically.

        bids / asks: list of (price, volume) tuples.

        svr_bid_ts / svr_ask_ts are the quote server's own clock for the push
        each side came from, per side because a push can refresh one side
        only. `ts` is still written exactly as before and still what the
        indexed range scans use; what changed is that reads now hand back the
        aligned time derived from these, so a row written without them falls
        back to `ts` and behaves as it always did.

        They exist because `ts` comes off an undisciplined local clock that
        free-runs at -39 ms/h, while the tape carries the exchange's -- an
        offset that measured -0.5 to -1.35 s across three sessions and grows
        between the irregular occasions Windows corrects it.

        Returns total rows inserted.
        """
        b = [[float(p), int(v)] for p, v in bids]
        a = [[float(p), int(v)] for p, v in asks]
        if not b and not a:
            return 0
        self._con.execute(
            "INSERT INTO ob_snapshots (code, ts, bids, asks, "
            "svr_bid_ts, svr_ask_ts) VALUES (?, ?, ?, ?, ?, ?)",
            (code, _ts_str(ts), json.dumps(b, separators=(",", ":")),
             json.dumps(a, separators=(",", ":")),
             (float(svr_bid_ts) if svr_bid_ts else None),
             (float(svr_ask_ts) if svr_ask_ts else None)),
        )
        self._con.commit()
        # Levels, not rows. One row is written now, but callers use this as
        # "how much book did I capture" -- the collector's session counter and
        # its first-snapshot log both read it that way.
        return len(b) + len(a)

    # ── read ───────────────────────────────────────────────────────────────────

    # Every read goes through one of the methods below rather than a
    # hand-rolled SELECT. The heatmap, the DOM window and two scripts each
    # used to open their own connection and spell out the same
    # "SELECT ts, side, price, volume FROM order_book_snapshots" -- six copies
    # of the row shape, which means six places to fix for any schema change.

    def _rows(self, sql: str, params: list) -> list[dict]:
        """Run a query returning (ts, bids, asks) and flatten it to level dicts.

        The single place that knows how a stored snapshot maps to the dicts
        readers expect -- which is why the schema change reaches exactly one
        function instead of the six hand-rolled SELECTs there used to be.
        Readers still see one dict per price level, bids before asks, so
        nothing downstream had to change.

        `ts` on the returned dicts is the ALIGNED time (see _aligned_ts), not
        the stored one, because every reader of these dicts goes on to put
        them beside tape data. The stored value is still what the SQL ranges
        filter on, so a query's two edges can admit or drop a row by up to the
        clock error; a range of any length wears that only at its ends, while
        filing every row under the local clock wore it throughout.
        """
        out: list[dict] = []
        for ts_s, bids_j, asks_j, sb, sa in self._con.execute(
                sql, params).fetchall():
            ts = _aligned_ts(ts_s, sb, sa)
            for side, payload in (("BID", bids_j), ("ASK", asks_j)):
                for price, volume in json.loads(payload):
                    out.append({"ts": ts, "side": side,
                                "price": float(price), "volume": float(volume)})
        return out

    def _snapshots(self, sql: str, params: list) -> list[list[dict]]:
        """Like _rows, but one list per stored snapshot rather than one flat
        list. Grouping happens by row, never by timestamp -- see the DDL note
        on why two snapshots can share one."""
        out: list[list[dict]] = []
        for ts_s, bids_j, asks_j, sb, sa in self._con.execute(
                sql, params).fetchall():
            ts = _aligned_ts(ts_s, sb, sa)
            group: list[dict] = []
            for side, payload in (("BID", bids_j), ("ASK", asks_j)):
                for price, volume in json.loads(payload):
                    group.append({"ts": ts, "side": side,
                                  "price": float(price), "volume": float(volume)})
            out.append(group)
        return out

    def query_snapshots(self, code: str, start: datetime, end: datetime,
                        end_inclusive: bool = False) -> list[dict]:
        """Rows for *code* between *start* and *end*, oldest first.

        end_inclusive exists because the callers disagreed: the viewer's
        date-window loads were written against a half-open end, the DOM
        window's against a closed one. Rather than quietly change either,
        both keep the bound they had.
        """
        op = "<=" if end_inclusive else "<"
        return self._rows(
            "SELECT ts, bids, asks, svr_bid_ts, svr_ask_ts FROM ob_snapshots "
            f"WHERE code = ? AND ts >= ? AND ts {op} ? ORDER BY ts",
            [code, _ts_str(start), _ts_str(end)],
        )

    def latest_snapshot(self, code: str) -> list[dict]:
        """Every row of the newest snapshot for *code* -- one full book."""
        # ORDER BY ... LIMIT 1, not ts = MAX(ts): duplicate timestamps are
        # possible (see the DDL note), and MAX(ts) would return both books
        # fused into one 240-level "snapshot".
        return self._rows(
            "SELECT ts, bids, asks, svr_bid_ts, svr_ask_ts FROM ob_snapshots "
            "WHERE code = ? ORDER BY ts DESC, rowid DESC LIMIT 1",
            [code],
        )

    def snapshot_at_or_before(self, code: str, ts: datetime) -> list[dict]:
        """The newest snapshot at or before *ts* -- the book as of that moment.

        *ts* is compared against the aligned time, since the callers ask this
        with a moment taken from the tape. The scan still runs on the stored
        `ts` so the index drives it: take a short run of candidates from past
        the margin and pick the newest whose aligned time qualifies.
        """
        groups = self._snapshots(
            "SELECT ts, bids, asks, svr_bid_ts, svr_ask_ts FROM ("
            "  SELECT rowid, ts, bids, asks, svr_bid_ts, svr_ask_ts "
            "  FROM ob_snapshots "
            "  WHERE code = ? AND ts <= ? "
            "  ORDER BY ts DESC, rowid DESC LIMIT ?)"
            " ORDER BY ts DESC, rowid DESC",
            [code, _ts_str(ts + _ALIGN_MARGIN), _AT_OR_BEFORE_SCAN],
        )
        best: list[dict] = []
        for group in groups:
            if group and group[0]["ts"] <= ts and (
                    not best or group[0]["ts"] > best[0]["ts"]):
                best = group
        return best

    def last_n_snapshots(self, code: str, n: int) -> list[list[dict]]:
        """The last *n* snapshots for *code*, oldest first, one list each.

        One query rather than the 1 + n the callers used to issue: pull every
        row whose ts is in the newest n timestamps, then group. At ~120 rows a
        snapshot that is a round trip per snapshot saved.
        """
        # One row IS one snapshot, so this groups by row rather than by ts.
        # Grouping by ts would fuse two same-timestamp books into one.
        return self._snapshots(
            "SELECT ts, bids, asks, svr_bid_ts, svr_ask_ts FROM ("
            "  SELECT rowid, ts, bids, asks, svr_bid_ts, svr_ask_ts "
            "  FROM ob_snapshots "
            "  WHERE code = ? ORDER BY ts DESC, rowid DESC LIMIT ?)"
            " ORDER BY ts, rowid",
            [code, n],
        )

    def bucketed_snapshots(self, code: str, start: datetime, end: datetime,
                           bucket_secs: int) -> list[list[dict]]:
        """One snapshot per *bucket_secs* window in [start, end), oldest first.

        For drawing a long stretch of the heatmap: a column covers bucket_secs
        of time and shows the book as it stood at the end of that window, so
        pulling every snapshot in the range and throwing almost all of them
        away is pure cost. The whole retained history is ~535k snapshots; at a
        30s bucket this returns ~3.8k of them.

        The bucket representative is MAX(rowid), which is the newest row in the
        window -- rowid is insertion order, and that matches what the live path
        paints, since it repaints a column in place as new pushes arrive.

        Measured against the full table (SQLite, 535k snapshots): 0.46s at a
        60s bucket, 0.49s at 30s, 0.61s at 5s. A 1s bucket is 73s and 192MB --
        it selects 105k snapshots, which is both unusable and undrawable, so
        callers are expected to coarsen the bucket for long ranges rather than
        this method silently capping it.

        Selecting by a materialised id list instead of this nested subquery
        collapses past a few thousand buckets: `rowid IN (?,?,...)` with 11k
        parameters measured 51s against 42ms for the same query at 3.8k.
        """
        if bucket_secs < 1:
            bucket_secs = 1
        # The bucket key is the ALIGNED time, not the stored one. Labelling a
        # column with one clock while deciding its contents with another is
        # how a 1 s grid ends up a column out: the two disagree by the clock
        # error, which is the whole thing being corrected here.
        #
        # The two frames are reconciled inside the CAST rather than outside
        # it. A server stamp is already a true epoch; strftime reads the
        # stored string as UTC, so the rows without a server stamp need the ET
        # offset added before the two can share a divisor.
        #
        # The scan bound stays on the stored `ts` so the (code, ts) index
        # still drives it, widened by _ALIGN_MARGIN and trimmed exactly
        # afterwards.
        groups = self._snapshots(
            "SELECT ts, bids, asks, svr_bid_ts, svr_ask_ts FROM ob_snapshots "
            "WHERE rowid IN ("
            "    SELECT MAX(rowid) FROM ob_snapshots "
            "    WHERE code = ? AND ts >= ? AND ts < ? "
            "    GROUP BY CAST(COALESCE("
            "        MAX(COALESCE(svr_bid_ts, svr_ask_ts),"
            "            COALESCE(svr_ask_ts, svr_bid_ts)) - ?,"
            "        strftime('%s', ts) + ?) / ? AS INT)"
            ") ORDER BY rowid",
            [code, _ts_str(start - _ALIGN_MARGIN),
             _ts_str(end + _ALIGN_MARGIN),
             SERVER_LEAD_SECS, _et_offset_secs(start), bucket_secs],
        )
        # Sorted by aligned time rather than returned in rowid order: a caller
        # painting these left to right needs them monotonic on the axis it
        # labels, and rowid is insertion order on the other clock.
        return sorted((g for g in groups if g and start <= g[0]["ts"] < end),
                      key=lambda g: g[0]["ts"])

    def latest_ts(self, code: str | None = None) -> datetime | None:
        """Newest snapshot timestamp, for *code* or across every code."""
        if code:
            r = self._con.execute(
                "SELECT MAX(ts) FROM ob_snapshots WHERE code = ?",
                [code]).fetchone()
        else:
            r = self._con.execute(
                "SELECT MAX(ts) FROM ob_snapshots").fetchone()
        return datetime.fromisoformat(r[0]) if r and r[0] else None

    def earliest_ts(self, code: str | None = None) -> datetime | None:
        """Oldest snapshot timestamp, for *code* or across every code."""
        if code:
            r = self._con.execute(
                "SELECT MIN(ts) FROM ob_snapshots WHERE code = ?",
                [code]).fetchone()
        else:
            r = self._con.execute(
                "SELECT MIN(ts) FROM ob_snapshots").fetchone()
        return datetime.fromisoformat(r[0]) if r and r[0] else None

    def missing_server_ts_rate(self, code: str,
                               since: datetime | None = None
                               ) -> tuple[int, int]:
        """(rows with no server stamp, rows looked at) for *code*.

        Qot_UpdateOrderBook.proto documents svrRecvTime*Timestamp as Hong Kong
        only. US.SOXL carries it on every push today, but that is behaviour
        outside what the protocol promises, not a guarantee. If it were
        withdrawn the rows would keep arriving and _aligned_ts would quietly
        fall back to the local clock, so the alignment would decay with
        nothing failing and no error logged. This is what notices.

        Counting rows written before 2026-11-03 00:46 as "missing" is correct
        -- they are -- so pass *since* when asking about the live feed.
        """
        sql = ("SELECT COUNT(*), "
               "SUM(CASE WHEN svr_bid_ts IS NULL AND svr_ask_ts IS NULL "
               "         THEN 1 ELSE 0 END) "
               "FROM ob_snapshots WHERE code = ?")
        params: list = [code]
        if since is not None:
            sql += " AND ts >= ?"
            params.append(_ts_str(since))
        total, missing = self._con.execute(sql, params).fetchone()
        return int(missing or 0), int(total or 0)

    def codes(self) -> list[str]:
        """Every code with at least one stored snapshot, sorted."""
        return [r[0] for r in self._con.execute(
            "SELECT DISTINCT code FROM ob_snapshots ORDER BY code")]

    def query_date(self, code: str, day: date) -> list[dict]:
        start = datetime(day.year, day.month, day.day)
        end   = datetime(day.year, day.month, day.day, 23, 59, 59, 999999)
        return self.query_snapshots(code, start, end)

    def available_dates(self, code: str) -> list[date]:
        cur = self._con.execute(
            "SELECT DISTINCT date(ts) FROM ob_snapshots "
            "WHERE code = ? ORDER BY 1",
            [code],
        )
        return [date.fromisoformat(r[0]) for r in cur.fetchall()]

    def snapshot_count(self, code: str | None = None) -> int:
        """Number of stored snapshots, for *code* or across every code.

        Renamed from row_count with the schema: a row is a snapshot now, so
        the old name would have quietly changed meaning by a factor of ~120
        while still reading as correct at every call site.
        """
        if code:
            return self._con.execute(
                "SELECT COUNT(*) FROM ob_snapshots WHERE code = ?",
                [code],
            ).fetchone()[0]
        return self._con.execute(
            "SELECT COUNT(*) FROM ob_snapshots"
        ).fetchone()[0]

    def prune(self, keep: int = _DEFAULT_KEEP_SNAPSHOTS,
              codes: list[str] | None = None) -> int:
        """Keep only the most recent *keep* SNAPSHOTS per code.

        Snapshots, not rows -- the distinction used to matter and was wrong.
        Under the row-per-level table a "keep 1000 rows" budget bought 1000/120
        = 8 snapshots of a 60+60 book, i.e. under 20 seconds of history, and
        the retained depth silently shrank as the book got deeper. That is why
        the heatmap's pre-fill could only ever find a handful of columns.

        codes: restrict pruning to this subset (e.g. so a caller can run this
        twice with different *keep* values per code group, mirroring
        TickStore.prune_older_than's codes-list pattern). Defaults to every
        code currently in the table.

        Returns snapshots deleted.
        """
        if codes is None:
            codes = [r[0] for r in self._con.execute(
                "SELECT DISTINCT code FROM ob_snapshots"
            ).fetchall()]
        deleted = 0
        for code in codes:
            # rowid, not ts: two snapshots can share a timestamp, and a
            # ts-based NOT IN would either keep or delete both of them.
            cur = self._con.execute(
                "DELETE FROM ob_snapshots WHERE code = ? AND rowid NOT IN ("
                "  SELECT rowid FROM ob_snapshots WHERE code = ?"
                "  ORDER BY ts DESC, rowid DESC LIMIT ?"
                ")",
                [code, code, keep],
            )
            deleted += cur.rowcount
        if deleted:
            self._con.commit()
        return deleted

    def drop_legacy_table(self) -> bool:
        """Drop the pre-schema-change order_book_snapshots table, if present.

        Not called automatically: it is the user's data, and the file does not
        shrink until a VACUUM anyway. Returns whether a table was dropped.
        """
        row = self._con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' "
            "AND name='order_book_snapshots'").fetchone()
        if not row:
            return False
        self._con.execute("DROP TABLE order_book_snapshots")
        self._con.commit()
        return True

    # ── lifecycle ──────────────────────────────────────────────────────────────

    def close(self) -> None:
        self._con.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

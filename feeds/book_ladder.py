"""Per-second order book LADDER, archived level by level.

`feeds/depth_bars.py` already archives the book, but only as aggregates --
band totals, the touch, the largest level. Those cannot answer the question
that matters most about a level that disappears: were the shares BOUGHT or
WITHDRAWN. Separating the two needs the size at each individual price, so
that a level's decrease can be matched against the volume printed at that
same price. On 2026-10-13 a 42,019-share level at 118.00 vanished while only
15,501 shares printed there; no aggregate column can show that.

The source, `db/order_book.db`, is pruned BY ROW COUNT -- about 535,000
snapshots, which at the current rate is roughly 1.4 days. Everything older
is gone for good. This module is therefore an ARCHIVE, exactly like
depth_bars: `append_day` only ever adds, and `load` reads the archive alone.
Rebuilding from the live database would silently truncate months of history
to the retention window.

Run it daily, BEFORE the collector prunes. A missed day is a lost day.

What is stored, one row per price level per second:
  ts     the second (the LAST snapshot inside it, so the level sizes are
         the book as that second ended)
  price  the price level
  size   total resting size at that price, as moomoo reports it -- the
         venue-aggregated total, NOT a single order. moomoo's feed carries
         no order count, which is the one thing MBO would add.
  side   'B' or 'A'

The WHOLE visible book is kept rather than a band around the midpoint. The
visible book is only about 146 levels (p90 154), so a band at 0.3% would
save barely half the space while fixing a choice that later work cannot
undo. About 10 MB a day, against 4.6 MB for the aggregates already archived.

One snapshot a second, not all 6.6: the full rate costs 66 MB a day and the
estimator that consumes this works on multi-second windows anyway. Faster
moves are netted into the second that contains them.
"""

from __future__ import annotations

import json
import pathlib
import sqlite3
from datetime import date

import numpy as np
import pandas as pd

_ROOT = pathlib.Path(__file__).parent.parent
_OB_DB = _ROOT / "db" / "order_book.db"
_CACHE = _ROOT / "db" / "cache" / "book_ladder"

COLUMNS = ["ts", "price", "size", "side"]


def _day_path(code: str, day: date | str) -> pathlib.Path:
    d = pd.Timestamp(day).date()
    return _CACHE / code.replace(".", "_") / f"{d.isoformat()}.parquet"


def _empty() -> pd.DataFrame:
    return pd.DataFrame({"ts": pd.Series(dtype="datetime64[ns]"),
                         "price": pd.Series(dtype="float64"),
                         "size": pd.Series(dtype="int64"),
                         "side": pd.Series(dtype="object")})


def build_day(code: str, day: date | str,
              db_path: pathlib.Path | None = None,
              skip_seconds: set | None = None) -> pd.DataFrame:
    """One row per level per second for *day*, from surviving snapshots.

    Returns an empty frame for a day already pruned -- the normal answer for
    anything past the retention window, not an error.

    *skip_seconds* holds seconds already archived, whose JSON is then never
    parsed. The scheduler re-runs this every few hours so that a missed run
    does not lose a day, and without the skip each run would rebuild all
    twelve million rows of a day to keep a few thousand new ones.
    """
    lo = pd.Timestamp(day).normalize()
    hi = lo + pd.Timedelta(days=1)
    con = sqlite3.connect(f"file:{db_path or _OB_DB}?mode=ro", uri=True)
    con.execute("PRAGMA busy_timeout=30000")
    try:
        # Timestamps first: only the last snapshot of each second is parsed,
        # which skips roughly five sixths of the JSON.
        stamps = con.execute(
            "SELECT rowid, ts FROM ob_snapshots WHERE code = ? "
            "AND ts >= ? AND ts < ? ORDER BY rowid",
            [code, lo.isoformat(sep=" "), hi.isoformat(sep=" ")],
        ).fetchall()
        if not stamps:
            return _empty()
        keep: dict[str, int] = {}
        for rid, ts in stamps:
            keep[str(ts)[:19]] = rid
        if skip_seconds:
            keep = {s: r for s, r in keep.items() if s not in skip_seconds}
        if not keep:
            return _empty()
        # Fetched in batches: a day holds tens of thousands of seconds and
        # SQLite caps host parameters well below that.
        rows = []
        rids = list(keep.values())
        for i in range(0, len(rids), 800):
            chunk = rids[i:i + 800]
            rows.extend(con.execute(
                "SELECT rowid, bids, asks FROM ob_snapshots WHERE rowid IN "
                f"({','.join('?' * len(chunk))})", chunk).fetchall())
    finally:
        con.close()

    by_rid = {rid: (b, a) for rid, b, a in rows}
    # A full day is around ten million rows. One pandas Timestamp per row
    # would cost half a gigabyte on its own, so each second is recorded once
    # and expanded with np.repeat at the end.
    secs, counts, px_out, sz_out, sd_out = [], [], [], [], []
    for sec, rid in keep.items():
        got = by_rid.get(rid)
        if got is None:
            continue
        try:
            bids, asks = json.loads(got[0]), json.loads(got[1])
        except (ValueError, TypeError):
            continue
        if not bids or not asks:
            continue
        n = 0
        for book, side in ((bids, "B"), (asks, "A")):
            for p, v in book:
                px_out.append(p)
                sz_out.append(v)
                sd_out.append(side)
                n += 1
        secs.append(sec)
        counts.append(n)

    if not secs:
        return _empty()
    ts = np.repeat(np.array(secs, dtype="datetime64[s]"),
                   np.array(counts, dtype=np.int64))
    df = pd.DataFrame({
        "ts": ts.astype("datetime64[ns]"),
        "price": np.round(np.asarray(px_out, dtype=np.float64), 4),
        "size": np.asarray(sz_out, dtype=np.int64),
        "side": pd.Categorical(sd_out, categories=["A", "B"]),
    })
    return df.sort_values(["ts", "side", "price"]).reset_index(drop=True)


def append_day(code: str, day: date | str,
               db_path: pathlib.Path | None = None,
               cache_dir: pathlib.Path | None = None) -> int:
    """Add *day* to the archive, keeping every second already stored.

    Seconds already archived win: a second run over a day the collector has
    since partly pruned must not replace good rows with the remains.
    Returns the number of rows added.
    """
    p = _day_path(code, day)
    if cache_dir is not None:
        p = pathlib.Path(cache_dir) / p.parent.name / p.name

    old = pd.read_parquet(p) if p.exists() else None
    have = ({pd.Timestamp(t).strftime("%Y-%m-%d %H:%M:%S")
             for t in old.ts.unique()} if old is not None else None)
    fresh = build_day(code, day, db_path=db_path, skip_seconds=have)
    if fresh.empty:
        return 0

    p.parent.mkdir(parents=True, exist_ok=True)
    if old is not None:
        before = len(old)
        merged = pd.concat([old, fresh])
    else:
        before = 0
        merged = fresh
    merged = merged.sort_values(["ts", "side", "price"]).reset_index(drop=True)
    merged.to_parquet(p, compression="zstd", index=False)
    return len(merged) - before


def load(code: str, start=None, end=None,
         cache_dir: pathlib.Path | None = None) -> pd.DataFrame:
    """Read the archive. It is never rebuilt from the live database.

    A range reaching past what was archived comes back short rather than
    triggering a build: the snapshots behind it no longer exist, so a build
    would return less than the archive already holds.
    """
    base = pathlib.Path(cache_dir) if cache_dir else _CACHE
    d = base / code.replace(".", "_")
    if not d.exists():
        return _empty()
    lo = pd.Timestamp(start) if start is not None else None
    hi = pd.Timestamp(end) if end is not None else None
    out = []
    for f in sorted(d.glob("*.parquet")):
        day = pd.Timestamp(f.stem)
        if lo is not None and day + pd.Timedelta(days=1) <= lo:
            continue
        if hi is not None and day > hi:
            continue
        out.append(pd.read_parquet(f))
    if not out:
        return _empty()
    df = pd.concat(out, ignore_index=True)
    if lo is not None:
        df = df[df.ts >= lo]
    if hi is not None:
        df = df[df.ts <= hi]
    return df.sort_values(["ts", "side", "price"]).reset_index(drop=True)


def book_at(df: pd.DataFrame, when) -> tuple[dict, dict]:
    """The two sides as {price: size} for the second containing *when*."""
    s = pd.Timestamp(when).floor("s")
    w = df[df.ts == s]
    b = dict(zip(w.loc[w.side == "B", "price"], w.loc[w.side == "B", "size"]))
    a = dict(zip(w.loc[w.side == "A", "price"], w.loc[w.side == "A", "size"]))
    return b, a


def coverage(code: str, cache_dir: pathlib.Path | None = None) -> pd.DataFrame:
    """Per-day seconds and rows, for spotting a missed run."""
    base = pathlib.Path(cache_dir) if cache_dir else _CACHE
    d = base / code.replace(".", "_")
    if not d.exists():
        return pd.DataFrame(columns=["seconds", "rows", "first", "last", "MB"])
    rows = []
    for f in sorted(d.glob("*.parquet")):
        df = pd.read_parquet(f, columns=["ts"])
        rows.append({"day": f.stem, "seconds": df.ts.nunique(),
                     "rows": len(df),
                     "first": df.ts.min().strftime("%H:%M:%S"),
                     "last": df.ts.max().strftime("%H:%M:%S"),
                     "MB": round(f.stat().st_size / 1e6, 1)})
    return pd.DataFrame(rows).set_index("day")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--codes", nargs="+", default=["US.SOXL"])
    ap.add_argument("--days", type=int, default=2,
                    help="how many days back to sweep, today included")
    ap.add_argument("--coverage", action="store_true",
                    help="print what is archived and exit")
    args = ap.parse_args()

    for code in args.codes:
        if args.coverage:
            cov = coverage(code)
            print(f"{code}: {len(cov)} days archived")
            if not cov.empty:
                print(cov.to_string())
            continue
        today = pd.Timestamp.now().normalize()
        for k in range(args.days):
            day = (today - pd.Timedelta(days=k)).date()
            added = append_day(code, day)
            print(f"  {code} {day}: +{added:,} rows", flush=True)
        cov = coverage(code)
        print(f"{code}: archive now {len(cov)} days / "
              f"{int(cov.rows.sum()) if len(cov) else 0:,} rows / "
              f"{cov.MB.sum() if len(cov) else 0:.1f} MB", flush=True)


if __name__ == "__main__":
    main()

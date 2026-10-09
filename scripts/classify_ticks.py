"""Infer a side for NEUTRAL ticks from the prevailing midpoint.

NEUTRAL is the majority of the tape -- 57.5% of SOXL volume -- and it is
off-exchange flow, not noise: 43.9% of those prints carry sub-penny prices,
which only wholesaler price improvement produces. Signed against the midpoint
it tracks the minute's return about as well as the exchange's own tags (+0.253
against +0.276), so delta without it sees about an eighth of the flow. See
doc/ORDER_FLOW_GUIDE.md appendix A.

Run offline, not on the read path: assembling a day of midpoints from
order_book.db takes ~22s, which no fetch on a one-second timer can afford.

    uv run scripts/classify_ticks.py --code US.SOXL --days 3
    uv run scripts/classify_ticks.py --code US.SOXL --start 2026-10-05 --end 2026-10-07

Only rows the feed left NEUTRAL are touched. Where it says BUY or SELL the
exchange knows the aggressor and a midpoint inference does not improve on it.
Coverage is bounded by order_book.db's retention -- a few days -- while
ticks.db holds months, so older history stays unclassified and the readers
fall back to the feed's tags.
"""
from __future__ import annotations

import argparse
import bisect
import json
import pathlib
import sqlite3
import sys
import time
from datetime import datetime, timedelta

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from feeds.tick_store import TickStore          # noqa: E402
from feeds.tick_mid_store import TickMidStore   # noqa: E402

ROOT = pathlib.Path(__file__).parent.parent
TICK_DB = ROOT / "db" / "ticks.db"
OB_DB = ROOT / "db" / "order_book.db"

# A midpoint older than this is not used. 95.8% of NEUTRAL prints have a book
# within half a second (median staleness 157ms), and accuracy measured against
# the feed's own BUY/SELL tags is flat from 0.25s out to no limit at all --
# 77-78% -- so staleness is not what limits this; the midpoint rule itself is.
MAX_STALE_SECS = 1.0


def _midpoints(code: str, start: datetime, end: datetime) -> tuple[list, list]:
    """(timestamps, midpoints) for *code*, oldest first.

    Reads the JSON sides and keeps only each book's extremes. Expanding every
    level into a dict the way OrderBookStore does would build 140 of them per
    snapshot for two numbers.
    """
    con = sqlite3.connect(f"file:{OB_DB}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT ts, bids, asks FROM ob_snapshots "
            "WHERE code = ? AND ts >= ? AND ts < ? ORDER BY rowid",
            [code, start.isoformat(sep=" "), end.isoformat(sep=" ")],
        ).fetchall()
    finally:
        con.close()
    ts, mid = [], []
    for ts_s, bids_j, asks_j in rows:
        b = json.loads(bids_j)
        a = json.loads(asks_j)
        if not b or not a:
            continue
        bb = max(x[0] for x in b)
        ba = min(x[0] for x in a)
        if ba <= bb:          # crossed or locked -- no meaningful midpoint
            continue
        ts.append(datetime.fromisoformat(ts_s))
        mid.append((bb + ba) / 2.0)
    return ts, mid


def classify_day(code: str, day: datetime, store: TickStore,
                 mids: TickMidStore) -> dict:
    d0 = datetime(day.year, day.month, day.day)
    d1 = d0 + timedelta(days=1)

    t0 = time.time()
    bts, bmid = _midpoints(code, d0 - timedelta(seconds=MAX_STALE_SECS), d1)
    t_mid = time.time() - t0
    if not bts:
        return {"date": d0.date(), "ticks": 0, "neutral": 0, "signed": 0,
                "stale": 0, "at_mid": 0, "secs": t_mid, "note": "no book"}

    ticks = store.query_ticks(code, d0, d1)
    # "direction" is what query_ticks returns; the store never renames it.
    neutral = [t for t in ticks if t["direction"] == "NEUTRAL"]

    updates, stale, at_mid = [], 0, 0
    for t in neutral:
        i = bisect.bisect_right(bts, t["ts"]) - 1
        if i < 0 or (t["ts"] - bts[i]).total_seconds() > MAX_STALE_SECS:
            stale += 1
            continue
        m = bmid[i]
        if t["price"] > m:
            side = "BUY"
        elif t["price"] < m:
            side = "SELL"
        else:
            at_mid += 1
            continue
        # ts_raw, not the parsed datetime: ticks.db stores the feed's own
        # string and re-serialising a datetime pads the microseconds, which
        # makes every key miss.
        updates.append((code, t["ts_raw"], t["price"], t["volume"], side))

    for i in range(0, len(updates), 20000):
        mids.put(updates[i:i + 20000])
    return {"date": d0.date(), "ticks": len(ticks), "neutral": len(neutral),
            "signed": len(updates), "stale": stale, "at_mid": at_mid,
            "secs": time.time() - t0, "note": ""}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--code", default="US.SOXL")
    ap.add_argument("--days", type=int, default=3,
                    help="how many days back from today (default 3)")
    ap.add_argument("--start", help="YYYY-MM-DD, overrides --days")
    ap.add_argument("--end", help="YYYY-MM-DD inclusive, defaults to today")
    args = ap.parse_args()

    if args.start:
        d0 = datetime.strptime(args.start, "%Y-%m-%d")
        d1 = (datetime.strptime(args.end, "%Y-%m-%d") if args.end
              else datetime.now().replace(hour=0, minute=0, second=0,
                                          microsecond=0))
    else:
        d1 = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        d0 = d1 - timedelta(days=args.days - 1)

    print(f"{args.code}  {d0:%Y-%m-%d} .. {d1:%Y-%m-%d}   "
          f"max staleness {MAX_STALE_SECS}s")
    print(f"{'date':>12} {'ticks':>10} {'neutral':>10} {'signed':>10} "
          f"{'% of neu':>9} {'no book':>9} {'at mid':>8} {'secs':>7}")
    # ticks.db read-only: the collector holds its write lock whenever a session
    # is open, and nothing here needs to write to it.
    with TickStore(TICK_DB, read_only=True) as store, TickMidStore() as mids:
        day = d0
        while day <= d1:
            r = classify_day(args.code, day, store, mids)
            pct = (100 * r["signed"] / r["neutral"]) if r["neutral"] else 0.0
            print(f"{str(r['date']):>12} {r['ticks']:>10,} {r['neutral']:>10,} "
                  f"{r['signed']:>10,} {pct:>8.1f}% {r['stale']:>9,} "
                  f"{r['at_mid']:>8,} {r['secs']:>7.1f}"
                  + (f"  {r['note']}" if r["note"] else ""))
            day += timedelta(days=1)


if __name__ == "__main__":
    main()

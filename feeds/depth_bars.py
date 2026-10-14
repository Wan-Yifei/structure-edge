"""Per-second resting depth, aggregated from order book snapshots and archived.

This exists because `db/order_book.db` is a ROLLING WINDOW. The collector
prunes by row count -- 24 hours for SOXL, 6.5 for everything else -- so the
raw snapshots are gone a day after they arrive. Nothing that needs order book
history can be answered by querying that file; by the time the question comes
up the data has been deleted.

So this module is an ARCHIVE, not a cache. `delta_bars.py` can rebuild itself
from ticks.db at any time because the ticks are still there. This one cannot:
`append_day` adds whatever snapshots currently exist to a parquet that only
ever grows, and `load` reads that parquet alone. Rebuilding from the live
database would silently truncate the archive to the retention window, which is
why there is no refresh flag.

Run it daily, BEFORE the collector prunes. Missing a day loses that day for
good.

Size: one row per second, sixteen columns, about 5 MB a day uncompressed and
rather less in parquet -- roughly 0.4 GB for sixty sessions, against 66 GB for
the raw snapshots that produced them.

What each row holds, measured from the snapshot's own midpoint:
  resting size summed within 0.1%, 0.2%, 0.5% and 1% on each side -- the
  0.2% band is where a one-minute move lives, the 1% band is the whole
  visible book;
  the size at the touch, which is what a market order meets first;
  the largest single level on each side and how far away it sits, which is
  the "wall" a chart shows;
  the midpoint twice -- the second's mean, and its LAST value, which is what
  a classifier needs when it asks what the book looked like before a print.

Totals matter as much as the imbalance between them. On 2026-10-13 the
failed probe at 11:35 carried 1.57x the usual depth and was broken through;
the low a minute later had 0.90x and held. Depth collapsing is liquidity
being used up, and a page that stored only (bid - ask) / (bid + ask) could
not see it.
"""

from __future__ import annotations

import json
import pathlib
import sqlite3
from datetime import date, datetime

import numpy as np
import pandas as pd

_ROOT = pathlib.Path(__file__).parent.parent
_OB_DB = _ROOT / "db" / "order_book.db"
_CACHE = _ROOT / "db" / "cache" / "depth_bars"

# Bands as a fraction of the midpoint, and the column suffix for each.
BANDS = ((0.001, "01"), (0.002, "02"), (0.005, "05"), (0.010, "10"))

# mid_last, not just the mean: scripts/classify_ticks.py signs an
# off-exchange print against the book that stood BEFORE it, and a second's
# mean sits in the middle of whatever moved during that second. Signing from
# the mean agreed with the exchange's own tags 75.3% of the time against the
# raw stream's 79.3%, and the two disagreed on a third of the prints they
# both signed. The last book of the previous second is the closest thing to
# "the book before the print" that a per-second row can carry.
COLUMNS = (["mid", "mid_last", "spread", "b_top", "a_top"]
           + [f"{s}{n}" for n in (b[1] for b in BANDS) for s in ("b", "a")]
           + ["b_max", "b_max_d", "a_max", "a_max_d", "n"])


def _cache_path(code: str) -> pathlib.Path:
    return _CACHE / f"{code.replace('.', '_')}_1s.parquet"


def _snapshot_row(bids: list, asks: list) -> dict | None:
    """Collapse one snapshot to the numbers worth keeping."""
    if not bids or not asks:
        return None
    bp = np.asarray([p for p, _ in bids], dtype=float)
    bv = np.asarray([v for _, v in bids], dtype=float)
    ap = np.asarray([p for p, _ in asks], dtype=float)
    av = np.asarray([v for _, v in asks], dtype=float)
    if not bp.size or not ap.size:
        return None
    mid = (bp.max() + ap.min()) / 2.0
    if mid <= 0:
        return None

    row = {"mid": mid, "spread": float(ap.min() - bp.max()),
           "b_top": float(bv[bp.argmax()]), "a_top": float(av[ap.argmin()])}
    for frac, name in BANDS:
        row["b" + name] = float(bv[bp >= mid * (1 - frac)].sum())
        row["a" + name] = float(av[ap <= mid * (1 + frac)].sum())
    bi, ai = int(bv.argmax()), int(av.argmax())
    row["b_max"] = float(bv[bi])
    row["b_max_d"] = float(100.0 * (mid - bp[bi]) / mid)
    row["a_max"] = float(av[ai])
    row["a_max_d"] = float(100.0 * (ap[ai] - mid) / mid)
    return row


def build_day(code: str, day: date | str,
              db_path: pathlib.Path | None = None) -> pd.DataFrame:
    """One row per second for *day*, from whatever snapshots still exist.

    Returns an empty frame with the right columns when the day has been
    pruned already -- an empty result is the normal answer for anything
    older than the retention window, not an error.
    """
    lo = pd.Timestamp(day).normalize()
    con = sqlite3.connect(f"file:{db_path or _OB_DB}?mode=ro", uri=True)
    con.execute("PRAGMA busy_timeout=30000")
    try:
        rows = con.execute(
            "SELECT ts, bids, asks FROM ob_snapshots "
            "WHERE code = ? AND ts >= ? AND ts < ? ORDER BY ts",
            [code, lo.isoformat(sep=" "),
             (lo + pd.Timedelta(days=1)).isoformat(sep=" ")],
        ).fetchall()
    finally:
        con.close()
    if not rows:
        return pd.DataFrame(columns=COLUMNS,
                            index=pd.DatetimeIndex([], name="ts"))

    out, stamps = [], []
    for ts, b, a in rows:
        try:
            r = _snapshot_row(json.loads(b), json.loads(a))
        except (ValueError, TypeError):
            continue
        if r is None:
            continue
        out.append(r)
        stamps.append(ts)
    if not out:
        return pd.DataFrame(columns=COLUMNS,
                            index=pd.DatetimeIndex([], name="ts"))

    df = pd.DataFrame(out)
    # The collector writes the feed's own timestamp string: some carry
    # milliseconds and some do not, so a single fixed format rejects half.
    df["ts"] = pd.to_datetime(pd.Series(stamps), format="ISO8601").dt.floor("s")
    by = df.groupby("ts")
    g = by.mean(numeric_only=True)
    # The rows arrive ORDER BY ts, so "last" within a second is the latest
    # book that second held.
    g["mid_last"] = by["mid"].last()
    g["n"] = by.size()
    return g[list(COLUMNS)]


def append_day(code: str, day: date | str,
               db_path: pathlib.Path | None = None,
               cache_path: pathlib.Path | None = None) -> int:
    """Add *day* to the archive, keeping everything already there.

    Rows already archived for that second win: a second run over a day the
    collector has since partly pruned must not replace good rows with the
    remains. Returns the number of rows added.
    """
    fresh = build_day(code, day, db_path=db_path)
    if fresh.empty:
        return 0
    p = cache_path or _cache_path(code)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        old = pd.read_parquet(p)
        before = len(old)
        merged = pd.concat([old, fresh[~fresh.index.isin(old.index)]])
    else:
        before = 0
        merged = fresh
    merged = merged.sort_index()
    merged.to_parquet(p)
    return len(merged) - before


def load(code: str, start: datetime | str | None = None,
         end: datetime | str | None = None,
         cache_path: pathlib.Path | None = None) -> pd.DataFrame:
    """Read the archive. It is never rebuilt from the live database.

    A range that reaches past what was archived comes back short rather than
    triggering a build: the snapshots behind it no longer exist, so a build
    would return less than the archive already holds.
    """
    p = cache_path or _cache_path(code)
    if not p.exists():
        return pd.DataFrame(columns=COLUMNS,
                            index=pd.DatetimeIndex([], name="ts"))
    df = pd.read_parquet(p)
    if start is not None:
        df = df[df.index >= pd.Timestamp(start)]
    if end is not None:
        df = df[df.index <= pd.Timestamp(end)]
    return df


def coverage(code: str, cache_path: pathlib.Path | None = None) -> pd.DataFrame:
    """Per-day row count and seconds covered, for spotting a missed run."""
    df = load(code, cache_path=cache_path)
    if df.empty:
        return pd.DataFrame(columns=["rows", "first", "last", "hours"])
    g = df.groupby(df.index.date).agg(rows=("mid", "size"))
    g["first"] = df.groupby(df.index.date).apply(
        lambda s: s.index.min().strftime("%H:%M:%S"), include_groups=False)
    g["last"] = df.groupby(df.index.date).apply(
        lambda s: s.index.max().strftime("%H:%M:%S"), include_groups=False)
    g["hours"] = (g.rows / 3600).round(2)
    return g


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
        total = 0
        for k in range(args.days):
            day = (today - pd.Timedelta(days=k)).date()
            added = append_day(code, day)
            total += added
            print(f"  {code} {day}: +{added:,} rows", flush=True)
        cov = coverage(code)
        print(f"{code}: added {total:,} rows, archive now "
              f"{len(cov)} days / {int(cov.rows.sum()) if len(cov) else 0:,} rows",
              flush=True)


if __name__ == "__main__":
    main()

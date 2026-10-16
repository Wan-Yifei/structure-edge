"""Per-minute signed volume, aggregated from ticks and cached.

The tick table holds 49M rows for SOXL alone, so a backtest that recomputes
delta per parameter combination spends all its time in SQLite. This collapses
the ticks to one row per minute once and keeps the result in a parquet file
beside the other caches.

Two limits that travel with every number this module produces:

**Coverage starts 2026-05-18.** Ticks older than that were pruned, so a
backtest wanting delta cannot run before then. SOXS was pruned harder still --
the 30-day retention exempts SOXL only -- and has two days in July and nothing
else, which is why this is SOXL-only in practice.

**Direction is known for about 43% of volume.** The rest prints NEUTRAL: it is
off-exchange flow rather than noise, and signing it against the midpoint is a
separate offline job (scripts/classify_ticks.py) that has only been run over a
few days. So `delta` here is exchange flow, a biased subset -- its slope is
more trustworthy than its level, and the level should never be compared across
days.
"""

from __future__ import annotations

import pathlib
import sqlite3
from datetime import datetime

import pandas as pd

_ROOT = pathlib.Path(__file__).parent.parent
_TICK_DB = _ROOT / "db" / "ticks.db"
_CACHE = _ROOT / "db" / "cache" / "delta_bars"

TICKS_START = pd.Timestamp("2026-05-18")   # anything earlier was pruned


def _cache_path(code: str) -> pathlib.Path:
    return _CACHE / f"{code.replace('.', '_')}_1m.parquet"


def build(code: str, start: datetime | str, end: datetime | str,
          db_path: pathlib.Path | None = None) -> pd.DataFrame:
    """Aggregate ticks to one row per minute: buy, sell, neutral, delta, n.

    Reads in day-sized chunks so a multi-month range never holds the whole
    tick table in memory.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    con = sqlite3.connect(f"file:{db_path or _TICK_DB}?mode=ro", uri=True)
    con.execute("PRAGMA busy_timeout=30000")
    out = []
    day = start.normalize()
    try:
        while day <= end:
            nxt = day + pd.Timedelta(days=1)
            rows = con.execute(
                "SELECT ts, volume, direction FROM ticks "
                "WHERE code = ? AND ts >= ? AND ts < ? ORDER BY ts",
                [code, day.isoformat(sep=" "), nxt.isoformat(sep=" ")]
            ).fetchall()
            day = nxt
            if not rows:
                continue
            d = pd.DataFrame(rows, columns=["ts", "volume", "direction"])
            # ticks.db keeps the feed's own timestamp string: some carry
            # milliseconds and some do not, so a single fixed format rejects
            # half of them.
            d["t"] = pd.to_datetime(d.ts, format="ISO8601").dt.floor("min")
            d["buy"] = d.volume.where(d.direction == "BUY", 0)
            d["sell"] = d.volume.where(d.direction == "SELL", 0)
            d["neutral"] = d.volume.where(
                ~d.direction.isin(["BUY", "SELL"]), 0)
            g = d.groupby("t").agg(buy=("buy", "sum"), sell=("sell", "sum"),
                                   neutral=("neutral", "sum"),
                                   n=("volume", "size"))
            out.append(g)
    finally:
        con.close()
    if not out:
        return pd.DataFrame(columns=["buy", "sell", "neutral", "n", "delta"])
    df = pd.concat(out).sort_index()
    df["delta"] = df.buy - df.sell
    return df


def load(code: str, start: datetime | str, end: datetime | str,
         refresh: bool = False) -> pd.DataFrame:
    """Cached `build`. Rebuilds whenever the request reaches outside the cache.

    The cache is whole-file rather than per-range: a partial cache that
    silently returned a short series would make a backtest look like it had
    data it did not.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    # A bare date means midnight, so `end="2026-10-14"` used to cut that whole
    # session's intraday bars out of the answer. Callers pass dates far more
    # often than timestamps, and the silent version of this bug is the worse
    # one: the caller gets a frame, just without the last day in it.
    if end == end.normalize():
        end = end + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
    p = _cache_path(code)
    cached = None
    if p.exists() and not refresh:
        cached = pd.read_parquet(p)
        if len(cached) and cached.index.min() <= start and cached.index.max() >= end:
            return cached.loc[start:end]

    # Build only what is missing and MERGE. Overwriting with the built range
    # shrank the cache to whatever the last caller asked for -- the nightly
    # holdout job runs with --days 3 and cut five months down to four days,
    # after which every other caller silently got a near-empty frame.
    if cached is not None and len(cached) and not refresh \
            and start >= cached.index.min():
        gap_start = cached.index.max()
        fresh = build(code, gap_start, end)
        df = pd.concat([cached, fresh])
        df = df[~df.index.duplicated(keep="last")].sort_index()
    else:
        df = build(code, start, end)
        if cached is not None and len(cached) and not refresh:
            df = pd.concat([cached, df])
            df = df[~df.index.duplicated(keep="last")].sort_index()
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p)
    return df.loc[start:end]


def coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Per-day bar count and the share of volume carrying a direction."""
    if df.empty:
        return pd.DataFrame()
    g = df.groupby(df.index.date).agg(
        bars=("n", "size"), ticks=("n", "sum"),
        buy=("buy", "sum"), sell=("sell", "sum"), neutral=("neutral", "sum"))
    tot = g.buy + g.sell + g.neutral
    g["directional_pct"] = (100 * (g.buy + g.sell) / tot.where(tot > 0)).round(1)
    return g

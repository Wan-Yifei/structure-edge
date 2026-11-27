"""Fetch historical klines from moomoo API and cache to DuckDB.

Handles pagination automatically (moomoo caps each call at ~960 bars).
On subsequent calls, returns cached data without reconnecting to moomoo.
"""

from __future__ import annotations

import time
import pathlib

import pandas as pd

from feeds.kline_store import KlineStore

_HOST = "127.0.0.1"
_PORT = 11111
_MAX_PER_CALL = 960
_RATE_LIMIT_S = 0.25   # sleep between paginated API calls

# String keys only — KLType resolved lazily when moomoo is actually needed
_KTYPE_KEYS = {"1m", "3m", "5m", "15m", "30m", "60m", "1d"}


def _ktype_map():
    """Return the moomoo KLType enum map (lazily imported to avoid import-time errors)."""
    from moomoo import KLType
    return {
        "1m":  KLType.K_1M,
        "3m":  KLType.K_3M,
        "5m":  KLType.K_5M,
        "15m": KLType.K_15M,
        "30m": KLType.K_30M,
        "60m": KLType.K_60M,
        "1d":  KLType.K_DAY,
    }

# Synthetic TFs built by resampling 60m data — not stored in DuckDB cache.
# key → pandas freq string passed to resample()
_SYNTH_TF: dict[str, str] = {
    "2h": "2h",
    "3h": "3h",
    "4h": "4h",
}


def _ktype_obj(ktype: str):
    """Convert a timeframe string to the corresponding moomoo KLType enum value."""
    m = _ktype_map()
    if ktype not in m:
        raise ValueError(f"Unknown ktype {ktype!r}. Choose from {sorted(m)}")
    return m[ktype]


def _resample(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Resample a 60m OHLCV DataFrame to a coarser frequency (e.g. '2h', '4h')."""
    import pandas as _pd
    tmp = df.copy()
    tmp["dt"] = _pd.to_datetime(tmp["time_key"])
    tmp = tmp.set_index("dt").sort_index()
    r = tmp.resample(freq, closed="right", label="right").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).dropna(subset=["open"])
    r = r.reset_index()
    r["time_key"] = r["dt"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return r[["time_key", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


def _describe_missing(days: list) -> str:
    """Short, readable summary of the days a cache is short of."""
    if len(days) <= 3:
        return ", ".join(str(d) for d in days)
    return f"{days[0]} … {days[-1]} ({len(days)} days)"


def fetch_klines(
    code: str,
    ktype: str,
    start: str,
    end: str,
    force_refresh: bool = False,
    db_path: str | pathlib.Path | None = None,
    live_refresh: bool = False,
) -> pd.DataFrame:
    """Return OHLCV DataFrame for the given code, timeframe, and date range.

    Data is cached in DuckDB. A cache hit avoids any moomoo connection.

    Args:
        code:          moomoo stock code, e.g. 'US.SNDK'
        ktype:         timeframe string: '1m','5m','15m','30m','60m','1d'
        start:         'YYYY-MM-DD'  (inclusive)
        end:           'YYYY-MM-DD'  (inclusive)
        force_refresh: re-fetch from API and overwrite cache
        db_path:       override the default DuckDB path
        live_refresh:  this call is a small foreground refresh on a timer,
                       not a one-off load. Two things change, and they are
                       one flag because setting either alone is a mistake:
                       the buffer pool is capped (see KlineStore), and the
                       coverage rows are left alone.

                       Skipping the coverage DELETE is what the flag is
                       really for. Measured on the 1 GB cache, a refresh
                       cycle costs 486 ms, of which close() is 344 -- and
                       close() on its own is 2.5 ms. It is the DELETE that
                       makes the checkpoint rewrite the file. Dropping it
                       takes the cycle to 103 ms.

                       It is also pointless on this path: force_refresh
                       already skips the missing_days check, so nothing
                       here reads coverage, and mark_covered re-inserts the
                       same rows a few lines later. The viewer was paying a
                       whole-database checkpoint to delete rows it was
                       about to write back.

                       Only valid together with force_refresh. Without it
                       the coverage rows are what decides whether the API
                       is called at all, and leaving a stale one in place
                       would pin the cache shut.
    """
    if live_refresh and not force_refresh:
        raise ValueError(
            "live_refresh is only meaningful with force_refresh: without it "
            "the coverage rows are what decides whether the API is called")
    # Synthetic TFs are derived on-the-fly from cached 60m data
    if ktype in _SYNTH_TF:
        df_60m = fetch_klines(code, "60m", start, end, force_refresh, db_path)
        resampled = _resample(df_60m, _SYNTH_TF[ktype])
        print(f"[fetcher] Resampled {code} {ktype} from 60m  ({len(resampled)} bars)")
        return resampled

    # 256MB is far more than three statements need and far less than the
    # default 80%-of-RAM pool, which is what made this call swing the
    # viewer's resident set by ~690 MB every 30 seconds.
    mem = "256MB" if live_refresh else None
    store = (KlineStore(db_path, memory_limit=mem) if db_path
             else KlineStore(memory_limit=mem))

    if force_refresh and not live_refresh:
        # The caller is overriding the cache, so forget what we think we know
        # about this range -- otherwise a span recorded from a fetch that came
        # back empty for the wrong reason could never be retried.
        #
        # Skipped for a live refresh, where it costs a whole-database
        # checkpoint to delete rows mark_covered writes straight back. See
        # the live_refresh argument.
        store.clear_coverage(code, ktype, start, end)
    else:
        missing = store.missing_days(code, ktype, start, end)
        if not missing:
            cached = store.load(code, ktype, start, end)
            if not cached.empty:
                print(f"[fetcher] Cache hit: {code} {ktype}  ({len(cached)} bars)")
                store.close()
                return cached
        elif store.has_data(code, ktype):
            print(
                f"[fetcher] Cache short: {code} {ktype}  "
                f"missing {_describe_missing(missing)} — re-fetching"
            )

    print(f"[fetcher] Fetching {code} {ktype} from {start} to {end} ...")
    frames: list[pd.DataFrame] = []

    from moomoo import OpenQuoteContext, RET_OK, AuType
    ctx = OpenQuoteContext(host=_HOST, port=_PORT)
    try:
        page_req_key = None
        while True:
            if page_req_key is None:
                ret, data, next_key = ctx.request_history_kline(
                    code,
                    start=start,
                    end=end,
                    ktype=_ktype_obj(ktype),
                    autype=AuType.QFQ,
                    max_count=_MAX_PER_CALL,
                    extended_time=True,
                    session="ALL",
                )
            else:
                ret, data, next_key = ctx.request_history_kline(
                    code,
                    ktype=_ktype_obj(ktype),
                    autype=AuType.QFQ,
                    max_count=_MAX_PER_CALL,
                    page_req_key=page_req_key,
                    extended_time=True,
                    session="ALL",
                )

            if ret != RET_OK:
                raise RuntimeError(f"request_history_kline failed: {data}")

            if not data.empty:
                frames.append(
                    data[["time_key", "open", "high", "low", "close", "volume"]].copy()
                )
                print(f"  page: +{len(data)} bars (total {sum(len(f) for f in frames)})")

            if next_key is None:
                break
            page_req_key = next_key
            time.sleep(_RATE_LIMIT_S)
    finally:
        ctx.close()

    if not frames:
        print("[fetcher] No data returned from API.")
        # The request completed, so the market was shut for all of it. Record
        # that or every later call repeats the round trip.
        store.mark_covered(code, ktype, start, end)
        store.close()
        return pd.DataFrame(columns=["time_key", "open", "high", "low", "close", "volume"])

    df = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates("time_key")
        .sort_values("time_key")
        .reset_index(drop=True)
    )
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype(float)
    df["volume"] = df["volume"].astype("int64")

    print(f"[fetcher] Saving {len(df)} bars to cache ...")
    store.save(code, ktype, df)
    store.mark_covered(code, ktype, start, end)
    store.close()

    return df[
        (df["time_key"] >= start) & (df["time_key"] <= end + " 23:59:59")
    ].reset_index(drop=True)

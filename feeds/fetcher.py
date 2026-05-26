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


def fetch_klines(
    code: str,
    ktype: str,
    start: str,
    end: str,
    force_refresh: bool = False,
    db_path: str | pathlib.Path | None = None,
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
    """
    # Synthetic TFs are derived on-the-fly from cached 60m data
    if ktype in _SYNTH_TF:
        df_60m = fetch_klines(code, "60m", start, end, force_refresh, db_path)
        resampled = _resample(df_60m, _SYNTH_TF[ktype])
        print(f"[fetcher] Resampled {code} {ktype} from 60m  ({len(resampled)} bars)")
        return resampled

    store = KlineStore(db_path) if db_path else KlineStore()

    if not force_refresh and store.has_data(code, ktype):
        cached = store.load(code, ktype, start, end)
        if not cached.empty:
            print(f"[fetcher] Cache hit: {code} {ktype}  ({len(cached)} bars)")
            store.close()
            return cached

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
                )
            else:
                ret, data, next_key = ctx.request_history_kline(
                    code,
                    ktype=_ktype_obj(ktype),
                    autype=AuType.QFQ,
                    max_count=_MAX_PER_CALL,
                    page_req_key=page_req_key,
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
    store.close()

    return df[
        (df["time_key"] >= start) & (df["time_key"] <= end + " 23:59:59")
    ].reset_index(drop=True)

"""The tick bucketing that SQL replaced, kept as the thing it is checked against.

`_query_ticks` -> `_size_thresholds` -> `_bucket_ticks` used to live in
analysis/trade_viewer_qt.py. They read every row of the tape into Python --
1.33M dicts for one SOXL day, ~861 MB of them -- only to sum them and throw
them away. `_query_tick_bins` now does the same summing inside SQLite.

They are copied here verbatim rather than deleted outright because the new
code's only real specification is "what the old code did", and an assertion
written by hand about what it did is a restatement, not the thing itself.
Tests drive both over the same database and compare every cell.

Nothing in the viewer imports this. If the production behaviour is meant to
change, this file changes with it and the change shows up as a test edit.
"""
from __future__ import annotations

import pathlib
import sqlite3
from collections import defaultdict
from datetime import datetime

import numpy as np

import analysis.trade_viewer_qt as tv
from core.time_utils import candle_start


# ── the reference implementation ──────────────────────────────────────────
def ref_rows(db_path: pathlib.Path, code: str, start: datetime,
             end: datetime, with_offex: bool) -> list[dict]:
    """What TickStore.query_ticks handed back, one dict per print."""
    from feeds.tick_store import TickStore
    with TickStore(db_path, read_only=True) as store:
        return store.query_ticks(code, start, end, with_mid=with_offex)


def ref_thresholds(rows) -> tuple[float, float]:
    """The day's own p33 and p67 by volume; (inf, inf) under six prints."""
    vols = np.array([r["volume"] for r in rows if r["volume"] > 0],
                    dtype=float)
    if len(vols) < 6:
        return float("inf"), float("inf")
    return float(np.percentile(vols, 33)), float(np.percentile(vols, 67))


def ref_bucket(rows, candle_mins: int,
               thresh_m: float, thresh_l: float) -> dict:
    """{bucket: {price: bin}} with GIVEN thresholds."""
    def _size(v) -> str:
        if v <= thresh_m:
            return "s"
        if v <= thresh_l:
            return "m"
        return "l"

    buckets: dict = defaultdict(lambda: defaultdict(tv._new_tick_bin))
    for r in rows:
        ts = (r["ts"] if isinstance(r["ts"], datetime)
              else datetime.fromisoformat(str(r["ts"])))
        bucket = candle_start(ts, candle_mins)
        key = {"BUY": "buy", "SELL": "sell"}.get(
            r["direction"].upper(), "neutral")
        vol = r["volume"]
        buckets[bucket][r["price"]][key] += vol
        buckets[bucket][r["price"]][f"{key}_{_size(vol)}"] += vol
        if key == "neutral":
            md = r.get("mid_dir")
            if md == "BUY":
                buckets[bucket][r["price"]]["off_buy"] += vol
            elif md == "SELL":
                buckets[bucket][r["price"]]["off_sell"] += vol
    return dict(buckets)


def ref_load(db_path: pathlib.Path, code: str, start: datetime,
             end: datetime, with_offex: bool, candle_mins: int):
    """(buckets, thresholds, last_ts) the way the old chain produced them."""
    rows = ref_rows(db_path, code, start, end, with_offex)
    if not rows:
        return None
    tm, tl = ref_thresholds(rows)
    last = max((r["ts"] if isinstance(r["ts"], datetime)
                else datetime.fromisoformat(str(r["ts"]))) for r in rows)
    return ref_bucket(rows, candle_mins, tm, tl), (tm, tl), last


# ── a database to drive both over ─────────────────────────────────────────
def build_db(tmp_path: pathlib.Path, ticks, mids=()) -> pathlib.Path:
    """Write a ticks.db (and tick_mid.db beside it) from plain tuples.

    ticks: (code, ts, price, volume, direction) -- ts a datetime or a string
    mids:  (code, ts, price, volume, mid_dir)

    The schemas come from the production stores, not from a copy of their
    DDL here, so a column added there is a column these tests get too.
    """
    db = tmp_path / "db" / "ticks.db"
    db.parent.mkdir(parents=True, exist_ok=True)

    from feeds.tick_store import TickStore
    with TickStore(db):                        # creates the table
        pass

    con = sqlite3.connect(str(db))
    con.executemany(
        "INSERT OR REPLACE INTO ticks (code, ts, price, volume, direction)"
        " VALUES (?, ?, ?, ?, ?)",
        [(c, t.isoformat(sep=" ") if isinstance(t, datetime) else str(t),
          float(p), int(v), d) for c, t, p, v, d in ticks])
    con.commit()
    con.close()

    if mids:
        from feeds.tick_mid_store import TickMidStore
        with TickMidStore(db.parent / "tick_mid.db") as ms:
            ms.put(mids)
    return db


def cells_equal(got: dict, ref: dict) -> list[str]:
    """Every difference between two {bucket: {price: bin}} mappings."""
    out = []
    for bk in set(got) | set(ref):
        if bk not in got:
            out.append(f"bucket missing from new: {bk}")
            continue
        if bk not in ref:
            out.append(f"bucket only in new: {bk}")
            continue
        pn, pr = got[bk], ref[bk]
        for p in set(pn) | set(pr):
            if p not in pn:
                out.append(f"level missing from new: {bk} {p}")
                continue
            if p not in pr:
                out.append(f"level only in new: {bk} {p}")
                continue
            if set(pn[p]) != set(tv._BIN_KEYS):
                out.append(f"bin keys wrong at {bk} {p}: {sorted(pn[p])}")
                continue
            for k in tv._BIN_KEYS:
                if int(pn[p][k]) != int(pr[p][k]):
                    out.append(f"{bk} {p} {k}: new={pn[p][k]} ref={pr[p][k]}")
    return out

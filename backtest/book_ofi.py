"""Multi-level order flow imbalance from the archived book, and how deep to go.

Everything in this project so far measures flow on the TRADE side: delta_r
is (buy - sell) / (buy + sell) over prints the feed tagged. The book side
-- what was added to and pulled from each price level -- has never been
computed, and it is the half the literature actually calls OFI.

The question here is not "does it make money". It is the narrower one that
can be answered with five days: **how many levels are worth including**.
Cont, Cucuringu & Zhang report that in-sample fit rises monotonically with
depth while out-of-sample fit turns over past some level -- the deeper
levels carry redundant information, so adding them buys overfitting. Their
follow-up integrates the top levels into one variable instead. Both of
those are reproducible here, and the answer is a property of the data, not
of a strategy, so it does not need a hundred sessions.

OFI at level m between consecutive snapshots, the standard construction:

    e_m = 1{P^b_m(t) >= P^b_m(t-1)} q^b_m(t) - 1{P^b_m(t) <= P^b_m(t-1)} q^b_m(t-1)
        - 1{P^a_m(t) <= P^a_m(t-1)} q^a_m(t) + 1{P^a_m(t) >= P^a_m(t-1)} q^a_m(t-1)

Equal prices give the size difference, a more aggressive price gives the
new size, a retreating one gives up the old size. Summed over a bucket and
scaled by the average depth over that bucket, which is what makes the
coefficient comparable across days and levels.

    uv run backtest/book_ofi.py
    uv run backtest/book_ofi.py --bucket 60 --levels 10

RESULTS, first run (2026-10-20): SOXL, 5 full sessions 10-13..10-19,
10-second buckets, RTH only, leave-one-day-out.

Contemporaneous (OFI against the move it happens with):

      levels   1      2      3      5      7     10     12     20     30
      OOS R²  .303   .395   .432   .458   .469   .472   .472   .470   .476
      PC1 over 30 levels: .4795

    The curve is flat from level 7 on: everything between 7 and 30 sits in
    .469-.476 against a fold sd of ~.026, so they are one number. Seven
    levels is where it arrives, and the first principal component of the
    thirty does marginally better than any of them with ONE regressor.
    Level 1 alone gives .303 -- the inside of the book is well under half
    the story, which is the result that justifies building this at all.

Next interval (OFI against the move that follows it):

      levels   1      2      5     10     20     30
      IS  R²  .0003  .0005  .0012  .0016  .0025  .0038
      OOS R² -.0005 -.0005 -.0002 -.0015 -.0030 -.0048
      PC1 over 30 levels: +.0001

    In-sample rises twelvefold with depth; out-of-sample falls monotonically
    and is negative everywhere. This is the Cont-Cucuringu-Zhang overfitting
    result reproduced exactly: deeper levels carry redundant information and
    buy nothing but variance. Fold sd grows with depth too, .0018 to .0064.

    Read together, the two tables say OFI is a measure of impact, not a
    forecast. It explains 47% of the move it is part of and none of the next
    one. Nothing here is tradeable on its own, and five sessions could not
    establish that it was even if the sign had gone the other way -- this
    answers the construction question (how deep, what form), which is a
    property of the data, and leaves the incremental-value question to a
    sample that can carry a session bootstrap.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import duckdb
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).parent.parent
LADDER = ROOT / "db" / "cache" / "book_ladder"

# Regular session only for a first pass. The overnight book is thin and its
# own thing; mixing it in would let one regime speak for the other.
RTH_START, RTH_END = "09:30:00", "16:00:00"


def _levels_sql(path: str, levels: int) -> str:
    """Top-`levels` rows per side per snapshot, pivoted side by side.

    row_number with a flipped sort key on the bid side: level 1 is the
    highest bid and the lowest ask, i.e. the inside of the book.
    """
    return f"""
    WITH r AS (
        SELECT ts, side, price, size,
               row_number() OVER (
                   PARTITION BY ts, side
                   ORDER BY CASE WHEN side = 'A' THEN price ELSE -price END
               ) AS lvl
        FROM read_parquet('{path}')
        WHERE CAST(ts AS TIME) >= TIME '{RTH_START}'
          AND CAST(ts AS TIME) <  TIME '{RTH_END}'
    )
    SELECT ts, lvl,
           max(CASE WHEN side = 'B' THEN price END) AS bp,
           max(CASE WHEN side = 'B' THEN size  END) AS bq,
           max(CASE WHEN side = 'A' THEN price END) AS ap,
           max(CASE WHEN side = 'A' THEN size  END) AS aq
    FROM r
    WHERE lvl <= {levels}
    GROUP BY ts, lvl
    ORDER BY ts, lvl
    """


def load_day(path: pathlib.Path, levels: int):
    """(ts, bp, bq, ap, aq) with the four arrays shaped (snapshots, levels)."""
    df = duckdb.sql(_levels_sql(path.as_posix(), levels)).df()
    if df.empty:
        return None
    wide = {}
    for col in ("bp", "bq", "ap", "aq"):
        w = df.pivot(index="ts", columns="lvl", values=col)
        wide[col] = w.reindex(columns=range(1, levels + 1)).to_numpy(float)
    ts = pd.to_datetime(df["ts"].drop_duplicates().sort_values().to_numpy())
    return ts, wide["bp"], wide["bq"], wide["ap"], wide["aq"]


def ofi_per_snapshot(bp, bq, ap, aq):
    """e_m between each pair of consecutive snapshots. Shape (n-1, levels)."""
    bp0, bp1 = bp[:-1], bp[1:]
    bq0, bq1 = bq[:-1], bq[1:]
    ap0, ap1 = ap[:-1], ap[1:]
    aq0, aq1 = aq[:-1], aq[1:]
    e = (np.where(bp1 >= bp0, bq1, 0.0) - np.where(bp1 <= bp0, bq0, 0.0)
         - np.where(ap1 <= ap0, aq1, 0.0) + np.where(ap1 >= ap0, aq0, 0.0))
    # A missing level on either side makes its own term undefined, not zero.
    bad = ~np.isfinite(bp0) | ~np.isfinite(bp1) | ~np.isfinite(ap0) | \
          ~np.isfinite(ap1)
    e[bad] = np.nan
    return e


def bucket(ts, e, bq, aq, bp, ap, seconds: int):
    """Aggregate to fixed intervals.

    Returns the summed OFI per level, scaled by the average depth over the
    interval, plus the mid return across it in basis points.
    """
    t = pd.Series(ts[1:])                       # e is between t-1 and t
    key = t.dt.floor(f"{seconds}s")
    mid = (bp[:, 0] + ap[:, 0]) / 2.0

    g = pd.DataFrame({"k": key.to_numpy()})
    out_ofi, levels = [], e.shape[1]
    for m in range(levels):
        out_ofi.append(g.assign(v=e[:, m]).groupby("k")["v"].sum(min_count=1))
    ofi = pd.concat(out_ofi, axis=1)
    ofi.columns = [f"ofi{m+1}" for m in range(levels)]

    # Average depth per level over the interval. Kept per level rather than
    # collapsed here: the scale Cont, Cucuringu & Zhang use is the mean
    # depth over the M levels ACTUALLY IN THE MODEL, so it has to be
    # recomputed for each depth rather than fixed at whatever this run
    # happened to load. Collapsing it here made the M=1 regressor depend on
    # --levels, which is a yardstick changing under the measurement.
    for m in range(levels):
        d = (bq[1:, m] + aq[1:, m]) / 2.0
        ofi[f"q{m+1}"] = g.assign(v=d).groupby("k")["v"].mean()

    first = pd.Series(mid[1:]).groupby(key.to_numpy()).first()
    last = pd.Series(mid[1:]).groupby(key.to_numpy()).last()
    ofi["ret"] = (last - first) / first * 1e4          # basis points
    ofi["n"] = g.groupby("k").size()
    return ofi.dropna(subset=["ret"])


def ols(X, y):
    """Least squares with an intercept; returns the coefficient vector."""
    A = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return beta


def r2(y, pred, base):
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - base) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def design(df: pd.DataFrame, M: int) -> np.ndarray:
    """The M scaled OFI columns, divided by the depth of those same M."""
    raw = df[[f"ofi{m+1}" for m in range(M)]].to_numpy(float)
    q = np.nanmean(df[[f"q{m+1}" for m in range(M)]].to_numpy(float), axis=1)
    q = np.where(q > 0, q, np.nan)
    return raw / q[:, None]


def evaluate(data: dict[str, pd.DataFrame], levels: int, horizon: int):
    """In-sample and leave-one-day-out R^2 for depth 1..levels.

    The fold is a whole DAY. Splitting inside a day would put neighbouring
    intervals on both sides of the split, and the book is autocorrelated
    over far more than one interval -- the test set would be reading the
    training set's mind.
    """
    days = sorted(data)
    rows = []
    for M in range(1, levels + 1):
        ins, oos = [], []
        # In-sample: one fit over everything, the number that always rises.
        allX = np.vstack([design(data[d], M) for d in days])
        ally = np.concatenate([_target(data[d], horizon) for d in days])
        ok = np.isfinite(allX).all(1) & np.isfinite(ally)
        b = ols(allX[ok], ally[ok])
        pred = b[0] + allX[ok] @ b[1:]
        ins = r2(ally[ok], pred, ally[ok].mean())

        for held in days:
            tr = [d for d in days if d != held]
            Xtr = np.vstack([design(data[d], M) for d in tr])
            ytr = np.concatenate([_target(data[d], horizon) for d in tr])
            mtr = np.isfinite(Xtr).all(1) & np.isfinite(ytr)
            b = ols(Xtr[mtr], ytr[mtr])
            Xte = design(data[held], M)
            yte = _target(data[held], horizon)
            mte = np.isfinite(Xte).all(1) & np.isfinite(yte)
            pred = b[0] + Xte[mte] @ b[1:]
            # Baseline is the TRAINING mean: predicting the test set's own
            # mean is not available out of sample.
            oos.append(r2(yte[mte], pred, ytr[mtr].mean()))
        rows.append((M, ins, float(np.mean(oos)), float(np.std(oos))))
    return rows


def _target(df: pd.DataFrame, horizon: int) -> np.ndarray:
    """horizon 0 = contemporaneous impact, 1 = the next interval."""
    y = df["ret"].to_numpy()
    if horizon == 0:
        return y
    out = np.full(len(y), np.nan)
    out[:-horizon] = y[horizon:]
    return out


def integrated(data: dict[str, pd.DataFrame], levels: int, horizon: int):
    """The follow-up's suggestion: collapse the levels into one variable.

    First principal component of the multi-level OFI vector, fitted on the
    training days only -- fitting it on everything would leak the test day
    into the very definition of the predictor.
    """
    days = sorted(data)
    oos = []
    for held in days:
        tr = [d for d in days if d != held]
        Xtr = np.vstack([design(data[d], levels) for d in tr])
        ytr = np.concatenate([_target(data[d], horizon) for d in tr])
        m = np.isfinite(Xtr).all(1) & np.isfinite(ytr)
        Xtr, ytr = Xtr[m], ytr[m]
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd[sd == 0] = 1.0
        Z = (Xtr - mu) / sd
        _, _, vt = np.linalg.svd(Z, full_matrices=False)
        w = vt[0]
        b = ols((Z @ w).reshape(-1, 1), ytr)

        Xte = design(data[held], levels)
        yte = _target(data[held], horizon)
        mt = np.isfinite(Xte).all(1) & np.isfinite(yte)
        zte = ((Xte[mt] - mu) / sd) @ w
        oos.append(r2(yte[mt], b[0] + b[1] * zte, ytr.mean()))
    return float(np.mean(oos)), float(np.std(oos))


def _pc1(Xtr):
    """First principal component of the training design, with its scaler."""
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1.0
    _, _, vt = np.linalg.svd((Xtr - mu) / sd, full_matrices=False)
    return mu, sd, vt[0]


def _binned(xtr, ytr, xte, bins=10):
    """Non-parametric fit: predict each test point the training mean of its
    decile. Edges from the training half only."""
    edges = np.quantile(xtr, np.linspace(0, 1, bins + 1)[1:-1])
    itr = np.digitize(xtr, edges)
    means = np.array([ytr[itr == b].mean() if (itr == b).any() else ytr.mean()
                      for b in range(bins)])
    return means[np.digitize(xte, edges)]


def shape_test(data: dict[str, pd.DataFrame], levels: int, horizon: int):
    """Leave-one-day-out R^2 for five functional forms at a fixed depth."""
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
    except ImportError:
        HistGradientBoostingRegressor = None

    days = sorted(data)
    names = ["线性", "signed sqrt", "signed log", "分箱（非参）", "GBM（多档）"]
    acc = {n: [] for n in names}
    for held in days:
        tr = [d for d in days if d != held]
        Xtr = np.vstack([design(data[d], levels) for d in tr])
        ytr = np.concatenate([_target(data[d], horizon) for d in tr])
        m = np.isfinite(Xtr).all(1) & np.isfinite(ytr)
        Xtr, ytr = Xtr[m], ytr[m]
        Xte = design(data[held], levels)
        yte = _target(data[held], horizon)
        mt = np.isfinite(Xte).all(1) & np.isfinite(yte)
        Xte, yte = Xte[mt], yte[mt]
        base = ytr.mean()

        mu, sd, w = _pc1(Xtr)
        ztr = ((Xtr - mu) / sd) @ w
        zte = ((Xte - mu) / sd) @ w

        for nm, f in (("线性", lambda v: v),
                      ("signed sqrt", lambda v: np.sign(v) * np.sqrt(np.abs(v))),
                      ("signed log", lambda v: np.sign(v) * np.log1p(np.abs(v)))):
            b = ols(f(ztr).reshape(-1, 1), ytr)
            acc[nm].append(r2(yte, b[0] + b[1] * f(zte), base))

        acc["分箱（非参）"].append(r2(yte, _binned(ztr, ytr, zte), base))

        if HistGradientBoostingRegressor is not None:
            g = HistGradientBoostingRegressor(
                max_depth=3, max_iter=200, learning_rate=0.05,
                random_state=0).fit(Xtr, ytr)
            acc["GBM（多档）"].append(r2(yte, g.predict(Xte), base))
        else:
            acc["GBM（多档）"].append(float("nan"))
    return [(n, float(np.mean(acc[n])), float(np.std(acc[n]))) for n in names]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--code", default="US_SOXL")
    ap.add_argument("--levels", type=int, default=10)
    ap.add_argument("--bucket", type=int, default=10, help="seconds")
    ap.add_argument("--min-snapshots", type=int, default=15000,
                    help="skip partial days below this many RTH snapshots")
    ap.add_argument("--shape", type=int, default=0, metavar="LEVELS",
                    help="also test non-linear forms at this depth (0 = off)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    d = LADDER / args.code
    files = sorted(d.glob("*.parquet"))
    if not files:
        raise SystemExit(f"no ladder archive under {d}")

    print(f"{args.code}  levels<={args.levels}  bucket={args.bucket}s  "
          f"RTH {RTH_START}-{RTH_END}\n", flush=True)

    data = {}
    for f in files:
        got = load_day(f, args.levels)
        if got is None:
            print(f"  {f.stem}: 无盘中数据，跳过", flush=True)
            continue
        ts, bp, bq, ap_, aq = got
        if len(ts) < args.min_snapshots:
            print(f"  {f.stem}: 盘中快照 {len(ts):,} < "
                  f"{args.min_snapshots:,}，当作不完整跳过", flush=True)
            continue
        e = ofi_per_snapshot(bp, bq, ap_, aq)
        df = bucket(ts, e, bq, aq, bp, ap_, args.bucket)
        data[f.stem] = df
        print(f"  {f.stem}: 快照 {len(ts):,} → 区间 {len(df):,}  "
              f"收益 sd {df['ret'].std():.2f} bp", flush=True)

    if len(data) < 3:
        raise SystemExit("少于 3 个完整交易日，留一法没有意义")

    for horizon, name in ((0, "同期（价格冲击）"), (1, "下一区间（预测）")):
        print(f"\n=== {name} ===", flush=True)
        print(f"{'档数':>5}{'样本内 R²':>12}{'留一 R²':>12}{'折间 sd':>10}",
              flush=True)
        rows = evaluate(data, args.levels, horizon)
        best = max(rows, key=lambda r: r[2])
        for M, ins, oo, sd in rows:
            mark = "  <<< 最优" if M == best[0] else ""
            print(f"{M:>5}{ins:>12.4f}{oo:>12.4f}{sd:>10.4f}{mark}",
                  flush=True)
        pc, pcsd = integrated(data, args.levels, horizon)
        print(f"{'整合':>5}{'':>12}{pc:>12.4f}{pcsd:>10.4f}"
              f"   （前 {args.levels} 档的第一主成分）", flush=True)

        if args.shape:
            print(f"\n  形状检验（整合变量，{args.shape} 档；GBM 用原始多档）",
                  flush=True)
            print(f"  {'形式':<16}{'留一 R²':>12}{'折间 sd':>10}", flush=True)
            for nm, mu_, sd_ in shape_test(data, args.shape, horizon):
                print(f"  {nm:<16}{mu_:>12.4f}{sd_:>10.4f}", flush=True)


if __name__ == "__main__":
    main()

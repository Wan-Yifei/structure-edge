"""Standing book pressure: the resting size, and the fate of what leaves.

backtest/book_ofi.py measured the CHANGE in resting size (OFI) and found
it explains 47% of the move it happens with and nothing of the next one:
"a measure of impact, not a forecast". This file measures the two
quantities that study did not, to see whether the same shape holds.

QUEUE IMBALANCE -- the stock rather than the flow:

    QI_N = (bid size over the N levels nearest the touch
            - the same on the ask) / their total

FATE OF REMOVED SIZE -- generalising the per-wall attribution that
analysis/liq_hm_window.py's _wall_fate draws on the heatmap. Size that
leaves a level either traded away or was withdrawn, and a snapshot shows
only the sum; matching the loss against what printed AT that price, in
the direction that could consume it, separates them:

    removed(p)  sum over consecutive snapshots of max(0, prev - now)
    eaten(p)    min(removed, volume printed at p in the consuming
                direction during that minute) -- SELLs into a bid, BUYs
                into an ask
    pulled      removed - eaten

    pulled_frac  pulled / removed, both sides together
    eaten_imb    (bid eaten - ask eaten) / their total
    pulled_imb   the same for the withdrawn share

An ESTIMATE, not an observation, and the per-wall version says so for
the same reasons: a level refilled and hit again inside one minute nets
out, a print cannot be tied to the order it hit without per-order data,
and a level that appears and fills between two snapshots is invisible.
Every one of those biases toward calling size "pulled", so the LEVEL of
pulled_frac means little -- its variation across minutes is what is
being tested. NEUTRAL volume is excluded throughout: off-exchange prints
consume no displayed size, and counting them would read withdrawal as
consumption.

RESULTS, first run (2026-10-24): SOXL, the 10 archived sessions
2026-10-12..10-23, 3,690 RTH minutes, leave-one-day-out.

    depth  measure        now corr   now OOS R²   next corr   next OOS R²
        1  qi               +0.046      -0.0021      +0.011       -0.0034
        5  qi               -0.167      +0.0256      -0.016       -0.0043
       20  qi               -0.326      +0.0648      -0.016       -0.0041
       60  qi               -0.209      +0.0360      +0.003       -0.0042
        5  eaten_imb        -0.400      +0.1490      -0.009       -0.0043
       20  eaten_imb        -0.385      +0.1331      -0.008       -0.0043
       60  eaten_imb        -0.376      +0.1290      -0.005       -0.0042
        5  pulled_imb       -0.163      +0.0235      -0.013       -0.0040
       20  pulled_imb       -0.220      +0.0139      -0.032       -0.0039
       60  pulled_imb       -0.040      -0.0023      -0.015       -0.0035
     5-60  pulled_frac   +0.03..+0.06   all negative   ~0          all negative

  Three things in that table.

  Contemporaneously the fate of removed size beats everything else here:
  eaten_imb at 14.9% against queue imbalance's 6.5%. Its sign is
  negative, which is the arithmetic rather than a finding -- a minute
  where bid-side size is eaten is a minute price fell.

  Queue imbalance is also negative, and that one is worth a second look:
  a thicker resting bid goes with a DOWN minute, the opposite of reading
  depth as demand. Depth carries it -- the touch alone is +0.046,
  essentially nothing -- but it peaks at 20 levels and is weaker at 60.

  Which half of the split carries the information is clear: eaten_imb
  does, pulled_frac does not (negative out-of-sample at every depth). It
  matters how much was consumed and on which side, not what fraction was
  withdrawn. Consistent with how little is left to consume out deep: the
  median pulled share runs 72% at 5 levels, 87% at 20 and 96% at 60.

  Forecasting: nothing, anywhere. Thirteen variable/depth combinations,
  next-minute correlations all inside |0.03|, and the out-of-sample R² is
  negative in every one. Together with OFI that is four independent
  constructions of book pressure whose contemporaneous fit ranges from 6%
  to 47% and whose forward fit is uniformly zero.

  Ten sessions cannot prove a null. What they bound is the size of what
  could be hiding: the day-level bootstrap puts the next-minute
  high-minus-low difference for QI20 at -0.033 ATR, 95% [-0.076, +0.043].
  At a 1m ATR near $0.47 the undetected residue is under a cent a minute
  against a 3-cent spread, so the question the archive cannot settle is
  one whose answer would not pay either way. Re-running this on a longer
  archive halves that interval; it does not change that arithmetic.

    uv run backtest/book_pressure.py
    uv run backtest/book_pressure.py --measure qi --depths 1,5,20
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import duckdb
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

LADDER = ROOT / "db" / "cache" / "book_ladder"
TICKS = ROOT / "db" / "ticks.db"

# Regular session only, matching book_ofi.py: the overnight book is thin
# and its own thing, and mixing it in lets one regime speak for the other.
RTH_LO, RTH_HI = 9.5, 16.0


def _ladder_glob(code: str) -> str:
    return str(LADDER / code.replace(".", "_") / "*.parquet").replace("\\", "/")


def _rank_cte(code: str) -> str:
    """Levels numbered outward from the touch, per snapshot per side."""
    return f"""
    select ts, side, price, size,
           row_number() over (
             partition by ts, side
             order by case when side='B' then -price else price end
           ) as rk
    from read_parquet('{_ladder_glob(code)}')
    """


def queue_imbalance(code: str, depths: tuple[int, ...]) -> pd.DataFrame:
    """Mean QI per minute at each depth."""
    sel = ", ".join(
        f"sum(case when rk <= {n} and side='B' then size else 0 end) as b{n}, "
        f"sum(case when rk <= {n} and side='A' then size else 0 end) as a{n}"
        for n in depths)
    snap = duckdb.sql(f"""
        with ranked as ({_rank_cte(code)})
        select ts, {sel} from ranked group by ts order by ts
    """).df()
    out = pd.DataFrame({"m": pd.to_datetime(snap["ts"]).dt.floor("1min")})
    for n in depths:
        tot = snap[f"b{n}"] + snap[f"a{n}"]
        out[f"qi{n}"] = np.where(tot > 0,
                                 (snap[f"b{n}"] - snap[f"a{n}"]) / tot, np.nan)
    return out.groupby("m").mean()


def _trades(code: str) -> pd.DataFrame:
    """Signed volume per price per minute. NEUTRAL is excluded -- an
    off-exchange print consumes no displayed size."""
    con = duckdb.connect()
    con.execute("install sqlite; load sqlite;")
    con.execute(f"attach '{str(TICKS).replace(chr(92), '/')}' as t (type sqlite)")
    df = con.execute("""
        select date_trunc('minute', cast(ts as timestamp)) as m,
               cast(price as double) as price,
               upper(direction) as dir,
               sum(cast(volume as double)) as vol
        from t.ticks
        where code = ? and upper(direction) in ('BUY','SELL')
        group by 1,2,3
    """, [code]).df()
    con.close()
    return df


def removed_fate(code: str, depth: int, tr: pd.DataFrame) -> pd.DataFrame:
    """Per minute: how much size left each side, and how it left."""
    rm = duckdb.sql(f"""
        with ranked as ({_rank_cte(code)}),
        near as (select * from ranked where rk <= {depth}),
        diffs as (
          select side, price, date_trunc('minute', ts) as m,
                 greatest(0, lag(size) over (partition by side, price
                                             order by ts) - size) as gone
          from near
        )
        select m, side, price, sum(gone) as removed
        from diffs where gone > 0 group by m, side, price
    """).df()
    # A bid is consumed by SELLs, an ask by BUYs.
    rm["dir"] = np.where(rm["side"] == "B", "SELL", "BUY")
    j = rm.merge(tr, on=["m", "price", "dir"], how="left")
    j["vol"] = j["vol"].fillna(0.0)
    j["eaten"] = np.minimum(j["removed"], j["vol"])
    j["pulled"] = j["removed"] - j["eaten"]
    w = j.groupby(["m", "side"])[["removed", "eaten", "pulled"]].sum().unstack(
        "side")
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    return w.fillna(0.0)


def minute_returns(code: str, lo: str, hi: str) -> pd.DataFrame:
    from feeds.fetcher import fetch_klines

    k = fetch_klines(code, "1m", lo, hi)
    k["t"] = pd.to_datetime(k.time_key)
    k = k.sort_values("t").drop_duplicates("t").set_index("t")
    hh = k.index.hour + k.index.minute / 60
    k = k[(hh > RTH_LO) & (hh <= RTH_HI)]
    tr = np.maximum(k.high - k.low,
                    np.maximum((k.high - k.close.shift()).abs(),
                               (k.low - k.close.shift()).abs()))
    k["atr"] = tr.rolling(14, min_periods=5).mean()
    k["ret"] = (k.close - k.close.shift()) / k["atr"]
    k["fwd"] = k["ret"].shift(-1)
    return k[["atr", "ret", "fwd"]]


def loo_r2(x: np.ndarray, y: np.ndarray, day: np.ndarray) -> float:
    """Leave-one-day-out R², the yardstick book_ofi.py used.

    A single regressor through the origin, fitted on the other days. Days
    rather than rows because minutes inside a session are anything but
    independent, and an i.i.d. split would report a fit that no out-of-
    sample day could reproduce.
    """
    out = []
    for d in sorted(set(day)):
        te = day == d
        trn = ~te
        if te.sum() < 20 or trn.sum() < 100:
            continue
        xt, yt = x[trn], y[trn]
        den = float((xt * xt).sum())
        b = float((xt * yt).sum() / den) if den else 0.0
        ss = float(((y[te] - b * x[te]) ** 2).sum())
        tt = float(((y[te] - y[te].mean()) ** 2).sum())
        out.append(1 - ss / tt if tt else np.nan)
    return float(np.nanmean(out)) if out else float("nan")


def day_bootstrap(x: np.ndarray, y: np.ndarray, day: np.ndarray,
                  reps: int = 4000, seed: int = 11) -> tuple:
    """High-minus-low in y, resampling DAYS. Ten sessions make for a wide
    interval, which is the point of showing it."""
    rng = np.random.default_rng(seed)
    days = sorted(set(day))
    cut = float(np.median(x))
    obs = float(y[x > cut].mean() - y[x <= cut].mean())
    stats = []
    for _ in range(reps):
        pick = rng.choice(len(days), len(days), replace=True)
        idx = np.concatenate([np.flatnonzero(day == days[i]) for i in pick])
        xs, ys = x[idx], y[idx]
        c = float(np.median(xs))
        hi, lo = xs > c, xs <= c
        if hi.sum() < 5 or lo.sum() < 5:
            continue
        stats.append(ys[hi].mean() - ys[lo].mean())
    if not stats:
        return obs, float("nan"), float("nan")
    return obs, float(np.percentile(stats, 2.5)), float(
        np.percentile(stats, 97.5))


def evaluate(name: str, depth: int, x: pd.Series, k: pd.DataFrame,
             boot: bool) -> None:
    d = k.join(x.rename("x"), how="inner").dropna(subset=["atr", "x"])
    day = np.array([t.date() for t in d.index])
    xv = d["x"].to_numpy(float)
    for tag, col in (("now", "ret"), ("next", "fwd")):
        y = d[col].to_numpy(float)
        m = np.isfinite(xv) & np.isfinite(y)
        if m.sum() < 100:
            continue
        r = float(np.corrcoef(xv[m], y[m])[0, 1])
        r2 = loo_r2(xv[m], y[m], day[m])
        print(f"{depth:>6}  {name:<12} {tag:<5}  corr {r:>+7.3f}"
              f"   OOS R² {r2:>+8.4f}", end="")
        if boot and tag == "next":
            o, lo, hi = day_bootstrap(xv[m], y[m], day[m])
            print(f"   高−低 {o:>+7.4f}  95% [{lo:+.4f}, {hi:+.4f}]")
        else:
            print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--code", default="US.SOXL")
    ap.add_argument("--depths", default="1,5,20,60")
    ap.add_argument("--measure", choices=("qi", "fate", "both"),
                    default="both")
    ap.add_argument("--no-bootstrap", dest="boot", action="store_false",
                    help="skip the day-level interval (it is the slow part)")
    args = ap.parse_args()
    depths = tuple(int(x) for x in args.depths.split(","))

    d = LADDER / args.code.replace(".", "_")
    files = sorted(d.glob("*.parquet"))
    if not files:
        print(f"没有挂单归档：{d}")
        return
    print(f"{args.code}  归档 {len(files)} 天  {files[0].stem} .. "
          f"{files[-1].stem}", flush=True)
    k = minute_returns(args.code, files[0].stem, files[-1].stem)

    print(f"\n{'深度':>6}  {'变量':<12} {'视野':<5}")
    if args.measure in ("qi", "both"):
        qi = queue_imbalance(args.code, depths)
        # A book minute stamped 10:30 belongs to the bar labelled 10:31,
        # since time_key is the bar END. Shift rather than join on the
        # label, which would hand each bar the book from inside itself.
        qi.index = qi.index + pd.Timedelta(minutes=1)
        for n in depths:
            evaluate("qi", n, qi[f"qi{n}"], k, args.boot)

    if args.measure in ("fate", "both"):
        tr = _trades(args.code)
        for n in depths:
            w = removed_fate(args.code, n, tr)
            w.index = pd.to_datetime(w.index) + pd.Timedelta(minutes=1)
            rem = w.get("removed_B", 0) + w.get("removed_A", 0)
            pul = w.get("pulled_B", 0) + w.get("pulled_A", 0)
            eb, ea = w.get("eaten_B", 0), w.get("eaten_A", 0)
            pb, pa = w.get("pulled_B", 0), w.get("pulled_A", 0)
            feats = {
                "pulled_frac": np.where(rem > 0, pul / rem, np.nan),
                "eaten_imb": np.where((eb + ea) > 0,
                                      (eb - ea) / (eb + ea), np.nan),
                "pulled_imb": np.where(pul > 0, (pb - pa) / pul, np.nan),
            }
            for nm, v in feats.items():
                evaluate(nm, n, pd.Series(v, index=w.index), k, args.boot)
            med = float(np.nanmedian(feats["pulled_frac"]))
            print(f"{'':>6}  （depth {n}：撤单占比中位 {med:.1%}）")


if __name__ == "__main__":
    main()

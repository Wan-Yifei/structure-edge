"""What the account would actually have done, one position at a time.

Every number so far is per OPPORTUNITY: each minute is scored as if it
could be taken, and the two sides together fire on 36% of minutes while
each trade holds up to 30 bars. Those overlap almost completely. An
account holds one position, so most of those signals arrive while it is
already busy and are simply never taken -- "+0.247 R a trade" describes
the average quality of an opportunity, not a return.

This walks the sessions in order and takes a signal only when flat. It is
the one test expected to make the numbers WORSE, and it is the only one
that produces an equity curve, a drawdown and a trade count a person can
actually plan around.

TWO FILL ASSUMPTIONS, BOTH REPORTED. delta_r for a minute exists only once
that minute has closed, so the close of the signal bar is a price that
cannot be traded on -- the earliest real fill is the next bar's open. The
labelling used the close throughout, so both are run here: the gap between
them is what that assumption was worth, and with an edge near 0.2 R it does
not take much slippage to matter.

Measured on the walk-forward out-of-sample blocks, so no session is judged
by a rule that was fitted to it.

Usage:
    uv run python -m backtest.serial_sim
"""

from __future__ import annotations

import pathlib
import sys
from datetime import datetime

import numpy as np
import pandas as pd

_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from backtest.delta_atr_matrix import (CODE, END, FWD, MIN_SIGNED,  # noqa
                                       SPREAD, START, STOP, atr)
from backtest.walk_forward import TEST_WIN, TRAIN_MIN, folds  # noqa: E402

THRESH = 0.4
TGT = 1.5
rng = np.random.default_rng(307)


def build():
    from feeds import delta_bars
    from feeds.fetcher import fetch_klines

    k = fetch_klines(CODE, "1m", START, END)
    k["t"] = pd.to_datetime(k.time_key)
    k = k.sort_values("t").drop_duplicates("t").set_index("t")
    dl = delta_bars.load(CODE, START, END)
    k = k.join(dl[["buy", "sell"]], how="left")
    k[["buy", "sell"]] = k[["buy", "sell"]].fillna(0.0)

    days = {}
    for day, g in k.groupby(k.index.normalize()):
        hh = g.index.hour + g.index.minute / 60
        r = g[(hh > 9.5) & (hh <= 16.0)]
        if len(r) < 380:
            continue
        o, h, l, c = (r.open.to_numpy(float), r.high.to_numpy(float),
                      r.low.to_numpy(float), r.close.to_numpy(float))
        b, s = r.buy.to_numpy(float), r.sell.to_numpy(float)
        tot = b + s
        with np.errstate(invalid="ignore", divide="ignore"):
            dr = np.where(tot >= MIN_SIGNED, (b - s) / tot, np.nan)
        days[str(day.date())] = {
            "o": o, "h": h, "l": l, "c": c, "dr": dr,
            "a": atr(h, l, c, 14), "t": r.index}
    return days


def run_day(d, next_open: bool):
    """One session, one position at a time. Returns (trades, signals)."""
    o, h, l, c = d["o"], d["h"], d["l"], d["c"]
    dr, a = d["dr"], d["a"]
    n = len(c)
    out, signals = [], 0
    i = 0
    while i < n - 1:
        v, av = dr[i], a[i]
        if not np.isfinite(v) or not np.isfinite(av) or av <= 0:
            i += 1
            continue
        side = 1 if v >= THRESH else (-1 if v <= -THRESH else 0)
        if side == 0:
            i += 1
            continue
        signals += 1
        # The signal is known only once bar i has closed, so the earliest
        # tradeable price is bar i+1's open. The close of bar i is the
        # assumption every earlier run made; both are run.
        if next_open:
            entry, first = o[i + 1], i + 1
        else:
            entry, first = c[i], i + 1
        stop = entry - side * STOP * av
        target = entry + side * TGT * av
        exit_i, r_mult = None, None
        for j in range(first, min(n, first + FWD)):
            hit_stop = (l[j] <= stop) if side > 0 else (h[j] >= stop)
            hit_tgt = (h[j] >= target) if side > 0 else (l[j] <= target)
            if hit_stop:                 # adverse first within one bar
                exit_i, r_mult = j, -STOP
                break
            if hit_tgt:
                exit_i, r_mult = j, TGT
                break
        if exit_i is None:               # time stop, at the close
            exit_i = min(n - 1, first + FWD - 1)
            r_mult = side * (c[exit_i] - entry) / av
        out.append({"i": i, "side": side, "dr": v, "atr": av,
                    "r": r_mult - SPREAD / (STOP * av),
                    "bars": exit_i - i, "exit": exit_i})
        # Flat only once the trade is out; everything in between is a
        # signal the account could not have taken.
        i = exit_i + 1
    return out, signals


def summarise(name, rows, signals, days_n):
    if not rows:
        print(f"{name:<28}   没有成交")
        return None
    r = np.array([x["r"] for x in rows])
    eq = np.cumsum(r)
    dd = float(np.max(np.maximum.accumulate(eq) - eq))
    print(f"{name:<28}{len(r):>7}{len(r)/days_n:>9.1f}"
          f"{100*(r > 0).mean():>8.1f}%{r.mean():>+9.3f}{r.sum():>+9.1f}"
          f"{dd:>9.1f}{100*len(r)/max(1, signals):>9.1f}%")
    return r


def boot_days(per_day, reps=4000):
    """Resample SESSIONS, as everywhere else."""
    tot = np.array([v.sum() for v in per_day])
    cnt = np.array([len(v) for v in per_day], float)
    if len(tot) < 5:
        return np.nan, np.nan
    i = rng.integers(0, len(tot), size=(reps, len(tot)))
    n = cnt[i].sum(1)
    ok = n > 0
    v = tot[i].sum(1)[ok] / n[ok]
    return np.percentile(v, 2.5), np.percentile(v, 97.5)


def main():
    days = build()
    all_days = sorted(days)
    oos = []
    for _, te in folds(all_days, anchored=True):
        oos += te
    oos = [d for d in oos if d in days]
    print(f"样本外 {len(oos)} 个交易日（与前推验证同一批区块）")
    print(f"规则：|delta_r(1)| >= {THRESH} 进场，止损 1 ATR14，"
          f"目标 {TGT} ATR，{FWD} 根时间止损")
    print(f"价差 {SPREAD} 往返已扣；同一时刻只持一个仓位，持仓期间的信号全部放过\n")

    print(f"{'':<28}{'笔数':>7}{'每天':>9}{'胜率':>9}"
          f"{'每笔 R':>9}{'合计 R':>9}{'最大回撤':>9}{'信号接受率':>10}")
    results = {}
    for label, nxt in (("在信号那根收盘成交", False),
                       ("在下一根开盘成交（可执行）", True)):
        rows_by_day, total_sig = [], 0
        for d in oos:
            rr, sg = run_day(days[d], next_open=nxt)
            total_sig += sg
            rows_by_day.append(rr)
        flat = [x for rr in rows_by_day for x in rr]
        r = summarise(label, flat, total_sig, len(oos))
        results[label] = ([np.array([x["r"] for x in rr]) for rr in rows_by_day],
                          flat, r)

    print()
    for label, (per_day, flat, r) in results.items():
        if r is None:
            continue
        lo, hi = boot_days(per_day)
        print(f"{label:<28}每笔 R 的 95% 区间 [{lo:+.3f}, {hi:+.3f}]"
              + ("  *" if lo > 0 else ""))

    a = results["在信号那根收盘成交"][2]
    b = results["在下一根开盘成交（可执行）"][2]
    if a is not None and b is not None:
        print(f"\n两种成交假设之差：{b.mean() - a.mean():+.3f} R/笔"
              f"（{100*(b.mean() - a.mean())/abs(a.mean()):+.0f}%）")

    flat = results["在下一根开盘成交（可执行）"][1]
    if flat:
        print("\n=== 可执行口径的细节 ===")
        bars = np.array([x["bars"] for x in flat])
        rr = np.array([x["r"] for x in flat])
        print(f"  持仓时长：中位 {np.median(bars):.0f} 根，"
              f"均值 {bars.mean():.1f} 根，占满 {FWD} 根的 "
              f"{100*(bars >= FWD - 1).mean():.0f}%")
        for nm, m in (("做多", np.array([x["side"] for x in flat]) > 0),
                      ("做空", np.array([x["side"] for x in flat]) < 0)):
            if m.sum():
                print(f"  {nm}：{int(m.sum())} 笔，胜率 "
                      f"{100*(rr[m] > 0).mean():.1f}%，每笔 {rr[m].mean():+.3f} R")
        print("\n  对照：按机会计价（每分钟独立评估）"
              "多头 +0.207 / 空头 +0.247 R")
        print("  差额就是「同一时刻只能持一个仓位」的代价。")


if __name__ == "__main__":
    main()

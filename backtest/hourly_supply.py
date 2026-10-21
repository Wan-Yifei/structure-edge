"""How many scalps a session actually offers, and when they are worth taking.

"How many trades is reasonable" has two different ceilings and they are
easy to confuse. One is SUPPLY -- how often the rule fires at all, which is
a property of the tape. The other is ECONOMICS -- every entry pays the
spread whether or not it works, so the edge per trade has to clear that
cost before a higher count is an improvement rather than a leak.

Both are measured per hour here, because the premise worth testing is that
the morning is different. Everything is imported from the modules that
produced the existing results: a second implementation of the label or the
threshold would make any hourly difference uninterpretable.

    uv run backtest/hourly_supply.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from backtest.absorption import TGT, outcomes                 # noqa: E402
from backtest.delta_atr_matrix import (CODE, END, MIN_SIGNED,  # noqa: E402
                                       SPREAD, START, STOP, atr)

DR_GATE = 0.40


def build() -> pd.DataFrame:
    from feeds import delta_bars
    from feeds.fetcher import fetch_klines

    k = fetch_klines(CODE, "1m", START, END)
    k["t"] = pd.to_datetime(k.time_key)
    k = k.sort_values("t").drop_duplicates("t").set_index("t")
    dl = delta_bars.load(CODE, START, END)
    k = k.join(dl[["buy", "sell"]], how="left")
    k[["buy", "sell"]] = k[["buy", "sell"]].fillna(0.0)

    parts = []
    for day, g in k.groupby(k.index.normalize()):
        hh = g.index.hour + g.index.minute / 60
        r = g[(hh > 9.5) & (hh <= 16.0)]
        if len(r) < 380:
            continue
        h, l, c = (r.high.to_numpy(float), r.low.to_numpy(float),
                   r.close.to_numpy(float))
        a = atr(h, l, c, 14)
        buy, sell = r.buy.to_numpy(float), r.sell.to_numpy(float)
        tot = buy + sell
        ok = tot >= MIN_SIGNED
        with np.errstate(invalid="ignore", divide="ignore"):
            dr = np.where(ok, (buy - sell) / tot, np.nan)
        parts.append(pd.DataFrame({
            "dr1": dr,
            "dr_prev": np.concatenate([[np.nan], dr[:-1]]),
            "atr": a,
            "r": outcomes(h, l, c, a, +1),
            "day": day.strftime("%Y-%m-%d"),
            "hour": r.index.hour,
        }, index=r.index))
    f = pd.concat(parts)
    f["sig"] = (f.dr1 >= DR_GATE) & (f.dr_prev < 0)
    return f


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    f = build()
    sessions = f.day.nunique()
    sig = f[f.sig & np.isfinite(f.r)]
    print(f"{CODE}  {START}..{END}  {sessions} 个交易日，"
          f"盘中分钟 {len(f):,}", flush=True)
    print(f"规则 dr1 >= {DR_GATE} 且前一根 < 0，"
          f"止损 {STOP} ATR14 / 目标 {TGT} R，"
          f"已扣 {SPREAD} 价差\n", flush=True)

    print(f"{'时段':>8}{'信号/日':>9}{'笔数':>7}{'每笔 R':>9}"
          f"{'95% 区间':>20}{'胜率':>7}{'累计 R/日':>11}", flush=True)
    for hr in sorted(sig.hour.unique()):
        s = sig[sig.hour == hr]
        r = s.r.to_numpy()
        per_day = len(s) / sessions
        se = r.std(ddof=1) / np.sqrt(len(r)) if len(r) > 1 else np.nan
        print(f"{hr:>5}:00{per_day:>9.1f}{len(r):>7}{r.mean():>9.3f}"
              f"   [{r.mean()-1.96*se:>+6.3f}, {r.mean()+1.96*se:>+6.3f}]"
              f"{(r > 0).mean():>7.0%}{per_day*r.mean():>11.3f}", flush=True)

    r = sig.r.to_numpy()
    se = r.std(ddof=1) / np.sqrt(len(r))
    print(f"\n{'全天':>8}{len(sig)/sessions:>9.1f}{len(r):>7}{r.mean():>9.3f}"
          f"   [{r.mean()-1.96*se:>+6.3f}, {r.mean()+1.96*se:>+6.3f}]"
          f"{(r > 0).mean():>7.0%}{len(sig)/sessions*r.mean():>11.3f}",
          flush=True)

    # The question as asked: before 1 PM against the rest.
    print("\n按「13 点之前 / 之后」切", flush=True)
    for name, m in (("09:30-13:00", sig.hour < 13),
                    ("13:00-16:00", sig.hour >= 13)):
        s = sig[m]
        r = s.r.to_numpy()
        se = r.std(ddof=1) / np.sqrt(len(r))
        print(f"  {name}  {len(s)/sessions:>5.1f} 信号/日   "
              f"每笔 {r.mean():+.3f} R  "
              f"95% [{r.mean()-1.96*se:+.3f}, {r.mean()+1.96*se:+.3f}]   "
              f"合计 {len(s)/sessions*r.mean():+.3f} R/日", flush=True)

    # What one trade costs, in the unit the edge is quoted in.
    cost = sig.apply(lambda x: SPREAD / (STOP * x.atr), axis=1)
    print(f"\n每笔的价差成本：中位 {cost.median():.3f} R，"
          f"均值 {cost.mean():.3f} R  "
          f"（已扣在上面的 R 里，列出来是为了看它占多少）", flush=True)

    # Taking only the strongest N per session: does selectivity pay?
    print("\n每天只做最强的 N 笔（按 dr1 排序，当天之内选）", flush=True)
    for n in (1, 2, 3, 5, 8, 999):
        top = (sig.sort_values("dr1", ascending=False)
                  .groupby("day").head(n))
        r = top.r.to_numpy()
        se = r.std(ddof=1) / np.sqrt(len(r))
        label = "全部" if n == 999 else f"前 {n}"
        print(f"  {label:<6} {len(top)/sessions:>5.1f} 笔/日  "
              f"每笔 {r.mean():+.3f} R  "
              f"95% [{r.mean()-1.96*se:+.3f}, {r.mean()+1.96*se:+.3f}]  "
              f"合计 {len(top)/sessions*r.mean():+.3f} R/日", flush=True)


if __name__ == "__main__":
    main()

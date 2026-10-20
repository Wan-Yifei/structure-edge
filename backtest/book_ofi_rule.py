"""Does book OFI say anything the dr1 + flip rule does not?

Phase 1 (book_ofi.py) answered the construction question: seven levels,
integrated, and the relation is contemporaneous -- it explains 47% of the
move it is part of and forecasts none of the next one. That was a property
of the data and five sessions could carry it.

This asks the trading question, which five sessions CANNOT carry. The
established rule fires perhaps a dozen times a day, so a week holds a few
dozen trades and any per-trade mean has an interval wider than the effect
being looked for. It is run anyway, for one reason: a sign. If the book
adds something, five days should at least not point the wrong way, and if
it points the wrong way there is nothing to buy data for.

Everything about the trade side is imported rather than re-derived --
outcomes(), STOP, TGT, FWD, SPREAD, MIN_SIGNED and atr() all come from the
modules that produced the existing results. A second implementation of the
label would make any difference here uninterpretable: it could be the book,
or it could be that the two labellers disagree.

The book variable is the UNWEIGHTED sum of the seven normalised levels, not
the first principal component. PC1 has to be fitted, and fitting it on five
days to then split those same five days is the leak this script exists to
avoid. Phase 1 showed the two are within noise of each other.

    uv run backtest/book_ofi_rule.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from backtest.absorption import TGT, outcomes                  # noqa: E402
from backtest.book_ofi import (LADDER, bucket, design,         # noqa: E402
                               load_day, ofi_per_snapshot)
from backtest.delta_atr_matrix import (CODE, MIN_SIGNED,       # noqa: E402
                                       STOP, atr)

LEVELS = 7            # phase 1's answer
DR_GATE = 0.40        # the pre-registered threshold
# Low enough to admit a short session. A day that starts at 13:00 is not a
# damaged day, it is a short one, and its bars carry the same information.
# 4,000 snapshots is a bit over an hour of regular trading.
MIN_RTH = 4000


def book_minutes(day: str):
    """Integrated book OFI per minute, keyed by the minute's START."""
    got = load_day(LADDER / CODE.replace(".", "_") / f"{day}.parquet", LEVELS)
    if got is None:
        return None
    ts, bp, bq, ap, aq = got
    if len(ts) < MIN_RTH:
        return None
    e = ofi_per_snapshot(bp, bq, ap, aq)
    df = bucket(ts, e, bq, aq, bp, ap, 60)
    X = design(df, LEVELS)
    return pd.Series(np.nansum(X, axis=1), index=df.index, name="ofi")


def trade_minutes(days: list[str]) -> pd.DataFrame:
    """The established per-minute frame: dr1, flip, ATR and the long R."""
    from feeds import delta_bars
    from feeds.fetcher import fetch_klines

    lo, hi = min(days), max(days)
    k = fetch_klines(CODE, "1m", lo, hi)
    k["t"] = pd.to_datetime(k.time_key)
    k = k.sort_values("t").drop_duplicates("t").set_index("t")
    dl = delta_bars.load(CODE, lo, hi)
    k = k.join(dl[["buy", "sell"]], how="left")
    k[["buy", "sell"]] = k[["buy", "sell"]].fillna(0.0)

    parts = []
    for day, g in k.groupby(k.index.normalize()):
        d = day.strftime("%Y-%m-%d")
        if d not in days:
            continue
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
        out = pd.DataFrame({
            "dr1": dr,
            "dr_prev": np.concatenate([[np.nan], dr[:-1]]),
            "atr": a,
            "r": outcomes(h, l, c, a, +1),
        }, index=r.index)
        out["day"] = d
        parts.append(out)
    f = pd.concat(parts)
    f["flip"] = (f.dr1 >= DR_GATE) & (f.dr_prev < 0)
    # The book bucket is keyed by the minute's START; a kline's time_key is
    # its END, so they are one minute apart and joining them raw would pair
    # every bar with the wrong minute's flow.
    f["bkey"] = f.index - pd.Timedelta(minutes=1)
    return f


def line(name, r: np.ndarray):
    r = r[np.isfinite(r)]
    if not len(r):
        print(f"  {name:<34}      —", flush=True)
        return
    # Interval on the mean, so a count of 12 cannot masquerade as a finding.
    se = r.std(ddof=1) / np.sqrt(len(r)) if len(r) > 1 else float("nan")
    win = (r > 0).mean()
    print(f"  {name:<34}{len(r):>5} 笔  {r.mean():+.3f} R  "
          f"95% [{r.mean()-1.96*se:+.3f}, {r.mean()+1.96*se:+.3f}]  "
          f"胜率 {win:.0%}", flush=True)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    d = LADDER / CODE.replace(".", "_")
    days = sorted(p.stem for p in d.glob("*.parquet"))

    book = {}
    for day in days:
        b = book_minutes(day)
        if b is not None:
            book[day] = b
    if not book:
        raise SystemExit("没有完整的盘口交易日")
    use = sorted(book)
    print(f"{CODE}  盘口完整的交易日 {len(use)} 个: {', '.join(use)}",
          flush=True)
    print(f"规则：dr1 >= {DR_GATE} 且前一根 dr1 < 0；"
          f"止损 {STOP} ATR14 / 目标 {TGT} R\n", flush=True)

    f = trade_minutes(use)
    allb = pd.concat(book.values())
    f = f.join(allb, on="bkey")
    f = f[np.isfinite(f.r)]
    print(f"盘中分钟 {len(f):,}，其中有盘口 OFI 的 "
          f"{int(np.isfinite(f.ofi).sum()):,}", flush=True)

    sig = f[f.flip & np.isfinite(f.ofi)]
    print(f"\n规则触发 {len(sig)} 笔 —— 这就是本脚本能说的话的上限\n",
          flush=True)

    print("基准", flush=True)
    line("全部盘中分钟（做多）", f.r.to_numpy())
    line(f"规则（dr1≥{DR_GATE} 且翻转）", sig.r.to_numpy())

    if len(sig) >= 6:
        med = sig.ofi.median()
        print("\n在规则触发的那些笔里，按盘口 OFI 切开", flush=True)
        line("盘口 OFI 为正（挂单同向）", sig[sig.ofi > 0].r.to_numpy())
        line("盘口 OFI 为负", sig[sig.ofi <= 0].r.to_numpy())
        line("盘口 OFI 高于中位", sig[sig.ofi > med].r.to_numpy())
        line("盘口 OFI 低于中位", sig[sig.ofi <= med].r.to_numpy())

    print("\n盘口 OFI 单独选（不看 dr1），作为对照", flush=True)
    ok = f[np.isfinite(f.ofi)]
    q = ok.ofi.quantile([0.2, 0.8])
    line("OFI 最高的五分之一", ok[ok.ofi >= q.iloc[1]].r.to_numpy())
    line("OFI 最低的五分之一", ok[ok.ofi <= q.iloc[0]].r.to_numpy())

    if len(sig) >= 20:
        pos = sig[sig.ofi > 0].r.to_numpy()
        neg = sig[sig.ofi <= 0].r.to_numpy()
        gap = float(np.nanmean(pos) - np.nanmean(neg))

        print(f"\n控制一：按天拆开（{len(sig)} 笔分到 "
              f"{sig.day.nunique()} 天，看是不是某一天的事）", flush=True)
        for day, g in sig.groupby("day"):
            p = g[g.ofi > 0].r.to_numpy()
            q = g[g.ofi <= 0].r.to_numpy()
            dd = (np.nanmean(p) - np.nanmean(q)) if len(p) and len(q) \
                else float("nan")
            print(f"  {day}  正 {len(p):>2} 笔 {np.nanmean(p):+.2f} R   "
                  f"负 {len(q):>2} 笔 {np.nanmean(q):+.2f} R   "
                  f"差 {dd:+.2f}", flush=True)

        print("\n控制二：留一天（差值撑不撑得住丢掉任意一天）", flush=True)
        for day in sorted(sig.day.unique()):
            s = sig[sig.day != day]
            p = s[s.ofi > 0].r.to_numpy()
            q = s[s.ofi <= 0].r.to_numpy()
            print(f"  去掉 {day}: 差 {np.nanmean(p)-np.nanmean(q):+.3f} R  "
                  f"（{len(p)}+{len(q)} 笔）", flush=True)

        print("\n控制三：置换检验（把 OFI 标签在同样 77 个结果上打乱）",
              flush=True)
        rs = sig.r.to_numpy()
        npos = len(pos)
        rng = np.random.default_rng(97)
        N = 20000
        cnt = 0
        for _ in range(N):
            idx = rng.permutation(len(rs))
            a, b = rs[idx[:npos]], rs[idx[npos:]]
            if abs(np.nanmean(a) - np.nanmean(b)) >= abs(gap):
                cnt += 1
        print(f"  实际差 {gap:+.3f} R，置换 {N:,} 次中 "
              f"{cnt/N:.1%} 达到或超过这个幅度  → p = {cnt/N:.3f}",
              flush=True)

        print("\n控制四：这个差值要多少笔才测得动", flush=True)
        sd = float(np.sqrt((np.nanvar(pos, ddof=1) * (len(pos) - 1) +
                            np.nanvar(neg, ddof=1) * (len(neg) - 1)) /
                           (len(pos) + len(neg) - 2)))
        per_day = len(sig) / sig.day.nunique()
        print(f"  合并标准差 {sd:.2f} R，每个交易日约 {per_day:.0f} 笔信号",
              flush=True)
        print(f"  {'真实效应':>10}{'每组所需':>10}{'合计':>8}{'约需交易日':>12}",
              flush=True)
        for frac in (1.0, 0.75, 0.5, 0.33, 0.25):
            delta = abs(gap) * frac
            # Two-sample, alpha .05 two-sided, 80% power: n = 16 (sd/delta)^2
            n = 16 * (sd / delta) ** 2
            print(f"  {delta:>9.2f} R{n:>10.0f}{2*n:>8.0f}"
                  f"{2*n/per_day:>12.0f}", flush=True)
        print("  观测到的 0.70 R 是在「看到之后才去测」的同一批数据上量的，"
              "作为真值估计偏高；", flush=True)
        print("  所以规划应当按它的一半或三分之一来排。", flush=True)

    print("\n相关", flush=True)
    sub = ok[np.isfinite(ok.dr1)]
    print(f"  盘口 OFI vs dr1 : {np.corrcoef(sub.ofi, sub.dr1)[0,1]:+.3f}"
          f"   （两者说的是不是同一件事）", flush=True)
    print(f"  盘口 OFI vs 后续 R: "
          f"{np.corrcoef(ok.ofi, ok.r)[0,1]:+.3f}", flush=True)


if __name__ == "__main__":
    main()

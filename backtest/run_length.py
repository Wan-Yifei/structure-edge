"""Does the RUN before a delta_r flip carry anything the flip does not?

PRE-REGISTERED. Written before any number was read.

The observation: several red DCVD bars, then a green one, looks like the
start of a reversal. That description contains two separate things, and
only one of them is new:

  the flip   delta_r crossing from negative to positive. Already measured:
             dr1 >= 0.4 is worth +0.207 R a trade long, +0.247 short.
  the run    how many consecutive negative bars preceded it. This is the
             only part that is not already in hand, so it is the only
             thing this study asks about.

THE QUESTION, therefore: conditional on a flip, does the outcome improve
with run length k? If R is flat in k, the run is something the eye added.

THREE CONTROLS, ALL REGISTERED HERE.

  PRICE. k consecutive negative bars means price has probably fallen, so
  "long red run then green" overlaps with "fell a lot, bounced" -- plain
  mean reversion, nothing to do with order flow. Reported: the mean
  preceding return by k, and R within buckets of that return. If R tracks
  the drop rather than k, the run is a proxy for the drop.

  THE FLIP ALONE. A flip after a red run is compared against a flip at the
  same delta_r level that did NOT follow one. Same signal strength, one
  with the pattern and one without.

  THE NULL. Run lengths in an independent-sign sequence are geometric:
  k=5 turns up once every 32 bars in pure noise, and the eye finds runs in
  noise reliably. The observed distribution is printed against that null.
  If they match there is no clustering to trade.

Both directions, since the mirror image (greens then a red) is the same
claim and a signal that works one way only is usually an artefact.

Usage:
    uv run python -m backtest.run_length
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

_ROOT = pathlib.Path(__file__).parent.parent
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(_ROOT))

from backtest.absorption import outcomes                    # noqa: E402
from backtest.delta_atr_matrix import (CODE, END, MIN_SIGNED,  # noqa
                                       SPREAD, START, STOP, atr)
from backtest.walk_forward import folds                     # noqa: E402

FLIP = 0.0            # a flip is dr1 crossing to >= this
STRONG = 0.4          # the validated threshold, used as a second cut
LOOKBACK = 5          # bars of preceding return, for the price control
rng = np.random.default_rng(509)


def build():
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

        d = pd.DataFrame(index=r.index)
        d["day"] = str(day.date())
        d["atr"] = a
        d["dr1"] = dr

        # Run of the OPPOSITE sign ending at t-1. Counted directly rather
        # than by the helper above, which the first version got wrong.
        n = len(dr)
        neg_run = np.zeros(n, dtype=int)   # consecutive dr<0 before t
        pos_run = np.zeros(n, dtype=int)   # consecutive dr>=0 before t
        for i in range(1, n):
            p = dr[i - 1]
            if not np.isfinite(p):
                neg_run[i] = pos_run[i] = 0
            elif p < 0:
                neg_run[i] = neg_run[i - 1] + 1
                pos_run[i] = 0
            else:
                pos_run[i] = pos_run[i - 1] + 1
                neg_run[i] = 0
        d["neg_run"] = neg_run
        d["pos_run"] = pos_run

        # Price over the preceding LOOKBACK bars, in ATR -- the control.
        prev = np.full(n, np.nan)
        for i in range(LOOKBACK, n):
            if np.isfinite(a[i]) and a[i] > 0:
                prev[i] = (c[i] - c[i - LOOKBACK]) / a[i]
        d["prev_ret"] = prev

        d["r_long"] = outcomes(h, l, c, a, +1)
        d["r_short"] = outcomes(h, l, c, a, -1)
        parts.append(d)
    return pd.concat(parts)


def agg(sub, col):
    g = sub.dropna(subset=[col]).groupby("day")
    a = g.agg(n=(col, "size"), sr=(col, "sum"))
    return a.n.to_numpy(float), a.sr.to_numpy()


def net(n, sr):
    return np.nan if n.sum() == 0 else sr.sum() / n.sum()


def ci(sub, col, reps=3000):
    n, sr = agg(sub, col)
    if len(n) < 5:
        return np.nan, np.nan
    i = rng.integers(0, len(n), size=(reps, len(n)))
    t = n[i].sum(1)
    ok = t > 0
    v = sr[i].sum(1)[ok] / t[ok]
    return np.percentile(v, 2.5), np.percentile(v, 97.5)


def row(label, sub, col, extra=""):
    s = sub.dropna(subset=[col])
    if len(s) < 80:
        print(f"{label:<22}{len(s):>7}   样本太少")
        return
    n, sr = agg(s, col)
    lo, hi = ci(s, col)
    star = "  *" if lo > 0 else ""
    print(f"{label:<22}{len(s):>7}{100*(s[col] > 0).mean():>8.1f}%"
          f"{net(n, sr):>+9.3f}   [{lo:+.3f}, {hi:+.3f}]{star}{extra}")


def main():
    d = build()
    days = sorted(d.day.unique())
    oos = []
    for _, te_days in folds(days, anchored=True):
        oos += te_days
    te = d[d.day.isin(oos)]
    print(f"样本外 {te.day.nunique()} 个交易日 / {len(te):,} 分钟")
    print(f"止损 1 ATR，目标 1.5 ATR，30 根时间止损，价差已扣")
    print("k = 翻转前连续反向 bar 的根数；NaN 打断游程，不延长\n")

    hdr = f"{'':<22}{'笔数':>7}{'胜率':>8}{'每笔 R':>9}{'95% 区间':>22}"

    for name, run_col, flip_cmp, col, mirror in (
            ("看涨：红→绿", "neg_run", lambda s: s.dr1 >= FLIP, "r_long",
             "红"),
            ("看跌：绿→红", "pos_run", lambda s: s.dr1 < FLIP, "r_short",
             "绿")):
        flip = te[flip_cmp(te) & te.dr1.notna()]
        print(f"=== {name} ===")
        print(hdr)
        row("全部翻转（k≥1）", flip[flip[run_col] >= 1], col)
        for k in (1, 2, 3, 4, 5):
            row(f"  前面 {k} 根{mirror}", flip[flip[run_col] == k], col)
        row(f"  前面 ≥6 根{mirror}", flip[flip[run_col] >= 6], col)
        print()

    print("=== 控制一：游程只是跌幅的代理吗 ===")
    print("  k 越大，前 5 根的累计跌幅通常越深。若 R 跟着跌幅走而不跟 k，")
    print("  那「多根红」就是跌幅的代理变量。")
    flip = te[(te.dr1 >= FLIP) & te.dr1.notna()]
    print(f"{'k':>4}{'笔数':>8}{'前5根均值(ATR)':>16}{'每笔 R':>10}")
    for k in (1, 2, 3, 4, 5):
        s = flip[flip.neg_run == k].dropna(subset=["r_long", "prev_ret"])
        if len(s) < 80:
            continue
        n, sr = agg(s, "r_long")
        print(f"{k:>4}{len(s):>8}{s.prev_ret.mean():>16.2f}"
              f"{net(n, sr):>+10.3f}")

    print("\n  同一跌幅档内比 k（跌幅按五分位分档）：")
    s = flip.dropna(subset=["r_long", "prev_ret"]).copy()
    s["q"] = pd.qcut(s.prev_ret, 5, labels=False, duplicates="drop")
    print(f"{'跌幅档':>8}{'k≤2 笔数':>10}{'k≤2 R':>9}"
          f"{'k≥4 笔数':>10}{'k≥4 R':>9}{'差':>8}")
    for q in sorted(s.q.dropna().unique()):
        a_ = s[(s.q == q) & (s.neg_run <= 2)]
        b_ = s[(s.q == q) & (s.neg_run >= 4)]
        if len(a_) < 50 or len(b_) < 50:
            continue
        ra, rb = net(*agg(a_, "r_long")), net(*agg(b_, "r_long"))
        print(f"{int(q)+1:>8}{len(a_):>10}{ra:>+9.3f}"
              f"{len(b_):>10}{rb:>+9.3f}{rb-ra:>+8.3f}")

    print("\n=== 控制二：有没有游程，差别在哪 ===")
    print("  同样的信号强度，一个前面有连红，一个没有。")
    print(hdr)
    strong = te[(te.dr1 >= STRONG) & te.dr1.notna()]
    row("dr1≥0.4 且前面≥3 根红", strong[strong.neg_run >= 3], "r_long")
    row("dr1≥0.4 且前面≤1 根红", strong[strong.neg_run <= 1], "r_long")

    print("\n=== 控制三：游程分布 vs 独立符号的零假设 ===")
    v = te.dr1.dropna()
    p = float((v < 0).mean())
    print(f"  dr1<0 的占比 p = {p:.3f}；若符号独立，连续 k 根红的")
    print(f"  条件概率应恒为 p，与已经连了几根无关。")
    print(f"{'已连红根数':>12}{'样本':>9}{'下一根仍为红':>14}{'零假设':>9}")
    sgn = te.dr1.to_numpy()
    nr = te.neg_run.to_numpy()
    nxt = np.isfinite(sgn) & (sgn < 0)
    for k in range(1, 7):
        m = (nr == k) & np.isfinite(sgn)
        if m.sum() < 200:
            continue
        print(f"{k:>12}{int(m.sum()):>9}{nxt[m].mean():>14.3f}{p:>9.3f}")
    print("\n  若右两列逐行接近，游程里没有聚集，k 不含额外信息。")
    print("  * = 95% 区间（按交易日重采样）整个在 0 以上")


if __name__ == "__main__":
    main()

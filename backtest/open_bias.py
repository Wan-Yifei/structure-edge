"""Does the opening 15 minutes or the opening 30 minutes call the day?

The trap this is built around: correlating the opening window against the
FULL day's return is mostly correlating the window with itself, since the
window is part of the day. A 30-minute window is a larger part than a
15-minute one, so that comparison hands the win to whichever window is
longer regardless of what the market does.

So both are reported and labelled. The one that answers the question is
the REST of the day -- from the end of the window to the close -- because
that is the only part still tradeable once the window has told you
something.

Direction agreement is reported alongside correlation, because "日内偏见
方向" is a question about sign, not about magnitude, and a correlation can
be carried by a handful of large days while the sign is a coin flip.

    uv run backtest/open_bias.py
    uv run backtest/open_bias.py --code US.SOXS --start 2025-01-01
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

WINDOWS = (5, 15, 30, 60, 90)


def sessions(code: str, start: str, end: str) -> list[pd.DataFrame]:
    from feeds.fetcher import fetch_klines

    k = fetch_klines(code, "1m", start, end)
    k["t"] = pd.to_datetime(k.time_key)
    k = k.sort_values("t").drop_duplicates("t").set_index("t")
    out = []
    for _, g in k.groupby(k.index.normalize()):
        hh = g.index.hour + g.index.minute / 60
        r = g[(hh > 9.5) & (hh <= 16.0)]
        # A short session is a half day, not a normal one; its open-to-close
        # behaviour is a different animal and would be read as noise here.
        if len(r) >= 380:
            out.append(r)
    return out


def boot_ci(x, y, reps=4000, seed=7):
    """Bootstrap interval on the correlation, over SESSIONS.

    A t-interval on r assumes normality that daily returns do not have;
    resampling days keeps whatever shape they actually carry.
    """
    rng = np.random.default_rng(seed)
    n = len(x)
    out = np.empty(reps)
    for i in range(reps):
        j = rng.integers(0, n, n)
        out[i] = np.corrcoef(x[j], y[j])[0, 1]
    return np.percentile(out, 2.5), np.percentile(out, 97.5)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--code", default="US.SOXL")
    ap.add_argument("--start", default="2025-01-01")
    ap.add_argument("--end", default="2026-10-22")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    days = sessions(args.code, args.start, args.end)
    print(f"{args.code}  {len(days)} 个完整交易日  "
          f"{days[0].index[0].date()} .. {days[-1].index[0].date()}\n",
          flush=True)

    rows = []
    for w in WINDOWS:
        opens, rest, full = [], [], []
        for r in days:
            c = r["close"].to_numpy(float)
            o = float(r["open"].iloc[0])
            if len(c) <= w:
                continue
            opens.append((c[w - 1] - o) / o * 100)        # the window itself
            rest.append((c[-1] - c[w - 1]) / c[w - 1] * 100)  # what is left
            full.append((c[-1] - o) / o * 100)            # open to close
        a = np.array(opens)
        b = np.array(rest)
        f = np.array(full)
        r_rest = np.corrcoef(a, b)[0, 1]
        r_full = np.corrcoef(a, f)[0, 1]
        lo, hi = boot_ci(a, b)
        same = float(np.mean(np.sign(a) == np.sign(b)))
        rows.append((w, len(a), r_rest, lo, hi, r_full, same,
                     float(np.std(a))))

    print("开盘窗口 → 窗口之后到收盘（这才是能用的那个）", flush=True)
    print(f"{'窗口':>6}{'天数':>6}{'相关':>8}{'95% 区间':>20}"
          f"{'同向率':>8}{'窗口幅度 sd':>12}", flush=True)
    for w, n, rr, lo, hi, _, same, sd in rows:
        print(f"{w:>4}分{n:>6}{rr:>8.3f}   [{lo:>+6.3f}, {hi:>+6.3f}]"
              f"{same:>8.1%}{sd:>11.2f}%", flush=True)

    print("\n对照：开盘窗口 → 全天（含窗口自身，机械相关，不可用）",
          flush=True)
    for w, n, _, _, _, rf, _, _ in rows:
        print(f"{w:>4}分  {rf:>6.3f}", flush=True)

    # Does a bigger opening move call the rest of the day better?
    print("\n按开盘幅度分档（15 分钟窗口），看幅度大是否更可信", flush=True)
    w = 15
    a, b = [], []
    for r in days:
        c = r["close"].to_numpy(float)
        o = float(r["open"].iloc[0])
        a.append((c[w - 1] - o) / o * 100)
        b.append((c[-1] - c[w - 1]) / c[w - 1] * 100)
    a, b = np.array(a), np.array(b)
    mag = np.abs(a)
    q = np.quantile(mag, [1 / 3, 2 / 3])
    for name, m in (("小", mag <= q[0]),
                    ("中", (mag > q[0]) & (mag <= q[1])),
                    ("大", mag > q[1])):
        x, y = a[m], b[m]
        same = float(np.mean(np.sign(x) == np.sign(y)))
        print(f"  {name}  {len(x):>4} 天  "
              f"|开盘| 中位 {np.median(np.abs(x)):.2f}%   "
              f"相关 {np.corrcoef(x, y)[0, 1]:+.3f}   "
              f"同向率 {same:.1%}", flush=True)


if __name__ == "__main__":
    main()

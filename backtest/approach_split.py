"""Inside "reached it from below and did not close above": which half pays?

The state is judged on a 1-minute close, and a minute's close is a weak
statement about a level. Two very different bars satisfy it:

    close 97.42 against a level of 97.43, high 97.50
        a cent short -- this is nearly a breakout
    close 97.10 against the same level, high 97.50
        pushed up and sold back -- this is a rejection

Both are currently one cell, and that cell came out at +0.17 to +0.41 R.
The reading offered for it was "buying is pressing, the level has not gone
yet". That reading is only right if the near-misses are NOT what carries
it -- if they are, the honest description is "almost a breakout", and the
instruction that follows is different: wait for the break rather than
treat the failure as bullish.

Measured, not re-timed. Moving the state to a 5-minute close would also
separate them, but the signal is validated at one minute and the two do
not have to share a clock -- and a coarser clock answers a different
question than the one asked here.

    gap = (L - close) / ATR          how far short the close fell
    pos = (close - low) / (high - low)   where in the bar the close sits

A near-miss is a small gap and a high pos. A rejection is the opposite.

Usage:
    uv run python -m backtest.approach_split
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

_ROOT = pathlib.Path(__file__).parent.parent
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(_ROOT))

from backtest.level_states import LEVELS, agg, ci, net      # noqa: E402
from backtest.walk_forward import folds                     # noqa: E402

rng = np.random.default_rng(941)


def build_with_geometry():
    """level_states.build(), plus gap and pos for each level."""
    import backtest.level_states as ls
    from feeds import delta_bars
    from feeds.fetcher import fetch_klines
    from strategy.session_vp.profile import (bins_for_price,
                                             compute_value_area, find_nodes)
    from strategy.smc.fvg import compute_volume_profile
    from backtest.absorption import outcomes
    from backtest.delta_atr_matrix import CODE, END, MIN_SIGNED, START, atr

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

        n = len(r)
        lv = {key: np.full(n, np.nan) for _, key in LEVELS}
        for i in range(ls.WARMUP, n):
            up = r.iloc[:i + 1]
            nb = bins_for_price(float(up.low.min()), float(up.high.max()))
            if not nb:
                continue
            edges, vols = compute_volume_profile(up, n_bins=nb)
            if edges is None:
                continue
            va = compute_value_area(edges, vols, va_pct=0.70)
            lv["poc"][i], lv["val"][i], lv["vah"][i] = (
                va["poc"], va["val"], va["vah"])
            nodes = find_nodes(edges, vols)
            if not nodes:
                continue
            px = c[i]
            inside = [nd for nd in nodes if nd.lo <= px <= nd.hi]
            pick = (inside[0] if inside
                    else min(nodes, key=lambda z: abs(z.peak - px)))
            lv["ndlo"][i], lv["ndpk"][i], lv["ndhi"][i] = (
                pick.lo, pick.peak, pick.hi)

        d = pd.DataFrame(index=r.index)
        d["day"] = str(day.date())
        d["dr1"] = dr
        rng_bar = np.where(h > l, h - l, np.nan)
        d["pos"] = (c - l) / rng_bar
        prev_c = np.r_[np.nan, c[:-1]]
        for _, key in LEVELS:
            L = lv[key]
            d[f"{key}_approach"] = (prev_c < L) & (h >= L) & (c <= L)
            with np.errstate(invalid="ignore", divide="ignore"):
                d[f"{key}_gap"] = (L - c) / a
        d["r_long"] = outcomes(h, l, c, a, +1)
        parts.append(d)
    return pd.concat(parts)


def row(label, sub):
    s = sub.dropna(subset=["r_long"])
    if len(s) < 60:
        print(f"{label:<30}{len(s):>7}   样本太少")
        return
    n, sr = agg(s)
    lo, hi = ci(s)
    star = "  *" if lo > 0 else ""
    print(f"{label:<30}{len(s):>7}{100*(s.r_long > 0).mean():>8.1f}%"
          f"{net(n, sr):>+9.3f}   [{lo:+.3f}, {hi:+.3f}]{star}")


def main():
    d = build_with_geometry()
    days = sorted(d.day.unique())
    oos = []
    for _, te_days in folds(days, anchored=True):
        oos += te_days
    te = d[d.day.isin(oos)]
    base = te[te.dr1.notna() & (te.dr1 >= 0)]

    print("把「自下方到达未破」按「差多远」和「收在哪」拆开。")
    print("全程 1 分钟，只改量，不改尺度。只做多。\n")

    # Pooled across levels first: the pattern was a pattern precisely
    # because it held at all six, so the split should be asked of it too.
    pool = []
    for _, key in LEVELS:
        m = base[f"{key}_approach"].fillna(False)
        s = base[m].dropna(subset=["r_long", f"{key}_gap"]).copy()
        if s.empty:
            continue
        s["gap"] = s[f"{key}_gap"]
        s["lvl"] = key
        pool.append(s[["day", "r_long", "gap", "pos", "lvl"]])
    pool = pd.concat(pool, ignore_index=True)
    print(f"六个价位合并后共 {len(pool):,} 笔")

    print(f"\n{'':<30}{'笔数':>7}{'胜率':>8}{'每笔 R':>9}{'95% 区间':>22}")
    row("全部「自下方到达未破」", pool)

    print("\n  —— 按 gap（收盘离价位差多远，ATR）——")
    qs = pool.gap.quantile([0.25, 0.5, 0.75]).to_list()
    row(f"  差最少 25%（gap ≤ {qs[0]:.2f}）", pool[pool.gap <= qs[0]])
    row(f"  中间 50%", pool[(pool.gap > qs[0]) & (pool.gap <= qs[2])])
    row(f"  差最多 25%（gap > {qs[2]:.2f}）", pool[pool.gap > qs[2]])

    print("\n  —— 按 pos（收盘在本根区间的位置）——")
    row("  收在上 1/3（接近高点）", pool[pool.pos >= 2/3])
    row("  收在中 1/3", pool[(pool.pos >= 1/3) & (pool.pos < 2/3)])
    row("  收在下 1/3（冲高回落）", pool[pool.pos < 1/3])

    print("\n  —— 两者交叉：近似突破 vs 真被拒绝 ——")
    near = (pool.gap <= qs[0]) & (pool.pos >= 2/3)
    rej = (pool.gap > qs[1]) & (pool.pos < 1/3)
    row("  近似突破（差得少+收在高位）", pool[near])
    row("  真被拒绝（差得多+收在低位）", pool[rej])

    print("\n=== 对照：同期的「向上突破」 ===")
    brk = []
    for _, key in LEVELS:
        m = te[f"{key}_approach"].fillna(False)
        _ = m  # approach only; breakout lives in level_states
    print("  （突破状态见 level_states.py，此处只拆未破的那一格）")
    print("\n  * = 95% 区间（按交易日重采样）整个在 0 以上")


if __name__ == "__main__":
    main()

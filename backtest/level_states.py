"""How price ARRIVED at a level, not just that it is near one.

EXPLORATORY, NOT A TEST. Said plainly up front: the sixty out-of-sample
sessions this runs on have been looked at repeatedly today, and this adds
eighteen cells to that. Whatever comes out is a hypothesis to register and
check on data that does not exist yet -- it is not a result.

WHY ARRIVAL DIRECTION. price_location.py found VAL and shelf lower edges
worth +0.43 and +0.23 R, while POC and shelf peaks were worth nothing
(+0.015, +0.058). The reading was "lower edges hold bids, centres carry no
direction". There is a simpler explanation, and it is testable:

  VAL and a shelf's lower edge can essentially only be reached by falling
  to them -- they are lower boundaries, so being there means price came
  down. One arrival direction, nothing to cancel.

  POC, VAH and a shelf's peak are reachable both ways. For a LONG, falling
  to a level and reclaiming it is support; rising into it is a push against
  overhead supply. Those are opposite events, and averaged together they
  produce a null indistinguishable from "this level does nothing".

So the existing nulls are consistent with cancellation, and splitting by
arrival is the way to tell the two apart.

THE STATES are built from the decision bar and the one before it, both
already closed. No tolerance band, no lookback window -- the two tunable
parameters price_location.py needed are gone, which also shrinks the
surface a search could fit to:

    reclaim   close[i-1] > L,  low[i] <= L,  close[i] > L
              fell to it and closed back above -- support held
    breakout  close[i-1] <= L, close[i] > L
              closed above having been below -- the level gave way
    approach  close[i-1] < L,  high[i] >= L, close[i] <= L
              reached it from below and failed to close above
    (neither) excluded, as asked

LONG ONLY, and every state still requires the delta_r flip -- this asks
what the level adds to a signal already in hand, not whether the level
alone is tradeable.

Usage:
    uv run python -m backtest.level_states
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

WARMUP = 60
rng = np.random.default_rng(839)

LEVELS = [("VAL", "val"), ("POC", "poc"), ("VAH", "vah"),
          ("筹码峰下界", "ndlo"), ("筹码峰峰值", "ndpk"),
          ("筹码峰上界", "ndhi")]


def build():
    from feeds import delta_bars
    from feeds.fetcher import fetch_klines
    from strategy.session_vp.profile import (bins_for_price,
                                             compute_value_area, find_nodes)
    from strategy.smc.fvg import compute_volume_profile

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
        o, h, l, c = (r.open.to_numpy(float), r.high.to_numpy(float),
                      r.low.to_numpy(float), r.close.to_numpy(float))
        a = atr(h, l, c, 14)
        buy, sell = r.buy.to_numpy(float), r.sell.to_numpy(float)
        tot = buy + sell
        ok = tot >= MIN_SIGNED
        with np.errstate(invalid="ignore", divide="ignore"):
            dr = np.where(ok, (buy - sell) / tot, np.nan)
        tot_s = pd.Series(np.where(ok, tot, np.nan))
        scale = tot_s.rolling(60, min_periods=20).median().shift(1)
        x = (pd.Series(np.where(ok, buy - sell, np.nan)) / scale).to_numpy()

        n = len(r)
        lv = {key: np.full(n, np.nan) for _, key in LEVELS}
        for i in range(WARMUP, n):
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
        d["atr"] = a
        d["dr1"] = dr
        d["x"] = x
        prev_c = np.r_[np.nan, c[:-1]]
        for _, key in LEVELS:
            L = lv[key]
            # The level is the one known at i; comparing close[i-1] against
            # it rather than against the level as it stood at i-1 keeps the
            # three states mutually exclusive on a moving boundary.
            d[f"{key}_reclaim"] = (prev_c > L) & (l <= L) & (c > L)
            d[f"{key}_break"] = (prev_c <= L) & (c > L)
            d[f"{key}_approach"] = (prev_c < L) & (h >= L) & (c <= L)
            d[f"{key}_ok"] = np.isfinite(L) & np.isfinite(prev_c)
        d["r_long"] = outcomes(h, l, c, a, +1)
        parts.append(d)
    return pd.concat(parts)


def agg(sub):
    g = sub.dropna(subset=["r_long"]).groupby("day")
    a = g.agg(n=("r_long", "size"), sr=("r_long", "sum"))
    return a.n.to_numpy(float), a.sr.to_numpy()


def net(n, sr):
    return np.nan if n.sum() == 0 else sr.sum() / n.sum()


def ci(sub, reps=3000):
    n, sr = agg(sub)
    if len(n) < 5:
        return np.nan, np.nan
    i = rng.integers(0, len(n), size=(reps, len(n)))
    t = n[i].sum(1)
    ok = t > 0
    v = sr[i].sum(1)[ok] / t[ok]
    return np.percentile(v, 2.5), np.percentile(v, 97.5)


def row(label, sub):
    s = sub.dropna(subset=["r_long"])
    if len(s) < 60:
        print(f"{label:<28}{len(s):>7}   样本太少")
        return None
    n, sr = agg(s)
    lo, hi = ci(s)
    star = "  *" if lo > 0 else ""
    r = net(n, sr)
    print(f"{label:<28}{len(s):>7}{100*(s.r_long > 0).mean():>8.1f}%"
          f"{r:>+9.3f}   [{lo:+.3f}, {hi:+.3f}]{star}")
    return (label, len(s), r, lo)


def main():
    d = build()
    days = sorted(d.day.unique())
    oos = []
    for _, te_days in folds(days, anchored=True):
        oos += te_days
    te = d[d.day.isin(oos)]

    print("探索性，非检验。这 60 天今天已被反复查看，本轮再加 18 个格子；")
    print("看着好的只能进预先登记清单，等新数据验，不能当结论。\n")

    for gate_name, gate in (("dr1≥0", te.dr1 >= 0),
                            ("dr1≥0.4", te.dr1 >= 0.4)):
        base = te[te.dr1.notna() & gate]
        print(f"=== 条件：{gate_name}（{len(base):,} 笔）===")
        print(f"{'状态':<28}{'笔数':>7}{'胜率':>8}{'每笔 R':>9}"
              f"{'95% 区间':>22}")
        row("全部（不分位置）", base)
        found = []
        for nm, key in LEVELS:
            ok = base[f"{key}_ok"]
            for state, zh in (("reclaim", "跌到此并收复"),
                              ("break", "向上突破"),
                              ("approach", "自下方到达未破")):
                got = row(f"  {nm} · {zh}",
                          base[ok & base[f"{key}_{state}"]])
                if got:
                    found.append(got)
        print()

    print("=== 本轮候选（样本外区间下界 > 0，按下界排序）===")
    print("  下界高的优先登记——它对多重比较更稳，而不只是均值好看。")
    base = te[te.dr1.notna() & (te.dr1 >= 0)]
    cands = []
    for nm, key in LEVELS:
        ok = base[f"{key}_ok"]
        for state, zh in (("reclaim", "跌到此并收复"),
                          ("break", "向上突破"),
                          ("approach", "自下方到达未破")):
            s = base[ok & base[f"{key}_{state}"]].dropna(subset=["r_long"])
            if len(s) < 60:
                continue
            lo, _ = ci(s)
            if lo > 0:
                cands.append((lo, net(*agg(s)), len(s), f"{nm} · {zh}"))
    for lo, r, n, nm in sorted(cands, reverse=True):
        print(f"  {nm:<26}{n:>7}{r:>+9.3f}   下界 {lo:+.3f}")
    if not cands:
        print("  （无）")
    print("\n  * = 95% 区间（按交易日重采样）整个在 0 以上")


if __name__ == "__main__":
    main()

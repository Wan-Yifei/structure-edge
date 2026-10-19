"""Where price sits in the session's own volume profile, on top of the flip.

PRE-REGISTERED. Written before any number was read. LONG ONLY, as asked.

What is already in hand: the flip's strength (|dr1|, coefficient +0.38 R)
and the flow behind it (|x|, +0.05). Run length added nothing once those
were held, and the sign sequence turned out to be first-order Markov.

What this asks: does it matter WHERE the flip happens? A flip at the lower
edge of a volume shelf, or at VAL, is supposed to be a different event
from the same flip in thin air -- there is size resting there, and a
reversal that starts from it has something to lean on.

TWO FAMILIES OF LEVEL, and they are not the same object.

  VALUE AREA   one band holding 70% of the session's volume around the
               POC. On a day that traded two separate shelves it stretches
               across the gap, so its VAL can sit at a price where nothing
               changed hands.
  NODES        local shelves from find_nodes(): each has its own lo, hi
               and peak. A session with two shelves has two nodes. This is
               the "筹码峰" of the question, and its lo is the lower edge.

Both are reported. Conflating them is how a level that means something
gets averaged with one that does not.

NO LOOKAHEAD: the profile at bar i is built from the session open through
bar i, rebinned each bar. Nothing after i is visible.

"AT" A LEVEL means within AT_ATR of it, measured in ATR so the tolerance
travels across volatility regimes. 0.3 is a pre-specified guess, not a
measured one; 0.2 and 0.5 are reported beside it so the reader can see
whether the answer turns on that choice.

THE CONTROL THAT MATTERS: position correlates with the drop that preceded
it -- price is at VAL because it fell to VAL. The regression therefore
carries sig and xmag alongside, and the question is whether the position
terms add anything once the signal's own strength is held.

Usage:
    uv run python -m backtest.price_location
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

AT_ATR = 0.3          # "at" a level: within this many ATR
WARMUP = 60           # bars before a profile is worth computing
rng = np.random.default_rng(727)


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
        h, l, c = (r.high.to_numpy(float), r.low.to_numpy(float),
                   r.close.to_numpy(float))
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
        poc = np.full(n, np.nan)
        val = np.full(n, np.nan)
        vah = np.full(n, np.nan)
        nd_lo = np.full(n, np.nan)      # lower edge of the containing or
        nd_pk = np.full(n, np.nan)      # nearest shelf, and its peak
        nd_in = np.zeros(n, dtype=bool)
        for i in range(WARMUP, n):
            up = r.iloc[:i + 1]
            nb = bins_for_price(float(up.low.min()), float(up.high.max()))
            if not nb:
                continue
            edges, vols = compute_volume_profile(up, n_bins=nb)
            if edges is None:
                continue
            va = compute_value_area(edges, vols, va_pct=0.70)
            poc[i], val[i], vah[i] = va["poc"], va["val"], va["vah"]
            nodes = find_nodes(edges, vols)
            if not nodes:
                continue
            px = c[i]
            inside = [nd for nd in nodes if nd.lo <= px <= nd.hi]
            pick = (inside[0] if inside
                    else min(nodes, key=lambda z: abs(z.peak - px)))
            nd_lo[i], nd_pk[i] = pick.lo, pick.peak
            nd_in[i] = bool(inside)

        d = pd.DataFrame(index=r.index)
        d["day"] = str(day.date())
        d["atr"] = a
        d["dr1"] = dr
        d["x"] = x
        d["close"] = c
        with np.errstate(invalid="ignore", divide="ignore"):
            d["d_poc"] = (c - poc) / a
            d["d_val"] = (c - val) / a
            d["d_vah"] = (c - vah) / a
            d["d_ndlo"] = (c - nd_lo) / a
            d["d_ndpk"] = (c - nd_pk) / a
        d["in_node"] = nd_in
        d["r_long"] = outcomes(h, l, c, a, +1)
        parts.append(d)
    return pd.concat(parts)


def agg(sub, col="r_long"):
    g = sub.dropna(subset=[col]).groupby("day")
    a = g.agg(n=(col, "size"), sr=(col, "sum"))
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


def row(label, sub, base_n):
    s = sub.dropna(subset=["r_long"])
    if len(s) < 80:
        print(f"{label:<26}{len(s):>7}   样本太少")
        return
    n, sr = agg(s)
    lo, hi = ci(s)
    star = "  *" if lo > 0 else ""
    print(f"{label:<26}{len(s):>7}{len(s)/base_n:>7.1%}"
          f"{100*(s.r_long > 0).mean():>8.1f}%{net(n, sr):>+9.3f}"
          f"   [{lo:+.3f}, {hi:+.3f}]{star}")


def va_bucket(s, tol):
    """Mutually exclusive position labels, low to high."""
    return [
        (f"低于 VAL 超过 {tol} ATR", s.d_val < -tol),
        ("贴 VAL", s.d_val.abs() <= tol),
        ("VAL 与 POC 之间", (s.d_val > tol) & (s.d_poc < -tol)),
        ("贴 POC", s.d_poc.abs() <= tol),
        ("POC 与 VAH 之间", (s.d_poc > tol) & (s.d_vah < -tol)),
        ("贴 VAH", s.d_vah.abs() <= tol),
        (f"高于 VAH 超过 {tol} ATR", s.d_vah > tol),
    ]


def node_bucket(s, tol):
    return [
        ("贴筹码峰下边界", s.d_ndlo.abs() <= tol),
        ("贴筹码峰峰值", s.d_ndpk.abs() <= tol),
        ("在筹码峰内，不贴边",
         s.in_node & (s.d_ndlo.abs() > tol) & (s.d_ndpk.abs() > tol)),
        ("不在任何筹码峰内", ~s.in_node),
    ]


def fit(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]


def regress(df, cols, title, reps=1500):
    days = df.day.unique()
    by = {d: g for d, g in df.groupby("day")}
    X = np.column_stack([np.ones(len(df))] + [df[c].to_numpy() for c in cols])
    y = df.r_long.to_numpy()
    beta = fit(X, y)
    out = np.full((reps, len(cols) + 1), np.nan)
    for j in range(reps):
        pick = rng.integers(0, len(days), len(days))
        g = pd.concat([by[days[i]] for i in pick], ignore_index=True)
        Xb = np.column_stack(
            [np.ones(len(g))] + [g[c].to_numpy() for c in cols])
        try:
            out[j] = fit(Xb, g.r_long.to_numpy())
        except np.linalg.LinAlgError:
            pass
    lo = np.nanpercentile(out, 2.5, axis=0)
    hi = np.nanpercentile(out, 97.5, axis=0)
    print(f"\n{title}")
    print(f"{'变量':<14}{'系数':>10}{'95% 区间':>24}")
    for i, nm in enumerate(["常数"] + cols):
        star = "  *" if (lo[i] > 0 or hi[i] < 0) else ""
        print(f"{nm:<14}{beta[i]:>+10.4f}   [{lo[i]:+.4f}, {hi[i]:+.4f}]{star}")
    resid = y - X @ beta
    print(f"  R² = {1.0 - resid.var() / y.var():.5f}")


def main():
    d = build()
    days = sorted(d.day.unique())
    oos = []
    for _, te_days in folds(days, anchored=True):
        oos += te_days
    te = d[d.day.isin(oos)]
    have = te.dr1.notna() & te.d_poc.notna()
    print(f"样本外 {te.day.nunique()} 个交易日 / {int(have.sum()):,} 分钟"
          f"（有剖面且有 delta_r）")
    print("只做多。止损 1 ATR，目标 1.5 ATR，30 根时间止损，价差已扣")
    print(f"「贴」= 距该价位 {AT_ATR} ATR 以内；剖面自开盘算到当根，无前视\n")

    for gate_name, gate in (("翻转 dr1≥0", te.dr1 >= 0),
                            ("验证过的 dr1≥0.4", te.dr1 >= 0.4)):
        s = te[have & gate]
        base_n = len(s)
        print(f"=== 条件：{gate_name}（{base_n:,} 笔）===")
        print(f"{'位置':<26}{'笔数':>7}{'占比':>7}{'胜率':>8}"
              f"{'每笔 R':>9}{'95% 区间':>22}")
        row("全部（不分位置）", s, base_n)
        print("  —— 价值区 ——")
        for nm, m in va_bucket(s, AT_ATR):
            row("  " + nm, s[m], base_n)
        print("  —— 筹码峰 ——")
        for nm, m in node_bucket(s, AT_ATR):
            row("  " + nm, s[m], base_n)
        print()

    print("=== 容差敏感度：「贴」的定义换成 0.2 / 0.5 ATR ===")
    s = te[have & (te.dr1 >= 0.4)]
    print(f"{'':<26}{'笔数':>7}{'每笔 R':>9}")
    for tol in (0.2, 0.3, 0.5):
        for nm, m in (("贴 VAL", s.d_val.abs() <= tol),
                      ("贴 POC", s.d_poc.abs() <= tol),
                      ("贴筹码峰下边界", s.d_ndlo.abs() <= tol)):
            sub = s[m].dropna(subset=["r_long"])
            if len(sub) < 80:
                print(f"{f'{nm} @{tol}':<26}{len(sub):>7}   样本太少")
                continue
            print(f"{f'{nm} @{tol}':<26}{len(sub):>7}"
                  f"{net(*agg(sub)):>+9.3f}")

    print("\n=== 控制：位置在信号强度之外还剩多少 ===")
    reg = te[have & (te.dr1 >= 0)].dropna(
        subset=["r_long", "d_poc", "d_val", "d_ndlo", "d_ndpk", "x"]).copy()
    reg["sig"] = reg.dr1.abs()
    reg["xmag"] = reg.x.abs()
    reg["at_val"] = (reg.d_val.abs() <= AT_ATR).astype(float)
    reg["at_poc"] = (reg.d_poc.abs() <= AT_ATR).astype(float)
    reg["at_ndlo"] = (reg.d_ndlo.abs() <= AT_ATR).astype(float)
    reg["at_ndpk"] = (reg.d_ndpk.abs() <= AT_ATR).astype(float)
    reg["below_val"] = (reg.d_val < -AT_ATR).astype(float)
    regress(reg, ["sig", "xmag"], "只有信号强度（基准模型）")
    regress(reg, ["sig", "xmag", "at_val", "at_poc", "at_ndlo", "at_ndpk",
                  "below_val"], "加入位置")
    print("\n  * = 95% 区间（按交易日重采样）不跨 0")


if __name__ == "__main__":
    main()

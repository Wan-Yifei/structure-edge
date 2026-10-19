"""Does run length survive once everything it correlates with is held?

PRE-REGISTERED. Written before any coefficient was read.

The bucket test in run_length.py put k against the preceding drop in five
quantiles and came back two buckets up, two down -- with 402 / 208 / 137 /
77 trades in the k>=4 cells. That is "not shown", not "shown absent": a
coarse control with thin cells cannot separate a small effect from none.

Here the confounder goes in as a CONTINUOUS variable alongside k, so the
coefficient on k is what one more bar of run is worth with the drop, the
signal strength and the flow magnitude all held fixed. One number with an
interval, rather than four cells pointing different ways.

VARIABLES, all expressed relative to the signal's own direction so long
and short flips pool:

    k      opposite-sign run length immediately before the flip, capped
    sig    |dr1| on the flip bar -- how strong the signal itself is
    adv    how far price moved AGAINST the coming direction over the
           prior 5 bars, in ATR. Positive = fell before a long. This is
           the confounder: k and adv are nearly the same variable
    xmag   |x|, flow size against its own recent normal
    short  direction dummy, in case the two sides sit at different levels

RESPONSE: R net of spread, in the signal's direction.

INTERVALS BY SESSION BOOTSTRAP, not OLS standard errors. Minutes inside a
day are not independent -- consecutive trades share most of their forward
window -- so the textbook standard error would be far too narrow. The
whole fit is repeated on resampled days.

R-SQUARED WILL BE TINY and that is not a verdict. Per-trade returns are
mostly noise; a coefficient can be real and stable while explaining a
thousandth of the variance. The interval on the coefficient is the thing
to read, not the fit.

THE TIME-SERIES PART. run_length.py already showed the sign sequence is
clustered: P(red | k reds) runs 0.54 -> 0.64 against a null of 0.50. The
question a sequence model would answer is whether memory beyond the last
bar helps, and the direct form of that is a MARKOV ORDER TEST -- does
P(next | last m signs) keep improving as m grows? That is reported here
instead of an LSTM, which was already tried this session on the same tape
and lost to delta_r alone (AUC 0.528 against 0.639).

Usage:
    uv run python -m backtest.run_regress
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

_ROOT = pathlib.Path(__file__).parent.parent
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(_ROOT))

from backtest.run_length import build                       # noqa: E402
from backtest.walk_forward import folds                     # noqa: E402

K_CAP = 8
rng = np.random.default_rng(613)


def design(te):
    """One row per flip, pooled across directions."""
    rows = []
    for side, run_col, col in ((+1, "neg_run", "r_long"),
                               (-1, "pos_run", "r_short")):
        m = te.dr1.notna() & (te[run_col] >= 1)
        m &= (te.dr1 >= 0) if side > 0 else (te.dr1 < 0)
        s = te[m].dropna(subset=[col, "prev_ret", "x"])
        if s.empty:
            continue
        rows.append(pd.DataFrame({
            "day": s.day.to_numpy(),
            "R": s[col].to_numpy(),
            "k": np.minimum(s[run_col].to_numpy(), K_CAP),
            "sig": s.dr1.abs().to_numpy(),
            # Adverse: price moving against the coming trade.
            "adv": (-side * s.prev_ret).to_numpy(),
            "xmag": s.x.abs().to_numpy(),
            "short": 0.0 if side > 0 else 1.0,
        }))
    return pd.concat(rows, ignore_index=True)


def fit(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]


def boot_coefs(df, cols, reps=2000):
    """Refit on resampled SESSIONS; returns (coef, lo, hi) per column."""
    days = df.day.unique()
    by = {d: g for d, g in df.groupby("day")}
    X = np.column_stack([np.ones(len(df))] + [df[c].to_numpy() for c in cols])
    beta = fit(X, df.R.to_numpy())
    out = np.full((reps, len(cols) + 1), np.nan)
    for j in range(reps):
        pick = rng.integers(0, len(days), len(days))
        g = pd.concat([by[days[i]] for i in pick], ignore_index=True)
        Xb = np.column_stack(
            [np.ones(len(g))] + [g[c].to_numpy() for c in cols])
        try:
            out[j] = fit(Xb, g.R.to_numpy())
        except np.linalg.LinAlgError:
            pass
    lo = np.nanpercentile(out, 2.5, axis=0)
    hi = np.nanpercentile(out, 97.5, axis=0)
    return beta, lo, hi


def r2(df, cols):
    X = np.column_stack([np.ones(len(df))] + [df[c].to_numpy() for c in cols])
    y = df.R.to_numpy()
    resid = y - X @ fit(X, y)
    return 1.0 - resid.var() / y.var()


def report(df, cols, title):
    beta, lo, hi = boot_coefs(df, cols)
    print(f"\n{title}")
    print(f"{'变量':<10}{'系数':>10}{'95% 区间':>24}")
    for i, nm in enumerate(["常数"] + cols):
        star = "  *" if (lo[i] > 0 or hi[i] < 0) else ""
        print(f"{nm:<10}{beta[i]:>+10.4f}   [{lo[i]:+.4f}, {hi[i]:+.4f}]{star}")
    print(f"  R² = {r2(df, cols):.5f}")
    return beta, lo, hi


def markov_order(te, max_m=4):
    """Does knowing more than the last sign help predict the next one?"""
    print("\n=== 时序：马尔可夫阶数检验 ===")
    print("  只看符号序列。m 阶模型用前 m 根的符号组合去预测下一根。")
    print("  命中率若随 m 停止上升，说明更长的记忆不含额外信息。")
    seqs = []
    for _, g in te.groupby("day"):
        v = g.dr1.to_numpy()
        s = np.where(np.isfinite(v), (v >= 0).astype(int), -1)
        seqs.append(s)
    print(f"{'阶数 m':>8}{'状态数':>8}{'样本':>9}{'命中率':>9}{'较 m−1':>9}")
    prev_acc = None
    for m in range(0, max_m + 1):
        hit = tot = 0
        table: dict[tuple, list[int]] = {}
        for s in seqs:                       # first pass: counts
            for i in range(m, len(s)):
                if s[i] < 0 or (m and (s[i-m:i] < 0).any()):
                    continue
                key = tuple(s[i-m:i]) if m else ()
                table.setdefault(key, [0, 0])[s[i]] += 1
        for s in seqs:                       # second pass: score
            for i in range(m, len(s)):
                if s[i] < 0 or (m and (s[i-m:i] < 0).any()):
                    continue
                key = tuple(s[i-m:i]) if m else ()
                c = table.get(key)
                if not c or sum(c) < 30:
                    continue
                hit += int((c[1] > c[0]) == bool(s[i]))
                tot += 1
        acc = hit / tot if tot else float("nan")
        delta = "" if prev_acc is None else f"{acc - prev_acc:>+9.4f}"
        print(f"{m:>8}{len(table):>8}{tot:>9}{acc:>9.4f}{delta:>9}")
        prev_acc = acc
    print("  注：同一批数据既建表又计分，命中率天然偏高，阶数越大偏得越多。")
    print("  要看的是增量在哪一阶变平，不是绝对值。")


def main():
    d = build()
    days = sorted(d.day.unique())
    oos = []
    for _, te_days in folds(days, anchored=True):
        oos += te_days
    te = d[d.day.isin(oos)]
    df = design(te)
    print(f"样本外 {te.day.nunique()} 个交易日，翻转 {len(df):,} 笔"
          f"（做多 {int((df['short'] == 0).sum()):,} / "
          f"做空 {int((df['short'] == 1).sum()):,}）")
    print("响应变量 R = 每笔净 R（止损 1 ATR，目标 1.5 ATR，30 根时间止损）")

    print("\n=== 一、逐个加入，看 k 的系数怎么变 ===")
    report(df, ["k"], "只有 k —— 这是没有任何控制的版本")
    report(df, ["k", "adv"], "加入 adv（翻转前的逆向走幅）")
    report(df, ["k", "adv", "sig", "xmag", "short"], "全模型")

    print("\n=== 二、k 是不是线性的 ===")
    print("  把 k 拆成哑变量，基准是 k=1。若系数不单调，线性设定本身就不对。")
    dd = df.copy()
    for kk in range(2, K_CAP + 1):
        dd[f"k{kk}"] = (dd.k == kk).astype(float) if kk < K_CAP \
            else (dd.k >= kk).astype(float)
    kcols = [f"k{kk}" for kk in range(2, K_CAP + 1)]
    report(dd, kcols + ["adv", "sig", "xmag", "short"],
           "k 哑变量（相对 k=1）+ 全部控制")

    print("\n=== 三、k 到底贡献了多少解释力 ===")
    base = ["adv", "sig", "xmag", "short"]
    print(f"  不含 k：R² = {r2(df, base):.5f}")
    print(f"  含  k：R² = {r2(df, base + ['k']):.5f}")
    print(f"  增量  ：{r2(df, base + ['k']) - r2(df, base):+.5f}")

    markov_order(te)
    print("\n  * = 系数的 95% 区间不跨 0（按交易日重采样 2000 次）")


if __name__ == "__main__":
    main()

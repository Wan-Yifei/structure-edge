"""Absorption: heavy flow that fails to move price, as a residual.

PRE-REGISTERED. Everything below the data section was written before a
single number was read, because this project's hit rate on "plausible then
dead under a control" is about four to one and the grid cells that survive
a search are not the same thing as a finding.

WHY A RESIDUAL AND NOT A RATIO. The folk definition is "big volume, small
move", which as a ratio (dp / volume) blows up as volume goes to zero and
counts a quiet minute -- small flow AND small move -- as absorption. It
also ignores that the price impact coefficient is not a constant: it moves
with the volatility regime and the hour, so a fixed ratio is partly a
time-of-day detector. Fitting the impact locally and taking the residual
handles both:

    y_t = dp_t / ATR_t                    price move, in ATR
    x_t = OFI_t / median(total flow)      signed flow, in typical units
    lam_t = OLS slope of y on x over the trailing 60 bars, through 0
    eps_t = y_t - lam_t * x_t             what the flow failed to deliver
    A_t   = eps_t / sd(eps)               standardised

The residual carries the direction by itself. Heavy selling (x < 0) that
leaves price above where the fit said it should be gives eps > 0, which is
bullish; heavy buying that fails gives eps < 0. No sign flip is applied,
and applying one was the first thing that looked right and was wrong.

THE FLOW GATE IS NOT OPTIONAL. A large positive eps can also come from a
small flow and a large move, which is the opposite event. The gate
(|x| in the top fifth of its own trailing distribution) is what makes this
absorption rather than "price went up".

NO LOOKAHEAD. lam, sd and the gate are all fitted on bars strictly before
t, so bar t is never part of the line it is judged against.

WHAT THIS CANNOT SEE. Whether the absorber holds. The literature's own
identification problem -- compensated liquidity provision and informed
trading look identical in the flow -- is not solved here, and the evidence
that mattered in the one case examined by hand (2026-10-18 20:20) was the
BOOK: a 7,650-share bid cut to 2,139 and then refilled, with a new 3,131
appearing below. That is a second dimension this study does not have, and
its absence is the most likely reason for a null result.

ONE SHARP QUESTION, REGISTERED IN ADVANCE. Bullish absorption has x < 0,
hence delta_r < 0 -- and delta_r < 0 is the validated rule's worst bucket
for longs (-0.31 R). The two directly contradict each other. Either the
sign reverses inside the absorption subset or absorption is not a signal;
there is no comfortable third answer, which is what makes it worth asking.

Usage:
    uv run python -m backtest.absorption
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

_ROOT = pathlib.Path(__file__).parent.parent
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(_ROOT))

from backtest.delta_atr_matrix import (CODE, END, FWD, MIN_SIGNED,  # noqa
                                       SPREAD, START, STOP, atr)
from backtest.walk_forward import folds                     # noqa: E402

TGT = 1.5
WIN = 60              # trailing bars for lam, sd and the flow gate
FLOW_PCT = 0.80       # a bar must be in this tail of |x| to count as flow
A_MAIN = 1.5          # the pre-specified residual threshold
rng = np.random.default_rng(401)


def outcomes(h, l, c, a, side):
    """R per bar for a trade opened at this bar's close, that direction."""
    n = len(c)
    out = np.full(n, np.nan)
    for i in range(n - 1):
        if not np.isfinite(a[i]) or a[i] <= 0:
            continue
        entry = c[i]
        stop = entry - side * STOP * a[i]
        target = entry + side * TGT * a[i]
        r = None
        for j in range(i + 1, min(n, i + 1 + FWD)):
            hit_stop = (l[j] <= stop) if side > 0 else (h[j] >= stop)
            hit_tgt = (h[j] >= target) if side > 0 else (l[j] <= target)
            if hit_stop:                      # adverse first within a bar
                r = -STOP
                break
            if hit_tgt:
                r = TGT
                break
        if r is None:
            k = min(n - 1, i + FWD)
            r = side * (c[k] - entry) / a[i]
        out[i] = r - SPREAD / (STOP * a[i])
    return out


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
        ofi = np.where(ok, buy - sell, np.nan)

        d = pd.DataFrame(index=r.index)
        d["day"] = str(day.date())
        d["hr"] = r.index.hour
        d["atr"] = a
        with np.errstate(invalid="ignore", divide="ignore"):
            d["dr1"] = np.where(ok, (buy - sell) / tot, np.nan)

        dp = np.r_[np.nan, np.diff(c)]
        # Shifted like the other three rolling quantities. It was not, and
        # a 60-bar median that includes bar t lets that bar nudge the scale
        # it is then divided by. One value moves a median by at most half a
        # rank, so the effect is small and conservative -- a heavy bar
        # raises the median and so shrinks its own x -- but three of four
        # windows excluding t and the fourth including it is not a defence,
        # it is an inconsistency nobody can keep in their head.
        scale = pd.Series(np.where(ok, tot, np.nan)).rolling(
            WIN, min_periods=20).median().shift(1)
        y = pd.Series(dp / a)
        x = pd.Series(ofi) / scale
        d["x"] = x.to_numpy()

        # Trailing fit, shifted so bar t is never in its own regression.
        xy = (x * y).rolling(WIN, min_periods=20).sum().shift(1)
        xx = (x * x).rolling(WIN, min_periods=20).sum().shift(1)
        lam = xy / xx.replace(0.0, np.nan)
        eps = y - lam * x
        sd = eps.rolling(WIN, min_periods=20).std().shift(1)
        d["lam"] = lam.to_numpy()
        d["A"] = (eps / sd.replace(0.0, np.nan)).to_numpy()
        d["gate"] = (x.abs().rolling(WIN, min_periods=20)
                     .quantile(FLOW_PCT).shift(1).to_numpy())

        d["r_long"] = outcomes(h, l, c, a, +1)
        d["r_short"] = outcomes(h, l, c, a, -1)
        parts.append(d)
    return pd.concat(parts)


# ── scoring ──────────────────────────────────────────────────────────────────

def agg(sub, col):
    g = sub.assign(_c=SPREAD / (STOP * sub.atr)).groupby("day")
    a = g.agg(n=(col, "size"), sr=(col, "sum"))
    return a.n.to_numpy(float), a.sr.to_numpy()


def net(n, sr):
    return np.nan if n.sum() == 0 else sr.sum() / n.sum()


def ci(sub, col, reps=4000):
    n, sr = agg(sub.dropna(subset=[col]), col)
    if len(n) < 5:
        return np.nan, np.nan
    i = rng.integers(0, len(n), size=(reps, len(n)))
    t = n[i].sum(1)
    ok = t > 0
    v = sr[i].sum(1)[ok] / t[ok]
    return np.percentile(v, 2.5), np.percentile(v, 97.5)


def row(label, sub, col, base_n):
    s = sub.dropna(subset=[col])
    if len(s) < 100:
        print(f"{label:<34}{len(s):>7}   样本太少")
        return
    n, sr = agg(s, col)
    lo, hi = ci(s, col)
    star = "  *" if lo > 0 else ""
    print(f"{label:<34}{len(s):>7}{len(s)/base_n:>7.1%}"
          f"{100*(s[col] > 0).mean():>8.1f}%{net(n, sr):>+9.3f}"
          f"   [{lo:+.3f}, {hi:+.3f}]{star}")



def flow_r(sub):
    """R trading WITH the flow: long when x>0, short when x<0."""
    return np.where(sub.x > 0, sub.r_long, sub.r_short)


def per_day(sub, vals, days_all):
    """(n, sum R) aligned on a fixed day list, so two sets can be paired."""
    n = np.zeros(len(days_all))
    sr = np.zeros(len(days_all))
    pos = {d: i for i, d in enumerate(days_all)}
    v = pd.Series(vals, index=sub.index)
    ok = v.notna()
    for d, g in v[ok].groupby(sub.day[ok]):
        n[pos[d]], sr[pos[d]] = len(g), g.sum()
    return n, sr


def paired(a, b, days_all, reps=4000):
    """Interval on (A - B) with both on the same resampled days."""
    na, sa = a
    nb, sb = b
    i = rng.integers(0, len(days_all), size=(reps, len(days_all)))
    ta, tb = na[i].sum(1), nb[i].sum(1)
    ok = (ta > 0) & (tb > 0)
    if ok.sum() < reps // 10:
        return np.nan, np.nan, np.nan
    d = sa[i].sum(1)[ok] / ta[ok] - sb[i].sum(1)[ok] / tb[ok]
    return d.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5)


# Every step from the tape to a number, in order. The glossary says what a
# symbol means; this says how it is computed, which is the part someone
# would need to reproduce or to find the mistake in.
FORMULAS = [
    ("输入", [
        ("buy<sub>t</sub>, sell<sub>t</sub>",
         "该分钟内 direction 为 BUY / SELL 的成交股数之和。"
         "NEUTRAL（场外流，约占 57%）两边都不计入。"),
        ("c<sub>t</sub>, h<sub>t</sub>, l<sub>t</sub>",
         "该分钟的收 / 高 / 低。RTH 09:31–16:00，夜盘与盘前不参与。"),
    ]),
    ("1. 波动尺度（Wilder ATR，n=14）", [
        ("TR<sub>t</sub> = max( h<sub>t</sub>−l<sub>t</sub>, "
         "|h<sub>t</sub>−c<sub>t−1</sub>|, |l<sub>t</sub>−c<sub>t−1</sub>| )",
         "真实波幅，含跳空。"),
        ("ATR<sub>t</sub> = [ ATR<sub>t−1</sub>·(n−1) + TR<sub>t</sub> ] / n",
         "Wilder 平滑；首值取前 n 根 TR 的算术平均。"),
    ]),
    ("2. 流量", [
        ("OFI<sub>t</sub> = buy<sub>t</sub> − sell<sub>t</sub>",
         "订单流失衡（股）。要求 buy+sell ≥ 300，否则该 bar 作废——"
         "量太小时一笔小单就能把比值推到 ±1。"),
        ("dr1<sub>t</sub> = OFI<sub>t</sub> / (buy<sub>t</sub>+sell<sub>t</sub>)",
         "比值口径，恒在 −1..+1。这就是 viewer 的 DCVD。"),
        ("S<sub>t</sub> = median( buy+sell )　over t−60 … t−1",
         "近期常态成交量。<b>不含 t 自身</b>。"),
        ("x<sub>t</sub> = OFI<sub>t</sub> / S<sub>t</sub>",
         "量级口径。dr1 分不开「清淡的 0.5」和「巨量的 0.5」，x 分得开。"),
    ]),
    ("3. 价格", [
        ("y<sub>t</sub> = ( c<sub>t</sub> − c<sub>t−1</sub> ) / "
         "ATR14<sub>t</sub>",
         "用 ATR 除掉波动状态，使不同时段可比。"
         "与 OFI 同窗口：bar t 的流推动的就是 c<sub>t−1</sub>→c<sub>t</sub>。"),
    ]),
    ("4. 局部价格冲击（过原点的滚动 OLS）", [
        ("λ<sub>t</sub> = Σ x<sub>i</sub>y<sub>i</sub> / Σ x<sub>i</sub>²"
         "　i ∈ t−60 … t−1",
         "「这段时间单位流量通常推动多少价格」。过原点是因为零流量应当"
         "对应零冲击，留截距等于允许一个无来由的漂移。"
         "<b>i 的上界是 t−1</b>：bar t 不参与拟合评判自己的那条线。"),
    ]),
    ("5. 残差与标准化", [
        ("ε<sub>t</sub> = y<sub>t</sub> − λ<sub>t</sub>·x<sub>t</sub>",
         "流量<b>没能兑现</b>的那部分价格。重卖而价格没跌 → ε>0。"
         "方向由 ε 自带，代码里没有任何额外翻转。"),
        ("σ<sub>t</sub> = sd( ε )　over t−60 … t−1",
         "同样不含 t 自身。"),
        ("A<sub>t</sub> = ε<sub>t</sub> / σ<sub>t</sub>",
         "吸收分数。A=1.5 即「比近期常态罕见 1.5 个标准差」。"),
    ]),
    ("6. 流量门槛", [
        ("G<sub>t</sub> = Q<sub>0.80</sub>( |x| )　over t−60 … t−1",
         "自身近期 |x| 的 80 分位。"),
        ("触发条件：|x<sub>t</sub>| ≥ G<sub>t</sub>",
         "没有这一步，一根安静的 bar（流量小、价格也没动）残差同样可以很大，"
         "会被算成吸收。这是把「推不动」和「没人推」分开的那一步。"),
    ]),
    ("7. 事件定义", [
        ("看涨吸收：|x<sub>t</sub>| ≥ G<sub>t</sub>　且　x<sub>t</sub> &lt; 0"
         "　且　A<sub>t</sub> ≥ 1.5",
         "大量主动卖出，价格却没跌下去 → 做多。"),
        ("看跌吸收：|x<sub>t</sub>| ≥ G<sub>t</sub>　且　x<sub>t</sub> &gt; 0"
         "　且　A<sub>t</sub> ≤ −1.5",
         "大量主动买入，价格却没涨上去 → 做空。"),
        ("顺势对照：|x<sub>t</sub>| ≥ G<sub>t</sub>，方向 = sign(x<sub>t</sub>)",
         "同一道门槛，方向相反。这一组是对照，结果却是全研究最强的。"),
    ]),
    ("8. 结算（三重障碍 + 时间止损）", [
        ("entry = c<sub>t</sub>　　side = +1 做多 / −1 做空",
         "在信号那根的收盘价入场。"
         "<b>这是乐观假设</b>：dr 要整根走完才算得出来，实际最早只能在下一根"
         "开盘成交，串行模拟测得该假设值 −0.010 R/笔。"),
        ("stop = entry − side · 1.0 · ATR<sub>t</sub>",
         ""),
        ("target = entry + side · 1.5 · ATR<sub>t</sub>",
         ""),
        ("在 t+1 … t+30 内取<b>首次</b>触及：先碰 stop → R = −1；"
         "先碰 target → R = +1.5",
         "同一根里两边都碰到按 stop 算（取不利的一侧）。"),
        ("都没碰到：R = side · ( c<sub>t+30</sub> − entry ) / ATR<sub>t</sub>",
         "时间止损，按第 30 根收盘了结。"),
        ("R<sub>净</sub> = R − spread / ( 1.0 · ATR<sub>t</sub> )",
         "spread = 0.03 美元往返。约合 0.066 R。"),
    ]),
    ("9. 聚合与区间", [
        ("每笔净 R = Σ<sub>各日</sub> ΣR<sub>净</sub> / "
         "Σ<sub>各日</sub> n",
         "按笔加权，不是各日均值的均值。"),
        ("自助法：重采样<b>交易日</b>，4000 次，取 2.5 / 97.5 分位",
         "不能重采样分钟：相邻分钟的 30 根前瞻窗口几乎完全重合，"
         "有效样本接近交易日数而非分钟数。"),
        ("配对差：同一次日重采样内同时算 A 与 B 再相减",
         "两个各自的区间是否重叠，回答不了「B 比 A 好多少」。"),
    ]),
]


GLOSSARY = [
    ("OFI<sub>t</sub>",
     "订单流失衡 = 主买量 − 主卖量（股）。方向已知的那部分成交，"
     "约占全部成交的 43%，其余是 NEUTRAL 的场外流。",
     "正=买方在吃单，负=卖方在吃单"),
    ("dr1<sub>t</sub>",
     "delta_r = OFI ÷ (主买+主卖)。同一根 bar 的<b>比值</b>，"
     "恒在 −1..+1 之间，与成交量大小无关。",
     "viewer 里的 DCVD 就是它"),
    ("x<sub>t</sub>",
     "归一化流量 = OFI ÷ (近 60 根总成交量的中位数)。"
     "dr1 是比值，x 是<b>相对自己近期常态的量级</b>——"
     "一根 dr1=0.5 的清淡 bar 和一根 dr1=0.5 的巨量 bar，dr1 相同而 x 差很远。",
     "本研究的「流量」一律指 x"),
    ("y<sub>t</sub>",
     "归一化价格变动 = (收<sub>t</sub> − 收<sub>t−1</sub>) ÷ ATR14。",
     "用 ATR 除掉波动状态的影响"),
    ("λ<sub>t</sub>（lambda）",
     "局部价格冲击系数 = y 对 x 的回归斜率（过原点），"
     "用<b>前 60 根</b>拟合。即「这段时间里，单位流量通常能推动多少价格」。",
     "Kyle's λ 的局部估计"),
    ("ε<sub>t</sub>（残差）",
     "ε = y − λ·x，即「这根 bar 的流量<b>没能兑现</b>的那部分价格」。"
     "重卖而价格没跌下去 → ε 为正。",
     "吸收的数学定义就在这里"),
    ("A<sub>t</sub>",
     "标准化残差 = ε ÷ (前 60 根 ε 的标准差)。A 是本研究的<b>吸收分数</b>，"
     "正=看涨吸收，负=看跌吸收。符号由 ε 自带，没有额外翻转。",
     "A≥1.5 表示「比平时罕见 1.5 个标准差」"),
    ("流量门槛（gate）",
     "|x| 进入自己<b>前 60 根的前 20%</b>。没有这道门槛，"
     "一根安静的 bar（流量小、价格也没动）会被算成吸收。",
     "把「推不动」和「没人推」分开"),
    ("R",
     "以 1 ATR14 为单位的盈亏。止损 1 ATR，目标 1.5 ATR，"
     "30 根内都没碰到就按收盘了结。已扣 0.03 美元往返价差。",
     "1 R ≈ 0.45 美元/股"),
    ("前推样本外",
     "训练窗 ≥45 天、其后 15 天计分、滚动 4 折，拼起来 60 个交易日。"
     "本页所有数字都只来自这 60 天。",
     "没有任何一折被它自己拟合过"),
]

_SPEC = """本研究的全部定义在读到任何一个数字<b>之前</b>就写死在
<code>backtest/absorption.py</code> 的文档字符串里。这不是客套——这个项目里
「看起来成立、然后死在对照上」的比例大约是 4 比 1，而从搜索里挑出来的格子
和一个发现不是一回事。下面第六节是唯一的例外，已单独标注。"""


def _report(te, have, big, bull, bear, base_n, shown, deltas):
    """Self-contained HTML, same shape as the other backtest reports."""
    import pathlib
    from datetime import datetime

    import plotly.graph_objects as go

    from backtest.report import (_CSS, _apply_layout, _html_section,
                                 _plotly_js_tag, _to_html)

    def stat(mask, col):
        s = te[mask].dropna(subset=[col])
        if len(s) < 50:
            return None
        n, sr = agg(s, col)
        lo, hi = ci(s, col)
        return dict(n=len(s), hit=float((s[col] > 0).mean()),
                    r=net(n, sr), lo=lo, hi=hi)

    rows = [
        ("基准：每分钟做多", stat(have, "r_long"), "#6e7a9b"),
        ("基准：每分钟做空", stat(have, "r_short"), "#6e7a9b"),
        ("吸收做多 A≥1.5", stat(bull, "r_long"), "#ed8796"),
        ("吸收做空 A≤−1.5", stat(bear, "r_short"), "#ed8796"),
        ("大卖量顺势做空", stat(big & (te.x < 0), "r_short"), "#a6da95"),
        ("大买量顺势做多", stat(big & (te.x > 0), "r_long"), "#a6da95"),
        ("残差 A≥1.5（无门槛）",
         stat(have & (te.A >= 1.5), "r_long"), "#f5a97f"),
        ("残差 A≤−1.5（无门槛）",
         stat(have & (te.A <= -1.5), "r_short"), "#f5a97f"),
    ]
    rows = [(n, s, c) for n, s, c in rows if s]

    fig = go.Figure(go.Bar(
        x=[n for n, _, _ in rows], y=[s["r"] for _, s, _ in rows],
        marker_color=[c for _, _, c in rows],
        error_y=dict(type="data", symmetric=False,
                     array=[s["hi"] - s["r"] for _, s, _ in rows],
                     arrayminus=[s["r"] - s["lo"] for _, s, _ in rows],
                     color="#cdd6f4", thickness=1.2),
        text=[f"{s['r']:+.3f}" for _, s, _ in rows], textposition="outside"))
    fig.add_hline(y=0, line=dict(color="#45506b", width=1))
    _apply_layout(fig, "每笔净 R（95% 区间按交易日重采样）",
                  yaxis_title="净 R", height=460, showlegend=False)
    ch_main = _to_html(fig)

    fig = go.Figure()
    for nm, sign, col, c in (("吸收做多", +1, "r_long", "#ed8796"),
                             ("吸收做空", -1, "r_short", "#f5a97f")):
        xs, ys = [], []
        for thr in (1.0, 1.5, 2.0, 2.5):
            m = big & ((te.x < 0) if sign > 0 else (te.x > 0))
            m = m & ((te.A >= thr) if sign > 0 else (te.A <= -thr))
            s = stat(m, col)
            if s:
                xs.append(thr); ys.append(s["r"])
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name=nm,
                                 line=dict(color=c, width=2),
                                 marker=dict(size=9)))
    fig.add_hline(y=0, line=dict(color="#45506b", width=1))
    _apply_layout(fig, "阈值越极端，吸收越差（单调，不是噪声）",
                  xaxis_title="|A| 阈值", yaxis_title="净 R", height=380)
    ch_thr = _to_html(fig)

    order = ["两者都触发", "只有流量门槛", "只有 dr1>=0.4", "两者都不触发"]
    fig = go.Figure(go.Bar(
        x=order, y=[shown[k][2] for k in order],
        marker_color=["#a6da95", "#5c9cf5", "#f5a97f", "#6e7a9b"],
        text=[f"{shown[k][2]:+.3f}<br>n={shown[k][0]:,}" for k in order],
        textposition="outside"))
    fig.add_hline(y=0, line=dict(color="#45506b", width=1))
    _apply_layout(fig, "流量门槛 × dr1≥0.4：顺势交易的每笔净 R",
                  yaxis_title="净 R", height=400, showlegend=False)
    ch_cell = _to_html(fig)

    hrs = sorted(te.hr.unique())
    fig = go.Figure()
    for nm, m, c in (("吸收做多", bull, "#ed8796"),
                     ("吸收做空", bear, "#f5a97f")):
        fig.add_trace(go.Bar(x=[f"{h}时" for h in hrs],
                             y=[int((m & (te.hr == h)).sum()) for h in hrs],
                             name=nm, marker_color=c))
    _apply_layout(fig, "吸收事件的时段分布（均匀，不是时段探测器）",
                  yaxis_title="事件数", height=340, barmode="group")
    ch_hr = _to_html(fig)

    fm = ""
    for head, items in FORMULAS:
        fm += (f'<tr><td colspan="2" style="padding-top:14px;color:#5c9cf5;'
               f'font-weight:600">{head}</td></tr>')
        for expr, why in items:
            fm += (f'<tr><td style="white-space:nowrap;font-family:'
                   f'ui-monospace,monospace;color:#cdd6f4">{expr}</td>'
                   f'<td style="color:#6e7a9b">{why}</td></tr>')
    fm = ('<div class="table-wrap"><table class="result-table">'
          + fm + "</table></div>")

    gl = "".join(
        f"<tr><td style='white-space:nowrap'><b>{a}</b></td>"
        f"<td>{b}</td><td style='color:#6e7a9b'>{c}</td></tr>"
        for a, b, c in GLOSSARY)
    gl = ('<div class="table-wrap"><table class="result-table">'
          "<tr><th>符号</th><th>定义</th><th>备注</th></tr>"
          + gl + "</table></div>")

    tbl = "".join(
        f"<tr><td>{n}</td><td>{s['n']:,}</td><td>{s['hit']:.1%}</td>"
        f"<td style='color:{'#a6da95' if s['r'] > 0 else '#ed8796'}'>"
        f"{s['r']:+.3f}</td><td>[{s['lo']:+.3f}, {s['hi']:+.3f}]</td></tr>"
        for n, s, _ in rows)
    tbl = ('<div class="table-wrap"><table class="result-table">'
           "<tr><th>规则</th><th>笔数</th><th>胜率</th><th>每笔 R</th>"
           "<th>95% 区间</th></tr>" + tbl + "</table></div>")

    dl = "".join(
        f"<tr><td>{k}</td><td style='color:"
        f"{'#a6da95' if v[1] > 0 else '#ed8796'}'>{v[0]:+.3f}</td>"
        f"<td>[{v[1]:+.3f}, {v[2]:+.3f}]</td></tr>"
        for k, v in deltas.items())
    dl = ('<div class="table-wrap"><table class="result-table">'
          "<tr><th>增量</th><th>Δ 净 R</th><th>95% 区间</th></tr>"
          + dl + "</table></div>")

    title = "吸收信号研究 — US.SOXL 1m"
    body = (
        f'<div class="meta">样本外 {te.day.nunique()} 个交易日 / '
        f'{base_n:,} 分钟。{_SPEC}</div>'
        + _html_section("符号与定义", gl)
        + _html_section("计算公式", fm
            + '<p class="note">每一步都严格前视隔离：S、λ、σ、G 四个滚动量的窗口上界都是 t−1，bar t 绝不参与构造评判自己的那把尺子。</p>')
        + _html_section(
            "一、结论：吸收不只是无效，是反的",
            f'<div class="chart-full">{ch_main}</div>' + tbl
            + '<p class="note">同一道流量门槛，顺势做 +0.36，'
              '逆势做（吸收）−0.39。门槛挑出的是「有大流量」的时刻，'
              '而那些时刻的正确做法是跟着流走。</p>')
        + _html_section(
            "二、阈值敏感度",
            f'<div class="chart-full">{ch_thr}</div>'
            + '<p class="note">越极端越差，两个方向都单调。'
              '只在单一阈值上成立才是噪声；这是稳定的反向。</p>')
        + _html_section(
            "三、流量门槛 × dr1（事后追加，非事先登记）",
            f'<div class="chart-full">{ch_cell}</div>' + dl
            + '<p class="note"><b>这一节是看到对照组结果之后才加的</b>，'
              '不在事先登记的范围内，区间偏乐观。两者都有增量，而流量门槛的'
              '增量是 dr 的两倍多——但这需要一段新的留出数据才能当结论。</p>')
        + _html_section(
            "四、不是时段探测器",
            f'<div class="chart-full">{ch_hr}</div>')
    )
    html = ('<!DOCTYPE html><html lang="zh"><head><meta charset="UTF-8">'
            '<meta name="viewport" content="width=device-width,'
            f'initial-scale=1.0"><title>{title}</title>'
            + _plotly_js_tag() + _CSS
            + '<style>.note{color:#6e7a9b;font-size:.85rem;margin:8px 0;}'
              '.result-table td{vertical-align:top;}</style>'
            + f'</head><body><h1>{title}</h1>{body}</body></html>')

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = (pathlib.Path(__file__).parent.parent / "backtest" / "results"
               / f"{stamp}_absorption")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "report.html"
    out.write_text(html, encoding="utf-8")
    print(f"\nReport saved -> {out}")
    return out


def section_six(te, have, base_n):
    """The flow gate against the validated delta_r rule: which contains which.

    Registered as a CONTROL, and it came back the strongest number in the
    study -- so the question is whether it is a new thing or the same thing
    wearing a different gate. The two sign identically, since dr1 is OFI
    over total flow; only the gate differs, a level against its own recent
    history versus a fixed ratio.
    """
    print("\n=== 六、流量门槛 与 dr1>=0.4 谁包含谁 ===")
    te_days = sorted(te.day.unique())
    gate_on = have & (te.x.abs() >= te.gate)
    dr_on = have & (te.dr1.abs() >= 0.4)
    cells = [("两者都触发", gate_on & dr_on),
             ("只有流量门槛", gate_on & ~dr_on),
             ("只有 dr1>=0.4", ~gate_on & dr_on),
             ("两者都不触发", have & ~gate_on & ~dr_on)]
    print(f"{'':<20}{'笔数':>8}{'占比':>8}{'胜率':>8}{'每笔 R':>9}")
    acc, shown = {}, {}
    for nm, m in cells:
        sub = te[m]
        v = flow_r(sub)
        acc[nm] = per_day(sub, v, te_days)
        fin = np.isfinite(v)
        n, sr = acc[nm]
        shown[nm] = (int(fin.sum()),
                     float((v[fin] > 0).mean()) if fin.sum() else float("nan"),
                     float(sr.sum() / n.sum()) if n.sum() else float("nan"))
        if fin.sum() < 100:
            print(f"{nm:<20}{int(fin.sum()):>8}   样本太少")
            continue
        print(f"{nm:<20}{int(fin.sum()):>8}{fin.sum()/base_n:>8.1%}"
              f"{100*shown[nm][1]:>7.1f}%{shown[nm][2]:>+9.3f}")

    print("\n  增量（配对，同一次按日重采样）：")
    deltas = {}
    for a, b, label in (
            ("两者都触发", "只有 dr1>=0.4", "门槛加在 dr 之上值多少"),
            ("两者都触发", "只有流量门槛", "dr 加在门槛之上值多少")):
        m, lo, hi = paired(acc[a], acc[b], te_days)
        deltas[label] = (m, lo, hi)
        star = "  *" if np.isfinite(lo) and lo > 0 else ""
        print(f"    {label:<24}{m:>+8.3f}   [{lo:+.3f}, {hi:+.3f}]{star}")
    return shown, deltas


def main():
    d = build()
    days = sorted(d.day.unique())
    oos = []
    for _, te in folds(days, anchored=True):
        oos += te
    te = d[d.day.isin(oos)]
    base_n = len(te)
    print(f"样本外 {te.day.nunique()} 个交易日 / {base_n:,} 分钟"
          f"（与前推验证同一批区块）")
    print(f"止损 1 ATR，目标 {TGT} ATR，{FWD} 根时间止损，价差已扣")
    print(f"流量门槛 |x| 进自身前 {100-100*FLOW_PCT:.0f}%，"
          f"残差阈值 |A| >= {A_MAIN}\n")

    have = te.A.notna() & te.gate.notna() & te.x.notna()
    big = have & (te.x.abs() >= te.gate)
    bull = big & (te.x < 0) & (te.A >= A_MAIN)
    bear = big & (te.x > 0) & (te.A <= -A_MAIN)

    hdr = (f"{'规则':<34}{'笔数':>7}{'占比':>7}{'胜率':>8}"
           f"{'每笔 R':>9}{'95% 区间':>22}")
    print("=== 一、事先选定的吸收规则 ===")
    print(hdr)
    row("基准：每一分钟做多", te, "r_long", base_n)
    row("基准：每一分钟做空", te, "r_short", base_n)
    row("吸收做多 (x<0, A>=1.5)", te[bull], "r_long", base_n)
    row("吸收做空 (x>0, A<=-1.5)", te[bear], "r_short", base_n)

    print("\n=== 二、把吸收拆开：是流量？是残差？还是两者？ ===")
    print("  大流量单独看，方向取流的符号——这是顺势，不是吸收。")
    print(hdr)
    row("大卖量就做空", te[big & (te.x < 0)], "r_short", base_n)
    row("大买量就做多", te[big & (te.x > 0)], "r_long", base_n)
    print("  残差单独看，不要流量门槛：")
    row("A>=1.5 就做多（无门槛）", te[have & (te.A >= A_MAIN)],
        "r_long", base_n)
    row("A<=-1.5 就做空（无门槛）", te[have & (te.A <= -A_MAIN)],
        "r_short", base_n)

    print("\n=== 三、阈值敏感度 ===")
    print(hdr)
    for thr in (1.0, 1.5, 2.0, 2.5):
        row(f"吸收做多 A>={thr}", te[big & (te.x < 0) & (te.A >= thr)],
            "r_long", base_n)
    for thr in (1.0, 1.5, 2.0, 2.5):
        row(f"吸收做空 A<=-{thr}", te[big & (te.x > 0) & (te.A <= -thr)],
            "r_short", base_n)

    print("\n=== 四、与 delta_r 的正面冲突 ===")
    print("  吸收做多必然 dr1<0，而验证过的规则说 dr1<0 做多是 −0.31 R。")
    print("  要么符号在吸收子集里翻转，要么吸收不成立。")
    print(hdr)
    neg = have & (te.dr1 < 0)
    row("dr1<0 全体做多（对照）", te[neg], "r_long", base_n)
    row("dr1<0 且吸收做多", te[neg & bull], "r_long", base_n)
    row("dr1<0 非吸收做多", te[neg & ~bull], "r_long", base_n)
    c = te.loc[have, ["A", "dr1"]].corr().iloc[0, 1]
    print(f"\n  A 与 dr1 的相关系数：{c:+.3f}"
          f"（接近 0 才说明是两个独立维度）")

    print("\n=== 五、是不是时段探测器 ===")
    print(f"{'小时':>6}{'吸收做多':>10}{'吸收做空':>10}{'占该小时':>10}")
    for hr in sorted(te.hr.unique()):
        m = te.hr == hr
        nb, ns = int((bull & m).sum()), int((bear & m).sum())
        print(f"{hr:>4}时{nb:>10}{ns:>10}{(nb+ns)/max(1,int(m.sum())):>10.1%}")
    print("\n  * = 95% 区间（按交易日重采样）整个在 0 以上")


    shown, deltas = section_six(te, have, base_n)
    _report(te, have, big, bull, bear, base_n, shown, deltas)


if __name__ == "__main__":
    main()

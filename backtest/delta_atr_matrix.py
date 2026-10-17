"""delta_r x ATR matrix sweep over every RTH minute, with an HTML report.

WHAT IS SWEPT. ATR enters a rule in four places that earlier runs kept
confusing with one another, so each is its own axis here:

  PERIOD      the Wilder n of the ATR itself (5..30)
  SMOOTHING   a second average laid over the ATR series (none, SMA, EMA) --
              how steady the reading is, which is not the same question as
              how much history it carries
  FORM        how "rising" is decided: a k-bar difference, the sign of an
              OLS slope, or no direction at all but a level above its own
              median
  RISK UNIT   the ATR period that defines the stop, and so the target, the
              label and the spread's share of R -- the one choice with
              consequences for what a "win" even means

Crossed with delta_r (window and threshold) and the reward ratio from 1:1
to 1:2.

R IS NOT COMPARABLE ACROSS RISK UNITS. +0.08 R on a 7-period stop and on a
21-period stop are different amounts of money, because R is the stop and
the stop is what changed. Both are reported; cents per share settles it.

HOW THE NUMBERS ARE PROTECTED.

  TIME SPLIT. Cells are ranked on the first 2/3 of sessions and reported
  on the last 1/3, untouched until then. A handful of rules are written
  down before the sweep runs and read on that same test set; only those
  intervals mean what they appear to.

  SESSION BOOTSTRAP. Consecutive minutes share nearly all of their forward
  window, so the effective sample is far below the minute count.
  Resampling sessions, not minutes, is what keeps intervals honest.

  PAIRED CONTRASTS. "Does this term earn its place" is a difference, and a
  difference needs both sides measured on the same resampled days. Two
  separate intervals overlap whenever each is wide, however consistent the
  gap.

  THE THIRD OUTCOME. A trade touching neither barrier within the horizon is
  closed at that bar's close -- a time stop, and a real instruction.
  Dropping those minutes would change the sample with every target and
  would score a rule that only knows which minutes resolved after the fact.

Usage:
    uv run python -m backtest.delta_atr_matrix [--open]
"""

from __future__ import annotations

import pathlib
import sys
from datetime import datetime

import numpy as np
import pandas as pd

_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

CODE, START, END = "US.SOXL", "2026-05-19", "2026-10-16"
FWD = 30                      # bars the trade is given before the time stop
STOP = 1.0                    # stop distance in ATR, fixed so R is one unit
# Up to 1:3 because the first run's optimum sat on the 1:2 boundary, and an
# optimum at the edge of the range means the range stopped too early.
TGTS = (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0)
SPREAD = 0.03                 # round trip, dollars per share
MIN_SIGNED = 300.0            # signed volume a bar needs for delta_r to exist
TRAIN_FRAC = 0.67
MIN_TRAIN_N, MIN_TEST_N = 400, 150
BIG = 10**9

PERIODS = (5, 7, 10, 14, 21, 30)          # filter ATR period
SMOOTHS = ("raw", "SMA5", "SMA10", "EMA10", "EMA20")
FORMS = ("d1", "d5", "d10", "slope10", "slope20", "hi60")
RISKS = (7, 14, 21)                       # ATR period used as the risk unit
DRS = (("off", None), ("dr1", 0.0), ("dr1", 0.2), ("dr1", 0.4),
       ("dr5", 0.0), ("dr5", 0.2), ("dr5", 0.4))

FORM_LABEL = {"d1": "差分 1 根", "d5": "差分 5 根", "d10": "差分 10 根",
              "slope10": "回归斜率 10", "slope20": "回归斜率 20",
              "hi60": "高于 60 根中位"}

rng = np.random.default_rng(97)


# ── indicators ───────────────────────────────────────────────────────────────

def atr(h, l, c, n):
    tr = np.full(len(h), np.nan)
    for i in range(1, len(h)):
        tr[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
    out = np.full(len(h), np.nan)
    if len(h) > n:
        out[n] = np.nanmean(tr[1:n+1])
        for i in range(n+1, len(h)):
            out[i] = (out[i-1]*(n-1) + tr[i]) / n
    return out


def smooth(s: pd.Series, how: str) -> pd.Series:
    if how == "raw":
        return s
    kind, k = how[:3], int(how[3:])
    return s.rolling(k).mean() if kind == "SMA" else \
        s.ewm(span=k, adjust=False).mean()


def slope_positive(s: pd.Series, k: int) -> np.ndarray:
    """Sign of the OLS slope over a trailing k bars, in closed form.

    A rolling polyfit over 30 series is minutes of work; the slope of y on
    0..k-1 is a ratio of rolling sums, and only its sign is wanted, so the
    positive denominator can be dropped.
    """
    t = np.arange(len(s), dtype=float)
    s0 = s.rolling(k).sum()
    s1 = (s * t).rolling(k).sum()
    start = pd.Series(t, index=s.index) - k + 1
    num = k*(s1 - start*s0) - (k*(k-1)/2.0)*s0
    return (num > 0).to_numpy()


def form_mask(s: pd.Series, form: str) -> np.ndarray:
    if form.startswith("d"):
        return (s.diff(int(form[1:])) > 0).to_numpy()
    if form.startswith("slope"):
        return slope_positive(s, int(form[5:]))
    return (s > s.rolling(60).median()).to_numpy()


def outcomes(h, l, c, a):
    """First touch of each barrier; the close when neither is reached."""
    n = len(c)
    ok = np.zeros(n, bool)
    jd = np.full(n, BIG)
    ju = {t: np.full(n, BIG) for t in TGTS}
    ret = np.full(n, np.nan)
    for i in range(n):
        if not np.isfinite(a[i]) or a[i] <= 0 or i + 1 >= n:
            continue
        hi, lo = h[i+1:i+1+FWD], l[i+1:i+1+FWD]
        ok[i] = True
        td = np.flatnonzero(lo <= c[i] - STOP*a[i])
        jd[i] = td[0] if len(td) else BIG
        for t in TGTS:
            tu = np.flatnonzero(hi >= c[i] + t*a[i])
            ju[t][i] = tu[0] if len(tu) else BIG
        ret[i] = (c[min(i+FWD, n-1)] - c[i]) / a[i]
    return ok, jd, ju, ret


def rcol(p, t):
    return f"R{p}_{int(t*100)}"


def fcol(p, sm, form):
    return f"f{p}_{sm}_{form}"


# ── data ─────────────────────────────────────────────────────────────────────

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
        hh = g.index.hour + g.index.minute/60
        r = g[(hh > 9.5) & (hh <= 16.0)]
        if len(r) < 380:
            continue
        h, l, c = (r.high.to_numpy(float), r.low.to_numpy(float),
                   r.close.to_numpy(float))
        d = {"day": np.repeat(str(day.date()), len(r))}
        keep = np.zeros(len(r), bool)

        for p in RISKS:
            a = atr(h, l, c, p)
            d[f"atr{p}"] = a
            ok, jd, ju, ret = outcomes(h, l, c, a)
            d[f"ok{p}"] = ok & np.isfinite(a)
            keep |= d[f"ok{p}"]
            for t in TGTS:
                d[rcol(p, t)] = np.where(ju[t] < jd, t,
                                         np.where(jd < BIG, -STOP, ret))

        for p in PERIODS:
            base = pd.Series(atr(h, l, c, p))
            for sm in SMOOTHS:
                s = smooth(base, sm)
                for form in FORMS:
                    d[fcol(p, sm, form)] = form_mask(s, form)

        buy = pd.Series(r.buy.to_numpy(float))
        sell = pd.Series(r.sell.to_numpy(float))
        for n in (1, 5):
            b = buy.rolling(n, min_periods=1).sum().to_numpy()
            s_ = sell.rolling(n, min_periods=1).sum().to_numpy()
            tot = b + s_
            with np.errstate(invalid="ignore", divide="ignore"):
                d[f"dr{n}"] = np.where(tot >= MIN_SIGNED, (b-s_)/tot, np.nan)

        parts.append(pd.DataFrame(d, index=r.index)[keep])
    return pd.concat(parts)


# ── evaluation ───────────────────────────────────────────────────────────────

class Scope:
    """One time slice, with per-session sums done by bincount.

    A groupby per cell is the whole runtime at this grid size; the day
    index is fixed, so the aggregate is three bincounts.
    """

    def __init__(self, d: pd.DataFrame, days_all):
        self.d = d
        codes = pd.Categorical(d["day"], categories=days_all).codes
        self.code = codes.astype(np.intp)
        self.nday = len(days_all)
        self.cost = {p: (SPREAD/(STOP*d[f"atr{p}"].to_numpy())) for p in RISKS}
        self.dollars = {p: d[f"atr{p}"].to_numpy()*STOP for p in RISKS}

    def sums(self, m, p, t):
        cd = self.code[m]
        n = np.bincount(cd, minlength=self.nday).astype(float)
        r = self.d[rcol(p, t)].to_numpy()[m]
        sr = np.bincount(cd, weights=r, minlength=self.nday)
        sc = np.bincount(cd, weights=self.cost[p][m], minlength=self.nday)
        sd = np.bincount(cd, weights=r*self.dollars[p][m], minlength=self.nday)
        return n, sr, sc, sd

    def mask(self, cell):
        risk, p, sm, form, drk, thr = cell
        m = self.d[f"ok{risk}"].to_numpy()
        if form is not None:
            m = m & self.d[fcol(p, sm, form)].to_numpy()
        if drk != "off":
            v = self.d[drk].to_numpy()
            m = m & np.isfinite(v) & (v >= thr)
        return m


def net_r(n, sr, sc, sd):
    tot = n.sum()
    return np.nan if tot == 0 else (sr.sum() - sc.sum()) / tot


def net_cents(n, sr, sc, sd):
    tot = n.sum()
    return np.nan if tot == 0 else 100*(sd.sum() - SPREAD*tot) / tot


def hit_rate(sc_: Scope, m, p, t):
    return float((sc_.d[rcol(p, t)].to_numpy()[m] > 0).mean())


def boot_ci(n, sr, sc, sd, fn, reps=3000):
    live = n > 0
    if live.sum() < 5:
        return np.nan, np.nan
    idx = rng.integers(0, len(n), size=(reps, len(n)))
    out = np.empty(reps)
    for j in range(reps):
        i = idx[j]
        out[j] = fn(n[i], sr[i], sc[i], sd[i])
    good = np.isfinite(out)
    if good.sum() < reps // 10:
        return np.nan, np.nan
    return np.percentile(out[good], 2.5), np.percentile(out[good], 97.5)


def paired_ci(sa, sb, fn, reps=3000):
    """Interval on (A - B) with both sides on the same resampled days."""
    na, ra, ca, da = sa
    nb, rb, cb, db = sb
    idx = rng.integers(0, len(na), size=(reps, len(na)))
    out = np.full(reps, np.nan)
    for j in range(reps):
        i = idx[j]
        if na[i].sum() == 0 or nb[i].sum() == 0:
            continue
        out[j] = fn(na[i], ra[i], ca[i], da[i]) - fn(nb[i], rb[i], cb[i], db[i])
    good = np.isfinite(out)
    if good.sum() < reps // 10:
        return np.nan, np.nan, np.nan
    v = out[good]
    return v.mean(), np.percentile(v, 2.5), np.percentile(v, 97.5)


def all_cells():
    cells = []
    for risk in RISKS:
        for drk, thr in DRS:
            cells.append((risk, None, None, None, drk, thr))
            for p in PERIODS:
                for sm in SMOOTHS:
                    for form in FORMS:
                        cells.append((risk, p, sm, form, drk, thr))
    return cells


def sweep(tr: Scope, te: Scope) -> pd.DataFrame:
    rows = []
    for cell in all_cells():
        mtr = tr.mask(cell)
        ntr = int(mtr.sum())
        if ntr < MIN_TRAIN_N:
            continue
        mte = te.mask(cell)
        nte = int(mte.sum())
        risk, p, sm, form, drk, thr = cell
        for t in TGTS:
            a_tr = tr.sums(mtr, risk, t)
            a_te = te.sums(mte, risk, t) if nte >= MIN_TEST_N else None
            rows.append({
                "risk": risk, "period": p, "smooth": sm, "form": form,
                "dr": drk, "thr": thr, "rr": t,
                "n_tr": ntr, "n_te": nte,
                "r_tr": net_r(*a_tr),
                "r_te": net_r(*a_te) if a_te else np.nan,
                "c_te": net_cents(*a_te) if a_te else np.nan,
                "hit_te": hit_rate(te, mte, risk, t) if a_te else np.nan,
            })
    return pd.DataFrame(rows)


# ── charts ───────────────────────────────────────────────────────────────────

def _charts(df: pd.DataFrame, base_te: dict, pre: list, abl: list):
    import plotly.graph_objects as go
    from backtest.report import _apply_layout, _to_html

    ok = df[df.n_te >= MIN_TEST_N].copy()
    out = {}

    # 1. Train against test -- the shrinkage picture. One point per cell.
    fig = go.Figure()
    for dr_, col in (("off", "#6e7a9b"), ("dr1", "#5c9cf5"), ("dr5", "#f5a97f")):
        s = ok[ok.dr == dr_]
        fig.add_trace(go.Scatter(
            x=s.r_tr, y=s.r_te, mode="markers", name=f"delta_r: {dr_}",
            marker=dict(size=4, color=col, opacity=.55)))
    lim = [min(ok.r_tr.min(), ok.r_te.min()), max(ok.r_tr.max(), ok.r_te.max())]
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="完全不缩水",
                             line=dict(color="#45506b", dash="dash", width=1)))
    fig.add_hline(y=0, line=dict(color="#45506b", width=1))
    _apply_layout(fig, "训练集净 R → 测试集净 R（每点一个组合）",
                  xaxis_title="训练集 净 R", yaxis_title="测试集 净 R",
                  height=420)
    out["scatter"] = _to_html(fig)

    # 2. Period x smoothing, held at the delta_r setting that carries most
    #    of the cells, so the panel reads as one comparison.
    sub = ok[(ok.dr == "dr1") & (ok.thr == 0.0) & (ok.form.notna())]
    piv = sub.pivot_table(index="smooth", columns="period", values="r_te",
                          aggfunc="median").reindex(SMOOTHS)
    fig = go.Figure(go.Heatmap(
        z=piv.to_numpy(), x=[str(c) for c in piv.columns], y=list(piv.index),
        colorscale="RdBu", zmid=0, colorbar=dict(title="净 R"),
        text=np.round(piv.to_numpy(), 3), texttemplate="%{text}",
        textfont=dict(size=10)))
    _apply_layout(fig, "周期 × 平滑度（dr1≥0，各判定形式与 RR 的中位数）",
                  xaxis_title="ATR 周期", yaxis_title="平滑度", height=360)
    out["heat"] = _to_html(fig)

    # 3. Form, with intervals. Direction tests and the level test side by side.
    fig = go.Figure()
    fs = [f for f in FORMS]
    med = [ok[ok.form == f].r_te.median() for f in fs]
    fig.add_trace(go.Bar(x=[FORM_LABEL[f] for f in fs], y=med,
                         marker_color=["#5c9cf5"]*5 + ["#f5a97f"],
                         text=[f"{v:+.3f}" for v in med],
                         textposition="outside", name="测试集净 R 中位数"))
    fig.add_hline(y=0, line=dict(color="#45506b", width=1))
    _apply_layout(fig, "判定形式（蓝=方向，橙=水平）",
                  yaxis_title="测试集 净 R 中位数", height=360, showlegend=False)
    out["form"] = _to_html(fig)

    # 4. The pre-specified rules, which are the only intervals that read
    #    straight. Error bars, not bars alone.
    names = [p["name"] for p in pre]
    vals = [p["r"] for p in pre]
    los = [p["r"] - p["lo"] for p in pre]
    his = [p["hi"] - p["r"] for p in pre]
    fig = go.Figure(go.Bar(
        x=names, y=vals, marker_color="#5c9cf5",
        error_y=dict(type="data", symmetric=False, array=his, arrayminus=los,
                     color="#cdd6f4", thickness=1.2),
        text=[f"{v:+.3f}" for v in vals], textposition="outside"))
    fig.add_hline(y=0, line=dict(color="#45506b", width=1))
    _apply_layout(fig, "事先选定的规则（测试集净 R，95% 区间按交易日重采样）",
                  yaxis_title="净 R", height=400, showlegend=False)
    out["pre"] = _to_html(fig)

    # 5. Leave-one-out, paired. A term that earns nothing sits at or below 0.
    fig = go.Figure(go.Bar(
        x=[a["name"] for a in abl], y=[a["d"] for a in abl],
        marker_color=["#5c9cf5" if a["lo"] > 0 else "#6e7a9b" for a in abl],
        error_y=dict(type="data", symmetric=False,
                     array=[a["hi"]-a["d"] for a in abl],
                     arrayminus=[a["d"]-a["lo"] for a in abl],
                     color="#cdd6f4", thickness=1.2),
        text=[f"{a['d']:+.3f}" for a in abl], textposition="outside"))
    fig.add_hline(y=0, line=dict(color="#45506b", width=1))
    _apply_layout(fig, "留一消融：去掉该项的代价（配对，RR 1:1.5）",
                  yaxis_title="Δ 净 R", height=400, showlegend=False)
    out["abl"] = _to_html(fig)

    # 6. RR, one line per risk unit. Net R only -- cents live in the table,
    #    because two scales on one axis is the chart that lies.
    fig = go.Figure()
    for p, col in zip(RISKS, ("#5c9cf5", "#a6da95", "#f5a97f")):
        s = ok[(ok.risk == p) & (ok.dr == "dr1") & (ok.thr == 0.0)
               & ok.form.isna()]
        s = s.sort_values("rr")
        fig.add_trace(go.Scatter(x=[f"1:{v:g}" for v in s.rr], y=s.r_te,
                                 mode="lines+markers", name=f"风险单位 ATR{p}",
                                 line=dict(color=col, width=2),
                                 marker=dict(size=8)))
    fig.add_hline(y=0, line=dict(color="#45506b", width=1))
    _apply_layout(fig, "盈亏比 × 风险单位（dr1≥0，无 ATR 筛选）",
                  xaxis_title="盈亏比", yaxis_title="测试集 净 R", height=360)
    out["rr"] = _to_html(fig)
    return out


def _kpis(pairs) -> str:
    cards = "".join(
        f'<div class="kpi-card"><div class="kpi-label">{k}</div>'
        f'<div class="kpi-value">{v}</div></div>' for k, v in pairs)
    return f'<div class="kpi-row">{cards}</div>'


def _table(df: pd.DataFrame, n: int = 25) -> str:
    s = df.sort_values("r_tr", ascending=False).head(n)
    head = ("组合", "RR", "风险单位", "训练 n", "训练净 R",
            "测试 n", "测试胜率", "测试净 R", "测试每股")
    rows = []
    for _, r in s.iterrows():
        name = ("无 ATR 筛选" if r.form is None else
                f"ATR{r.period} {r['smooth']} {FORM_LABEL[r.form]}")
        dr = "—" if r.dr == "off" else f"{r.dr}≥{r.thr:g}"
        col = "#a6da95" if r.r_te > 0 else "#ed8796"
        rows.append(
            f"<tr><td>{name} + {dr}</td><td>1:{r.rr:g}</td>"
            f"<td>ATR{r.risk}</td><td>{r.n_tr}</td><td>{r.r_tr:+.3f}</td>"
            f"<td>{r.n_te}</td><td>{r.hit_te:.1%}</td>"
            f"<td style='color:{col}'>{r.r_te:+.3f}</td>"
            f"<td>{r.c_te:+.2f}¢</td></tr>")
    return ('<div class="table-wrap"><table class="result-table"><tr>'
            + "".join(f"<th>{h}</th>" for h in head) + "</tr>"
            + "".join(rows) + "</table></div>")


def _note(text: str) -> str:
    return f'<p class="note">{text}</p>'


def main(open_browser: bool = False) -> pathlib.Path:
    from backtest.report import _CSS, _html_section, _plotly_js_tag

    print("building features ...", flush=True)
    d = build()
    days = sorted(d["day"].unique())
    cut = days[int(len(days)*TRAIN_FRAC)]
    tr_days = [x for x in days if x < cut]
    te_days = [x for x in days if x >= cut]
    tr = Scope(d[d["day"] < cut], tr_days)
    te = Scope(d[d["day"] >= cut], te_days)
    print(f"  {len(d):,} minutes / {len(days)} sessions"
          f"  train {len(tr_days)}  test {len(te_days)}", flush=True)

    print("sweeping ...", flush=True)
    df = sweep(tr, te)
    print(f"  {len(df):,} cells cleared the training minimum", flush=True)

    # Pre-specified rules, fixed before any test number was looked at.
    PRE = [
        ("基准：每一分钟", (14, None, None, None, "off", None)),
        ("dr1≥0", (14, None, None, None, "dr1", 0.0)),
        ("dr1≥0.2", (14, None, None, None, "dr1", 0.2)),
        ("dr5≥0", (14, None, None, None, "dr5", 0.0)),
        ("ATR14 差分1根", (14, 14, "raw", "d1", "off", None)),
        ("ATR14 差分1根 + dr1≥0", (14, 14, "raw", "d1", "dr1", 0.0)),
        ("ATR14 斜率10 + dr1≥0", (14, 14, "raw", "slope10", "dr1", 0.0)),
        ("ATR14 SMA5 斜率10 + dr1≥0",
         (14, 14, "SMA5", "slope10", "dr1", 0.0)),
    ]
    RR_MAIN = 1.5
    pre = []
    for name, cell in PRE:
        m = te.mask(cell)
        a = te.sums(m, cell[0], RR_MAIN)
        lo, hi = boot_ci(*a, net_r)
        pre.append({"name": name, "n": int(m.sum()), "r": net_r(*a),
                    "c": net_cents(*a), "lo": lo, "hi": hi,
                    "hit": hit_rate(te, m, cell[0], RR_MAIN)})
        print(f"  pre {name:<28} n={m.sum():>6} "
              f"netR={pre[-1]['r']:+.3f} [{lo:+.3f},{hi:+.3f}]", flush=True)

    FULL = (14, 14, "raw", "slope10", "dr1", 0.0)
    DROPS = [("去掉 delta_r", (14, 14, "raw", "slope10", "off", None)),
             ("去掉 ATR 条件", (14, None, None, None, "dr1", 0.0))]
    a_full = te.sums(te.mask(FULL), 14, RR_MAIN)
    abl = []
    for name, cell in DROPS:
        a_b = te.sums(te.mask(cell), 14, RR_MAIN)
        dmean, lo, hi = paired_ci(a_full, a_b, net_r)
        abl.append({"name": name, "d": dmean, "lo": lo, "hi": hi})
        print(f"  abl {name:<28} delta={dmean:+.3f} "
              f"[{lo:+.3f},{hi:+.3f}]", flush=True)

    base_m = te.mask((14, None, None, None, "off", None))
    base_te = {t: net_r(*te.sums(base_m, 14, t)) for t in TGTS}

    print("rendering ...", flush=True)
    ch = _charts(df, base_te, pre, abl)
    ok = df[df.n_te >= MIN_TEST_N]
    surv = int((ok.r_te > 0).sum())
    pre_dr = next(p for p in pre if p["name"] == "dr1≥0")

    kpi = _kpis([
        ("分钟数", f"{len(d):,}"),
        ("交易日", f"{len(days)}（训练 {len(tr_days)} / 测试 {len(te_days)}）"),
        ("扫过的组合", f"{len(df):,}"),
        ("测试集净 R > 0", f"{surv:,} / {len(ok):,}"),
        ("基准净 R（1:1.5）", f"{base_te[1.5]:+.3f}"),
        ("dr1≥0 净 R（1:1.5）", f"{pre_dr['r']:+.3f}"),
    ])

    pre_rows = "".join(
        f"<tr><td>{p['name']}</td><td>{p['n']}</td><td>{p['hit']:.1%}</td>"
        f"<td>{p['r']:+.3f}</td><td>[{p['lo']:+.3f}, {p['hi']:+.3f}]</td>"
        f"<td>{p['c']:+.2f}¢</td></tr>" for p in pre)
    pre_tbl = ('<div class="table-wrap"><table class="result-table">'
               "<tr><th>规则</th><th>n</th><th>胜率</th><th>净 R</th>"
               "<th>95% 区间</th><th>每股</th></tr>"
               + pre_rows + "</table></div>")

    title = (f"delta_r × ATR 矩阵回测 — {CODE} 1m "
             f"{START} → {END}")
    body = (
        f'<div class="meta">每一个 RTH 分钟都是候选，止损 1 ATR，'
        f'{FWD} 根未触及则按收盘了结；价差 {SPREAD:.2f} 美元往返已扣。'
        f'区间一律按交易日重采样，因为相邻分钟的前瞻窗口几乎重合。</div>'
        + kpi
        + _html_section("一、事先选定的规则（唯一可以照字面读的区间）",
                        pre_tbl + _note(
                            "这些规则在看测试集之前就写定了。下面第三节的"
                            "排名是从上万个组合里挑出来的，它们的区间天然偏"
                            "乐观，不能同样对待。")
                        + f'<div class="chart-full">{ch["pre"]}</div>')
        + _html_section("二、留一消融：每一项各自值多少",
                        f'<div class="chart-full">{ch["abl"]}</div>'
                        + _note("两边用同一次按日重采样配对算差。"
                                "单独看两个区间是否重叠回答不了这个问题。"))
        + _html_section("三、训练到测试的缩水",
                        f'<div class="chart-full">{ch["scatter"]}</div>'
                        + _note("点越靠下、离虚线越远，说明训练集上的那个"
                                "数字越是挑出来的。"))
        + _html_section("四、四个维度",
                        f'<div class="chart-full">{ch["heat"]}</div>'
                        f'<div class="chart-grid">'
                        f'<div>{ch["form"]}</div><div>{ch["rr"]}</div></div>')
        + _html_section("五、训练集排名前 25（测试集一并列出）",
                        _table(df) + _note(
                            "按训练集净 R 排序，测试列是同一组合在没见过的"
                            "35 个交易日上的表现。两列的落差就是过拟合的量。"))
    )

    html = (f'<!DOCTYPE html><html lang="zh"><head><meta charset="UTF-8">'
            f'<meta name="viewport" content="width=device-width,'
            f'initial-scale=1.0"><title>{title}</title>'
            f'{_plotly_js_tag()}{_CSS}'
            '<style>.note{color:#6e7a9b;font-size:.82rem;margin:6px 0 2px;}'
            '.chart-grid>div{min-width:0;}</style>'
            f'</head><body><h1>{title}</h1>{body}</body></html>')

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = _ROOT / "backtest" / "results" / f"{stamp}_delta_atr_matrix"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "report.html"
    out.write_text(html, encoding="utf-8")
    df.to_csv(out_dir / "cells.csv", index=False, encoding="utf-8-sig")
    print(f"Report saved -> {out}", flush=True)
    print(f"Cells saved  -> {out_dir / 'cells.csv'}", flush=True)

    if open_browser:
        import webbrowser
        webbrowser.open(out.resolve().as_uri())
    return out


if __name__ == "__main__":
    main(open_browser="--open" in sys.argv)

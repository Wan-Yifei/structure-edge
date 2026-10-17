"""Walk-forward validation: more out-of-sample days, shorter training windows.

The single split spends 70 sessions choosing and leaves 35 to judge with.
Those 35 are the entire evidence base, and because consecutive minutes
share nearly all of their forward window, the effective sample is closer to
35 than to the 13,125 minutes it looks like.

Walk-forward trades training length for evidence length: fit on a window,
judge on the block that follows, roll forward, and pool the blocks. Four
folds of 15 sessions give 60 out-of-sample sessions instead of 35, every
one of them judged by a rule chosen without seeing it.

TWO THINGS ARE MEASURED, AND THEY ARE DIFFERENT QUESTIONS.

  A RULE. The pre-specified rules are the same in every fold, so their
  pooled result is simply a longer out-of-sample read.

  THE PROCEDURE. "Rank cells on the training window, trade the best one"
  is itself a strategy, and re-fitting it each fold is what someone
  actually does. Pooling the chosen cells' results answers whether the
  SELECTION works, which no single cell's number can. This is usually
  where a grid dies: the per-fold winner keeps changing and the pooled
  result collapses toward the baseline.

ANCHORED vs ROLLING. Anchored trains on everything before the block, so
late folds get more history; rolling keeps a fixed recent window, which
matters if the market or the data quality drifts -- and here the tick
coverage does drift, from 89% early to 99% late. Both are run; a rule that
only works under one of them has not been established.

Usage:
    uv run python -m backtest.walk_forward [--open]
"""

from __future__ import annotations

import pathlib
import sys
from datetime import datetime

import numpy as np
import pandas as pd

_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from backtest.delta_atr_matrix import (  # noqa: E402
    CODE, END, FORM_LABEL, MIN_SIGNED, PERIODS, SMOOTHS, FORMS, SPREAD,
    START, STOP, TGTS, DRS, Scope, build, net_cents, net_r, rcol)

RISK = 14              # the risk-unit sweep found 7/14/21 indistinguishable
TRAIN_MIN = 45         # sessions in the first training window
TEST_WIN = 15          # sessions judged per fold
MIN_TRAIN_N = 300
RR_MAIN = 1.5

# Written down before any fold is scored.
PRE = [
    ("基准：每一分钟", (RISK, None, None, None, "off", None)),
    ("dr1≥0", (RISK, None, None, None, "dr1", 0.0)),
    ("dr1≥0.2", (RISK, None, None, None, "dr1", 0.2)),
    ("dr1≥0.4", (RISK, None, None, None, "dr1", 0.4)),
    ("ATR14 原始 斜率10 + dr1≥0", (RISK, 14, "raw", "slope10", "dr1", 0.0)),
    ("ATR30 EMA20 斜率20 + dr1≥0", (RISK, 30, "EMA20", "slope20", "dr1", 0.0)),
]

rng = np.random.default_rng(131)


def cells_for_selection():
    out = []
    for drk, thr in DRS:
        if drk == "off":
            continue                      # selection should earn its filter
        out.append((RISK, None, None, None, drk, thr))
        for p in PERIODS:
            for sm in SMOOTHS:
                for form in FORMS:
                    out.append((RISK, p, sm, form, drk, thr))
    return out


def folds(days, anchored: bool):
    """(train_days, test_days) blocks rolling forward over the sessions."""
    out = []
    start = TRAIN_MIN
    while start + TEST_WIN <= len(days):
        tr = days[:start] if anchored else days[max(0, start-TRAIN_MIN):start]
        out.append((tr, days[start:start+TEST_WIN]))
        start += TEST_WIN
    return out


def pooled_boot(n, sr, sc, reps=4000):
    """Resample the pooled out-of-sample SESSIONS."""
    live = n.sum()
    if live == 0 or len(n) < 5:
        return np.nan, np.nan
    i = rng.integers(0, len(n), size=(reps, len(n)))
    tot = n[i].sum(1)
    ok = tot > 0
    v = (sr[i].sum(1)[ok] - sc[i].sum(1)[ok]) / tot[ok]
    return np.percentile(v, 2.5), np.percentile(v, 97.5)


def cell_name(cell):
    _, p, sm, form, drk, thr = cell
    atr_part = "无 ATR 筛选" if form is None else \
        f"ATR{p} {sm} {FORM_LABEL[form]}"
    return f"{atr_part} + {drk}≥{thr:g}"


def run_mode(d: pd.DataFrame, days, anchored: bool):
    """Returns per-fold records plus pooled per-session sums for each track."""
    fl = folds(days, anchored)
    cands = cells_for_selection()
    tracks = {name: {"n": [], "sr": [], "sc": []} for name, _ in PRE}
    tracks["前推挑选（每折重选）"] = {"n": [], "sr": [], "sc": []}
    recs = []

    for k, (tr_days, te_days) in enumerate(fl, 1):
        tr = Scope(d[d["day"].isin(tr_days)], tr_days)
        te = Scope(d[d["day"].isin(te_days)], te_days)

        best, best_r = None, -np.inf
        for cell in cands:
            m = tr.mask(cell)
            if int(m.sum()) < MIN_TRAIN_N:
                continue
            for t in TGTS:
                v = net_r(*tr.sums(m, RISK, t))
                if np.isfinite(v) and v > best_r:
                    best_r, best = v, (cell, t)

        for name, cell in PRE:
            m = te.mask(cell)
            n, sr, sc, _ = te.sums(m, RISK, RR_MAIN)
            tracks[name]["n"].append(n)
            tracks[name]["sr"].append(sr)
            tracks[name]["sc"].append(sc)

        rec = {"fold": k, "train": f"{tr_days[0]}→{tr_days[-1]}",
               "test": f"{te_days[0]}→{te_days[-1]}",
               "n_train_days": len(tr_days)}
        if best is None:
            for key in ("n", "sr", "sc"):
                tracks["前推挑选（每折重选）"][key].append(
                    np.zeros(len(te_days)))
            rec.update({"pick": "（训练窗无合格组合）", "rr": np.nan,
                        "r_tr": np.nan, "r_te": np.nan, "n_te": 0})
        else:
            cell, t = best
            m = te.mask(cell)
            n, sr, sc, _ = te.sums(m, RISK, t)
            tracks["前推挑选（每折重选）"]["n"].append(n)
            tracks["前推挑选（每折重选）"]["sr"].append(sr)
            tracks["前推挑选（每折重选）"]["sc"].append(sc)
            rec.update({"pick": cell_name(cell), "rr": t, "r_tr": best_r,
                        "r_te": net_r(n, sr, sc, None), "n_te": int(m.sum())})
        # The pre-specified reference, same fold, for a like-for-like read.
        mref = te.mask(PRE[1][1])
        rec["r_te_dr1"] = net_r(*te.sums(mref, RISK, RR_MAIN))
        rec["r_te_base"] = net_r(*te.sums(te.mask(PRE[0][1]), RISK, RR_MAIN))
        recs.append(rec)

    pooled = {}
    for name, acc in tracks.items():
        n = np.concatenate(acc["n"]) if acc["n"] else np.zeros(0)
        sr = np.concatenate(acc["sr"]) if acc["sr"] else np.zeros(0)
        sc = np.concatenate(acc["sc"]) if acc["sc"] else np.zeros(0)
        lo, hi = pooled_boot(n, sr, sc)
        pooled[name] = {"days": int((n > 0).sum()), "n": int(n.sum()),
                        "r": net_r(n, sr, sc, None), "lo": lo, "hi": hi}
    return pd.DataFrame(recs), pooled


# ── report ───────────────────────────────────────────────────────────────────

def _charts(res):
    import plotly.graph_objects as go
    from backtest.report import _apply_layout, _to_html
    out = {}

    # Pooled, both schemes, with intervals. Error bars are the point here:
    # the whole exercise is about how wide they are.
    names = list(res["anchored"][1].keys())
    fig = go.Figure()
    for (mode, col) in (("anchored", "#5c9cf5"), ("rolling", "#f5a97f")):
        p = res[mode][1]
        vals = [p[n]["r"] for n in names]
        fig.add_trace(go.Bar(
            name="锚定（训练窗递增）" if mode == "anchored" else "滚动（固定 45 天）",
            x=names, y=vals, marker_color=col,
            error_y=dict(type="data", symmetric=False,
                         array=[p[n]["hi"]-p[n]["r"] for n in names],
                         arrayminus=[p[n]["r"]-p[n]["lo"] for n in names],
                         color="#cdd6f4", thickness=1.2)))
    fig.add_hline(y=0, line=dict(color="#45506b", width=1))
    _apply_layout(fig, f"汇总样本外净 R（RR 1:{RR_MAIN:g}，95% 区间按交易日重采样）",
                  yaxis_title="净 R", height=460, barmode="group")
    out["pooled"] = _to_html(fig)

    # Per fold. A rule that only works in one block has not been shown to
    # work; this is the chart that reveals it.
    df = res["anchored"][0]
    fig = go.Figure()
    for col, nm, c in (("r_te_base", "基准：每一分钟", "#6e7a9b"),
                       ("r_te_dr1", "dr1≥0", "#5c9cf5"),
                       ("r_te", "前推挑选", "#a6da95")):
        fig.add_trace(go.Bar(name=nm, x=[f"第 {k} 折" for k in df.fold],
                             y=df[col], marker_color=c))
    fig.add_hline(y=0, line=dict(color="#45506b", width=1))
    _apply_layout(fig, "逐折样本外净 R（锚定模式）",
                  yaxis_title="净 R", height=400, barmode="group")
    out["folds"] = _to_html(fig)

    # Training number against the block that followed it, for the picks.
    fig = go.Figure()
    for mode, c in (("anchored", "#5c9cf5"), ("rolling", "#f5a97f")):
        s = res[mode][0].dropna(subset=["r_tr", "r_te"])
        fig.add_trace(go.Scatter(
            x=s.r_tr, y=s.r_te, mode="markers+text",
            text=[f"第{k}折" for k in s.fold], textposition="top center",
            textfont=dict(size=9, color="#6e7a9b"),
            marker=dict(size=11, color=c),
            name="锚定" if mode == "anchored" else "滚动"))
    allv = pd.concat([res[m][0][["r_tr", "r_te"]] for m in res]).dropna()
    if len(allv):
        lim = [min(allv.min()), max(allv.max())]
        fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="完全不缩水",
                                 line=dict(color="#45506b", dash="dash",
                                           width=1)))
    fig.add_hline(y=0, line=dict(color="#45506b", width=1))
    _apply_layout(fig, "被挑中的组合：训练窗上的数字 → 它之后那一段的真实表现",
                  xaxis_title="训练窗 净 R", yaxis_title="样本外 净 R",
                  height=420)
    out["shrink"] = _to_html(fig)
    return out


def _fold_table(df: pd.DataFrame) -> str:
    head = ("折", "训练窗", "样本外窗", "被挑中的组合", "RR",
            "训练净 R", "样本外净 R", "同段 dr1≥0", "同段基准")
    rows = []
    for _, r in df.iterrows():
        col = "#a6da95" if r.r_te > 0 else "#ed8796"
        rr = "—" if not np.isfinite(r.rr) else f"1:{r.rr:g}"
        rows.append(
            f"<tr><td>{r.fold}</td><td>{r.train}</td><td>{r.test}</td>"
            f"<td>{r.pick}</td><td>{rr}</td><td>{r.r_tr:+.3f}</td>"
            f"<td style='color:{col}'>{r.r_te:+.3f}</td>"
            f"<td>{r.r_te_dr1:+.3f}</td><td>{r.r_te_base:+.3f}</td></tr>")
    return ('<div class="table-wrap"><table class="result-table"><tr>'
            + "".join(f"<th>{h}</th>" for h in head) + "</tr>"
            + "".join(rows) + "</table></div>")


def _pooled_table(pooled: dict) -> str:
    rows = []
    for name, p in pooled.items():
        star = " *" if np.isfinite(p["lo"]) and p["lo"] > 0 else ""
        col = "#a6da95" if p["r"] > 0 else "#ed8796"
        rows.append(
            f"<tr><td>{name}</td><td>{p['days']}</td><td>{p['n']}</td>"
            f"<td style='color:{col}'>{p['r']:+.3f}{star}</td>"
            f"<td>[{p['lo']:+.3f}, {p['hi']:+.3f}]</td></tr>")
    return ('<div class="table-wrap"><table class="result-table">'
            "<tr><th>规则</th><th>样本外交易日</th><th>笔数</th>"
            "<th>净 R</th><th>95% 区间</th></tr>"
            + "".join(rows) + "</table></div>")


def main(open_browser: bool = False) -> pathlib.Path:
    from backtest.report import _CSS, _html_section, _plotly_js_tag

    print("building features ...", flush=True)
    d = build()
    days = sorted(d["day"].unique())
    print(f"  {len(d):,} minutes / {len(days)} sessions", flush=True)

    res = {}
    for anchored, key in ((True, "anchored"), (False, "rolling")):
        print(f"walk-forward [{key}] ...", flush=True)
        recs, pooled = run_mode(d, days, anchored)
        res[key] = (recs, pooled)
        for _, r in recs.iterrows():
            print(f"  fold {r.fold} {r.test}  pick={r.pick[:46]:<46}"
                  f" train={r.r_tr:+.3f} oos={r.r_te:+.3f}", flush=True)
        for n, p in pooled.items():
            print(f"  pooled {n:<30} days={p['days']:>3} n={p['n']:>6} "
                  f"netR={p['r']:+.3f} [{p['lo']:+.3f},{p['hi']:+.3f}]",
                  flush=True)

    print("rendering ...", flush=True)
    ch = _charts(res)
    a_pool = res["anchored"][1]
    oos_days = a_pool["dr1≥0"]["days"]
    single = 35

    kpi = "".join(
        f'<div class="kpi-card"><div class="kpi-label">{k}</div>'
        f'<div class="kpi-value">{v}</div></div>'
        for k, v in [
            ("样本外交易日", f"{oos_days}（单次切分只有 {single}）"),
            ("折数", f"{len(res['anchored'][0])} × {TEST_WIN} 天"),
            ("训练窗", f"≥{TRAIN_MIN} 天"),
            ("dr1≥0 汇总净 R", f"{a_pool['dr1≥0']['r']:+.3f}"),
            ("前推挑选 汇总净 R",
             f"{a_pool['前推挑选（每折重选）']['r']:+.3f}"),
            ("基准 汇总净 R", f"{a_pool['基准：每一分钟']['r']:+.3f}"),
        ])

    title = f"前推验证 — {CODE} 1m {START} → {END}"
    body = (
        f'<div class="meta">把证据段从 35 个交易日加厚到 {oos_days} 个。'
        f'每折在训练窗上挑组合，只在其后 {TEST_WIN} 个交易日上计分，'
        f'再把各折拼起来。事先选定的规则每折相同，它们的汇总就是一次更长的'
        f'样本外读数。</div>'
        f'<div class="kpi-row">{kpi}</div>'
        + _html_section("一、汇总样本外结果",
                        _pooled_table(a_pool)
                        + f'<div class="chart-full">{ch["pooled"]}</div>'
                        + '<p class="note">* = 区间不跨 0。两种模式都报：'
                          '只在其中一种下成立的规则，不算站住。</p>')
        + _html_section("二、逐折",
                        _fold_table(res["anchored"][0])
                        + f'<div class="chart-full">{ch["folds"]}</div>'
                        + '<p class="note">只在某一折有效的规则没有被证明有效。'
                          '「被挑中的组合」一列如果每折都在变，说明网格挑出来的'
                          '是噪声而不是结构。</p>')
        + _html_section("三、挑选这件事本身有没有用",
                        f'<div class="chart-full">{ch["shrink"]}</div>'
                        + '<p class="note">横轴是挑它时训练窗上的数字，纵轴是'
                          '它接下来真实的表现。点落在虚线下方很远，说明训练窗上'
                          '那个高分主要是挑出来的。</p>')
    )
    # No .format() here: _CSS and the inline style are full of braces.
    html = ('<!DOCTYPE html><html lang="zh"><head><meta charset="UTF-8">'
            '<meta name="viewport" content="width=device-width,'
            f'initial-scale=1.0"><title>{title}</title>'
            + _plotly_js_tag() + _CSS
            + '<style>.note{color:#6e7a9b;font-size:.82rem;margin:6px 0 2px;}'
              '</style></head><body>'
            + f'<h1>{title}</h1>{body}</body></html>')

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = _ROOT / "backtest" / "results" / f"{stamp}_walk_forward"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "report.html"
    out.write_text(html, encoding="utf-8")
    for key in res:
        res[key][0].to_csv(out_dir / f"folds_{key}.csv", index=False,
                           encoding="utf-8-sig")
    print(f"Report saved -> {out}", flush=True)

    if open_browser:
        import webbrowser
        webbrowser.open(out.resolve().as_uri())
    return out


if __name__ == "__main__":
    main(open_browser="--open" in sys.argv)

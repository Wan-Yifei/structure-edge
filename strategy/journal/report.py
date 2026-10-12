"""Render the ledger as one HTML page: win rate across every setting.

The point of the page is to make small samples *look* small. Almost nothing in
two months of trading reaches significance, so a bare league table of win rates
would be actively misleading -- the eye ranks the rows and the n column gets
ignored. Every rate is therefore drawn as a dot with its 95% Wilson interval,
against a dashed line at the book's own rate. A bucket whose whisker crosses
that line says nothing, and says it visibly.

Three sections, because the three sources answer different questions:

  account   every position. The only unbiased view, and the only one that can
            speak to expectancy.
  journal   the write-ups. A selected sample -- trades thought worth reviewing
            -- so its rates describe the selection, not the system.
  design    what the screenshots show was planned. Coverage grows as the
            captures are transcribed.
"""

from __future__ import annotations

import html
import math

import pandas as pd

from strategy.journal.reconcile import session_of

# Palette: dataviz reference instance. One series, so the dot takes slot 1 and
# needs no legend. Money wears the delta text tokens and always carries its own
# sign, so hue is never doing the work alone.
CSS = """
:root{color-scheme:light;
  --plane:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
  --grid:#e1e0d9;--axis:#c3c2b7;--series:#2a78d6;--up:#006300;--down:#d03b3b;
  --ring:rgba(11,11,11,.10)}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme=light])){
  color-scheme:dark;
  --plane:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;
  --grid:#2c2c2a;--axis:#383835;--series:#3987e5;--up:#0ca30c;--down:#e66767;
  --ring:rgba(255,255,255,.10)}}
:root[data-theme=dark]{color-scheme:dark;
  --plane:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;
  --grid:#2c2c2a;--axis:#383835;--series:#3987e5;--up:#0ca30c;--down:#e66767;
  --ring:rgba(255,255,255,.10)}
*{box-sizing:border-box}
body{margin:0;background:var(--plane);color:var(--ink);
  font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1040px;margin:0 auto;padding:32px 16px 72px}
h1{font-size:22px;margin:0 0 4px;letter-spacing:-.01em}
h2{font-size:15px;margin:36px 0 2px;letter-spacing:-.01em}
h3{font-size:13px;margin:22px 0 8px;color:var(--ink2);font-weight:600}
p.sub{margin:0 0 20px;color:var(--ink2);font-size:13px;max-width:62ch}
p.note{margin:2px 0 14px;color:var(--muted);font-size:12px;max-width:66ch}
.card{background:var(--surface);border:1px solid var(--ring);border-radius:10px;
  padding:16px 18px 10px;margin:10px 0 0}
table{width:100%;border-collapse:collapse}
th{font-size:11px;font-weight:600;color:var(--muted);text-align:left;
  padding:0 8px 6px;text-transform:uppercase;letter-spacing:.04em}
th.r,td.r{text-align:right}
td{padding:3px 8px;border-top:1px solid var(--grid);vertical-align:middle}
tr:first-child td{border-top:none}
td.lab{white-space:nowrap;font-size:13px}
td.n{font-variant-numeric:tabular-nums;color:var(--ink2);width:46px}
td.wr{font-variant-numeric:tabular-nums;width:62px}
td.pl{font-variant-numeric:tabular-nums;width:84px}
.up{color:var(--up)}.down{color:var(--down)}
.thin td.lab,.thin td.n{color:var(--muted)}
.plot{width:100%;min-width:220px}
/* Positioned in CSS rather than an SVG viewBox: a stretched viewBox turns
   every dot into an ellipse, which is exactly the geometry the eye reads. */
.track{position:relative;height:18px;margin:0 7px}
.track i{position:absolute;display:block}
.base{top:0;bottom:0;width:0;border-left:1px dashed var(--muted)}
.ci{top:8px;height:2px;background:var(--axis);border-radius:1px}
.dot{top:5px;width:8px;height:8px;margin-left:-4px;border-radius:50%;
  background:var(--series);box-shadow:0 0 0 2px var(--surface)}
.thin .dot{background:var(--muted)}
.axis{position:relative;height:14px;margin:2px 7px 4px}
.axis span{position:absolute;transform:translateX(-50%);font-size:10px;
  color:var(--muted);top:0}
tbody tr:hover td{background:color-mix(in srgb,var(--series) 7%,transparent)}
/* histogram: one bar per bucket, sign carried by hue AND by side of zero */
.hist{display:flex;align-items:flex-end;gap:2px;height:132px;margin:4px 7px 0}
.hist div{flex:1;position:relative;border-radius:3px 3px 0 0;min-height:2px;
  background:var(--series)}
.hist div.neg{background:var(--down)}
.hist div.at1{outline:2px solid var(--ink2);outline-offset:1px}
.hx{display:flex;gap:2px;margin:3px 7px 0;color:var(--muted);font-size:10px}
.hx span{flex:1;text-align:center;overflow:hidden;white-space:nowrap}
/* scatter */
.scatwrap{display:flex;margin:4px 7px 0}
.sy{width:34px;position:relative;flex:none}
.sy span{position:absolute;right:4px;transform:translateY(-50%);font-size:10px;
  color:var(--muted)}
.scat{position:relative;height:230px;flex:1;
  border-left:1px solid var(--axis);border-bottom:1px solid var(--axis)}
.scat i{position:absolute;width:9px;height:9px;margin:-4.5px 0 0 -4.5px;
  border-radius:50%;background:var(--series);box-shadow:0 0 0 2px var(--surface)}
.scat i.neg{background:var(--down)}
.scat .zero{position:absolute;left:0;right:0;height:0;
  border-top:1px dashed var(--muted);margin:0}
.scat .ident{position:absolute;inset:0;pointer-events:none}
.tiles{display:flex;gap:10px;flex-wrap:wrap;margin:10px 0 0}
.tile{background:var(--surface);border:1px solid var(--ring);border-radius:10px;
  padding:10px 14px;min-width:150px;flex:1}
.tile b{display:block;font-size:22px;font-weight:600;letter-spacing:-.02em;
  font-variant-numeric:tabular-nums}
.tile span{display:block;font-size:11px;color:var(--muted);margin-top:2px}
.legend{display:flex;gap:18px;flex-wrap:wrap;color:var(--muted);font-size:12px;
  margin:6px 0 0;align-items:center}
.sw{display:inline-block;width:10px;height:10px;border-radius:50%;
  background:var(--series);margin-right:5px;vertical-align:-1px}
.sw.m{background:var(--muted)}
.head{display:flex;justify-content:space-between;align-items:flex-start;gap:16px}
.toggle{flex:none;background:var(--surface);border:1px solid var(--ring);
  color:var(--ink2);border-radius:999px;padding:5px 12px;font-size:12px;
  cursor:pointer}
@media(max-width:620px){.wrap{padding:20px 12px 56px}td.n{width:38px}
  td.wr{width:56px}td.pl{width:72px}}
"""

JS = """
const r=document.documentElement,b=document.getElementById('t');
b.onclick=()=>{const d=r.getAttribute('data-theme')==='dark'||
 (!r.getAttribute('data-theme')&&matchMedia('(prefers-color-scheme:dark)').matches);
 r.setAttribute('data-theme',d?'light':'dark');b.textContent=d?'Dark':'Light';};
"""


# ---------------------------------------------------------------- strings ---
# One table, two languages. Bucket labels come out of the data (Session values,
# ExitMech values and so on), so they are translated here too rather than at
# the point they are produced -- the ledger keeps its English values and only
# the page is localised.
T = {
    "title":      ("Trade Settings", "交易设置分析"),
    "h1":         ("Which settings move the win rate",
                   "哪些设置真正改变了胜率"),
    "lede":       ("Every rate is a dot with its 95% Wilson interval. The "
                   "dashed line is the book&rsquo;s own rate. <strong>A whisker "
                   "that crosses the dashed line is telling you nothing</strong> "
                   "&mdash; which is most of them, on two months of data. Rows "
                   "under 20 trades are greyed as a reminder.",
                   "每个比率是一个点加它的 95% Wilson 区间，虚线是账本自己的胜率。"
                   "<strong>须线只要穿过虚线，这一格就什么都没说</strong>"
                   "&mdash;&mdash; 两个月的数据里，绝大多数都穿过了。"
                   "不足 20 笔的行整行变灰，提醒你别去读它的胜率。"),
    "lg_hi":      ("win rate, n &ge; 20", "胜率，n &ge; 20"),
    "lg_lo":      ("n &lt; 20", "n &lt; 20"),
    "lg_ci":      ("95% interval", "95% 区间"),
    "lg_base":    ("book rate", "账本胜率"),

    "h2_acct":    ("From the account &mdash; every position",
                   "来自结单 &mdash; 每一个持仓"),
    "sub_acct":   ("{n} positions, {d} trading days, book win rate {wr}, {pnl} "
                   "total. This is the only unbiased section: it counts every "
                   "fill, not just the trades worth writing up.",
                   "{n} 个持仓，{d} 个交易日，账本胜率 {wr}，合计 {pnl}。"
                   "这是唯一无偏的一节：它统计每一笔成交，而不只是值得写进复盘的那些。"),
    "h2_jrnl":    ("From the journal &mdash; the write-ups",
                   "来自日志 &mdash; 被写下来的交易"),
    "sub_jrnl":   ("{n} trades. A <em>selected</em> sample: the deck records "
                   "the trades worth reviewing, so these rates describe that "
                   "selection rather than the system. Dollar figures appear "
                   "only where a write-up could be tied to a position beyond "
                   "doubt, so they under-count.",
                   "{n} 笔。这是一个<em>有选择</em>的样本：幻灯片只记录值得复盘的交易，"
                   "所以这些比率描述的是这个选择，不是这套系统。"
                   "美元数字只在复盘能确切对上某个持仓时才出现，因此是偏少的。"),
    "h2_exec":    ("Execution &mdash; plan against outcome",
                   "执行 &mdash; 计划对照结果"),
    "sub_exec":   ("Only the trades whose stop the account itself confirms: "
                   "the exit of a stopped-out trade should land on the stop, "
                   "and {conf} of them do, within 0.1% of price. R is a "
                   "ratio, so it is never summed here &mdash; risk-weighted "
                   "means sum(P&amp;L) over sum(dollars risked).",
                   "只统计止损被结单本身证实的交易：一笔被止损打掉的交易，"
                   "它的出场价应该落在止损上&mdash;&mdash;{conf} 笔落在了价格的 0.1% 以内。"
                   "R 是个比值，所以这里从不加总它&mdash;&mdash;"
                   "风险加权的意思是 sum(盈亏) 除以 sum(押上的美元)。"),
    "h2_shot":    ("From the screenshots &mdash; what was planned",
                   "来自截图 &mdash; 当时计划的是什么"),
    "sub_shot":   ("{cov}. Coverage grows as the captures are read.",
                   "{cov}。覆盖率随截图转录而增长。"),
    "cov":        ("{a} of {b} September screenshots transcribed",
                   "九月 {b} 张截图已转录 {a} 张"),
    "foot":       ("Equity {eq}. Win rate counts a position as a win at any "
                   "profit above zero, so scratches land on whichever side of "
                   "zero they finished. Interval is Wilson at 95%.",
                   "账户权益 {eq}。胜率把任何高于零的盈利都算作赢，"
                   "所以打平的单子落在它最终所在的那一侧。区间为 95% Wilson。"),

    "th_bucket":  ("bucket", "分组"),
    "th_wr":      ("win rate &amp; 95% interval", "胜率与 95% 区间"),
    "th_n":       ("n", "n"),
    "th_wr2":     ("WR", "胜率"),
    "th_pl":      ("$ / trade", "每笔"),
    "th_share":   ("share", "占比"),

    "p_sess":     ("By session (entry time)", "按时段（开仓时间）"),
    "p_hour":     ("By entry hour (ET)", "按开仓小时（美东）"),
    "n_hour":     ("18 hours tested against one book: a single hour needs p "
                   "below 0.003 to mean anything. None gets there.",
                   "18 个小时对同一个账本做检验：单个小时要有意义，p 需要低于 0.003。"
                   "没有一个达到。"),
    "p_sym":      ("By instrument", "按标的"),
    "p_size":     ("By position size (quartiles of notional)",
                   "按仓位大小（名义金额四分位）"),
    "p_held":     ("By holding time", "按持有时长"),
    "p_dow":      ("By weekday", "按星期"),
    "p_strat":    ("By strategy", "按策略"),
    "p_entry":    ("By profile level", "按成交量轮廓价位"),
    "p_jsess":    ("By session, as written up", "按时段（以复盘所写为准）"),
    "p_setup":    ("By setup -- what was traded", "按形态 —— 做的是什么"),
    "n_setup":    ("The pattern the write-up names. Buckets under 4 trades are "
                   "left out.",
                   "复盘里写明的形态。不足 4 笔的分组不显示。"),
    "p_ctx":      ("By context -- what the market, or he, was doing",
                   "按环境 —— 当时市场或自己在做什么"),
    "n_ctx":      ("Conditions the note flags rather than patterns traded. "
                   "\u5939\u89d2 is the KD bands pinching shut, a condition; "
                   "the trades taken inside it were ordinary IVB entries at "
                   "POC. The question it raises -- same setup inside the pinch "
                   "and outside it -- needs more than the four trades there are.",
                   "笔记标注的环境，而不是所做的形态。夹角是 KD 带收口，"
                   "是一个条件；在那里做的交易本身是普通的 IVB 在 POC 入场。"
                   "它提出的问题——同一形态在夹角内和夹角外——需要的样本远多于现有的四笔。"),
    "p_win":      ("How the winners were closed", "盈利单是怎么平的"),
    "n_win":      ("Win rate cannot answer this -- hitting the stop is what "
                   "makes a trade a loss, so a rate by exit mechanism restates "
                   "itself. The question an exit can answer is how often it "
                   "was the planned one.",
                   "胜率回答不了这个问题——碰到止损正是一笔交易成为亏损的定义，"
                   "所以按离场方式算胜率是在重复自己。离场方式能回答的问题是："
                   "有多少次是按计划走的。"),
    "p_lose":     ("How the losers were closed", "亏损单是怎么平的"),
    "n_lose":     ("stop blown is where the stop did not hold.",
                   "stop blown 指止损没有守住。"),
    "p_rr":       ("By planned R/R", "按计划 R/R"),
    "n_rr":       ("The ratio the R/R tool showed when the screenshot was "
                   "taken. The stop rarely moved afterwards; the target often "
                   "did.",
                   "截图当时 R/R 工具显示的比值。止损之后很少动，目标经常动。"),
    "p_tf":       ("By chart timeframe", "按图表周期"),
    "p_sw":       ("By stop width (entry to stop)", "按止损宽度（入场到止损）"),
    "n_sw":       ("Measured from the account&rsquo;s own fill to the stop on "
                   "the screenshot.",
                   "从结单自己的成交价量到截图上的止损。"),
    "p_hist":     ("Realised R", "实际 R"),
    "n_hist":     ("Red is a loss. The outlined bar is the -1R bucket: where "
                   "the stop did exactly what it was set to do. Buckets widen "
                   "to the right because the tail is long and thin.",
                   "红色是亏损。描边的那一格是 -1R：止损完全按设定执行的地方。"
                   "分组向右变宽，因为尾巴又长又薄。"),
    "p_scat":     ("Planned against realised", "计划对照实际"),
    "n_scat":     ("If the plan governed the outcome these would climb left to "
                   "right. They do not: the losers sit on a flat line at -1R "
                   "whatever was planned, because a stop is a stop.",
                   "如果计划决定结果，这些点应该从左向右爬升。它们没有："
                   "亏损单在 -1R 上排成一条平线，不管计划的是多少，因为止损就是止损。"),
    "p_risk":     ("Dollar risked per trade", "每笔押上的美元"),
    "n_risk":     ("This is why R cannot be summed or averaged across trades: "
                   "the smallest position risks a twentieth of the largest, so "
                   "an unweighted mean hands the loudest vote to the smallest "
                   "bet.",
                   "这就是 R 不能跨交易加总或取平均的原因：最小的仓位只押了最大仓位的"
                   "二十分之一，不加权的平均把最大的话语权交给了最小的注。"),
    "t_wr":       ("risk-weighted &mdash; what one dollar of risk returned",
                   "风险加权 &mdash; 每押一块钱风险拿回多少"),
    "t_med":      ("median trade", "中位单笔"),
    "t_risk":     ("dollar risked per trade (median {m})",
                   "每笔押上的美元（中位 {m}）"),
    "t_n":        ("trades with a stop the account confirms",
                   "止损被结单证实的交易"),
    "x_plan":     ("x: planned R/R &nbsp;&middot;&nbsp; y: realised R",
                   "x：计划 R/R &nbsp;&middot;&nbsp; y：实际 R"),
    "theme":      ("Dark", "深色"),
}

# Values that come out of the data and still need localising on the page.
V = {
    "premarket": "盘前", "regular": "日盘", "afterhours": "盘后",
    "overnight": "夜盘",
    "long (SOXL)": "做多 (SOXL)", "short (SOXS)": "做空 (SOXS)",
    "smallest 25%": "最小 25%", "2nd": "第二档", "3rd": "第三档",
    "largest 25%": "最大 25%",
    "< 5 min": "< 5 分钟", "5-15 min": "5-15 分钟",
    "15-60 min": "15-60 分钟", "> 1 h": "> 1 小时",
    "Mon": "周一", "Tue": "周二", "Wed": "周三", "Thu": "周四", "Fri": "周五",
    "target": "到达目标", "trailed stop": "移动止损被扫",
    "manual early": "提前手动平", "initial stop": "原始止损",
    "stop blown": "止损失效", "unstated": "未说明",
    "3 touch": "三触", "breakout": "突破", "range breakout": "箱体突破",
    "4th touch": "第四次触碰",
    "KD pinch": "KD 夹角", "liquidity sweep": "流动性扫荡",
    "against CVD": "逆 CVD", "impulse entry": "冲动入场",
    "overtrading": "过度交易", "rule broken": "违规",
    "RSI stretched": "RSI 过高", "did not wait": "没有等回踩",
    "fading a drop": "抄底", "fading a bounce": "反弹做空",
    "-": "-", "NA": "NA",
}


def t(key, lang, **kw):
    v = T[key][1 if lang == "zh" else 0]
    return v.format(**kw) if kw else v


def lv(label, lang):
    """Localise a bucket label that came out of the data."""
    if lang != "zh":
        return str(label)
    s = str(label)
    return V.get(s, "; ".join(V.get(x.strip(), x.strip())
                              for x in s.split(";"))) if s else s


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval for a proportion. Wilson, not normal: at n=3 the normal
    interval runs past 0 and 1 and stops being readable."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def _row(label, n, wins, pnl, money_n, base, lang="en", thin_at=20):
    lo, hi = wilson(wins, n)
    wr = wins / n if n else 0
    # Averaged over the trades whose P&L is known, not over the bucket: a
    # write-up with no matching position contributes nothing, and printing
    # "+0.00" for it would read as break-even rather than as unknown.
    # Needs at least three known P&Ls: an average over one trade is that
    # trade, and printing it invites reading it as the bucket's behaviour.
    per = pnl / money_n if money_n >= 3 else None
    thin = " thin" if n < thin_at else ""
    tip = (f"{label}: {wins}/{n} = {100*wr:.0f}% "
           f"(95% CI {100*lo:.0f}-{100*hi:.0f}%)"
           + (f", {per:+.2f} per trade over {money_n} with a known P&L"
              if per is not None else
              f", {money_n} with a known P&L -- too few to average"))
    cls = "" if per is None else ("up" if per > 0 else "down" if per < 0 else "")
    money = "&mdash;" if per is None else f"{per:+.2f}"
    return f"""<tr class="r{thin}"><td class="lab">{html.escape(lv(label, lang))}</td>
<td class="n r">{n}</td>
<td class="plot"><div class="track" title="{html.escape(tip)}">
<i class="base" style="left:{100*base:.2f}%"></i>
<i class="ci" style="left:{100*lo:.2f}%;width:{100*(hi-lo):.2f}%"></i>
<i class="dot" style="left:{100*wr:.2f}%"></i></div></td>
<td class="wr r">{100*wr:.0f}%</td>
<td class="pl r {cls}">{money}</td></tr>"""


def panel(title, rows, base, note="", lang="en"):
    """rows: iterable of (label, n, wins, pnl), already in a meaningful order."""
    rows = [r for r in rows if r[1] > 0]
    if not rows:
        return ""
    body = "".join(_row(*r, base=base, lang=lang) for r in rows)
    axis = ('<div class="axis">' + "".join(
        f'<span style="left:{p*100:.0f}%">{int(p*100)}%</span>'
        for p in (0, .25, .5, .75, 1)) + "</div>")
    return f"""<h3>{html.escape(title)}</h3>
{f'<p class="note">{html.escape(note)}</p>' if note else ''}
<div class="card"><table><thead><tr><th>{t("th_bucket", lang)}</th>
<th class="r">{t("th_n", lang)}</th><th>{t("th_wr", lang)}</th>
<th class="r">{t("th_wr2", lang)}</th>
<th class="r">{t("th_pl", lang)}</th></tr></thead><tbody>{body}</tbody></table>
<table><tr><td class="lab" style="border:none"></td>
<td class="n" style="border:none"></td>
<td class="plot" style="border:none">{axis}</td>
<td class="wr" style="border:none"></td>
<td class="pl" style="border:none"></td></tr></table></div>"""


def share_panel(title, counts, total, note="", lang="en"):
    """Composition, for questions a win rate cannot answer.

    Win rate by exit mechanism is circular -- a trade that hit its stop is a
    loss by definition, so the panel would just restate its own input. What is
    worth asking of an exit is how often it was the planned one.
    """
    if not total:
        return ""
    rows = "".join(f"""<tr class="r"><td class="lab">{html.escape(lv(k, lang))}</td>
<td class="n r">{v}</td>
<td class="plot"><div class="track" title="{html.escape(f'{k}: {v} of {total}')}">
<i class="ci" style="left:0;width:{100*v/total:.2f}%;height:8px;top:5px;
 background:var(--series);border-radius:2px"></i></div></td>
<td class="wr r">{100*v/total:.0f}%</td><td class="pl r"></td></tr>"""
                   for k, v in counts)
    return f"""<h3>{html.escape(title)}</h3>
{f'<p class="note">{html.escape(note)}</p>' if note else ''}
<div class="card"><table><thead><tr><th>{t("th_bucket", lang)}</th>
<th class="r">{t("th_n", lang)}</th><th>{t("th_share", lang)}</th><th class="r">%</th>
<th class="r"></th></tr></thead>
<tbody>{rows}</tbody></table></div>"""


def hist_panel(title, values, edges, note="", mark=None, lang="en"):
    """Distribution. A single number cannot describe realised R -- the shape is
    the finding: a wall at -1 where the stop did its job, and a long thin tail
    on the other side."""
    import numpy as np
    v = pd.Series(values).dropna()
    if v.empty:
        return ""
    cnt, _ = np.histogram(v.clip(edges[0], edges[-1] - 1e-9), bins=edges)
    top = max(1, cnt.max())
    bars = ""
    for i, c in enumerate(cnt):
        lo, hi = edges[i], edges[i + 1]
        cls = "neg" if hi <= 0 else ""
        if mark is not None and lo <= mark < hi:
            cls += " at1"
        bars += (f'<div class="{cls}" style="height:{100*c/top:.1f}%" '
                 f'title="{lo:+.1f} to {hi:+.1f}R: {c} trades"></div>')
    ticks = "".join(f"<span>{edges[i]:+g}</span>" for i in range(len(edges) - 1))
    return f"""<h3>{html.escape(title)}</h3>
{f'<p class="note">{html.escape(note)}</p>' if note else ''}
<div class="card"><div class="hist">{bars}</div><div class="hx">{ticks}</div></div>"""


def scatter_panel(title, xs, ys, labels, xmax, ymin, ymax, note="", lang="en"):
    """Planned against realised. Two measures of the same thing on one axis
    each -- never two y-scales."""
    pts = ""
    for x, y, lab in zip(xs, ys, labels):
        if pd.isna(x) or pd.isna(y):
            continue
        px = 100 * min(x, xmax) / xmax
        py = 100 * (1 - (min(max(y, ymin), ymax) - ymin) / (ymax - ymin))
        pts += (f'<i class="{"neg" if y <= 0 else ""}" '
                f'style="left:{px:.2f}%;top:{py:.2f}%" '
                f'title="{html.escape(lab)}"></i>')
    zero = 100 * (1 - (0 - ymin) / (ymax - ymin))
    # y ticks every whole R the range allows, at most six of them
    step = max(1, round((ymax - ymin) / 6))
    yt = ""
    v = ymin
    while v <= ymax:
        yt += (f'<span style="top:{100*(1-(v-ymin)/(ymax-ymin)):.2f}%">'
               f'{v:+g}R</span>')
        v += step
    return f"""<h3>{html.escape(title)}</h3>
{f'<p class="note">{html.escape(note)}</p>' if note else ''}
<div class="card"><div class="scatwrap"><div class="sy" style="height:230px">
{yt}</div><div class="scat">{pts}
<div class="zero" style="top:{zero:.2f}%"></div></div></div>
<div class="hx" style="margin-left:41px"><span>0</span><span>{xmax/4:g}</span>
<span>{xmax/2:g}</span><span>{3*xmax/4:g}</span><span>{xmax:g}</span></div>
<p class="note" style="margin:4px 0 8px">{t("x_plan", lang)}</p></div>"""


def _buckets(df, col, order=None):
    d = df.copy()
    d["_known"] = d.pnl.notna() & (d.get("has_pnl", True))
    g = d.groupby(col, dropna=False).agg(n=("pnl", "size"),
                                         wins=("win", "sum"),
                                         pnl=("pnl", "sum"),
                                         money_n=("_known", "sum"))
    if order:
        g = g.reindex([o for o in order if o in g.index])
    return [(i, int(r.n), int(r.wins), float(r.pnl), int(r.money_n))
            for i, r in g.iterrows()]


def build(positions: pd.DataFrame, ledger: pd.DataFrame, equity: float,
          screens_done: int, screens_total: int, lang: str = "en") -> str:
    pos = positions.copy()
    pos["win"] = pos.pnl > 0
    base = float(pos.win.mean())
    pos["Session"] = pos["open"].map(session_of)
    pos["hour"] = pos["open"].dt.hour
    pos["dow"] = pos["open"].dt.day_name().str[:3]
    pos["side"] = pos.sym.map({"SOXL": "long (SOXL)", "SOXS": "short (SOXS)"})
    pos["size"] = pd.qcut(pos.notional, 4,
                          labels=["smallest 25%", "2nd", "3rd", "largest 25%"])
    pos["held"] = pd.cut(pos.mins, [-1, 5, 15, 60, 1e9],
                         labels=["< 5 min", "5-15 min", "15-60 min", "> 1 h"])

    S = []
    S.append(panel(t("p_sess", lang), _buckets(
        pos, "Session", ["premarket", "regular", "afterhours", "overnight"]),
        base, lang=lang))
    hrs = [(f"{int(h):02d}:00", *rest)
           for h, *rest in _buckets(pos, "hour", sorted(pos.hour.unique()))]
    S.append(panel(t("p_hour", lang), hrs, base, t("n_hour", lang), lang=lang))
    S.append(panel(t("p_sym", lang), _buckets(pos, "side"), base, lang=lang))
    S.append(panel(t("p_size", lang), _buckets(pos, "size"), base, lang=lang))
    S.append(panel(t("p_held", lang), _buckets(pos, "held"), base, lang=lang))
    S.append(panel(t("p_dow", lang), _buckets(
        pos, "dow", ["Mon", "Tue", "Wed", "Thu", "Fri"]), base, lang=lang))

    j = ledger[ledger.Source.astype(str).str.startswith("journal")].copy()
    j = j[j["W/L"].isin(["win", "loss"])]
    j["win"] = j["W/L"].eq("win")
    known = pd.to_numeric(j["PnL$"], errors="coerce")
    j["has_pnl"] = known.notna()
    j["pnl"] = known.fillna(0.0)
    jbase = float(j.win.mean()) if len(j) else base
    J = []
    for col, title, order in (
            ("Strategy", t("p_strat", lang), None),
            ("Entry", t("p_entry", lang), ["VAL", "POC", "VAH", "-"]),
            ("Session", t("p_jsess", lang), None)):
        if col in j:
            J.append(panel(title, _buckets(j, col, order), jbase, lang=lang))
    prof = j[j.ExitMech.isin(["target", "trailed stop", "manual early"])]
    if len(prof):
        order = ["target", "trailed stop", "manual early"]
        vc = prof.ExitMech.value_counts()
        J.append(share_panel(
            t("p_win", lang), [(k, int(vc[k])) for k in order
                               if k in vc], int(len(prof)),
            t("n_win", lang), lang=lang))
    losers = j[j.ExitMech.isin(["initial stop", "stop blown"])]
    if len(losers):
        vc = losers.ExitMech.value_counts()
        J.append(share_panel(t("p_lose", lang),
                             [(k, int(vc[k])) for k in
                              ["initial stop", "stop blown"] if k in vc],
                             int(len(losers)), t("n_lose", lang), lang=lang))

    for col, title, note in (
            ("Setup", t("p_setup", lang), t("n_setup", lang)),
            ("Context", t("p_ctx", lang), t("n_ctx", lang))):
        x = j[j[col].astype(str).str.len() > 0]
        if not len(x):
            continue
        vc = x[col].value_counts()
        keep = [k for k in vc.index if vc[k] >= 4]
        if keep:
            J.append(panel(title, _buckets(x[x[col].isin(keep)], col), jbase,
                           note, lang=lang))

    D = []
    d = j[pd.to_numeric(j["INIT R/R"], errors="coerce").notna()].copy()
    if len(d):
        d["rr"] = pd.to_numeric(d["INIT R/R"], errors="coerce")
        d["rrb"] = pd.cut(d.rr, [0, 1, 1.5, 2.5, 100],
                          labels=["< 1.0", "1.0-1.5", "1.5-2.5", "> 2.5"])
        D.append(panel(t("p_rr", lang), _buckets(d, "rrb"), jbase,
                       t("n_rr", lang), lang=lang))
    tf = j[pd.to_numeric(j["Time frame"], errors="coerce").notna()].copy()
    if len(tf):
        tf["t"] = pd.to_numeric(tf["Time frame"], errors="coerce").astype(int)
        D.append(panel(t("p_tf", lang), _buckets(
            tf, "t", sorted(tf.t.unique())), jbase, lang=lang))
    sw = j[pd.to_numeric(j.Stop, errors="coerce").notna()].copy()
    if len(sw) and "Notional" in sw:
        px = pd.to_numeric(sw.Notional, errors="coerce") / pd.to_numeric(
            sw.Qty, errors="coerce")
        sw["swpct"] = 100 * (px - pd.to_numeric(sw.Stop, errors="coerce")) / px
        sw = sw[sw.swpct.between(0, 5)]
        if len(sw) > 6:
            sw["swb"] = pd.cut(sw.swpct, [0, .3, .6, 1.0, 5],
                               labels=["< 0.3%", "0.3-0.6%", "0.6-1.0%", "> 1.0%"])
            D.append(panel(t("p_sw", lang), _buckets(sw, "swb"), jbase,
                           t("n_sw", lang), lang=lang))

    # ---- execution: what the verified stops allow us to say ----
    E = []
    g = ledger[(ledger.Source == "journal+account") & ledger.Stop.notna()].copy()
    for c in ("Stop", "Notional", "Qty", "PnL$", "RealizedR", "INIT R/R"):
        g[c] = pd.to_numeric(g[c], errors="coerce")
    g["entry"] = g.Notional / g.Qty
    g["risk"] = g.Qty * (g.entry - g.Stop)
    g = g[(g.risk > 0) & g.RealizedR.notna()]
    tiles = ""
    if len(g):
        wR = (g.RealizedR * g.risk).sum() / g.risk.sum()
        tiles = f"""<div class="tiles">
<div class="tile"><b>{wR:+.2f}R</b><span>{t("t_wr", lang)}</span></div>
<div class="tile"><b>{g.RealizedR.median():+.2f}R</b>
<span>{t("t_med", lang)}</span></div>
<div class="tile"><b>${g.risk.min():.0f}&ndash;${g.risk.max():.0f}</b>
<span>{t("t_risk", lang, m=f"${g.risk.median():.0f}")}</span></div>
<div class="tile"><b>{len(g)}</b><span>{t("t_n", lang)}</span></div></div>"""
        E.append(hist_panel(
            t("p_hist", lang), g.RealizedR, [-3, -2, -1.5, -1, -0.5, 0, 0.5,
                                             1, 1.5, 2, 3, 5, 17],
            t("n_hist", lang), mark=-1, lang=lang)
        )
        lab = [f"{r.date} {r.Time}: planned {r['INIT R/R']:.2f}, "
               f"realised {r.RealizedR:+.2f}R, {r['PnL$']:+.2f} on "
               f"${r.risk:.0f} risked" for _, r in g.iterrows()]
        E.append(scatter_panel(
            t("p_scat", lang), g["INIT R/R"], g.RealizedR, lab,
            xmax=6, ymin=-3, ymax=6,
            note=t("n_scat", lang), lang=lang))
        E.append(hist_panel(
            t("p_risk", lang), g.risk, [0, 10, 20, 30, 40, 50, 60, 80],
            t("n_risk", lang), lang=lang))

    # How many stopped-out trades actually exited at the stop: the check
    # that says the transcribed stops are the real ones.
    conf = "0 of 0"
    if len(g):
        st = g[g.ExitMech == "initial stop"]
        if len(st):
            err = ((st.entry + st["PnL$"] / st.Qty) - st.Stop).abs() / st.entry
            # "23/26" reads the same in both languages
            conf = f"{int((100 * err < 0.1).sum())}/{len(st)}"

    cov = t("cov", lang, a=screens_done, b=screens_total) if screens_total else ""
    return f"""<!doctype html><html lang="{'zh' if lang == 'zh' else 'en'}">
<head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{t("title", lang)}</title><style>{CSS}</style></head><body>
<div class="wrap">
<div class="head"><h1>{t("h1", lang)}</h1>
<button class="toggle" id="t">{t("theme", lang)}</button></div>
<p class="sub">{t("lede", lang)}</p>
<div class="legend"><span><span class="sw"></span>{t("lg_hi", lang)}</span>
<span><span class="sw m"></span>{t("lg_lo", lang)}</span>
<span>&#8212; {t("lg_ci", lang)}</span>
<span>&#9482; {t("lg_base", lang)} {100*base:.0f}%</span></div>

<h2>{t("h2_acct", lang)}</h2>
<p class="sub">{t("sub_acct", lang, n=len(pos), d=pos.day.nunique(),
                  wr=f"{100*base:.1f}%", pnl=f"{pos.pnl.sum():+,.2f}")}</p>
{''.join(S)}

<h2>{t("h2_jrnl", lang)}</h2>
<p class="sub">{t("sub_jrnl", lang, n=len(j))}</p>
{''.join(J)}

<h2>{t("h2_exec", lang)}</h2>
<p class="sub">{t("sub_exec", lang, conf=conf)}</p>
{tiles}
{''.join(E)}

<h2>{t("h2_shot", lang)}</h2>
<p class="sub">{t("sub_shot", lang, cov=cov)}</p>
{''.join(D)}

<p class="note">{t("foot", lang, eq=f"{equity:,.0f}")}</p>
</div><script>{JS}</script></body></html>"""

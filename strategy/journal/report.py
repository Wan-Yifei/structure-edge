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
td.wr,td.pl{font-variant-numeric:tabular-nums;width:76px}
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
  td.wr,td.pl{width:64px}}
"""

JS = """
const r=document.documentElement,b=document.getElementById('t');
b.onclick=()=>{const d=r.getAttribute('data-theme')==='dark'||
 (!r.getAttribute('data-theme')&&matchMedia('(prefers-color-scheme:dark)').matches);
 r.setAttribute('data-theme',d?'light':'dark');b.textContent=d?'Dark':'Light';};
"""


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


def _row(label, n, wins, pnl, money_n, base, thin_at=20):
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
    return f"""<tr class="r{thin}"><td class="lab">{html.escape(str(label))}</td>
<td class="n r">{n}</td>
<td class="plot"><div class="track" title="{html.escape(tip)}">
<i class="base" style="left:{100*base:.2f}%"></i>
<i class="ci" style="left:{100*lo:.2f}%;width:{100*(hi-lo):.2f}%"></i>
<i class="dot" style="left:{100*wr:.2f}%"></i></div></td>
<td class="wr r">{100*wr:.0f}%</td>
<td class="pl r {cls}">{money}</td></tr>"""


def panel(title, rows, base, note=""):
    """rows: iterable of (label, n, wins, pnl), already in a meaningful order."""
    rows = [r for r in rows if r[1] > 0]
    if not rows:
        return ""
    body = "".join(_row(*r, base=base) for r in rows)
    axis = ('<div class="axis">' + "".join(
        f'<span style="left:{p*100:.0f}%">{int(p*100)}%</span>'
        for p in (0, .25, .5, .75, 1)) + "</div>")
    return f"""<h3>{html.escape(title)}</h3>
{f'<p class="note">{html.escape(note)}</p>' if note else ''}
<div class="card"><table><thead><tr><th>bucket</th><th class="r">n</th>
<th>win rate &amp; 95% interval</th><th class="r">WR</th>
<th class="r">$ / trade</th></tr></thead><tbody>{body}</tbody></table>
<table><tr><td class="lab" style="border:none"></td>
<td class="n" style="border:none"></td>
<td class="plot" style="border:none">{axis}</td>
<td class="wr" style="border:none"></td>
<td class="pl" style="border:none"></td></tr></table></div>"""


def share_panel(title, counts, total, note=""):
    """Composition, for questions a win rate cannot answer.

    Win rate by exit mechanism is circular -- a trade that hit its stop is a
    loss by definition, so the panel would just restate its own input. What is
    worth asking of an exit is how often it was the planned one.
    """
    if not total:
        return ""
    rows = "".join(f"""<tr class="r"><td class="lab">{html.escape(str(k))}</td>
<td class="n r">{v}</td>
<td class="plot"><div class="track" title="{html.escape(f'{k}: {v} of {total}')}">
<i class="ci" style="left:0;width:{100*v/total:.2f}%;height:8px;top:5px;
 background:var(--series);border-radius:2px"></i></div></td>
<td class="wr r">{100*v/total:.0f}%</td><td class="pl r"></td></tr>"""
                   for k, v in counts)
    return f"""<h3>{html.escape(title)}</h3>
{f'<p class="note">{html.escape(note)}</p>' if note else ''}
<div class="card"><table><thead><tr><th>bucket</th><th class="r">n</th>
<th>share</th><th class="r">%</th><th class="r"></th></tr></thead>
<tbody>{rows}</tbody></table></div>"""


def hist_panel(title, values, edges, note="", mark=None):
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


def scatter_panel(title, xs, ys, labels, xmax, ymin, ymax, note=""):
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
<p class="note" style="margin:4px 0 8px">x: planned R/R &nbsp;&middot;&nbsp;
y: realised R</p></div>"""


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
          screens_done: int, screens_total: int) -> str:
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
    S.append(panel("By session (entry time)", _buckets(
        pos, "Session", ["premarket", "regular", "afterhours", "overnight"]),
        base))
    hrs = [(f"{int(h):02d}:00", *rest)
           for h, *rest in _buckets(pos, "hour", sorted(pos.hour.unique()))]
    S.append(panel("By entry hour (ET)", hrs, base,
                   "18 hours tested against one book: a single hour needs "
                   "p below 0.003 to mean anything. None gets there."))
    S.append(panel("By instrument", _buckets(pos, "side"), base))
    S.append(panel("By position size (quartiles of notional)",
                   _buckets(pos, "size"), base))
    S.append(panel("By holding time", _buckets(pos, "held"), base))
    S.append(panel("By weekday", _buckets(
        pos, "dow", ["Mon", "Tue", "Wed", "Thu", "Fri"]), base))

    j = ledger[ledger.Source.astype(str).str.startswith("journal")].copy()
    j = j[j["W/L"].isin(["win", "loss"])]
    j["win"] = j["W/L"].eq("win")
    known = pd.to_numeric(j["PnL$"], errors="coerce")
    j["has_pnl"] = known.notna()
    j["pnl"] = known.fillna(0.0)
    jbase = float(j.win.mean()) if len(j) else base
    J = []
    for col, title, order in (
            ("Strategy", "By strategy", None),
            ("Entry", "By profile level", ["VAL", "POC", "VAH", "-"]),
            ("Session", "By session, as written up", None)):
        if col in j:
            J.append(panel(title, _buckets(j, col, order), jbase))
    prof = j[j.ExitMech.isin(["target", "trailed stop", "manual early"])]
    if len(prof):
        order = ["target", "trailed stop", "manual early"]
        vc = prof.ExitMech.value_counts()
        J.append(share_panel(
            "How the winners were closed", [(k, int(vc[k])) for k in order
                                            if k in vc], int(len(prof)),
            "Win rate cannot answer this -- hitting the stop is what makes a "
            "trade a loss, so a rate by exit mechanism restates itself. The "
            "question an exit can answer is how often it was the planned one."))
    losers = j[j.ExitMech.isin(["initial stop", "stop blown"])]
    if len(losers):
        vc = losers.ExitMech.value_counts()
        J.append(share_panel("How the losers were closed",
                             [(k, int(vc[k])) for k in
                              ["initial stop", "stop blown"] if k in vc],
                             int(len(losers)),
                             "stop blown is where the stop did not hold."))

    for col, title, note in (
            ("Setup", "By setup -- what was traded",
             "The pattern the write-up names. Buckets under 4 trades are left out."),
            ("Context", "By context -- what the market, or he, was doing",
             "Conditions the note flags rather than patterns traded. "
             "夹角 is the KD bands pinching shut, a condition; the "
             "trades taken inside it were ordinary IVB entries at POC. The "
             "question it raises -- same setup inside the pinch and outside it "
             "-- needs more than the four trades there are.")):
        x = j[j[col].astype(str).str.len() > 0]
        if not len(x):
            continue
        vc = x[col].value_counts()
        keep = [k for k in vc.index if vc[k] >= 4]
        if keep:
            J.append(panel(title, _buckets(x[x[col].isin(keep)], col), jbase,
                           note))

    D = []
    d = j[pd.to_numeric(j["INIT R/R"], errors="coerce").notna()].copy()
    if len(d):
        d["rr"] = pd.to_numeric(d["INIT R/R"], errors="coerce")
        d["rrb"] = pd.cut(d.rr, [0, 1, 1.5, 2.5, 100],
                          labels=["< 1.0", "1.0-1.5", "1.5-2.5", "> 2.5"])
        D.append(panel("By planned R/R", _buckets(d, "rrb"), jbase,
                       "The ratio the R/R tool showed when the screenshot was "
                       "taken. The stop rarely moved afterwards; the target "
                       "often did."))
    tf = j[pd.to_numeric(j["Time frame"], errors="coerce").notna()].copy()
    if len(tf):
        tf["t"] = pd.to_numeric(tf["Time frame"], errors="coerce").astype(int)
        D.append(panel("By chart timeframe", _buckets(
            tf, "t", sorted(tf.t.unique())), jbase))
    sw = j[pd.to_numeric(j.Stop, errors="coerce").notna()].copy()
    if len(sw) and "Notional" in sw:
        px = pd.to_numeric(sw.Notional, errors="coerce") / pd.to_numeric(
            sw.Qty, errors="coerce")
        sw["swpct"] = 100 * (px - pd.to_numeric(sw.Stop, errors="coerce")) / px
        sw = sw[sw.swpct.between(0, 5)]
        if len(sw) > 6:
            sw["swb"] = pd.cut(sw.swpct, [0, .3, .6, 1.0, 5],
                               labels=["< 0.3%", "0.3-0.6%", "0.6-1.0%", "> 1.0%"])
            D.append(panel("By stop width (entry to stop)",
                           _buckets(sw, "swb"), jbase,
                           "Measured from the account's own fill to the stop "
                           "on the screenshot."))

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
<div class="tile"><b>{wR:+.2f}R</b><span>risk-weighted &mdash; what one dollar
 of risk returned</span></div>
<div class="tile"><b>{g.RealizedR.median():+.2f}R</b><span>median trade</span></div>
<div class="tile"><b>${g.risk.min():.0f}&ndash;${g.risk.max():.0f}</b>
<span>dollar risked per trade (median ${g.risk.median():.0f})</span></div>
<div class="tile"><b>{len(g)}</b><span>trades with a stop the account
 confirms</span></div></div>"""
        E.append(hist_panel(
            "Realised R", g.RealizedR, [-3, -2, -1.5, -1, -0.5, 0, 0.5, 1,
                                        1.5, 2, 3, 5, 17],
            "Red is a loss. The outlined bar is the -1R bucket: where the stop "
            "did exactly what it was set to do. Buckets widen to the right "
            "because the tail is long and thin.", mark=-1)
        )
        lab = [f"{r.date} {r.Time}: planned {r['INIT R/R']:.2f}, "
               f"realised {r.RealizedR:+.2f}R, {r['PnL$']:+.2f} on "
               f"${r.risk:.0f} risked" for _, r in g.iterrows()]
        E.append(scatter_panel(
            "Planned against realised", g["INIT R/R"], g.RealizedR, lab,
            xmax=6, ymin=-3, ymax=6,
            note="If the plan governed the outcome these would climb left to right. "
            "They do not: the losers sit on a flat line at -1R whatever was "
            "planned, because a stop is a stop."))
        E.append(hist_panel(
            "Dollar risked per trade", g.risk,
            [0, 10, 20, 30, 40, 50, 60, 80],
            "This is why R cannot be summed or averaged across trades: the "
            "smallest position risks a twentieth of the largest, so an "
            "unweighted mean hands the loudest vote to the smallest bet."))

    # How many stopped-out trades actually exited at the stop: the check
    # that says the transcribed stops are the real ones.
    conf = "0 of 0"
    if len(g):
        st = g[g.ExitMech == "initial stop"]
        if len(st):
            err = ((st.entry + st["PnL$"] / st.Qty) - st.Stop).abs() / st.entry
            conf = f"{int((100 * err < 0.1).sum())} of {len(st)}"

    cov = (f"{screens_done} of {screens_total} September screenshots transcribed"
           if screens_total else "")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Trade Settings</title><style>{CSS}</style></head><body>
<div class="wrap">
<div class="head"><h1>Which settings move the win rate</h1>
<button class="toggle" id="t">Dark</button></div>
<p class="sub">Every rate is a dot with its 95% Wilson interval. The dashed line
is the book&rsquo;s own rate. <strong>A whisker that crosses the dashed line is
telling you nothing</strong> &mdash; which is most of them, on two months of
data. Rows under 20 trades are greyed as a reminder.</p>
<div class="legend"><span><span class="sw"></span>win rate, n &ge; 20</span>
<span><span class="sw m"></span>n &lt; 20</span>
<span>&#8212; 95% interval</span>
<span>&#9482; book rate {100*base:.0f}%</span></div>

<h2>From the account &mdash; every position</h2>
<p class="sub">{len(pos)} positions, {pos.day.nunique()} trading days, book win
rate {100*base:.1f}%, {pos.pnl.sum():+,.2f} total. This is the only unbiased
section: it counts every fill, not just the trades worth writing up.</p>
{''.join(S)}

<h2>From the journal &mdash; the write-ups</h2>
<p class="sub">{len(j)} trades. A <em>selected</em> sample: the deck records the
trades worth reviewing, so these rates describe that selection rather than the
system. Dollar figures appear only where a write-up could be tied to a position
beyond doubt, so they under-count.</p>
{''.join(J)}

<h2>Execution &mdash; plan against outcome</h2>
<p class="sub">Only the trades whose stop the account itself confirms: the exit
of a stopped-out trade should land on the stop, and on {conf} of them it does
within 0.1% of price. R is a ratio, so it is never summed here &mdash;
risk-weighted means sum(P&amp;L) over sum(dollars risked).</p>
{tiles}
{''.join(E)}

<h2>From the screenshots &mdash; what was planned</h2>
<p class="sub">{cov}. Coverage grows as the captures are read.</p>
{''.join(D)}

<p class="note">Equity {equity:,.0f}. Win rate counts a position as a win at any
profit above zero, so scratches land on whichever side of zero they finished.
Interval is Wilson at 95%.</p>
</div><script>{JS}</script></body></html>"""

"""Build decision drills: a chart that stops, and what the tape did next.

A question shows the bars up to a decision point and nothing after it. The
trader says enter or pass, sets a stop and a target, and the forward path
settles it. Scoring is in R and only in R, because on the events this module
selects the structural label carries no information: measured over
2026-05-19 to 10-12 on SOXL, a deep decline reached +2 ATR 33.3% of the time
and a shallow pullback 33.0%, intervals overlapping. What does separate is
delta_r -- 44.8% when the decision minute's 1m reading is >= 0 against 25.3%
when it is negative.

CONFIRM is why this module exists rather than another script. Deciding on the
low bar itself is not a decision anyone makes: you cannot know a bar is the
low until it closes, and the playbook's own entry asks for a turn first. With
CONFIRM=1 the chart runs one bar past the low and the entry is that bar's
close. Every measured base rate moves with it, so the two settings are
generated and measured separately rather than mixed.

Each question carries what the trader actually watches:
  delta_r at 1, 5 and 15 minutes, all trailing windows ending at the decision
  bar, so none of them reads a minute that has not happened;
  RSI(6) on 1m and 5m;
  the session's volume profile as it could have been drawn at that moment;
  a structural reading -- swings, trend, retracement, support tests;
  and, for the sessions `feeds/book_ladder.py` has archived, the resting book
  over the preceding 30 minutes as a price-by-time grid.

The figures above are CONFIRM=0 and do not carry over: at CONFIRM=1 the 1m
delta_r split narrows to 41.5/28.4 and the 15m split disappears entirely
(35.6/34.1). `--confirm` writes the setting into each question id so the two
sets never mix, and the page loads the rates belonging to the set in play.
"""

from __future__ import annotations

import base64
import json
import pathlib
from datetime import date

import numpy as np
import pandas as pd

_ROOT = pathlib.Path(__file__).parent.parent

RTH = (9.5, 16.0)
LOOK, MIN_DROP_ATR, GAP = 30, 1.0, 15
# Same pair the viewer uses, so a reading learned here means the same thing
# on the live chart. See analysis/trade_viewer_qt.py's _KD_FAST/_KD_SLOW.
DK_FAST, DK_SLOW = 25, 90
# A pinch: the fast midline falling over PINCH_SLOPE bars while the gap to
# the slow one is shrinking over PINCH_NARROW and has closed to within
# PINCH_ATR of an ATR -- or has crossed. 0.25 was chosen over 0.5 and 1.0
# because it is the tight pinch the request described; the wider settings
# differ in base rate only by an amount their extra samples cannot support.
PINCH_ATR, PINCH_SLOPE, PINCH_NARROW = 0.25, 3, 5
# How far a pinch must keep from the last dip mark. Pinches cluster
# right after dips -- the median gap is 6 bars -- so requiring the full
# GAP left only 26 of 136 over 110 sessions. 5 keeps 104 of them while
# still dropping the ones that would render as the same chart a bar or
# two along. Chosen for that, not for a base rate: picking it by win
# rate would be fitting the same data the questions come from.
PINCH_DIP_GAP = 5
FWD = 30
BEFORE, AFTER = 120, 60          # wider than the first set's 60/40
MIN_SIGNED = 300.0
PIVOT_K = 3


def atr(h, l, c, n=14):
    tr = np.full(len(h), np.nan)
    for i in range(1, len(h)):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    a = np.full(len(h), np.nan)
    if len(h) > n:
        a[n] = np.nanmean(tr[1:n + 1])
        for i in range(n + 1, len(h)):
            a[i] = (a[i - 1] * (n - 1) + tr[i]) / n
    return a


def label(H, L, C, i, a, up=2.0, dn=1.0, fwd=FWD):
    """First barrier touched wins; None when neither is reached in time."""
    u, d = C[i] + up * a, C[i] - dn * a
    for j in range(i + 1, min(len(C), i + 1 + fwd)):
        tu, td = H[j] >= u, L[j] <= d
        if tu and td:
            return 0
        if tu:
            return 1
        if td:
            return 0
    return None


def pivots(H, L, lo, i, k=PIVOT_K):
    """Confirmed swings in [lo, i]. The last k bars cannot be confirmed --
    the honest cost of using a pivot definition in real time.

    A bar must beat both bars k away, not merely tie the window's extreme:
    on a flat stretch every bar ties, which would mark all of them as swings
    and report a structure built from noise.
    """
    sh = [j for j in range(lo + k, i - k + 1)
          if H[j] >= max(H[j - k:j + k + 1]) and H[j] > H[j - k] and H[j] > H[j + k]]
    sl = [j for j in range(lo + k, i - k + 1)
          if L[j] <= min(L[j - k:j + k + 1]) and L[j] < L[j - k] and L[j] < L[j + k]]
    return sh, sl


def structure(O, H, L, C, i, a, span=60):
    """Signed structural components read off the visible bars."""
    lo = max(0, i - span)
    sh, sl = pivots(H, L, lo, i)
    if len(sh) < 2 or len(sl) < 2:
        return 0, [(0, "可见区间内摆动点不足，结构无法判定")], {"trend": "未知"}
    h1, h2, l1, l2 = H[sh[-1]], H[sh[-2]], L[sl[-1]], L[sl[-2]]
    trend = ("上升" if h1 > h2 and l1 > l2
             else "下降" if h1 < h2 and l1 < l2 else "区间")
    peak, trough = max(h1, h2), min(l1, l2)
    leg = max(peak - trough, 1e-9)
    retr = (peak - L[i]) / leg
    out = []

    if trend == "上升":
        out.append((2, f"上升趋势（高点 {h2:.2f}→{h1:.2f}，低点 {l2:.2f}→{l1:.2f}），"
                       f"当前是趋势中的回调"))
        if 0.38 <= retr <= 0.68:
            out.append((1, f"回调 {retr:.0%}，在 38–68% 的常见区间"))
        elif retr > 0.85:
            out.append((-1, f"回调 {retr:.0%}，已吃掉整段推动，上升结构存疑"))
    elif trend == "下降":
        out.append((-2, f"下降趋势（高点 {h2:.2f}→{h1:.2f}，低点 {l2:.2f}→{l1:.2f}），"
                        f"做多是逆势"))
    else:
        out.append((0, f"区间结构（高点 {h2:.2f}/{h1:.2f}，低点 {l2:.2f}/{l1:.2f}）"))

    if L[i] > l1:
        out.append((1, f"未跌破前摆动低 {l1:.2f}，低点抬高仍成立"))
    elif C[i] > l1:
        out.append((1, f"下破前摆动低 {l1:.2f} 后收回其上（假突破）"))
    else:
        out.append((-1, f"收在前摆动低 {l1:.2f} 之下，支撑失守"))

    # Both legs run from the MOST RECENT swing high. The current one used to
    # start at max(h1, h2) while the previous started at H[sh[-1]]; when the
    # older high was the higher of the two those are different points and
    # the ratio compared two moves measured from different anchors.
    hi_i = sh[-1]
    lo_i = sl[-1] if hi_i < sl[-1] else sl[-2]
    prev_from, prev_to = H[hi_i], L[lo_i]
    prev_leg = abs(prev_from - prev_to)
    cur_from = H[hi_i]
    cur_leg = cur_from - L[i]
    if cur_leg <= 0:
        # The low is ABOVE the swing high it would retrace from: a breakout,
        # not a pullback, with no retracement to compare. Reported as one it
        # read "本段下跌 -1.2 ATR，是前一段的 -29%，推动在减弱" -- a negative
        # drop, and exactly backwards.
        out.append((1, f"低点 {L[i]:.2f} 站在前高 {cur_from:.2f} 之上，"
                       f"是突破不是回调"))
    elif prev_leg > 0:
        ratio = cur_leg / prev_leg
        seg = (f"本段（前高 {cur_from:.2f} → 当前低 {L[i]:.2f}）"
               f"跌 {cur_leg/a:.1f} ATR，"
               f"前一段（{prev_from:.2f} → {prev_to:.2f}）"
               f"跌 {prev_leg/a:.1f} ATR")
        if ratio < 0.7:
            out.append((1, f"{seg}，本段只有 {ratio:.0%}，推动在减弱"))
        elif ratio > 1.4:
            out.append((-1, f"{seg}，本段是 {ratio:.0%}，推动在加速"))

    touches = sum(1 for j in sl if abs(L[j] - L[i]) <= 0.5 * a)
    if touches == 1:
        out.append((1, "该价位此前被测试 1 次并守住"))
    elif touches >= 3:
        out.append((-1, f"该价位已被测试 {touches} 次，反复测试的支撑易破"))

    r_now = np.mean([H[j] - L[j] for j in range(i - 9, i + 1)])
    r_bef = np.mean([H[j] - L[j] for j in range(i - 19, i - 9)])
    if r_bef > 0 and r_now < 0.65 * r_bef:
        out.append((1, f"近 10 根振幅收缩到之前的 {r_now/r_bef:.0%}，波动在压缩"))

    return sum(s for s, _ in out), out, {"trend": trend, "retr": float(retr)}


def call(score):
    return "看涨" if score >= 2 else ("看跌" if score <= -1 else "中性")


def flow_series(buy, sell, n, lo, hi):
    """delta_r over a trailing n bars, for every bar in [lo, hi).

    Each point reads only bars at or before itself, so the series carries
    the same no-lookahead property the single reading has.
    """
    out = []
    for i in range(lo, hi):
        a = max(0, i - n + 1)
        b, s = buy[a:i + 1].sum(), sell[a:i + 1].sum()
        tot = b + s
        out.append(None if tot < MIN_SIGNED
                   else round(float((b - s) / tot), 3))
    return out


FLOW_WIN = 60          # trailing bars the volume scale is taken over


def flow_mag_series(buy, sell, lo, hi):
    """Signed volume against its own recent normal, per bar in [lo, hi).

    delta_r is a ratio and says nothing about size: a quiet bar and a
    heavy one can share it. The viewer gates on this magnitude instead
    -- |x| in the top fifth of its trailing 60 -- and over 60
    out-of-sample sessions that gate was worth more than the 0.4 line
    on the ratio (+0.363 R a trade against +0.264). The drill could not
    show it at all, which is what this adds.

        x_i = (buy - sell)_i / median(buy + sell over the previous 60)

    The scale is taken from the whole session, not from the question's
    window, so a window's first bars are scaled against real history.
    Shifted, so a bar never sets the scale it is then divided by. None
    where the bar's signed volume is under MIN_SIGNED, matching drs1.
    """
    tot = buy + sell
    ok = tot >= MIN_SIGNED
    scale = pd.Series(np.where(ok, tot, np.nan)).rolling(
        FLOW_WIN, min_periods=20).median().shift(1).to_numpy()
    out = []
    for i in range(lo, hi):
        s = scale[i]
        out.append(None if not ok[i] or not np.isfinite(s) or s <= 0
                   else round(float((buy[i] - sell[i]) / s), 2))
    return out


def flow(buy, sell, i, n):
    """delta_r over the n bars ending at i. Trailing, so nothing unseen."""
    b, s = buy[i - n + 1:i + 1].sum(), sell[i - n + 1:i + 1].sum()
    tot = b + s
    return None if tot < MIN_SIGNED else round(float((b - s) / tot), 3)


def _profile(upto):
    from strategy.session_vp.profile import bins_for_price, compute_value_area
    from strategy.smc.fvg import compute_volume_profile
    nb = bins_for_price(float(upto.low.min()), float(upto.high.max()))
    edges, vols = compute_volume_profile(upto, n_bins=nb) if nb else (None, None)
    if edges is None:
        return [], {"poc": 0.0, "vah": 0.0, "val": 0.0}
    va = compute_value_area(edges, vols, va_pct=0.70)
    top = float(vols.max()) or 1.0
    # Drawn at a few hundred pixels tall, so bins below 1% of the peak are a
    # sub-pixel bar: they cost file size and show nothing.
    prof = [[round(float((edges[b] + edges[b + 1]) / 2), 2),
             round(float(vols[b] / top), 3)]
            for b in range(len(vols)) if vols[b] / top >= 0.01]
    return prof, va


def load_range(code: str, start: str, end: str) -> pd.DataFrame:
    """1m klines joined to the aggressor split, fetched once for the range."""
    from feeds import delta_bars
    from feeds.fetcher import fetch_klines

    k = fetch_klines(code, "1m", start, end)
    if k is None or k.empty:
        return pd.DataFrame()
    k["t"] = pd.to_datetime(k.time_key)
    k = k.sort_values("t").drop_duplicates("t").set_index("t")
    dl = delta_bars.load(code, start, end)
    cols = ["buy", "sell", "neutral"]
    if dl is None or dl.empty:
        full = k.assign(**{c: 0.0 for c in cols})
    else:
        full = k.join(dl[cols], how="left")
    full[cols] = full[cols].fillna(0.0)
    return full


HM_MINS, HM_SECS, HM_BINS = 30, 10, 96
# Net aggressor volume a 10s column needs to raise a bubble. The viewer
# defaults to 500, which on SOXL marks most columns and so says little;
# 5000 picks out the ones worth looking at.
BUBBLE_MIN = 5000.0


def heatmap(lad, t_end, lo, hi):
    """Resting size as a price x time grid for the 30 minutes up to `t_end`.

    Quantised to one byte a cell against the grid's own maximum, because the
    page draws it as opacity and a byte is finer than the eye reads. Returns
    None when the archive has nothing for that window -- the ladder only goes
    back as far as the first day it was collected, so most questions have no
    heatmap and should not carry an empty one.
    """
    if lad is None or lad.empty or hi <= lo:
        return None
    t0 = t_end - pd.Timedelta(minutes=HM_MINS)
    w = lad[(lad.ts > t0) & (lad.ts <= t_end) &
            (lad.price >= lo) & (lad.price <= hi)]
    if w.empty:
        return None
    ncol = HM_MINS * 60 // HM_SECS
    col = ((w.ts - t0).dt.total_seconds().to_numpy() - 1e-9) // HM_SECS
    row = (w.price.to_numpy() - lo) / (hi - lo) * HM_BINS
    col = np.clip(col.astype(int), 0, ncol - 1)
    row = np.clip(row.astype(int), 0, HM_BINS - 1)
    # Bid and ask go out as SEPARATE grids. Summed into one, a bid wall and
    # the ask wall above it render as the same colour and the panel can no
    # longer answer the question it exists for -- which side is holding. The
    # live tool has exactly this split behind its bid/ask toggle.
    sz = w["size"].to_numpy(float)
    sd = w["side"].to_numpy()
    gb = np.zeros((HM_BINS, ncol), float)
    ga = np.zeros((HM_BINS, ncol), float)
    for tgt, mark in ((gb, "B"), (ga, "A")):
        m = sd == mark
        if m.any():
            np.add.at(tgt, (row[m], col[m]), sz[m])

    # Sizes SUM across the price levels inside a bin -- that is the band's
    # resting liquidity -- but AVERAGE across the seconds inside a column, so
    # a column is the typical book rather than a total that grows with how
    # many snapshots happened to land in it. Hence the divisor is the number
    # of distinct seconds per column, not the number of rows per cell.
    #
    # Both sides divide by the SAME second count -- every snapshot carries
    # both books, so a column has one population of seconds, not one per
    # side. Counting them per side would divide each by about the same
    # number anyway, but only by accident.
    ts_i = w.ts.to_numpy().astype("datetime64[s]").astype(np.int64)
    pairs = np.unique(np.stack([col, ts_i]), axis=1)
    secs = np.bincount(pairs[0], minlength=ncol).astype(float)
    for gr in (gb, ga):
        np.divide(gr, secs[None, :], out=gr, where=secs[None, :] > 0)

    # One shared reference, taken AFTER the per-second average, so the two
    # layers stay comparable: scaled apart, a thin ask would look as solid
    # as a heavy bid.
    top = float(max(gb.max(), ga.max()))
    if top <= 0:
        return None
    ref = np.log1p(top)
    out = {"lo": round(float(lo), 3), "hi": round(float(hi), 3),
           "nb": HM_BINS, "nc": ncol, "sec": HM_SECS, "log": 1,
           "top": int(round(top))}
    for key, gr in (("db", gb), ("da", ga)):
        qq = np.round(np.log1p(gr) / ref * 255).astype(np.uint8)
        out[key] = base64.b64encode(qq.tobytes()).decode("ascii")
    return out


def day_ticks(code: str, day) -> pd.DataFrame:
    """Raw prints for one session, for the bubble columns."""
    import sqlite3
    d0 = pd.Timestamp(day).normalize()
    db = _ROOT / "db" / "ticks.db"
    if not db.exists():
        return pd.DataFrame()
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        d = pd.read_sql(
            "select ts,price,volume,direction from ticks where code=? "
            "and ts>=? and ts<? order by ts", con,
            params=(code, f"{d0:%Y-%m-%d} 00:00:00",
                    f"{d0 + pd.Timedelta(days=1):%Y-%m-%d} 00:00:00"))
    except Exception:
        return pd.DataFrame()
    finally:
        con.close()
    if d.empty:
        return d
    d["ts"] = pd.to_datetime(d.ts, errors="coerce")
    return d.dropna(subset=["ts"]).sort_values("ts")


def bubbles(tk, t_end, lo: float, hi: float) -> list:
    """Columns whose net aggressor volume clears BUBBLE_MIN.

    Returns [col, price, net, side], side +1 buying and -1 selling. The price
    is the column's last print, which always exists; the viewer uses the
    book's mid, which only exists for the days the ladder archive covers.
    """
    if tk is None or len(tk) == 0:
        return []
    t0 = t_end - pd.Timedelta(minutes=HM_MINS)
    w = tk[(tk.ts > t0) & (tk.ts <= t_end)]
    w = w[w.direction.isin(("BUY", "SELL"))]
    if w.empty:
        return []
    ncol = HM_MINS * 60 // HM_SECS
    col = ((w.ts - t0).dt.total_seconds().to_numpy() - 1e-9) // HM_SECS
    col = np.clip(col.astype(int), 0, ncol - 1)
    sign = np.where(w.direction.to_numpy() == "BUY", 1.0, -1.0)
    net = np.bincount(col, weights=sign * w.volume.to_numpy(float),
                      minlength=ncol)
    last = np.full(ncol, np.nan)
    for c, v in zip(col, w.price.to_numpy(float)):
        last[c] = v
    out = []
    for c in np.nonzero(np.abs(net) >= BUBBLE_MIN)[0]:
        pxc = last[c]
        if not np.isfinite(pxc) or pxc < lo or pxc > hi:
            continue
        out.append([int(c), round(float(pxc), 2), int(abs(net[c])),
                    1 if net[c] > 0 else -1])
    return out


def traded_profile(tk, t_end, lo: float, hi: float):
    """Volume executed at each price over the 30 minutes up to `t_end`.

    Same shape as `heatmap()` so the page can draw either. Summed within a
    cell -- unlike the resting book there is nothing to average, every print
    is a distinct event rather than a repeated snapshot of one order.
    """
    if tk is None or len(tk) == 0 or hi <= lo:
        return None
    t0 = t_end - pd.Timedelta(minutes=HM_MINS)
    w = tk[(tk.ts > t0) & (tk.ts <= t_end) &
           (tk.price >= lo) & (tk.price <= hi)]
    if w.empty:
        return None
    ncol = HM_MINS * 60 // HM_SECS
    col = ((w.ts - t0).dt.total_seconds().to_numpy() - 1e-9) // HM_SECS
    row = (w.price.to_numpy(float) - lo) / (hi - lo) * HM_BINS
    col = np.clip(col.astype(int), 0, ncol - 1)
    row = np.clip(row.astype(int), 0, HM_BINS - 1)
    # Split by aggressor, the traded analogue of the book's bid/ask: these
    # are prints, so the sides are who crossed the spread rather than who was
    # resting. Drawn in the same two colours, because the question a reader
    # asks of either panel is the same one -- which side is doing the work.
    vol = w.volume.to_numpy(float)
    dirn = w.direction.to_numpy() if "direction" in w else np.array([""] * len(w))
    gb = np.zeros((HM_BINS, ncol), float)
    ga = np.zeros((HM_BINS, ncol), float)
    for tgt, mark in ((gb, "BUY"), (ga, "SELL")):
        m = dirn == mark
        if m.any():
            np.add.at(tgt, (row[m], col[m]), vol[m])
    top = float(max(gb.max(), ga.max()))
    if top <= 0:
        return None
    ref = np.log1p(top)
    # SPARSE, unlike the resting heatmap. Price visits a narrow band at any
    # instant, so ~95% of this grid is zero; dense base64 would spend 11.5 KB
    # a question storing nothing and add 4 MB to the set.
    #
    # The floor is on the RAW grid, not on the quantised byte: in log space a
    # single share already codes above 6, so a byte test would keep every
    # cell price ever touched and the sparse form would stop being sparse.
    # 0.2% of the window's own maximum -- the old byte test cut at 2.4%,
    # which is what hid ordinary size next to a wall.
    floor = max(1.0, top * 0.002)
    out = {"lo": round(float(lo), 3), "hi": round(float(hi), 3),
           "nb": HM_BINS, "nc": ncol, "sec": HM_SECS,
           "top": int(round(top)), "log": 1}
    for key, gr in (("sb", gb), ("sa", ga)):
        qq = np.round(np.log1p(gr) / ref * 255).astype(int)
        rr, cc = np.nonzero(gr >= floor)
        out[key] = [[int(r), int(c), int(qq[r, c])] for r, c in zip(rr, cc)]
    if not out["sb"] and not out["sa"]:
        return None
    return out


def build_day(code: str, day: date | str, full: pd.DataFrame, confirm: int = 1,
              kinds=("deep", "shallow", "pinch"), lad=None, tk=None,
              light=False) -> list[dict]:
    """Questions for one session. `confirm` bars past the marked low.

    `light` drops the chart arrays, the volume profile and the heatmap and
    keeps only the fields a measurement reads. The profile is recomputed from
    the session open for every event, which dominates the runtime, so a study
    over months is many times faster without it.
    """
    from strategy.session_vp.reversal import compute_rsi

    d0 = pd.Timestamp(day).normalize()
    g = full[full.index.date == d0.date()]
    hh = g.index.hour + g.index.minute / 60
    reg = g[(hh > RTH[0]) & (hh <= RTH[1])].reset_index()
    if len(reg) < 380:
        return []

    O, H, L, C = (reg.open.to_numpy(float), reg.high.to_numpy(float),
                  reg.low.to_numpy(float), reg.close.to_numpy(float))
    buy, sell = reg.buy.to_numpy(float), reg.sell.to_numpy(float)
    A = atr(H, L, C)
    rsi1 = compute_rsi(reg.close.reset_index(drop=True), 6)
    # A 5m bar stamped 11:35 closes on the 1m bar stamped 11:35, so ffill at
    # minute i never reaches past i: at 11:33 the newest stamp at or before it
    # is 11:30, whose last 1m bar already closed. No shift is needed, and
    # shifting would serve a reading 5-7 minutes stale.
    five = reg.set_index("t").close.resample("5min", label="right",
                                             closed="right").last().dropna()
    rsi5s = pd.Series(compute_rsi(five, 6), index=five.index)
    rsi5 = rsi5s.reindex(reg["t"], method="ffill").to_numpy()

    # DK (多空通道): the fast/slow EMA channel midlines the viewer draws as
    # a ribbon. Computed over the whole session so each question's window
    # inherits the same warmup the chart has, then sliced per question.
    from strategy.smc.kd_trend import compute_kd
    _dk = compute_kd(reg, fast=DK_FAST, slow=DK_SLOW)
    dk1 = _dk["mid1"].to_numpy(float)
    dk2 = _dk["mid2"].to_numpy(float)

    spread = dk1 - dk2
    marks, last, last_p, armed = [], -99, -99, True
    for i in range(BEFORE, len(reg) - FWD - AFTER - confirm):
        a = A[i]
        if np.isnan(a) or a <= 0:
            continue

        # A pinch holds for many bars, so only its leading edge is an event.
        # Arming is evaluated on EVERY bar, including ones the spacing rules
        # below reject -- otherwise an episode that lapses during a cooldown
        # would never re-arm and the next one would be missed.
        pinch = bool(
            np.isfinite(spread[i]) and np.isfinite(spread[i - PINCH_NARROW])
            and dk1[i] < dk1[i - PINCH_SLOPE]
            and abs(spread[i]) < abs(spread[i - PINCH_NARROW])
            and (0 < spread[i] <= PINCH_ATR * a
                 or spread[i - 1] > 0 >= spread[i]))
        if not pinch:
            armed = True

        if i - last >= GAP:
            is_deep = (L[i] == L[i - LOOK:i + 1].min()
                       and (H[i - LOOK:i + 1].max() - L[i])
                       >= MIN_DROP_ATR * a)
            is_shal = (not is_deep) and L[i] == L[i - 10:i + 1].min()
            if is_deep and "deep" in kinds:
                last = i
                marks.append(("深回落", i))
                continue
            if is_shal and "shallow" in kinds:
                last = i
                marks.append(("浅回调", i))
                continue

        # Last, and on its own cursor: a pinch must never displace a dip,
        # whose id already carries saved answers.
        if (pinch and armed and "pinch" in kinds
                and i - last_p >= GAP and i - last >= PINCH_DIP_GAP):
            armed, last_p = False, i
            marks.append(("夹角", i))

    out = []
    for kind, i0 in marks:
        i = i0 + confirm                      # decide after the confirmation
        a = A[i]
        if np.isnan(a) or a <= 0:
            continue
        y = label(H, L, C, i, a)
        if y is None:
            continue
        s, e = i - BEFORE, min(len(reg), i + AFTER + 1)
        w = reg.iloc[s:e]
        sc, why, info = structure(O, H, L, C, i, a)
        if light:
            prof, va = [], {"poc": 0.0, "vah": 0.0, "val": 0.0}
        else:
            prof, va = _profile(reg.iloc[:i + 1])
        out.append({
            "id": f"{d0.date()}_{reg['t'].iloc[i]:%H%M}_{kind}_c{confirm}",
            "kind": kind, "sym": code.split(".")[-1], "dir": "long",
            "day": str(d0.date()), "tm": f"{reg['t'].iloc[i]:%H:%M}",
            # `lab` is the 2:1 barrier the study labels on; `lab11` is the
            # 1:1 one, which is a different question and cannot be derived
            # from it -- a path that reaches +1 ATR and turns back scores a
            # win on one and a loss on the other.
            "confirm": confirm, "mark": i - s, "lab": int(y),
            "lab11": (lambda v: None if v is None else int(v))(
                label(H, L, C, i, a, up=1.0, dn=1.0)),
            "atr": round(float(a), 4), "entry": round(float(C[i]), 3),
            "low0": round(float(L[i0]), 3),
            "dr1": flow(buy, sell, i, 1),
            "dr5": flow(buy, sell, i, 5),
            "dr15": flow(buy, sell, i, 15),
            "rsi1": None if not np.isfinite(rsi1[i]) else round(float(rsi1[i]), 1),
            "rsi5": None if not np.isfinite(rsi5[i]) else round(float(rsi5[i]), 1),
            # Two decimals: these are prices on a chart, not a statistic,
            # and the extra digits would cost ~0.5 MB a set for nothing.
            "dk1": [None if not np.isfinite(v) else round(float(v), 2)
                    for v in dk1[s:e]],
            "dk2": [None if not np.isfinite(v) else round(float(v), 2)
                    for v in dk2[s:e]],
            "sc": int(sc), "call": call(sc), "trend": info["trend"],
            "why": [[int(v), txt] for v, txt in why],
            "prof": prof,
            "poc": round(float(va["poc"]), 3),
            "vah": round(float(va["vah"]), 3),
            "val": round(float(va["val"]), 3),
        })
        if light:
            continue
        out[-1].update({
            "t": [f"{x:%H:%M}" for x in w["t"]],
            "o": [round(v, 2) for v in w.open], "h": [round(v, 2) for v in w.high],
            "l": [round(v, 2) for v in w.low], "c": [round(v, 2) for v in w.close],
            "rsi": [None if not np.isfinite(v) else round(float(v), 1)
                    for v in rsi1[s:e]],
            # The delta_r line. 1 bar is the sharpest reading and 5 keeps
            # most of it; 15 was measured at 1.5 points of separation and
            # stays a header number rather than earning a line.
            "drs1": flow_series(buy, sell, 1, s, e),
            "drs5": flow_series(buy, sell, 5, s, e),
            # The size behind the ratio. See flow_mag_series: this is
            # what the panel's lit/dim split reads, and it is the
            # stronger of the two signals the light reports.
            "xs": flow_mag_series(buy, sell, s, e),
        })
        vlo = float(min(L[s:i + 1])) - 0.02 * a
        vhi = float(max(H[s:i + 1])) + 0.02 * a
        hm = heatmap(lad, reg["t"].iloc[i], vlo, vhi)
        if hm:
            out[-1]["hm"] = hm
        bub = bubbles(tk, reg["t"].iloc[i], vlo, vhi)
        if bub:
            out[-1]["bub"] = bub
            out[-1]["bubMin"] = int(BUBBLE_MIN)
        # Only when there is no resting book to show -- it is the better
        # answer where it exists, and carrying both doubles the payload.
        if not hm:
            tvp = traded_profile(tk, reg["t"].iloc[i], vlo, vhi)
            if tvp:
                out[-1]["tvp"] = tvp
    return out


def _read_set(path: pathlib.Path) -> list[dict]:
    """Read a question set back, whether it was written as .json or as .js.

    The .js form is `window.DRILL_c1=[...];` so that the page can load it
    through a <script> tag under file://. That is not JSON, and --append read
    it with json.loads -- which is why the 21:00 job failed on its first real
    run. Parsing both shapes here keeps the two ends from drifting again.
    """
    t = path.read_text(encoding="utf-8").strip()
    if t.startswith("window."):
        t = t[t.index("=") + 1:].rstrip(";").strip()
    return json.loads(t) if t else []


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--code", default="US.SOXL")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--confirm", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--append", action="store_true",
                    help="keep the existing questions and add sessions newer "
                         "than the newest one already in the file")
    ap.add_argument("--keep", type=int, default=600,
                    help="cap on total questions; oldest sessions drop first")
    ap.add_argument("--no-rates", dest="rates", action="store_false",
                    help="do not regenerate rates_c{confirm}.js beside the "
                         "set. Only for writing a set somewhere scratch, "
                         "where a rates file would be noise.")
    ap.add_argument("--replace", action="store_true",
                    help="regenerate the sessions in --start..--end, dropping "
                         "what is already on file for those days. Use after a "
                         "data fix: --append would skip them as already done.")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    prior = []
    if (args.append or args.replace) and out.exists():
        prior = _read_set(out)
        if prior and args.replace:
            lo = str(pd.Timestamp(args.start).date())
            hi = str(pd.Timestamp(args.end).date())
            before = len(prior)
            prior = [r for r in prior if not (lo <= r["day"] <= hi)]
            print(f"replace: dropped {before - len(prior)} question(s) "
                  f"in {lo}..{hi}, {len(prior)} kept")
        elif prior:
            newest = max(r["day"] for r in prior)
            args.start = (pd.Timestamp(newest)
                          + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            print(f"append: {len(prior)} on file through {newest}, "
                  f"scanning from {args.start}")
            if pd.Timestamp(args.start) > pd.Timestamp(args.end):
                print("nothing new")
                return

    full = load_range(args.code, args.start, args.end)
    if full.empty:
        print("no klines in range")
        return
    # Loaded one session at a time rather than the whole range: the ladder is
    # ~7.5M rows a day and the archive only grows, so loading it all held a
    # gigabyte to read four days of it.
    try:
        from feeds import book_ladder
    except Exception as exc:
        print(f"ladder unavailable ({exc}); questions will have no heatmap")
        book_ladder = None
    rows = []
    for d in pd.date_range(args.start, args.end, freq="D"):
        if d.weekday() >= 5:
            continue
        lad = None
        if book_ladder is not None:
            try:
                # The end bound is a timestamp, not a date: passing `d` here
                # would keep only rows at or before midnight and silently
                # drop that session's entire trading day.
                lad = book_ladder.load(args.code, d,
                                       d + pd.Timedelta(days=1))
                if lad is None or lad.empty:
                    lad = None
            except Exception:
                lad = None
        tk = day_ticks(args.code, d)
        got = build_day(args.code, d, full, confirm=args.confirm,
                        lad=lad, tk=tk)
        if got:
            print(f"  {d.date()}: {len(got)}", flush=True)
            rows.extend(got)
    fresh = len(rows)
    if prior:
        seen = {r["id"] for r in prior}
        rows = prior + [r for r in rows if r["id"] not in seen]
    rows.sort(key=lambda r: (r["day"], r["tm"]))
    if len(rows) > args.keep:
        # Trim whole sessions, not a slice through one, so a day is either
        # entirely present or entirely gone.
        days = sorted({r["day"] for r in rows})
        while len(days) > 1 and len(rows) > args.keep:
            days.pop(0)
            rows = [r for r in rows if r["day"] in set(days)]

    out.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(rows, separators=(",", ":"))
    if out.suffix == ".js":
        # A .js file loads through a <script> tag, which file:// allows;
        # fetch() of a .json sibling does not. Writing the set as an
        # assignment is what lets the page be opened by double-clicking it
        # instead of needing a local server.
        body = f"window.DRILL_c{args.confirm}={body};"
    out.write_text(body, encoding="utf-8")
    from collections import Counter
    print(f"+{fresh} new, {len(rows)} total -> {args.out}  "
          f"{dict(Counter(r['kind'] for r in rows))}")

    if args.rates:
        _write_rates(rows, args.confirm, args.code, out.parent)


def _write_rates(rows, confirm, code, where: pathlib.Path) -> None:
    """Regenerate the base rates beside the set they describe.

    Computed from `rows` in memory, not by re-reading the file, so the two
    cannot disagree even if something goes wrong with the write above.
    """
    from strategy.drill_rates import build
    import json as _json
    try:
        data = build(rows, confirm, code)
    except Exception as exc:                      # a bad rates file is worse
        print(f"rates: 没有生成（{type(exc).__name__}: {exc}）")
        return
    p = where / f"rates_c{confirm}.js"
    p.write_text(f"window.RATES_c{confirm}=" +
                 _json.dumps(data, ensure_ascii=False,
                             separators=(",", ":")) + ";\n",
                 encoding="utf-8")
    b = data["buckets"]["_all"]
    print(f"rates: n={data['n']}，{data['sessions']} 个交易日 "
          f"{data['period'][0]}..{data['period'][1]}，"
          f"到 +2ATR {b[1]:.1%} [{b[2]:.1%}, {b[3]:.1%}] -> {p.name}")


if __name__ == "__main__":
    main()

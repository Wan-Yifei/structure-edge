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

    prev_leg = abs(H[sh[-1]] - L[sl[-1] if sh[-1] < sl[-1] else sl[-2]])
    cur_leg = peak - L[i]
    if prev_leg > 0:
        ratio = cur_leg / prev_leg
        if ratio < 0.7:
            out.append((1, f"本段下跌 {cur_leg/a:.1f} ATR，是前一段的 {ratio:.0%}，"
                           f"推动在减弱"))
        elif ratio > 1.4:
            out.append((-1, f"本段下跌 {cur_leg/a:.1f} ATR，是前一段的 {ratio:.0%}，"
                            f"推动在加速"))

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


HM_MINS, HM_SECS, HM_BINS = 30, 10, 48
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
    grid = np.zeros((HM_BINS, ncol), float)
    np.add.at(grid, (row, col), w["size"].to_numpy(float))
    # Sizes SUM across the price levels inside a bin -- that is the band's
    # resting liquidity -- but AVERAGE across the seconds inside a column, so
    # a column is the typical book rather than a total that grows with how
    # many snapshots happened to land in it. Hence the divisor is the number
    # of distinct seconds per column, not the number of rows per cell.
    ts_i = w.ts.to_numpy().astype("datetime64[s]").astype(np.int64)
    pairs = np.unique(np.stack([col, ts_i]), axis=1)
    secs = np.bincount(pairs[0], minlength=ncol).astype(float)
    grid = np.divide(grid, secs[None, :], out=grid, where=secs[None, :] > 0)
    top = grid.max()
    if top <= 0:
        return None
    q = np.round(grid / top * 255).astype(np.uint8)
    return {"lo": round(float(lo), 3), "hi": round(float(hi), 3),
            "nb": HM_BINS, "nc": ncol, "sec": HM_SECS,
            "top": int(round(top)),
            "d": base64.b64encode(q.tobytes()).decode("ascii")}


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
    grid = np.zeros((HM_BINS, ncol), float)
    np.add.at(grid, (row, col), w.volume.to_numpy(float))
    top = grid.max()
    if top <= 0:
        return None
    # SPARSE, unlike the resting heatmap. Price visits a narrow band at any
    # instant, so ~95% of this grid is zero; dense base64 would spend 11.5 KB
    # a question storing nothing and add 4 MB to the set.
    q = np.round(grid / top * 255).astype(int)
    rr, cc = np.nonzero(q >= 6)
    return {"lo": round(float(lo), 3), "hi": round(float(hi), 3),
            "nb": HM_BINS, "nc": ncol, "sec": HM_SECS, "top": int(round(top)),
            "s": [[int(r), int(c), int(q[r, c])] for r, c in zip(rr, cc)]}


def build_day(code: str, day: date | str, full: pd.DataFrame, confirm: int = 1,
              kinds=("deep", "shallow"), lad=None, tk=None,
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

    marks, last = [], -99
    for i in range(BEFORE, len(reg) - FWD - AFTER - confirm):
        a = A[i]
        if np.isnan(a) or a <= 0 or i - last < GAP:
            continue
        is_deep = (L[i] == L[i - LOOK:i + 1].min()
                   and (H[i - LOOK:i + 1].max() - L[i]) >= MIN_DROP_ATR * a)
        is_shal = (not is_deep) and L[i] == L[i - 10:i + 1].min()
        if is_deep and "deep" in kinds:
            last = i
            marks.append(("深回落", i))
        elif is_shal and "shallow" in kinds:
            last = i
            marks.append(("浅回调", i))

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


if __name__ == "__main__":
    main()

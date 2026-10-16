"""Drill questions built from trades the user actually took.

The structural drills ask whether a textbook location is worth entering. This
asks something harder and more useful: here is a moment you DID enter, with
the chart stopped exactly where you stopped it -- decide again, with no
memory of what happened, and see whether the second decision beats the first.

What the ledger supplies that the generated questions cannot:

  the entry is one a human chose, not one a scan found, so it is drawn from
  the distribution of setups actually traded rather than the two structural
  classes;
  the stop and target really used are on file, so after the reveal the two
  parameter sets can be compared on the same entry -- which separates "the
  location was wrong" from "the location was fine and the stop was wrong";
  both directions appear, where the generated set is long only.

What it cannot supply is a base rate. These entries are not the measured
classes, so the page must not quote 深回落/浅回调 numbers at them. The
question carries `nobase: True` to say so.

The trader's own stop and target are withheld until the reveal: showing them
up front would anchor the answer, which is the one thing this exercise is
supposed to be free of.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from strategy.drill_gen import (AFTER, BEFORE, FWD, RTH, _profile, _read_set,
                                atr, call, flow, heatmap, label, load_range,
                                structure)

LEDGER = pathlib.Path(__file__).parent / "journal" / "output" / "ledger.csv"
SESSIONS = {"regular", "overnight", "afterhours", "premarket"}
# Measured, not assumed: regression of 1m SOXS returns on SOXL returns over
# 2026-09-01..10-10 RTH. Used to carry a SOXS stop/target onto SOXL.
SOXS_BETA = -1.008


def read_ledger(path: pathlib.Path = LEDGER,
                symbols=("SOXL", "SOXS")) -> pd.DataFrame:
    """Trades with an entry time, a stop and a target, on `symbols`.

    SOXS trades are kept and expressed as SOXL. The decision is made on
    SOXL's order flow -- SOXS is only the vehicle -- and the two are inverse
    to within measurement error: over 10,920 shared RTH minutes the 1-minute
    return correlation is -0.9898, the regression slope -1.008, and on the
    largest 20% of SOXL moves the signs disagree 100% of the time. So a short
    in SOXS is read as a long in SOXL, and the stop and target convert as
    percentages of entry.

    That also rescues them: the collector's targets list holds only US.SOXL,
    so `ticks.db` has 52 million SOXL prints against 52 thousand SOXS ones --
    one day in July. Charted as SOXS these questions would have no delta at
    all, and delta_r is the one feature that has survived every control.
    """
    d = pd.read_csv(path)
    d.columns = [c.strip().lstrip("﻿") for c in d.columns]
    d = d[d.Symbol.isin(symbols)]
    d = d[d.ShotTime.notna() & d.Stop.notna() & d.Target.notna()]
    # ShotTime is ET, and it matches the ledger's own Session column: an
    # overnight fill reads 23:56 on the calendar day BEFORE its trading day.
    d = d.assign(st=pd.to_datetime(d.ShotTime, errors="coerce"))
    d = d.dropna(subset=["st"])
    d = d[d["W/L"].isin(("win", "loss"))]
    # itertuples cannot address "INIT R/R" -- the space and slash become a
    # positional name -- and the entry recovery needs it by name.
    d = d.rename(columns={"INIT R/R": "init_rr"})
    d = d[d.init_rr.notna()]
    return d.sort_values("st").reset_index(drop=True)


def recover_entry(stop: float, target: float, rr: float) -> float:
    """The fill price, from the stop, the target and the planned R/R.

    The ledger's `Entry` column is a dash on nearly every row, and the
    screenshot timestamp is not the fill -- using that minute's close put
    both the stop and the target on the same side of "entry" and produced
    levels that made sense for neither direction. But |target-entry| =
    rr * |entry-stop| solves for entry, and the result lands between the two
    levels on 100% of rows, which the close did not.
    """
    return (target + rr * stop) / (1.0 + rr)


def _dist_atr(level: float, entry_ref: float, soxl_entry: float,
              atr_val: float, on_soxs: bool) -> float:
    """How far a recorded level sat from the fill, in ATR of the SOXL chart.

    Distances transfer where absolute levels do not. The fill price is not
    the close of the screenshot's minute -- matching the two put the stop on
    the wrong side of "entry" on half the SOXL rows -- but how far the stop
    sat from wherever the fill actually was carries over cleanly, and in ATR
    it is directly comparable to the sliders, which are also in ATR.

    SOXS distances invert through the measured beta on the way across.
    """
    if entry_ref <= 0 or atr_val <= 0:
        return float("nan")
    pct = (float(level) - entry_ref) / entry_ref
    if on_soxs:
        pct = pct / SOXS_BETA
    return abs(pct) * soxl_entry / atr_val


def build(code: str, rows: pd.DataFrame, full: pd.DataFrame,
          book_ladder=None) -> list[dict]:
    """One question per ledger row whose minute is in `full`."""
    from strategy.session_vp.reversal import compute_rsi

    out = []
    for day, day_rows in rows.groupby(rows.st.dt.normalize()):
        g = full[full.index.normalize() == day]
        if len(g) < 200:
            continue
        # Per day, not per range: the ladder is ~7.5M rows a day. Most of
        # these trades predate the archive, so this is usually empty.
        lad = None
        if book_ladder is not None:
            try:
                lad = book_ladder.load(code, day, day + pd.Timedelta(days=1))
                if lad is not None and lad.empty:
                    lad = None
            except Exception:
                lad = None
        reg = g.reset_index()
        O, H, L, C = (reg.open.to_numpy(float), reg.high.to_numpy(float),
                      reg.low.to_numpy(float), reg.close.to_numpy(float))
        buy, sell = reg.buy.to_numpy(float), reg.sell.to_numpy(float)
        A = atr(H, L, C)
        rsi1 = compute_rsi(reg.close.reset_index(drop=True), 6)
        five = reg.set_index("t").close.resample("5min", label="right",
                                                 closed="right").last().dropna()
        rsi5 = (pd.Series(compute_rsi(five, 6), index=five.index)
                .reindex(reg["t"], method="ffill").to_numpy())
        # The bar CONTAINING the fill: time_key is the bar's end, so the
        # minute stamped 11:47 holds a fill at 11:46:08.
        tmap = {t: j for j, t in enumerate(reg["t"])}

        for _, r in day_rows.iterrows():
            key = r.st.ceil("1min")
            i = tmap.get(key)
            if i is None or i < BEFORE or i >= len(reg) - FWD - AFTER:
                continue
            a = A[i]
            if not np.isfinite(a) or a <= 0:
                continue
            # `Direction` is the view on the MARKET, not the side taken on
            # the instrument: a "short" row on SOXS is a long position in
            # SOXS expressing a bearish SOXL view. Checked against the data --
            # SOXS targets sit above the SOXS entry 92.5% of the time, which
            # is a long in SOXS. So the column already reads in SOXL terms
            # and needs no flip; only the price levels have to be carried
            # across, and those invert.
            # The instrument is decided by the PRICE, not the Symbol column:
            # some rows carry SOXS-level stops under Symbol=SOXL. `_on_soxs`
            # and `_entry_ref` are attached in main() by comparing the
            # recovered fill against both closes at that minute.
            on_soxs = bool(r._on_soxs)
            entry_ref = float(r._entry_ref)
            short = str(r.Direction).lower() == "short"
            y = label(H, L, C, i, a)
            s, e = i - BEFORE, min(len(reg), i + AFTER + 1)
            w = reg.iloc[s:e]
            sc, why, info = structure(O, H, L, C, i, a)
            prof, va = _profile(reg.iloc[:i + 1])
            q = {
                "id": f"mine_{r.st:%Y%m%d_%H%M}_{r.Symbol}",
                "kind": "我的交易", "sym": "SOXL", "vehicle": r.Symbol,
                "dir": "short" if short else "long",
                "day": str(day.date()), "tm": f"{reg['t'].iloc[i]:%H:%M}",
                "confirm": 0, "mark": i - s,
                "lab": None if y is None else int(y),
                "nobase": True,          # not one of the measured classes
                "sess": str(r.Session),
                "atr": round(float(a), 4), "entry": round(float(C[i]), 3),
                "low0": round(float(C[i]), 3),
                "dr1": flow(buy, sell, i, 1),
                "dr5": flow(buy, sell, i, 5),
                "dr15": flow(buy, sell, i, 15),
                "rsi1": None if not np.isfinite(rsi1[i]) else round(float(rsi1[i]), 1),
                "rsi5": None if not np.isfinite(rsi5[i]) else round(float(rsi5[i]), 1),
                "sc": int(sc), "call": call(sc), "trend": info["trend"],
                "why": [[int(v), t] for v, t in why],
                "prof": prof,
                "poc": round(float(va["poc"]), 3),
                "vah": round(float(va["vah"]), 3),
                "val": round(float(va["val"]), 3),
                # Revealed only after the answer is submitted. A SOXS
                # stop/target is carried over as a percentage of entry and
                # divided by the measured beta, which is the only honest way
                # to put the two instruments on one axis.
                # Their parameters as ATR distances, then placed on this
                # question's entry -- which puts them on the correct side by
                # construction instead of by luck.
                "myStopAtr": round(_dist_atr(r.Stop, entry_ref, C[i], a, on_soxs), 3),
                "myTgtAtr": round(_dist_atr(r.Target, entry_ref, C[i], a, on_soxs), 3),
                "myRR": round(float(r.init_rr), 2),
                "onSoxs": bool(on_soxs),
                "rawStop": round(float(r.Stop), 3),
                "rawTarget": round(float(r.Target), 3),
                "myWL": str(r["W/L"]),
                "myR": (None if pd.isna(r.get("RealizedR"))
                        else round(float(r["RealizedR"]), 2)),
                "note": ("" if pd.isna(r.get("Evaluation"))
                         else str(r["Evaluation"])[:300]),
                "setup": ("" if pd.isna(r.get("Setup")) else str(r["Setup"])),
                "t": [f"{x:%H:%M}" for x in w["t"]],
                "o": [round(v, 2) for v in w.open], "h": [round(v, 2) for v in w.high],
                "l": [round(v, 2) for v in w.low], "c": [round(v, 2) for v in w.close],
                "rsi": [None if not np.isfinite(v) else round(float(v), 1)
                        for v in rsi1[s:e]],
            }
            sgn = 1.0 if short else -1.0
            q["myStop"] = round(C[i] + sgn * q["myStopAtr"] * a, 3)
            q["myTarget"] = round(C[i] - sgn * q["myTgtAtr"] * a, 3)
            vlo = float(min(L[s:i + 1])) - 0.02 * a
            vhi = float(max(H[s:i + 1])) + 0.02 * a
            hm = heatmap(lad, reg["t"].iloc[i], vlo, vhi)
            if hm:
                q["hm"] = hm
            out.append(q)
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sessions", default="regular",
                    help="comma-separated ledger sessions to include. "
                         "Default 'regular' (RTH): the only one with a "
                         "comparable base rate measured on it. 'all' takes "
                         "every session.")
    ap.add_argument("--out", default="strategy/drill/questions_mine.js")
    ap.add_argument("--ledger", default=str(LEDGER))
    ap.add_argument("--symbols", default="SOXL,SOXS",
                    help="comma-separated ledger symbols to draw from. Both "
                         "by default; SOXS trades are charted as SOXL.")
    args = ap.parse_args()

    d = read_ledger(pathlib.Path(args.ledger),
                    symbols=tuple(args.symbols.split(",")))
    want = SESSIONS if args.sessions == "all" else set(args.sessions.split(","))
    d = d[d.Session.isin(want)]
    print(f"ledger: {len(d)} 笔  "
          f"{dict(d['W/L'].value_counts())}  {dict(d.Symbol.value_counts())}")
    if d.empty:
        return

    lo = (d.st.min() - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    hi = (d.st.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        from feeds import book_ladder
    except Exception:
        book_ladder = None

    # Every question is charted on SOXL, because that is the book the
    # decision was read from. A SOXS row needs its own close at the fill
    # minute to turn its stop and target into percentages.
    from feeds.fetcher import fetch_klines
    closes = {}
    for sym in ("SOXL", "SOXS"):
        k = fetch_klines(f"US.{sym}", "1m", lo, hi)
        k["t"] = pd.to_datetime(k.time_key)
        closes[sym] = k.set_index("t").close

    on_soxs, entry_ref, keep = [], [], []
    for r in d.itertuples():
        e = recover_entry(float(r.Stop), float(r.Target), float(r.init_rr))
        m = r.st.ceil("1min")
        cl, cs = closes["SOXL"].get(m), closes["SOXS"].get(m)
        dl = abs(e / cl - 1) if cl and cl > 0 else 9.9
        ds = abs(e / cs - 1) if cs and cs > 0 else 9.9
        is_s = ds < dl
        on_soxs.append(is_s)
        entry_ref.append(e)
        keep.append(min(dl, ds) <= 0.03)      # 3% of the close, or unusable
    d = d.assign(_on_soxs=on_soxs, _entry_ref=entry_ref, _keep=keep)
    bad = (~d._keep).sum()
    if bad:
        print(f"  {bad} 笔的解出入场价和两个标的当时价格都差 3% 以上，丢掉")
    d = d[d._keep]
    mism = (d._on_soxs != (d.Symbol == "SOXS")).sum()
    if mism:
        print(f"  {mism} 笔的 Symbol 列和价位对不上，按价位判定")

    full = load_range("US.SOXL", lo, hi)
    if full.empty:
        print("没有 SOXL K 线")
        return
    rows = build("US.SOXL", d, full, book_ladder=book_ladder)
    nhm = sum(1 for q in rows if "hm" in q)
    byv = {"SOXL 原单": sum(1 for q in rows if not q["onSoxs"]),
           "SOXS 换算": sum(1 for q in rows if q["onSoxs"])}
    print(f"  {len(rows)} 题（{nhm} 题有热力图） {byv}")

    rows.sort(key=lambda r: (r["day"], r["tm"]))
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(rows, separators=(",", ":"), ensure_ascii=False)
    if out.suffix == ".js":
        body = f"window.DRILL_mine={body};"
    out.write_text(body, encoding="utf-8")
    from collections import Counter
    print(f"{len(rows)} 题 -> {args.out}  "
          f"方向 {dict(Counter(r['dir'] for r in rows))}  "
          f"盈亏 {dict(Counter(r['myWL'] for r in rows))}")


if __name__ == "__main__":
    main()

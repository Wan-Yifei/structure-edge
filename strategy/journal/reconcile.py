"""Match journal trades to account positions and emit one ledger.

The journal numbers trades 1..n per day; the account knows only times.  The
rule that makes them line up is the one the user states himself: trades whose
profit or loss is negligible do not get logged.  Drop those and, on four of the
six days covered, what remains agrees with the journal on all three of count,
win/loss sequence and instrument sequence -- so the match is asserted only when
all three agree, and flagged for review otherwise.

The two days that do not match are both explained rather than mysterious:

  10/07  the journal's trade 4 is an add-on to trade 3 ("三次触碰入场加仓").
         The account shows the added lot losing $5.38 while the journal scores
         the combined idea a win, which it was: +21.20 - 5.38 = +15.82.
         The journal's grain is the trade idea; the account's is the lot.

  10/08  one real position (10:56, -13.40) is missing from the journal. It is
         not a scratch -- it is the fourth largest loss of that day -- so the
         journal's -85.71 understates the account's -98.51 by 13%.

Realised R is the point of the whole exercise: the account's own entry and exit
measured against the stop the screenshot shows.  -1.0 means the stop did its
job, and anything past it means it did not.
"""

from __future__ import annotations

import pathlib
import re

import pandas as pd

# Below this, a position is a scratch the journal does not record.  The gap in
# the data is clean: scratches come in at 0.03-0.04% of notional and the next
# smallest real trade is 0.13%, so the threshold is not a fitted parameter.
SCRATCH_USD = 2.0

# Read off the R/R tool overlay, which prints account equity under each
# outcome: stop 29,982.00 and target 30,088.61 on a 17-share position risking
# 1.019/share puts the base at 29,999.  Used only to express P&L as a percent.
EQUITY = 30_000.0

# The days 3 touch was being tested; everything else is IVB.  A strategy the
# journal names explicitly always wins over this default.
THREE_TOUCH_DAYS = {"2026-10-07", "2026-10-08"}

# Slide 10 is titled SOXS but its chart is SOXL (prices 144-155) and its note
# says "VAL 向 POC做多".  There is no SOXS position that afternoon.  Correcting
# it is what lets 10/01 match.
SYMBOL_FIXES = {10: "SOXL"}

_ZONE = re.compile(r"VAH|VAL|POC")

# Structural patterns, for the 13 trades whose note names no profile level.
_SETUPS = [("三触", "3 touch"), ("三次触碰", "3 touch"), ("第四次触碰", "4th touch"),
           ("箱体突破", "range breakout"), ("突破", "breakout"),
           ("夹角", "wedge"), ("流动性扫荡", "liquidity sweep"),
           ("抄底", "bottom fishing"), ("抓反弹", "bounce"),
           ("逆CVD", "against CVD"), ("反弹做空", "short the bounce")]


def entry_zone(note: str) -> str:
    """The first level named, not a fixed priority.

    "VAL 向 POC做多" enters at VAL and targets POC; ranking VAH > VAL > POC
    happens to get that one right and would read "POC 向 VAH" backwards.
    """
    m = _ZONE.search(str(note))
    # "-", not "NA": pandas reads the string NA back as a missing value, which
    # turns "the note names no level" into "this field was never filled in".
    return m.group(0) if m else "-"


def setup_tag(note: str) -> str:
    hits = [en for zh, en in _SETUPS if zh in str(note)]
    if "range breakout" in hits and "breakout" in hits:
        hits.remove("breakout")
    if "3 touch" in hits:
        hits = ["3 touch"] + [h for h in hits if h != "3 touch"]
    return "; ".join(dict.fromkeys(hits))


def exit_kind(note: str) -> str:
    """How the position was closed, from the note.

    Order matters: "抬高SL，被扫TP" is a trailed stop that caught a profit, not
    a stop loss, and testing for SL first would file it as one.
    """
    n = str(note)
    if "抬高SL" in n or "被扫TP" in n:
        return "Stopped out in profit"
    if "提前TP" in n:
        return "Early TP"
    if "TP" in n:
        return "Target"
    if "SL" in n or "被扫" in n:
        return "stop loss"
    return ""


def _blank(v) -> bool:
    return str(v).strip().lower() in ("", "nan", "none")


def build(trades: pd.DataFrame, positions: pd.DataFrame,
          screens: pd.DataFrame, hand: pd.DataFrame) -> pd.DataFrame:
    """One row per journal trade, plus a row per unlogged scratch."""
    trades = trades.copy()
    trades["day"] = trades.day.astype(str)
    for slide, sym in SYMBOL_FIXES.items():
        trades.loc[trades.slide == slide, "sym"] = sym
    positions = positions.copy()
    positions["day"] = positions.day.astype(str)
    screens = screens.copy()
    screens["day"] = screens.day.astype(str)
    shots = {(r.day, r.idx): r for _, r in screens.iterrows()}

    hand = hand.copy()
    hand.columns = [c.strip() for c in hand.columns]
    hand["day"] = pd.to_datetime(hand["date"], format="%m/%d/%Y").dt.date.astype(str)
    for c in hand.columns:
        hand[c] = hand[c].astype(str).str.strip()
    # The hand journal is a straight transcription of the deck, in slide order.
    by_slide: dict[int, dict] = {}
    for day, grp in trades.groupby("day"):
        grp = grp.sort_values("idx").reset_index(drop=True)
        h = hand[hand.day == day].reset_index(drop=True)
        for i, r in grp.iterrows():
            by_slide[r.slide] = h.loc[i].to_dict() if i < len(h) else {}

    rows: list[dict] = []
    for day in sorted(trades.day.unique()):
        logged = trades[trades.day == day].sort_values("idx").reset_index(drop=True)
        real = positions[(positions.day == day)
                         & (positions.pnl.abs() >= SCRATCH_USD)] \
            .sort_values("open").reset_index(drop=True)
        scratches = positions[(positions.day == day)
                              & (positions.pnl.abs() < SCRATCH_USD)]

        matched = (len(real) == len(logged)
                   and all((p > 0) == (r == "WIN")
                           for p, r in zip(real.pnl, logged.result))
                   and list(real.sym) == list(logged.sym))

        for i, t in logged.iterrows():
            m = real.loc[i] if matched else None
            k = shots.get((day, t.idx))
            own = by_slide.get(t.slide, {})

            realised = ""
            if m is not None and k is not None and pd.notna(k.stop):
                risk = m.px_in - k.stop
                if risk > 0:
                    realised = round((m.px_out - m.px_in) / risk, 2)

            strategy = own.get("Strategy", "")
            if _blank(strategy) or strategy.upper() == "NA":
                strategy = "3 touch" if day in THREE_TOUCH_DAYS else "IVB"
            zone = own.get("Entry", "")
            if _blank(zone) or zone.upper() == "NA":
                zone = entry_zone(t.note)

            rows.append({
                "date": day,
                "Time": f"{m['open']:%H:%M:%S}" if m is not None else "",
                "W/L": "win" if t.result == "WIN" else "loss",
                "Strategy": strategy,
                "Entry": zone,
                "Setup": setup_tag(t.note),
                "Direction": "short" if t.sym == "SOXS" else "long",
                "Time frame": (int(k.tf) if k is not None
                               else own.get("Time frame", "")),
                "INIT R/R": k.rr if k is not None else own.get("INIT R/R", ""),
                "RealizedR": realised,
                "Exit": (own["Exit"] if not _blank(own.get("Exit", ""))
                         else exit_kind(t.note)),
                "Evaluation": t.note,
                "Symbol": t.sym,
                "Qty": int(m.qty) if m is not None else "",
                "Notional": round(m.notional, 2) if m is not None else "",
                "PnL$": round(m.pnl, 2) if m is not None else "",
                "PnL%equity": round(100 * m.pnl / EQUITY, 3) if m is not None else "",
                "HoldMins": round(m.mins, 1) if m is not None else "",
                "Stop": k.stop if k is not None else "",
                "Target": (k.target if k is not None and pd.notna(k.target) else ""),
                "ShotTime": k.shot_ts if k is not None else "",
                "Scratch": "",
                "Match": "auto" if matched else "NEEDS REVIEW",
                "Slide": t.slide,
            })

        for _, m in scratches.iterrows():
            rows.append({
                "date": day, "Time": f"{m['open']:%H:%M:%S}", "Symbol": m.sym,
                "W/L": "win" if m.pnl > 0 else "loss",
                "Direction": "short" if m.sym == "SOXS" else "long",
                "Qty": int(m.qty), "Notional": round(m.notional, 2),
                "PnL$": round(m.pnl, 2),
                "PnL%equity": round(100 * m.pnl / EQUITY, 3),
                "HoldMins": round(m.mins, 1), "Scratch": "Y",
                "Match": "statement only (not in journal)",
            })

    cols = ["date", "Time", "W/L", "Strategy", "Entry", "Setup", "Direction",
            "Time frame", "INIT R/R", "RealizedR", "Exit", "Evaluation",
            "Symbol", "Qty", "Notional", "PnL$", "PnL%equity", "HoldMins",
            "Stop", "Target", "ShotTime", "Scratch", "Match", "Slide"]
    out = pd.DataFrame(rows).reindex(columns=cols).fillna("")
    return out.sort_values(["date", "Time"]).reset_index(drop=True)

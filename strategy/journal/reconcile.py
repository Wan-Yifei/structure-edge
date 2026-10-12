"""Match journal trades to account positions and emit one ledger.

**The deck is the record; the account is the money.** The journal is written
up for the trades worth reviewing, not for every fill, so a day with 29
positions and 22 write-ups is correct rather than incomplete. That makes the
ledger journal-driven: every trade in the deck gets a row, position data is
attached only where it can be identified beyond doubt, and the positions the
deck never mentions are carried as their own rows rather than discarded.

Two consequences worth keeping in mind when reading the output:

**Journal rows are a selected sample.** Anything computed over them -- win
rate, R multiples, which setup pays -- describes the trades that were thought
worth writing up, not the system. Account-level expectancy has to come from
every position, which is what output/account_days.csv carries.

**A journal trade is an idea, not a lot.** 10/07's trade 4 is an add-on to
trade 3; the account shows the added lot losing $5.38 while the deck scores the
combined idea a win, which it was (+21.20 - 5.38 = +15.82). So a write-up can
span several positions, which is why sign-sequence matching tops out at 26% of
trades uniquely placed and why the capture timestamp in each screenshot is the
only thing that resolves the rest.

Identification is asserted only when count, win/loss sequence and instrument
sequence all agree for a day. That holds on 10 of 28 days; the rest are written
out as `unresolved` rather than guessed.

Realised R is the point of the whole exercise: the account's own entry and exit
measured against the stop the screenshot shows. -1.0 means the stop did its
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

# ET session boundaries, the same ones config/schedule.json uses for the
# collectors.  Taken from the fill time rather than the note: "开盘" in a
# write-up means the cash open loosely, while the statement knows the minute.
SESSIONS = [(20.0, 24.0, "overnight"), (0.0, 4.0, "overnight"),
            (4.0, 9.5, "premarket"), (9.5, 16.0, "regular"),
            (16.0, 20.0, "afterhours")]

# What the notes call each session, for the trades no position could be tied to.
_NOTE_SESSION = [("夜盘", "overnight"), ("盘前", "premarket"),
                 ("盘后", "afterhours"), ("开盘", "regular"),
                 ("收盘", "regular")]

_ZONE = re.compile(r"VAH|VAL|POC")

# What was traded, and separately what the market was doing. The two were one
# column at first and it did not work: "夹角" is the KD bands pinching shut --
# a condition on an indicator, not a setup -- and the trades taken there were
# IVB entries at POC like any other. Ranking it beside "3 touch" compared two
# different axes, when the question worth asking is how the same setup does
# inside that condition and outside it.
_SETUPS = [("三触", "3 touch"), ("三次触碰", "3 touch"),
           ("第四次触碰", "4th touch"),
           ("箱体突破", "range breakout"), ("突破", "breakout")]

_CONTEXTS = [("夹角", "KD pinch"),            # the KD bands converging
             # Direction calls, not patterns: the setup underneath was an
             # ordinary IVB or 3 touch, only the way round differed.
             ("抄底", "fading a drop"), ("抓反弹", "fading a drop"),
             ("反弹做空", "fading a bounce"),
             ("流动性扫荡", "liquidity sweep"),
             ("逆CVD", "against CVD"),
             ("冲动", "impulse entry"),
             ("过度交易", "overtrading"),
             ("交易次数过多", "overtrading"),
             ("违规", "rule broken"),
             ("RSI过高", "RSI stretched"),
             ("RSI过大", "RSI stretched"),
             ("没有等", "did not wait")]


def session_of(ts) -> str:
    """Which session a fill time falls in."""
    if ts is None or pd.isna(ts):
        return ""
    h = ts.hour + ts.minute / 60
    for lo, hi, name in SESSIONS:
        if lo <= h < hi:
            return name
    return ""


def session_from_note(note: str) -> str:
    for zh, name in _NOTE_SESSION:
        if zh in str(note):
            return name
    return ""


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
        hits.remove("breakout")        # the specific one wins
    if "3 touch" in hits:
        hits = ["3 touch"] + [h for h in hits if h != "3 touch"]
    return "; ".join(dict.fromkeys(hits))


def context_tag(note: str) -> str:
    """Conditions the write-up noted about the market or about himself."""
    return "; ".join(dict.fromkeys(
        en for zh, en in _CONTEXTS if zh in str(note)))


# How a position was closed, read off the note.  Ordered: the first pattern
# that matches wins, because the phrasings nest -- "抬高SL，被扫TP" is a stop
# that was moved up and then caught a profit, and testing for SL or for TP
# first would file it as either a loss or a target hit.
#
# The distinction that matters is between `target` and `trailed stop`. Both end
# in profit and the old single label called both "Stopped out in profit", but
# only one of them reached the level the trade was designed around. Across 245
# trades: 61 target, 45 trailed stop, 12 manual early -- so roughly half the
# winners never saw their planned TP.
EXIT_MECHANISMS = [
    (("止损失效", "击穿"), "stop blown"),      # the stop did not hold
    (("抬高SL", "拉高SL", "被扫TP"), "trailed stop"),
    (("提前TP",), "manual early"),             # closed by hand, short of target
    (("TP",), "target"),
    (("SL", "被扫", "止损"), "initial stop"),
]


def exit_mechanism(note: str) -> str:
    n = str(note)
    for needles, label in EXIT_MECHANISMS:
        if any(x in n for x in needles):
            return label
    return "unstated"


def target_hit(realised, planned) -> str:
    """Did the trade reach what it was designed for?

    The note says how the position was closed; this says whether that close
    got the planned reward, which is the question the mechanism only implies.
    Blank where either number is missing.
    """
    try:
        r, p_ = float(realised), float(planned)
    except (TypeError, ValueError):
        return ""
    if p_ <= 0 or r <= 0:
        return "N" if r <= 0 else ""
    return "Y" if r >= 0.9 * p_ else "N"


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
    # Keyed on (day, slide). The index is not unique within a day, and the
    # slide number is not unique across decks -- each deck starts at 1, so
    # keying on the slide alone handed October's trades September's designs.
    # The statement caught it: four stopped-out trades came back with a stop
    # from the wrong instrument entirely (a 51.30 SOXS stop on a 149.45 SOXL
    # entry). Nothing internal to the journal would have noticed.
    shots = {(str(r.day), int(r.slide)): r for _, r in screens.iterrows()
             if pd.notna(r.get("slide"))}

    hand = hand.copy()
    hand.columns = [c.strip() for c in hand.columns]
    hand["day"] = pd.to_datetime(hand["date"], format="%m/%d/%Y").dt.date.astype(str)
    for c in hand.columns:
        hand[c] = hand[c].astype(str).str.strip()
    # The hand journal is a straight transcription of the deck, in slide order.
    by_slide: dict[int, dict] = {}
    for day, grp in trades.groupby("day"):
        grp = grp.sort_values("seq").reset_index(drop=True)
        h = hand[hand.day == day].reset_index(drop=True)
        for i, r in grp.iterrows():
            by_slide[r.slide] = h.loc[i].to_dict() if i < len(h) else {}

    rows: list[dict] = []
    for day in sorted(trades.day.unique()):
        day_rows = trades[trades.day == day]
        planned = day_rows[day_rows.result == "NOT TAKEN"]
        logged = (day_rows[day_rows.result != "NOT TAKEN"]
                  .sort_values("seq").reset_index(drop=True))
        day_pos = positions[positions.day == day].sort_values("open")
        # Trades too small to be worth a write-up are the ones the deck most
        # reliably leaves out, so they are set aside before testing the match.
        real = day_pos[day_pos.pnl.abs() >= SCRATCH_USD].reset_index(drop=True)

        matched = (len(real) == len(logged)
                   and all((p > 0) == (r == "WIN")
                           for p, r in zip(real.pnl, logged.result))
                   and list(real.sym) == list(logged.sym))

        for i, t in logged.iterrows():
            m = real.loc[i] if matched else None
            k = shots.get((day, int(t.slide)))
            own = by_slide.get(t.slide, {})

            realised = ""
            if m is not None and k is not None and pd.notna(k.stop):
                # Weighted average entry, not the first leg's price. A buy that
                # fills at two prices in the same second becomes one position
                # here, and taking the first print as the entry mis-states the
                # risk: on 09-21 it put the entry 9 cents below the average,
                # shrank the denominator to a third of the real stop distance
                # and turned a 16R trade into a 56R one. Deriving both ends
                # from the position's own notional and P&L makes
                # R x risk = P&L true by construction.
                entry = m.notional / m.qty
                exit_ = entry + m.pnl / m.qty
                risk = entry - k.stop
                if risk > 0:
                    realised = round((exit_ - entry) / risk, 2)

            strategy = own.get("Strategy", "")
            if _blank(strategy) or strategy.upper() == "NA":
                strategy = "3 touch" if day in THREE_TOUCH_DAYS else "IVB"
            zone = own.get("Entry", "")
            if _blank(zone) or zone.upper() == "NA":
                zone = entry_zone(t.note)

            sess = session_of(m["open"]) if m is not None else ""
            if not sess and k is not None and pd.notna(k.shot_ts):
                sess = session_of(pd.to_datetime(k.shot_ts))
            if not sess:
                sess = session_from_note(t.note)

            rows.append({
                "date": day,
                "Time": f"{m['open']:%H:%M:%S}" if m is not None else "",
                "Session": sess,
                "W/L": "win" if t.result == "WIN" else "loss",
                "Strategy": strategy,
                "Entry": zone,
                "Setup": setup_tag(t.note),
                "Context": context_tag(t.note),
                "Direction": "short" if t.sym == "SOXS" else "long",
                # A capture the deck reuses from another slide has a row but
                # no numbers, so every field here has to survive a NaN.
                "Time frame": (int(k.tf) if k is not None and pd.notna(k.tf)
                               else own.get("Time frame", "")),
                "INIT R/R": (k.rr if k is not None and pd.notna(k.rr)
                             else own.get("INIT R/R", "")),
                "RealizedR": realised,
                "Exit": own.get("Exit", ""),
                "ExitMech": exit_mechanism(t.note),
                "TargetHit": target_hit(realised, k.rr if k is not None
                                        else own.get("INIT R/R", "")),
                "Idx": int(t.seq),
                "IdxWritten": (int(t.idx) if pd.notna(t.idx) else ""),
                "Evaluation": t.note,
                "Symbol": t.sym,
                "Qty": int(m.qty) if m is not None else "",
                "Notional": round(m.notional, 2) if m is not None else "",
                "PnL$": round(m.pnl, 2) if m is not None else "",
                "PnL%equity": round(100 * m.pnl / EQUITY, 3) if m is not None else "",
                "HoldMins": round(m.mins, 1) if m is not None else "",
                "Stop": (k.stop if k is not None and pd.notna(k.stop) else ""),
                "Target": (k.target if k is not None and pd.notna(k.target) else ""),
                "ShotTime": (k.shot_ts if k is not None
                             and pd.notna(k.shot_ts) else ""),
                "Source": "journal+account" if matched else "journal only",
                "Slide": t.slide,
            })

        for _, t in planned.iterrows():
            rows.append({
                "date": day, "W/L": "not taken", "Symbol": t.sym,
                "Direction": "short" if t.sym == "SOXS" else "long",
                "Strategy": ("3 touch" if day in THREE_TOUCH_DAYS else "IVB"),
                "Entry": entry_zone(t.note), "Setup": setup_tag(t.note),
                "Context": context_tag(t.note),
                "Session": session_from_note(t.note),
                "Evaluation": t.note, "Slide": t.slide,
                "Source": "journal only (never entered)",
            })

        # On a resolved day the deck accounted for every position above the
        # threshold, so only the small ones went unwritten.  On an unresolved
        # day nothing is claimed, so the whole day's positions are emitted --
        # the journal rows carry no P&L there, so no money is counted twice and
        # the account's own total for the day stays recoverable from the ledger.
        unlogged = (day_pos[day_pos.pnl.abs() < SCRATCH_USD] if matched
                    else day_pos)
        tag = "account only" if matched else "account (day unresolved)"
        for _, m in unlogged.iterrows():
            rows.append({
                "date": day, "Time": f"{m['open']:%H:%M:%S}", "Symbol": m.sym,
                "Session": session_of(m["open"]),
                "W/L": "win" if m.pnl > 0 else "loss",
                "Direction": "short" if m.sym == "SOXS" else "long",
                "Qty": int(m.qty), "Notional": round(m.notional, 2),
                "PnL$": round(m.pnl, 2),
                "PnL%equity": round(100 * m.pnl / EQUITY, 3),
                "HoldMins": round(m.mins, 1),
                "Source": tag,
            })

    cols = ["date", "Time", "Session", "W/L", "Strategy", "Entry", "Setup",
            "Context", "Direction",
            "Time frame", "INIT R/R", "RealizedR", "ExitMech", "TargetHit",
            "Exit", "Evaluation",
            "Symbol", "Qty", "Notional", "PnL$", "PnL%equity", "HoldMins",
            "Stop", "Target", "ShotTime", "Source", "Idx", "IdxWritten",
            "Slide"]
    out = pd.DataFrame(rows).reindex(columns=cols).fillna("")
    return out.sort_values(["date", "Time"]).reset_index(drop=True)

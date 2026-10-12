"""Read moomoo trade confirmations into fills, and fills into positions.

The confirmation is the only source in this folder that knows what actually
happened to the money. The journal knows intent; this knows fills.

Two things about it are easy to get wrong:

**The statement date is a trading day, not a calendar day.** Page 6 of every
confirmation says transactions between 8:00 p.m. and 4:00 a.m. ET belong to the
overnight session, and the Oct 08 statement duly opens at 10/07 20:06. That is
the same 20:00 ET boundary the journal uses, so no conversion is needed -- but
a position opened before 20:00 and closed after it is split across two
statements, which is why `positions` chains inventory across the whole run
instead of processing each file alone.

**Short exposure is expressed by buying SOXS, never by shorting SOXL.** Naive
FIFO invents short positions out of the opening sells that close inventory
carried in from the previous day. The share counts settle it: starting SOXL at
the carried-in quantity keeps every intermediate position non-negative and ends
flat, which a genuine short book would not.
"""

from __future__ import annotations

import pathlib
import re
from datetime import date, datetime, timedelta

import pandas as pd
from pypdf import PdfReader

SYMBOLS = ("SOXL", "SOXS")
_DATE = re.compile(r"\d{4}/\d{2}/\d{2}")
_TIME = re.compile(r"\d\d:\d\d:\d\d")


def trading_day(ts: datetime) -> date:
    """The trading day a fill belongs to, under the 20:00 ET boundary.

    Derived from the fill, not from the filename: the current month arrives as
    one confirmation per day (`oct0826.pdf`) and earlier months as a single
    monthly statement (`september2026.pdf`), and only the timestamps are
    common to both.  Checked against the October filenames, which it
    reproduces exactly.

    The roll forward over a weekend never fires on real data -- the market is
    shut from Friday 20:00 to Sunday 20:00 -- but a Friday evening fill would
    otherwise be filed on a Saturday.
    """
    d = ts.date() + (timedelta(days=1) if ts.hour >= 20 else timedelta(0))
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def fills(pdf: pathlib.Path) -> pd.DataFrame:
    """Every execution in one confirmation.

    pypdf flattens the table to one cell per line, so the parse walks forward
    from each Buy/Sell and takes the first symbol, then the fixed run of
    date, time, price, quantity that follows "Agency".
    """
    text = "\n".join(p.extract_text() for p in PdfReader(str(pdf)).pages)
    lines = [l.strip() for l in text.splitlines()]
    out: list[dict] = []
    i = 0
    while i < len(lines):
        if lines[i] in ("Buy", "Sell"):
            side, j, sym = lines[i], i + 1, None
            while j < min(i + 8, len(lines)):
                if lines[j] in SYMBOLS:
                    sym = lines[j]
                    break
                j += 1
            if sym and j + 7 < len(lines) and lines[j + 1] == "Agency":
                d, t = lines[j + 3], lines[j + 4]
                if _DATE.fullmatch(d) and _TIME.fullmatch(t):
                    out.append({
                        "side": side, "sym": sym, "stmt": pdf.stem,
                        "ts": datetime.strptime(f"{d} {t}", "%Y/%m/%d %H:%M:%S"),
                        "price": float(lines[j + 5].replace(",", "")),
                        "qty": int(lines[j + 6].replace(",", "")),
                    })
                    i = j + 7
        i += 1
    return pd.DataFrame(out)


def load_fills(folder: pathlib.Path) -> pd.DataFrame:
    """Every confirmation in *folder*, oldest fill first."""
    pdfs = sorted(folder.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"no confirmations in {folder}")
    f = pd.concat([fills(p) for p in pdfs], ignore_index=True)
    return _drop_overlap(f).sort_values("ts").reset_index(drop=True)


def _drop_overlap(f: pd.DataFrame) -> pd.DataFrame:
    """Remove fills a monthly statement shares with a daily confirmation.

    The daily confirmation for 10/01 covers 09/30 20:00 onward, which the
    September monthly also reports, so those executions arrive twice.

    Deduplicating on the fill itself does not work and silently destroys data:
    a 150-share order routinely fills as nine prints of 19 shares at the same
    price in the same second, and those are nine real executions. Dropping
    them put September's P&L at -943 against a statement that says -170.

    As of the current data this finds nothing: moomoo's monthly statement
    already stops at the 20:00 boundary -- September ends at 09/30 10:33 and
    the 10/01 confirmation starts at 09/30 21:34 -- so the two do not in fact
    overlap. It stays as a guard for a broker that is less careful.

    The overlap is resolved by coverage instead. Each statement's span is
    its own first and last fill; where two spans intersect, the longer
    statement's fills inside the shorter one's window are the duplicates, and
    the more specific statement is kept.
    """
    spans = f.groupby("stmt").ts.agg(["min", "max"])
    spans["len"] = spans["max"] - spans["min"]
    drop = pd.Series(False, index=f.index)
    for long_, l in spans.sort_values("len", ascending=False).iterrows():
        for short, s_ in spans.iterrows():
            if short == long_ or s_["len"] >= l["len"]:
                continue
            # Intersection, not containment: the 10/01 confirmation starts
            # inside September and ends after it, so requiring the short span
            # to sit wholly inside the long one matches nothing.
            if s_["min"] <= l["max"] and l["min"] <= s_["max"]:
                drop |= ((f.stmt == long_)
                         & f.ts.between(s_["min"], s_["max"]))
    if drop.any():
        n = int(drop.sum())
        print(f"[confirmations] {n} fills covered by two statements; kept the "
              f"daily confirmation's copy")
    return f[~drop]


def positions(f: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """FIFO the fills into round turns on a long-only book.

    Returns (positions, unmatched).  Partial fills are folded back together by
    entry, because one decision that filled in three prints is one position in
    the journal.  `unmatched` holds shares sold with no buy in range (bought
    before the first statement) and anything still open at the end.
    """
    legs: list[dict] = []
    unmatched: list[dict] = []
    for sym, g in f.groupby("sym"):
        lots: list[list] = []
        for _, r in g.iterrows():
            if r.side == "Buy":
                lots.append([r.qty, r.price, r.ts, r.stmt])
                continue
            qty = r.qty
            while qty > 0 and lots:
                lot = lots[0]
                n = min(qty, lot[0])
                legs.append({"sym": sym, "qty": n, "open": lot[2], "close": r.ts,
                             "px_in": lot[1], "px_out": r.price,
                             "pnl": (r.price - lot[1]) * n,
                             "stmt_in": lot[3], "stmt_out": r.stmt})
                lot[0] -= n
                qty -= n
                if lot[0] == 0:
                    lots.pop(0)
            if qty > 0:
                unmatched.append({"sym": sym, "qty": qty, "close": r.ts,
                                  "px_out": r.price, "why": "entry predates the run"})
        for lot in lots:
            unmatched.append({"sym": sym, "qty": lot[0], "open": lot[2],
                              "px_in": lot[1], "why": "still open at the end"})

    t = pd.DataFrame(legs)
    pos = (t.assign(notional=t.px_in * t.qty)
             .groupby(["sym", "open"])
             .agg(qty=("qty", "sum"), px_in=("px_in", "first"),
                  px_out=("px_out", "last"), close=("close", "max"),
                  pnl=("pnl", "sum"), notional=("notional", "sum"),
                  stmt_in=("stmt_in", "first"), stmt_out=("stmt_out", "last"))
             .reset_index().sort_values("open").reset_index(drop=True))
    pos["day"] = pos["open"].map(trading_day)
    pos["mins"] = (pos.close - pos.open).dt.total_seconds() / 60
    return pos, unmatched

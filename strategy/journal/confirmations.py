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
from datetime import datetime

import pandas as pd
from pypdf import PdfReader

SYMBOLS = ("SOXL", "SOXS")
_DATE = re.compile(r"\d{4}/\d{2}/\d{2}")
_TIME = re.compile(r"\d\d:\d\d:\d\d")


def statement_day(pdf: pathlib.Path) -> datetime:
    """`oct0826.pdf` -> 2026-10-08.  The month lives in the filename prefix."""
    m = re.fullmatch(r"([a-z]{3})(\d{2})(\d{2})", pdf.stem.lower())
    if not m:
        raise ValueError(f"cannot read a date out of {pdf.name}")
    months = ["jan", "feb", "mar", "apr", "may", "jun",
              "jul", "aug", "sep", "oct", "nov", "dec"]
    return datetime(2000 + int(m.group(3)), months.index(m.group(1)) + 1,
                    int(m.group(2)))


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
    pdfs = sorted(folder.glob("*.pdf"), key=statement_day)
    if not pdfs:
        raise FileNotFoundError(f"no confirmations in {folder}")
    f = pd.concat([fills(p) for p in pdfs], ignore_index=True)
    return f.sort_values("ts").reset_index(drop=True)


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
    pos["day"] = pos.stmt_in.map(lambda s: statement_day(pathlib.Path(s + ".pdf")).date())
    pos["mins"] = (pos.close - pos.open).dt.total_seconds() / 60
    return pos, unmatched

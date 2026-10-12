#!/usr/bin/env python3
"""Build the trade ledger from the journal, the confirmations and the screens.

    uv run strategy/journal/build.py
    uv run strategy/journal/build.py --crops      # also re-cut the screenshots

Outputs output/ledger.csv plus a reconciliation summary on stdout.  Nothing
here writes to data/ -- the three sources are read-only records.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from strategy.journal import (confirmations, levels,   # noqa: E402
                              reconcile, report, slides)

HERE = pathlib.Path(__file__).parent
DATA = HERE / "data"
OUT = HERE / "output"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--crops", action="store_true",
                    help="re-extract and crop the slide screenshots")
    args = ap.parse_args()

    decks = sorted(DATA.glob("*.pptx"))
    if not decks:
        sys.exit(f"no .pptx in {DATA}")
    OUT.mkdir(exist_ok=True)

    journal = pd.concat([slides.trades(d) for d in decks], ignore_index=True)
    journal = journal.sort_values(["day", "idx"]).reset_index(drop=True)
    fills = confirmations.load_fills(DATA / "statements")
    positions, unmatched = confirmations.positions(fills)
    screens = (pd.read_csv(DATA / "screens.csv")
               if (DATA / "screens.csv").exists() else
               pd.DataFrame(columns=["day", "idx", "tf", "rr", "stop",
                                     "target", "qty_shown", "shot_ts"]))
    hand = pd.read_csv(DATA / "journal.csv")

    print(f"{len(journal)} journal trades over {journal.day.nunique()} days  "
          f"({', '.join(d.name for d in decks)})")
    not_taken = int((journal.result == "NOT TAKEN").sum())
    if not_taken:
        print(f"  {not_taken} planned but never entered -- no fill to match")
    print(f"{len(fills)} fills -> {len(positions)} positions  "
          f"({fills.ts.min():%m-%d %H:%M} .. {fills.ts.max():%m-%d %H:%M})")
    print(f"{len(screens)} R/R designs read off the screenshots\n")

    cross = positions[positions.stmt_in != positions.stmt_out]
    if len(cross):
        print(f"-- {len(cross)} positions cross a statement boundary --")
        for _, r in cross.iterrows():
            print(f"   {r.sym} {r.qty:>3}  {r['open']:%m-%d %H:%M} ({r.stmt_in})"
                  f" -> {r.close:%m-%d %H:%M} ({r.stmt_out})  {r.pnl:>+8.2f}")
        print()
    for u in unmatched:
        print(f"   unmatched: {u['sym']} {u['qty']} -- {u['why']}")

    led = reconcile.build(journal, positions, screens, hand)
    led.to_csv(OUT / "ledger.csv", index=False, encoding="utf-8-sig")

    # The account's own truth, uncontaminated by which trades were written up.
    days = (positions.assign(day=positions.day.astype(str))
            .groupby("day")
            .agg(positions=("pnl", "size"), wins=("pnl", lambda s: (s > 0).sum()),
                 pnl=("pnl", "sum"), turnover=("notional", "sum"))
            .assign(win_rate=lambda d: (d.wins / d.positions).round(3),
                    pct_equity=lambda d: (100 * d.pnl / reconcile.EQUITY).round(3))
            .round(2))
    days.to_csv(OUT / "account_days.csv", encoding="utf-8-sig")

    # Entry hour and session, again from every position rather than the
    # write-ups. Counts are small per bucket -- 18 hours tested against one
    # book means a single hour needs p < 0.003 to mean anything family-wide --
    # so the n column is there to be read before the win rate is.
    hr = positions.assign(hour=positions["open"].dt.hour,
                          session=positions["open"].map(reconcile.session_of),
                          win=positions.pnl > 0)
    hours = (hr.groupby(["session", "hour"])
             .agg(positions=("pnl", "size"), wins=("win", "sum"),
                  pnl=("pnl", "sum"), per_trade=("pnl", "mean"),
                  median_notional=("notional", "median"),
                  median_mins=("mins", "median"))
             .assign(win_rate=lambda d: (d.wins / d.positions).round(3))
             .round(2).reset_index().sort_values(["session", "hour"]))
    hours.to_csv(OUT / "account_hours.csv", index=False, encoding="utf-8-sig")

    # The index written on each slide is hand-typed and drifts: a skipped or
    # repeated number offsets every trade after it for the rest of the day.
    # Deck order is unambiguous, so it wins, and the disagreements are listed
    # for correcting the deck itself.
    mis = journal[(journal.result != "NOT TAKEN") & (journal.idx != journal.seq)]
    if len(mis):
        mis[["day", "slide", "idx", "seq", "sym", "result", "note"]].rename(
            columns={"idx": "written", "seq": "should_be"}
        ).to_csv(OUT / "journal_renumber.csv", index=False, encoding="utf-8-sig")
        by_day = mis.groupby(mis.day.astype(str)).size()
        print()
        print(f"   {len(mis)} slides numbered out of step with deck order "
              f"-> journal_renumber.csv")
        print("     " + ", ".join(f"{d} ({n})" for d, n in by_day.items()))

    j = led[led.Source.str.startswith("journal")]
    src = led.Source.value_counts()
    print()
    print(f"-- ledger: {len(led)} rows -> {OUT / 'ledger.csv'} --")
    for k, v in src.items():
        print(f"   {k:<30} {v:>4}")

    print()
    print(f"   {'day':<12} {'logged':>7} {'resolved':>9} {'positions':>10} "
          f"{'account $':>10}")
    for d, x in led.groupby("date"):
        lg = x[x.Source.str.startswith("journal")]
        res = (x.Source == "journal+account").sum()
        acct = days.pnl.get(d)
        print(f"   {d:<12} {len(lg):>7} {res:>9} "
              f"{int(days.positions.get(d, 0)):>10} "
              f"{('%+10.2f' % acct) if acct is not None else '  no stmt':>10}")
    print(f"   {'TOTAL':<12} {len(j):>7} "
          f"{(led.Source == 'journal+account').sum():>9} "
          f"{len(positions):>10} {positions.pnl.sum():>+10.2f}")

    print()
    print(f"   account: {len(positions)} positions over {len(days)} days, "
          f"{100 * (positions.pnl > 0).mean():.1f}% winners, "
          f"{positions.pnl.sum():+,.2f} on {positions.notional.sum():,.0f} turnover")
    print(f"   journal: {len(j)} write-ups, covering "
          f"{100 * len(j) / len(positions):.0f}% of positions -- a selected "
          f"sample, not the system")

    print()
    print("   exit mechanism (from the notes, all write-ups)")
    for k, v in j.ExitMech.value_counts().items():
        print(f"     {k:<14} {v:>4}")
    prof = j.ExitMech.isin(["target", "trailed stop", "manual early"]).sum()
    tgt = (j.ExitMech == "target").sum()
    print(f"     -> of {prof} profitable exits, {tgt} reached the planned "
          f"target ({100 * tgt / prof:.0f}%)")

    done = int(screens.day.astype(str).str.startswith("2026-09").sum())
    total = int((journal.day.astype(str).str.startswith("2026-09")
                 & (journal.result != "NOT TAKEN")).sum())
    for lang, name in (("en", "report.html"), ("zh", "report_zh.html")):
        page = report.build(positions, led, reconcile.EQUITY, done, total,
                            lang=lang)
        (OUT / name).write_text(page, encoding="utf-8")
        print(f"   wrote {OUT / name}  ({len(page)//1024}KB)")

    # R is a ratio, so it cannot be summed or averaged across trades whose
    # dollar risk differs 23-fold. Risk-weighted is the honest aggregate:
    # sum(R x risk) / sum(risk) reduces to sum(P&L) / sum(risk), i.e. what one
    # dollar of risk returned.
    g = led[(led.Source == "journal+account") & led.Stop.notna()].copy()
    for c in ("Stop", "Notional", "Qty", "PnL$", "RealizedR", "INIT R/R"):
        g[c] = pd.to_numeric(g[c], errors="coerce")
    g["entry"] = g.Notional / g.Qty
    g["risk"] = g.Qty * (g.entry - g.Stop)
    g = g[(g.risk > 0) & g.RealizedR.notna()]
    plan = pd.to_numeric(j["INIT R/R"], errors="coerce")
    print()
    print(f"   planned R/R   n={plan.notna().sum():>3}  median {plan.median():.2f}  "
          f"below 1.0: {(plan < 1).sum()}")
    if len(g):
        wR = (g.RealizedR * g.risk).sum() / g.risk.sum()
        w = g[g.RealizedR > 0]
        l = g[g.RealizedR <= 0]
        print(f"   realised R    n={len(g):>3}  median {g.RealizedR.median():+.2f}R  "
              f"risk-weighted {wR:+.2f}R  (${g['PnL$'].sum():,.2f} on "
              f"${g.risk.sum():,.2f} risked)")
        print(f"     winners n={len(w):>2}  weighted "
              f"{(w.RealizedR * w.risk).sum() / w.risk.sum():+.2f}R   "
              f"losers n={len(l):>2}  weighted "
              f"{(l.RealizedR * l.risk).sum() / l.risk.sum():+.2f}R")
        print(f"   risk per trade  ${g.risk.min():.2f} to ${g.risk.max():.2f}, "
              f"median ${g.risk.median():.2f} "
              f"({100 * g.risk.median() / reconcile.EQUITY:.3f}% of equity)")

    # Last step: the journal's level labels are claims about price, and the
    # statement knows the price. Test them.
    print()
    print("-- entries against the profile as it stood at the fill --")
    try:
        from feeds.fetcher import fetch_klines
        days = sorted(positions.day.astype(str))
        kl = fetch_klines("US.SOXL", "1m", days[0], days[-1])
        lv = levels.check(positions, led, kl)
        if not lv.empty:
            lv.to_csv(OUT / "entry_levels.csv", index=False, encoding="utf-8-sig")
            print(levels.summarise(lv))
            print(f"   wrote {OUT / 'entry_levels.csv'}")
    except Exception as exc:
        print(f"   skipped: {exc}")

    print()
    print(f"   wrote {OUT / 'account_days.csv'} and account_hours.csv")


if __name__ == "__main__":
    main()

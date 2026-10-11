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

from strategy.journal import confirmations, reconcile, slides   # noqa: E402

HERE = pathlib.Path(__file__).parent
DATA = HERE / "data"
OUT = HERE / "output"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--crops", action="store_true",
                    help="re-extract and crop the slide screenshots")
    args = ap.parse_args()

    pptx = next(DATA.glob("*.pptx"), None)
    if pptx is None:
        sys.exit(f"no .pptx in {DATA}")
    OUT.mkdir(exist_ok=True)

    journal = slides.trades(pptx)
    fills = confirmations.load_fills(DATA / "statements")
    positions, unmatched = confirmations.positions(fills)
    screens = pd.read_csv(DATA / "screens.csv")
    hand = pd.read_csv(DATA / "journal.csv")

    print(f"{len(journal)} journal trades over "
          f"{journal.day.nunique()} days  ({pptx.name})")
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

    j = led[led.Scratch != "Y"]
    print(f"\n-- ledger: {len(led)} rows -> {OUT / 'ledger.csv'} --")
    print(f"   journal trades {len(j)}   scratches recovered "
          f"{(led.Scratch == 'Y').sum()}")
    print(f"   auto-matched {(led.Match == 'auto').sum()}   "
          f"needs review {(led.Match == 'NEEDS REVIEW').sum()}")

    # "ledger" is only what a confirmed match could carry; "account" is every
    # position that day. They differ exactly where a day needs review, which is
    # the point of showing both.
    acct = positions.assign(day=positions.day.astype(str)).groupby("day").pnl.sum()
    print(f"\n   {'day':<12} {'rows':>5} {'auto':>5} {'scr':>4} "
          f"{'ledger $':>10} {'account $':>10}")
    for d, x in led.groupby("date"):
        pnl = pd.to_numeric(x["PnL$"], errors="coerce").sum()
        a = acct.get(d)
        print(f"   {d:<12} {len(x):>5} {(x.Match == 'auto').sum():>5} "
              f"{(x.Scratch == 'Y').sum():>4} {pnl:>+10.2f} "
              f"{('%+10.2f' % a) if a is not None else '  no stmt':>10}")
    print(f"   {'TOTAL':<12} {len(led):>5} {(led.Match == 'auto').sum():>5} "
          f"{(led.Scratch == 'Y').sum():>4} "
          f"{pd.to_numeric(led['PnL$'], errors='coerce').sum():>+10.2f} "
          f"{positions.pnl.sum():>+10.2f}")

    real = pd.to_numeric(j.RealizedR, errors="coerce").dropna()
    plan = pd.to_numeric(j["INIT R/R"], errors="coerce")
    if len(real):
        w = real[real > 0]
        pw = plan.reindex(w.index)
        print(f"\n   planned R/R   n={plan.notna().sum():>2}  "
              f"median {plan.median():.2f}  below 1.0: {(plan < 1).sum()}")
        print(f"   realised R    n={len(real):>2}  total {real.sum():+.2f}R  "
              f"mean {real.mean():+.3f}R")
        print(f"   plan adherence  winners {(w / pw).median():.2f} of plan   "
              f"losers median {real[real <= 0].median():+.2f}R")
        risk = ((pd.to_numeric(j.Notional, errors="coerce")
                 / pd.to_numeric(j.Qty, errors="coerce"))
                - pd.to_numeric(j.Stop, errors="coerce")) \
            * pd.to_numeric(j.Qty, errors="coerce")
        print(f"   risk per trade  median ${risk.median():.2f} = "
              f"{100 * risk.median() / reconcile.EQUITY:.3f}% of "
              f"${reconcile.EQUITY:,.0f}")


if __name__ == "__main__":
    main()

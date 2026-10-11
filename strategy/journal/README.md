# strategy/journal — trade ledger

Reconciles three records of the same trades into one table:

| Source | Grain | What only it has |
|--------|-------|------------------|
| `data/Trading Analysis Oct 2026.pptx` | trade idea, numbered 1..n per day | strategy, profile level, the written review |
| `data/statements/oct*.pdf` | fill | **what actually happened to the money** |
| screenshots inside the deck | trade idea | **the R/R design**: stop, target, ratio, timeframe |
| `data/journal.csv` | trade idea | the hand transcription (10/07–10/09 only) |

```bash
uv run strategy/journal/build.py            # -> output/ledger.csv
uv run strategy/journal/build.py --crops    # also re-cut the screenshots
```

## Why it is not a straight join

**The statement date is a trading day.** Page 6 of every confirmation puts
20:00–04:00 ET in the overnight session, and the Oct 08 statement opens at
10/07 20:06. That is the same boundary the journal uses, so no conversion is
needed — but a position opened before 20:00 and closed after it is split across
two files, so `confirmations.positions` chains inventory across the whole run
rather than processing each statement alone. Two such positions exist in the
current data and isolating the files would lose both.

**Short exposure is long SOXS, never short SOXL.** Naive FIFO invents short
positions out of the opening sells that close carried-in inventory. The share
counts settle it: starting SOXL at the carried-in quantity keeps every
intermediate position non-negative and ends flat.

**Trades with a negligible result are not logged.** `SCRATCH_USD = 2.0`
separates them. The gap in the data is clean — scratches come in at 0.03–0.04%
of notional and the next smallest real trade is 0.13% — so the threshold is not
a fitted parameter. The scratches are carried into the ledger with
`Scratch = Y` rather than dropped, because the behaviour they record is
sometimes the loudest thing about a day (see below).

**The trade index is not chronological.** The day session is grouped by
instrument: slide 6 of 10/08 is the 11:52 SOXL trade while slides 7 and 8 are
the SOXS trades that came before it. A match is therefore asserted only when
count, win/loss sequence *and* instrument sequence all agree; anything else is
written out as `NEEDS REVIEW` rather than guessed. The capture timestamp kept
in each crop is what resolves the order by hand.

## Known gaps in the current data

- **10/07** the journal's trade 4 is an add-on to trade 3 (`三次触碰入场加仓`).
  The account shows the added lot losing $5.38 while the journal scores the
  combined idea a win, which it was: `+21.20 - 5.38 = +15.82`. The journal's
  grain is the idea; the account's is the lot.
- **10/08** one real position (10:56, −13.40) is missing from the journal. It
  is not a scratch — it is the fourth largest loss of that day — so the
  journal's −85.71 understates the account's −98.51 by 13%.
- **10/09** has no confirmation, so its six trades carry no realised P&L.
- Slide 10 is titled SOXS but its chart is SOXL and there is no SOXS position
  that afternoon; `reconcile.SYMBOL_FIXES` corrects it, which is what lets
  10/01 match.

## Reading the output

`RealizedR` is the account's own entry and exit measured against the stop the
screenshot shows. `-1.0` means the stop did its job; past it, it did not.
`PnL%equity` uses `EQUITY = 30000`, read off the R/R tool overlay, which prints
account equity under each outcome. `Entry` is `-` where the note names no
profile level (`NA` would be read back as a missing value); `Setup` carries the
structural pattern for those, since 13 of 42 trades name a shape rather than a
level.

What the current 24 fully-reconciled trades say:

```
planned R/R      median 1.20          3 of 24 planned below 1.0
realised R       +8.53R total, +0.355R per trade
plan adherence   winners take 0.99 of plan; losers sit at -1.01R
risk per trade   median $13.73 = 0.046% of equity
```

Execution is close to exact. The binding constraints are the **size of the
plan** (median R/R 1.20 against a 0.81 break-even at the observed win rate) and
the **size of the risk unit** (0.046% where 0.5–2% is normal), which together
are why a real positive edge produces an almost flat equity curve.

## What is tracked

Code and these READMEs. **Nothing under `data/` or `output/`.** The
confirmations carry the account number and home address and this repository has
a remote; the rest is personal trading history that has no reason to be in
version control either.

A fresh clone has no `data/` and `build.py` will say so. `data/screens.csv` is
the one file here that no script can rebuild — it is hand-transcribed from the
deck's screenshots — so keep it with the deck rather than relying on git.

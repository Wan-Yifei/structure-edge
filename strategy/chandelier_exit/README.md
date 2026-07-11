# ATR Chandelier Exit — research tool

Tests whether an ATR-based chandelier (trailing) stop beats the smc_v2
engine's current static SL/TP, using the engine's own real entries so the
comparison is apples-to-apples (only the exit method changes).

**Standalone**: does not modify `backtest/engine.py` or `ALGO_VERSION` — same
pattern as `strategy/option/`. No `smc_v` version bump. If a chandelier combo
proves out and you want it live in the real engine, that's a separate,
future integration decision.

## What a chandelier exit is

```
Long:  stop = HighestHigh(N bars) − ATR(N) × multiplier   (ratchets up only)
Short: stop = LowestLow(N bars)  + ATR(N) × multiplier    (ratchets down only)
```

The stop trails price, tightening risk as a trade moves in your favor, and
never loosens. Unlike the engine's fixed SL/TP, there's no take-profit —
the trailing stop itself is the only exit.

## Quick start

```powershell
uv run strategy/chandelier_exit/run.py `
    --code US.SOXL `
    --start 2025-05-22 --end 2026-05-22 `
    --entry-csv backtest/results/<run>/US_SOXL/backtest_results.csv --entry-rank 1 `
    --atr-periods 10 14 20 22 `
    --multipliers 2.0 2.5 3.0 3.5 4.0
```

`--entry-csv/--entry-rank` picks one row (default sorted by `total_r`) out of
an existing grid-search results CSV and reconstructs its `BacktestParams` via
`BacktestParams.from_dict()` (unknown/metric columns are stripped
automatically — the same CSV a full grid backtest already produces works
directly, no new config format). `tf_pair` is **not** a separate flag: it's
read from the loaded params' `trend_tf`/`entry_tf` fields.

Alternative: `--entry-params-json <path.json>` for an ad hoc single combo
without a prior grid run.

### Quick calculator (no backtest, no position needed)

To just see where a chandelier stop would sit *right now* for a given
code/timeframe/ATR-period/multiplier (e.g. after picking a winning combo from
a grid run above, and wanting to know today's actual stop level):

```powershell
uv run strategy/chandelier_exit/calculator.py --code US.SOXL --tf 30m --period 20 --multiplier 2.0
```

This is a single latest-bar snapshot, not a ratcheted stop tracked since a
real entry date — see the script's own docstring for that distinction.

Output lands in `strategy/chandelier_exit/output/` (gitignored):
`grid_results.csv` (every combo, all metrics) and `REPORT.md` (baseline vs.
best combo, top-N table, caveats).

## How the comparison works

1. `entries.collect_entries()` fetches klines and calls the real
   `backtest.engine.run_backtest()` unmodified — reuses its entry/SL/TP logic
   as-is. This produces the baseline (static SL/TP) result *and* the batch of
   entries (price/time/direction/SL/TP) to re-exit under the chandelier rule.
2. For every `(atr_period, multiplier)` combo, `chandelier.simulate_chandelier_exit()`
   walks forward from each entry and finds where the trailing stop would have
   fired, using the **same** risk unit (`|entry_price − original_SL|`) as the
   baseline trade, so R-multiples are directly comparable.
3. `grid_search.run_grid_search()` aggregates each combo's trades into a
   synthetic `backtest.engine.BacktestResult` — reuses its existing
   win_rate/total_r/profit_factor/max_drawdown_r/sharpe/sortino formulas
   rather than re-deriving them.
4. Combos are ranked by `--rank-by total_r|profit_factor` (default `total_r`).
   A `noise_stopout_rate` diagnostic is also reported per combo — the
   fraction of trades that exited below the original TP but price later
   reached that TP anyway — but it is **informational only**, not part of
   ranking. A high rate on the winning combo can just mean a looser stop, not
   a better one; cross-check `max_drawdown_r`.

## Known simplifications

- Exit fills exactly at the stop price (no gap-through modeling) — matches
  `backtest/engine.py`'s own fixed-SL convention, so the two exit methods
  stay comparable, but is optimistic versus real fills.
- No-look-ahead: a bar's own new high/low can only affect the *next* bar's
  stop, never its own (the entry bar is the one exception — its own bar is
  used, matching how the engine's entry itself is decided at that bar's close).
- Single stock / single date range / single entry-params combo per run —
  small sample sizes make close differences between top combos easy to
  overfit to.

## File layout

```
strategy/chandelier_exit/
├── atr.py          — Wilder-smoothed ATR
├── chandelier.py    — trailing-stop simulator
├── entries.py         — wraps run_backtest() to produce real entries
├── grid_search.py       — grid loop + baseline comparison + noise diagnostic
├── report.py              — CSV + REPORT.md writer
├── run.py                   — CLI entrypoint
├── calculator.py              — point-in-time stop-level lookup (no backtest)
├── README.md / README_zh.md
├── .gitignore                — excludes output/ and README_zh.md
└── output/                    — grid_results.csv, REPORT.md (not committed)
```

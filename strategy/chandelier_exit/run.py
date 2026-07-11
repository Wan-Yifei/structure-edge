#!/usr/bin/env python3
"""ATR chandelier-exit research tool - grid-searches (ATR period, multiplier)
against real smc_v2 entries and compares against the engine's static SL/TP.

Standalone: does not modify backtest/engine.py or ALGO_VERSION. See README.md.

Usage:
    uv run strategy/chandelier_exit/run.py --code US.SOXL \\
        --start 2025-05-22 --end 2026-05-22 \\
        --entry-csv backtest/results/<run>/US_SOXL/backtest_results.csv --entry-rank 1 \\
        --atr-periods 10 14 20 22 --multipliers 2.0 2.5 3.0 3.5 4.0

    uv run strategy/chandelier_exit/run.py --code US.SOXL \\
        --start 2025-05-22 --end 2026-05-22 \\
        --entry-params-json my_params.json

Entry params (a single BacktestParams combo) are supplied one of two ways:
    --entry-csv <backtest_results.csv> --entry-rank N [--entry-rank-by total_r]
        Load an existing grid-search results CSV, sort by --entry-rank-by
        (default total_r), take row N (1-indexed), and reconstruct
        BacktestParams via BacktestParams.from_dict() (unknown/metric columns
        are stripped automatically).
    --entry-params-json <path.json>
        A plain JSON object of BacktestParams fields for an ad hoc combo.

tf_pair is NOT a separate flag -- it's read from the loaded params'
trend_tf/entry_tf fields (that's how the original grid search encoded it).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

import pandas as pd

from backtest.engine import BacktestParams
from strategy.chandelier_exit.entries import collect_entries
from strategy.chandelier_exit.grid_search import run_grid_search
from strategy.chandelier_exit.report import write_outputs

_DEFAULT_OUT = pathlib.Path(__file__).parent / "output"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ATR chandelier-exit grid search vs. static SL/TP baseline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--code", required=True, help="e.g. US.SOXL")
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")

    p.add_argument("--entry-csv", help="Path to an existing backtest_results.csv")
    p.add_argument("--entry-rank", type=int, default=1, help="1-indexed row to use (default 1)")
    p.add_argument("--entry-rank-by", default="total_r", help="Column to sort --entry-csv by (default total_r)")
    p.add_argument("--entry-params-json", help="Path to a JSON dict of BacktestParams fields")

    p.add_argument("--atr-periods", type=int, nargs="+", default=[10, 14, 20, 22])
    p.add_argument("--multipliers", type=float, nargs="+", default=[2.0, 2.5, 3.0, 3.5, 4.0])
    p.add_argument("--rank-by", choices=["total_r", "profit_factor"], default="total_r")
    p.add_argument("--max-bars-in-trade", type=int, default=200)
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--out", default=str(_DEFAULT_OUT))
    return p.parse_args()


def _load_entry_params(args: argparse.Namespace) -> tuple[BacktestParams, str]:
    if args.entry_csv:
        df = pd.read_csv(args.entry_csv)
        df = df.sort_values(args.entry_rank_by, ascending=False).reset_index(drop=True)
        if args.entry_rank < 1 or args.entry_rank > len(df):
            raise SystemExit(f"--entry-rank {args.entry_rank} out of range (1..{len(df)})")
        row = df.iloc[args.entry_rank - 1].to_dict()
        params = BacktestParams.from_dict(row)
        source = f"{args.entry_csv}  rank={args.entry_rank}  (sorted by {args.entry_rank_by})"
        return params, source
    if args.entry_params_json:
        with open(args.entry_params_json, encoding="utf-8") as f:
            d = json.load(f)
        return BacktestParams.from_dict(d), args.entry_params_json
    raise SystemExit("Must supply either --entry-csv/--entry-rank or --entry-params-json")


def main() -> None:
    args = _parse_args()
    params, entry_params_source = _load_entry_params(args)
    tf_pair = (params.trend_tf, params.entry_tf)

    print(f"Entry params: {params.label()}")
    print(f"tf_pair (from params): {tf_pair}")
    print(f"Fetching klines + running smc_v2 engine for {args.code} {args.start}..{args.end} ...")

    entries, baseline, ltf_df = collect_entries(
        args.code, tf_pair, args.start, args.end, params,
        max_bars_in_trade=args.max_bars_in_trade,
    )
    print(f"  -> {len(entries)} entries from the engine "
          f"(baseline total_r={baseline.total_r:.2f}, PF={baseline.profit_factor:.2f})")

    if not entries:
        raise SystemExit("Engine produced zero trades for this code/range/params -- nothing to compare.")

    print(f"Grid: atr_periods={args.atr_periods}  multipliers={args.multipliers}")
    grid_df = run_grid_search(
        entries, ltf_df, params, args.atr_periods, args.multipliers,
        max_bars_in_trade=args.max_bars_in_trade,
    )

    out_dir = pathlib.Path(args.out)
    write_outputs(
        grid_df, baseline,
        run_meta={
            "code": args.code, "tf_pair": tf_pair,
            "start": args.start, "end": args.end,
            "entry_params_source": entry_params_source,
            "n_entries": len(entries),
            "atr_periods": args.atr_periods, "multipliers": args.multipliers,
        },
        out_dir=out_dir, rank_by=args.rank_by, top_n=args.top_n,
    )
    print(f"Wrote {out_dir / 'grid_results.csv'} and {out_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()

"""SMC backtest runner with grid search.

Usage:
    uv run backtest/run.py --code US.SNDK --start 2025-02-13 --end 2025-12-31

Fetches klines (or loads from cache), runs every (TF pair × parameter) combination,
prints a ranked results table, saves results to backtest/results/backtest_results.csv,
and saves a four-panel visualisation to backtest/results/backtest_viz.png.

Flags:
  --fast      Smoke test — 2 TF pairs, 1 combo each
  --force     Re-fetch klines from moomoo API even if cached
  --no-viz    Skip the matplotlib chart
  --show-chart  Open chart in an interactive window (blocks until closed)
  --top N     Number of top runs to print / show in equity panel

Timeframes
----------
Native (fetched from moomoo):  1m  3m  5m  15m  30m  60m  1d
Synthetic (resampled from 60m): 2h  3h  4h

Note: 1m as entry_tf is ~15× slower per combo than 15m (~85 k bars vs ~5.7 k).
      It is excluded from the main grid; add it to TF_PAIRS manually for targeted runs.
"""

from __future__ import annotations

import argparse
import itertools
import os
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

from tqdm import tqdm

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from backtest.fetcher  import fetch_klines
from backtest.engine import BacktestParams, BacktestResult, run_backtest
from backtest.viz      import plot_backtest_results, plot_from_csv

_RESULTS_CSV = pathlib.Path(__file__).parent / "results" / "backtest_results.csv"

# ── Timeframe pairs (trend_tf, entry_tf) ──────────────────────────────────────
# Only pairs where trend_tf is strictly coarser than entry_tf make sense.
# 1m entry excluded from main grid — ~15× slower per combo than 15m.
# To test 1m entry add e.g. ("5m", "1m") or ("15m", "1m") here.

TF_PAIRS: list[tuple[str, str]] = [
    # ── 5m entry ──────────────────────────────────────────────────────────
    ("60m", "5m"),
    ("4h",  "5m"),
    # ── 15m entry (primary SMC intraday setup) ────────────────────────────
    ("60m", "15m"),
    ("2h",  "15m"),
    ("4h",  "15m"),
    ("1d",  "15m"),
    # ── 30m entry ─────────────────────────────────────────────────────────
    ("2h",  "30m"),
    ("4h",  "30m"),
    ("1d",  "30m"),
    # ── 60m entry ─────────────────────────────────────────────────────────
    ("4h",  "60m"),
    ("1d",  "60m"),
    # ── Longer entry TFs (swing / position) ───────────────────────────────
    ("1d",  "2h"),
    ("1d",  "4h"),
]

TF_PAIRS_FAST: list[tuple[str, str]] = [
    ("60m", "15m"),
    ("1d",  "60m"),
]

# ── Parameter grid ────────────────────────────────────────────────────────────
# trend_tf / entry_tf come from TF_PAIRS above, not from this grid.

PARAM_GRID: dict[str, list] = {
    "swing_lookback":        [2, 3],
    "bos_count":             [1, 2],
    "fvg_min_width_pct":     [0.001, 0.003, 0.005],
    "fvg_entry_depth_pct":   [0.10, 0.20, 0.50],
    "displacement_required": [False],
    "sl_buffer_pct":         [0.001, 0.003],
    "max_sl_pct":            [0.010],
    "min_rr":                [1.5, 2.0, 3.0],
}

PARAM_GRID_FAST: dict[str, list] = {
    "swing_lookback":        [2],
    "bos_count":             [1],
    "fvg_min_width_pct":     [0.002],
    "fvg_entry_depth_pct":   [0.20],
    "displacement_required": [False],
    "sl_buffer_pct":         [0.001],
    "max_sl_pct":            [0.010],
    "min_rr":                [1.5],
}


def build_param_list(
    pairs: list[tuple[str, str]],
    grid: dict,
) -> list[BacktestParams]:
    keys   = list(grid.keys())
    values = list(grid.values())
    result: list[BacktestParams] = []
    for trend_tf, entry_tf in pairs:
        for combo in itertools.product(*values):
            result.append(BacktestParams(
                trend_tf=trend_tf,
                entry_tf=entry_tf,
                **dict(zip(keys, combo)),
            ))
    return result


def _worker(args: tuple) -> tuple[int, BacktestResult]:
    """Top-level so ProcessPoolExecutor can pickle it on Windows (spawn mode)."""
    idx, params, htf, ltf = args
    return idx, run_backtest(htf, ltf, params)


def _fmt_row(d: dict) -> str:
    return (
        f"  trades={d['n_trades']:3d}  wr={d['win_rate']:.1%}  "
        f"R={d['total_r']:+.1f}  avgR={d['avg_r']:+.3f}  "
        f"PF={d['profit_factor']:.2f}  DD={d['max_drawdown_r']:.2f}  "
        f"maxL={d['max_loss_r']:.2f}"
    )


def run_grid(
    code: str,
    klines: dict[str, pd.DataFrame],
    params_list: list[BacktestParams],
    workers: int | None = None,
) -> list[BacktestResult]:
    n = len(params_list)

    # Build task list, dropping combos with missing kline data upfront
    tasks: list[tuple[int, BacktestParams, pd.DataFrame, pd.DataFrame]] = []
    for idx, params in enumerate(params_list, 1):
        htf = klines.get(params.trend_tf)
        ltf = klines.get(params.entry_tf)
        if htf is None or ltf is None:
            print(f"[{idx}/{n}] SKIP — missing data for {params.trend_tf}/{params.entry_tf}")
        else:
            tasks.append((idx, params, htf, ltf))

    bt_results: list[BacktestResult] = [None] * (n + 1)   # type: ignore

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_worker, t): t[0] for t in tasks}
        bar = tqdm(
            as_completed(futures),
            total=len(tasks),
            ncols=90,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        )
        for fut in bar:
            orig_idx, bt = fut.result()
            bt_results[orig_idx] = bt
            d = bt.summary_dict()
            bar.set_postfix_str(
                f"{bt.params.trend_tf}/{bt.params.entry_tf}"
                f"  T={d['n_trades']} WR={d['win_rate']:.0%}"
                f"  R={d['total_r']:+.1f} PF={d['profit_factor']:.2f}",
                refresh=False,
            )

    return [r for r in bt_results if r is not None]


def main() -> None:
    ap = argparse.ArgumentParser(description="SMC backtest grid search")
    ap.add_argument("--code",  default="US.SNDK", help="moomoo stock code")
    ap.add_argument("--start", default="2025-02-13", help="YYYY-MM-DD")
    ap.add_argument("--end",   default="2025-12-31", help="YYYY-MM-DD")
    ap.add_argument("--fast",       action="store_true", help="Smoke test — 2 TF pairs, minimal params")
    ap.add_argument("--force",      action="store_true", help="Re-fetch klines from API")
    ap.add_argument("--no-viz",     action="store_true", help="Skip the matplotlib visualisation")
    ap.add_argument("--show-chart", action="store_true", help="Open chart interactively (blocks)")
    ap.add_argument("--top",        type=int, default=20,
                    help="Print top N results ranked by profit factor")
    ap.add_argument("--workers",    type=int, default=None,
                    help="Parallel worker processes (default: CPU count)")
    ap.add_argument("--from-csv",   metavar="PATH",
                    help="Regenerate chart from an existing CSV (skips backtest)")
    args = ap.parse_args()

    if args.from_csv:
        csv_in = pathlib.Path(args.from_csv)
        if not csv_in.exists():
            print(f"ERROR: CSV not found: {csv_in}")
            sys.exit(1)
        plot_from_csv(
            csv_path=csv_in,
            show=args.show_chart,
            top_n=args.top,
        )
        return

    run_tag     = datetime.now().strftime("%Y%m%d_%H%M")
    results_dir = _RESULTS_CSV.parent / run_tag
    csv_path    = results_dir / "backtest_results.csv"
    viz_path    = results_dir / "backtest_viz.png"

    pairs  = TF_PAIRS_FAST if args.fast else TF_PAIRS
    grid   = PARAM_GRID_FAST if args.fast else PARAM_GRID
    params = build_param_list(pairs, grid)

    # Collect unique TFs needed across all pairs
    needed_tfs = {tf for trend_tf, entry_tf in pairs for tf in (trend_tf, entry_tf)}
    tf_order   = ["1m", "5m", "15m", "30m", "60m", "2h", "3h", "4h", "1d"]
    sorted_tfs = sorted(needed_tfs, key=lambda t: tf_order.index(t) if t in tf_order else 99)

    print(f"TF pairs: {len(pairs)}")
    print(f"Timeframes needed: {sorted_tfs}")
    print(f"Combinations: {len(params)}")

    klines: dict[str, pd.DataFrame] = {}
    for tf in sorted_tfs:
        df = fetch_klines(
            code=args.code, ktype=tf,
            start=args.start, end=args.end,
            force_refresh=args.force,
        )
        bar_range = (
            f"{df['time_key'].iloc[0]} … {df['time_key'].iloc[-1]}"
            if len(df) else "—"
        )
        print(f"  {tf}: {len(df)} bars  ({bar_range})")
        klines[tf] = df

    workers = args.workers or os.cpu_count()
    print(f"\n── Running backtest grid ({workers} workers) ──────────────────────────\n")
    bt_results = run_grid(args.code, klines, params, workers=workers)

    if not bt_results:
        print("No results.")
        return

    df_out = pd.DataFrame([r.summary_dict() for r in bt_results])

    # ── Print top N ───────────────────────────────────────────────────────
    df_ranked = df_out.sort_values(
        ["profit_factor", "total_r"], ascending=[False, False]
    ).head(args.top)

    print(f"\n── Top {args.top} by profit factor ──────────────────────────────────────\n")
    for _, row in df_ranked.iterrows():
        p = BacktestParams(**{k: row[k] for k in BacktestParams.__dataclass_fields__})   # type: ignore
        print(f"{p.label()}")
        print(_fmt_row(row.to_dict()))
        print()

    # ── Save CSV ──────────────────────────────────────────────────────────
    results_dir.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(csv_path, index=False)
    print(f"All {len(df_out)} results saved to {csv_path}")

    # ── Visualisation ─────────────────────────────────────────────────────
    if not args.no_viz:
        ranked_results = sorted(bt_results, key=lambda r: r.profit_factor, reverse=True)
        plot_backtest_results(
            ranked_results,
            df_out,
            top_n=min(args.top, 5),
            save_path=viz_path,
            show=args.show_chart,
        )


if __name__ == "__main__":
    main()

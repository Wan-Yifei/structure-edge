"""Reconstruct per-stock results CSVs from backtest.duckdb.

Use this when a run's CSV outputs were lost (e.g. Docker results-dir detection bug)
but the DB records were successfully merged.  The output CSV format is identical to
what backtest/run.py produces, so report.py and audit.py work unchanged.

Missing field: `final_value` is not stored in the DB; it is emitted as NaN.

Usage
─────
  # Export all SOXL runs matching a specific algo version and date range
  uv run backtest/export_csv_from_db.py \\
      --symbol US.SOXL \\
      --start 2025-05-22 --end 2026-05-22 \\
      --algo-version smc_unknown \\
      --out-dir backtest/results/recovered_soxl_grid_v2

  # Auto-read symbol/dates from a config JSON
  uv run backtest/export_csv_from_db.py \\
      --config config/backtest/soxl_grid_v2.json \\
      --algo-version smc_unknown \\
      --out-dir backtest/results/recovered_soxl_grid_v2

  # Then run post-processing as normal
  uv run backtest/report.py --from-csv backtest/results/recovered_soxl_grid_v2/US_SOXL/results_US_SOXL.csv
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import duckdb
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

_DEFAULT_DB = pathlib.Path(__file__).parent.parent / "db" / "backtest.duckdb"

# Stat columns stored in run_stats
_STAT_COLS = [
    "n_trades", "win_rate", "total_r", "avg_r", "profit_factor",
    "max_drawdown_r", "max_loss_r", "sharpe", "sortino",
]

# Column order for the output CSV (matches summary_dict + 'code' prefix)
_COL_ORDER_PREFIX = ["code"] + _STAT_COLS + [
    "bull_trades", "bear_trades", "bull_win_rate", "bear_win_rate",
    "bull_total_r", "bear_total_r", "final_value",
]


def _load_runs(
    con: duckdb.DuckDBPyConnection,
    symbols: list[str],
    start: str,
    end: str,
    algo_version: str,
) -> pd.DataFrame:
    placeholders = ", ".join("?" * len(symbols))
    df = con.execute(
        f"""
        SELECT
            r.run_id,
            r.symbol,
            r.config_json,
            r.trend_tf,
            r.entry_tf,
            s.n_trades,
            s.win_rate,
            s.total_r,
            s.avg_r,
            s.profit_factor,
            s.max_drawdown_r,
            s.max_loss_r,
            s.sharpe,
            s.sortino
        FROM runs r
        JOIN run_stats s ON r.run_id = s.run_id
        WHERE r.symbol IN ({placeholders})
          AND r.start_date = ?
          AND r.end_date   = ?
          AND r.algo_version = ?
        """,
        symbols + [start, end, algo_version],
    ).fetchdf()
    return df


def _load_bull_bear(
    con: duckdb.DuckDBPyConnection,
    run_ids: list[str],
) -> pd.DataFrame:
    if not run_ids:
        return pd.DataFrame(columns=[
            "run_id", "bull_trades", "bear_trades",
            "bull_wins", "bear_wins", "bull_total_r", "bear_total_r",
        ])
    placeholders = ", ".join("?" * len(run_ids))
    return con.execute(
        f"""
        SELECT
            run_id,
            COUNT(CASE WHEN direction = 'bull' THEN 1 END)               AS bull_trades,
            COUNT(CASE WHEN direction = 'bear' THEN 1 END)               AS bear_trades,
            SUM(CASE WHEN direction = 'bull' AND result = 'win'
                     THEN 1 ELSE 0 END)                                  AS bull_wins,
            SUM(CASE WHEN direction = 'bear' AND result = 'win'
                     THEN 1 ELSE 0 END)                                  AS bear_wins,
            ROUND(SUM(CASE WHEN direction = 'bull'
                           THEN r_multiple ELSE 0 END), 2)               AS bull_total_r,
            ROUND(SUM(CASE WHEN direction = 'bear'
                           THEN r_multiple ELSE 0 END), 2)               AS bear_total_r
        FROM trades
        WHERE run_id IN ({placeholders})
        GROUP BY run_id
        """,
        run_ids,
    ).fetchdf()


def export_symbol(
    df_runs: pd.DataFrame,
    symbol: str,
    out_dir: pathlib.Path,
    verbose: bool = True,
) -> pathlib.Path:
    code_slug = symbol.replace(".", "_")
    stock_dir = out_dir / code_slug
    stock_dir.mkdir(parents=True, exist_ok=True)
    csv_path = stock_dir / f"results_{code_slug}.csv"

    df = df_runs[df_runs["symbol"] == symbol].copy()

    # Expand config_json into individual param columns
    params_df = pd.json_normalize(df["config_json"].apply(
        lambda v: json.loads(v) if isinstance(v, str) else v
    ))
    params_df.index = df.index

    # Compute bull/bear win rates (avoid div-by-zero)
    df["bull_win_rate"] = (
        df["bull_wins"] / df["bull_trades"].replace(0, float("nan"))
    ).round(3).fillna(0.0)
    df["bear_win_rate"] = (
        df["bear_wins"] / df["bear_trades"].replace(0, float("nan"))
    ).round(3).fillna(0.0)
    df["final_value"] = float("nan")
    df["code"] = symbol

    combined = pd.concat(
        [df[["code"] + _STAT_COLS + [
            "bull_trades", "bear_trades", "bull_win_rate", "bear_win_rate",
            "bull_total_r", "bear_total_r", "final_value",
        ]].reset_index(drop=True),
         params_df.reset_index(drop=True)],
        axis=1,
    )

    combined.to_csv(csv_path, index=False)

    if verbose:
        print(f"  [{symbol}] {len(combined):,} rows → {csv_path}")

    return csv_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Reconstruct per-stock results CSVs from backtest.duckdb"
    )
    filter_group = ap.add_mutually_exclusive_group()
    filter_group.add_argument(
        "--config", metavar="PATH",
        help="Read symbol(s)/dates from a backtest config JSON",
    )
    filter_group.add_argument(
        "--symbol", nargs="+", metavar="CODE",
        help="One or more moomoo codes (e.g. US.SOXL US.AMD)",
    )
    ap.add_argument("--start",         metavar="YYYY-MM-DD", help="Backtest start date")
    ap.add_argument("--end",           metavar="YYYY-MM-DD", help="Backtest end date")
    ap.add_argument("--algo-version",  metavar="TAG", required=True,
                    help="algo_version to filter (e.g. smc_unknown, smc_v2.4)")
    ap.add_argument("--out-dir",       metavar="PATH", required=True,
                    help="Output directory (per-stock subdirs created automatically)")
    ap.add_argument("--db",            default=str(_DEFAULT_DB), metavar="PATH",
                    help=f"Master DB path (default: {_DEFAULT_DB})")
    ap.add_argument("--dry-run",       action="store_true",
                    help="Print row counts without writing files")
    args = ap.parse_args()

    # Resolve symbols / dates from config or CLI
    if args.config:
        cfg = json.loads(pathlib.Path(args.config).read_text(encoding="utf-8"))
        symbols = cfg.get("codes", [])
        start   = args.start or cfg.get("start") or cfg.get("start_date")
        end     = args.end   or cfg.get("end")   or cfg.get("end_date")
    else:
        symbols = args.symbol or []
        start   = args.start
        end     = args.end

    if not symbols:
        ap.error("Provide --symbol or --config with a 'codes' list.")
    if not start or not end:
        ap.error("Provide --start/--end (or use --config with start/end fields).")

    db_path = pathlib.Path(args.db)
    if not db_path.exists():
        print(f"ERROR: DB not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = pathlib.Path(args.out_dir)

    print(f"DB            : {db_path}")
    print(f"Symbols       : {symbols}")
    print(f"Date range    : {start} → {end}")
    print(f"Algo version  : {args.algo_version}")
    print(f"Output dir    : {out_dir}")
    print()

    con = duckdb.connect(str(db_path), read_only=True)

    df_runs = _load_runs(con, symbols, start, end, args.algo_version)
    if df_runs.empty:
        print("No matching runs found. Check --symbol, --start, --end, --algo-version.")
        sys.exit(1)

    print(f"Found {len(df_runs):,} runs across {df_runs['symbol'].nunique()} symbol(s).")

    # Load bull/bear trade aggregates
    bb = _load_bull_bear(con, df_runs["run_id"].tolist())
    con.close()

    df_runs = df_runs.merge(bb, on="run_id", how="left")
    for col in ["bull_trades", "bear_trades", "bull_wins", "bear_wins",
                "bull_total_r", "bear_total_r"]:
        df_runs[col] = df_runs[col].fillna(0)

    if args.dry_run:
        print("\nDry run — no files written.")
        for sym in sorted(df_runs["symbol"].unique()):
            n = (df_runs["symbol"] == sym).sum()
            print(f"  {sym}: {n:,} rows")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_paths: list[pathlib.Path] = []
    for sym in sorted(df_runs["symbol"].unique()):
        path = export_symbol(df_runs, sym, out_dir)
        csv_paths.append(path)

    print(f"\nDone. {len(csv_paths)} CSV file(s) written.")
    print("Next steps:")
    for p in csv_paths:
        print(f"  uv run backtest/report.py --from-csv {p}")


if __name__ == "__main__":
    main()

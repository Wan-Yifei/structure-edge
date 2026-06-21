"""Select the best FVG-sweep combo per (code, tf) under a width floor.

Given a results.csv from backtest/fvg_width_sweep.py, filters to combos
meeting a minimum mean_width_pct, then among those picks the one with the
most gaps (n_gaps) — "as many alerts as possible, but each gap at least this
wide". This is a different objective from the sweep's own total_width_pct
ranking (count-dominated, see fvg_width_sweep.py's print_top_n) and from a
pure max-mean_width_pct pick (tends to select very rare, very wide combos).

Usage:
    uv run backtest/fvg_width_select.py backtest/results/<run>/results.csv --min-mean-width-pct 0.0025
    uv run backtest/fvg_width_select.py backtest/results/<run>/results.csv --min-mean-width-pct 0.0025 --symbol US.SOXL
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

_PARAM_COLS = [
    "min_gap_pct", "require_displacement", "atr_mult", "body_ratio_min",
    "lookback", "require_lvn_overlap", "lvn_threshold",
]


def select_best_combos(df: pd.DataFrame, min_mean_width_pct: float) -> pd.DataFrame:
    """For each (code, tf), pick the combo with the most gaps among those
    meeting the width floor (ties broken by higher mean_width_pct). Groups
    with no qualifying combo are silently dropped from the result.
    """
    rows = []
    for (code, tf), group in df.groupby(["code", "tf"]):
        candidates = group[group["mean_width_pct"] >= min_mean_width_pct]
        if candidates.empty:
            continue
        best = candidates.sort_values(
            ["n_gaps", "mean_width_pct"], ascending=[False, False]
        ).iloc[0]
        rows.append(best)
    return pd.DataFrame(rows)


def to_watch_params(row: pd.Series) -> dict:
    """Convert a selected results-row into a config/scanner/fvg_watch_params.json entry."""
    out: dict = {"tf": row["tf"]}
    for col in _PARAM_COLS:
        if col not in row or pd.isna(row[col]):
            continue
        val = row[col]
        if col in ("require_displacement", "require_lvn_overlap"):
            val = bool(val)
        elif col == "lookback":
            val = int(val)
        out[col] = val
    return out


def main(argv=None) -> None:
    """CLI entry point for the FVG-sweep combo selector."""
    ap = argparse.ArgumentParser(
        description="Select the max-n_gaps FVG combo per (code, tf) under a width floor"
    )
    ap.add_argument("csv", help="results.csv from backtest/fvg_width_sweep.py")
    ap.add_argument(
        "--min-mean-width-pct", type=float, required=True,
        help="Minimum mean_width_pct to qualify, in the same units as the CSV column "
             "(e.g. 0.0025 for 0.25%%)",
    )
    ap.add_argument("--symbol", default=None, help="Filter to one code (the CSV's 'code' column)")
    args = ap.parse_args(argv)

    df = pd.read_csv(args.csv)
    if args.symbol:
        df = df[df["code"] == args.symbol]

    selected = select_best_combos(df, args.min_mean_width_pct)
    if selected.empty:
        print(f"No combo meets mean_width_pct >= {args.min_mean_width_pct} for any (code, tf).")
        return

    print(f"-- Best combo per (code, tf), mean_width_pct >= {args.min_mean_width_pct}, max n_gaps --\n")
    for _, row in selected.iterrows():
        watch = to_watch_params(row)
        print(
            f"{row['code']} {row['tf']}: n_gaps={row['n_gaps']:.0f} "
            f"mean_width_pct={row['mean_width_pct']:.4f} total_width_pct={row['total_width_pct']:.4f}"
        )
        print(f"  watch_params: {json.dumps(watch)}")
        print()


if __name__ == "__main__":
    main()

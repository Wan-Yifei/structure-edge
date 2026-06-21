"""Backscan: retrospectively list FVG-watch matches over a historical window.

Read-only research/debug tool — uses the same detection logic and config as
the live scanner's FVG-watch alert (analysis/fvg_watcher.py), but never
writes to SignalsDB. Useful for tuning config/scanner/fvg_watch_params.json
before enabling an alert live, or for debugging why a FVG was/wasn't flagged.

Usage:
    uv run analysis/fvg_backscan.py --symbol US.SOXL --start 2026-05-01
    uv run analysis/fvg_backscan.py --symbol US.SOXL --tf 15m --start 2026-05-01 --csv out.csv
    uv run main.py fvg_backscan --symbol US.SOXL --start 2026-05-01
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from datetime import datetime

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from analysis.fvg_watcher import load_fvg_watch_config, scan_symbol_tf

_RESULT_COLS = ["symbol", "tf", "direction", "formed_time", "zone_bottom", "zone_top", "width_pct", "filled"]


def _resolve_pairs(config: dict, symbol: str | None, tf: str | None) -> list[tuple[str, dict]]:
    """Resolve which (symbol, watch-entry) pairs to scan from CLI filters."""
    symbols = [symbol] if symbol else list(config.keys())
    pairs: list[tuple[str, dict]] = []
    for sym in symbols:
        entries = config.get(sym, [])
        if tf:
            entries = [e for e in entries if e["tf"] == tf]
        for entry in entries:
            pairs.append((sym, entry))
    return pairs


def run_backscan(
    config_path: pathlib.Path | None,
    symbol: str | None,
    tf: str | None,
    start: str,
    end: str,
) -> pd.DataFrame:
    """Scan every matching (symbol, tf) entry over [start, end] and return one row per FVG found."""
    config = load_fvg_watch_config(config_path)
    pairs = _resolve_pairs(config, symbol, tf)

    rows: list[dict] = []
    for sym, entry in pairs:
        rows.extend(scan_symbol_tf(sym, entry["tf"], entry, start, end, force_refresh=False))
    if not rows:
        return pd.DataFrame(columns=_RESULT_COLS)

    df = pd.DataFrame(rows)
    cols = [c for c in _RESULT_COLS if c in df.columns]
    return df[cols].sort_values(["symbol", "tf", "formed_time"]).reset_index(drop=True)


def main(argv=None) -> None:
    """CLI entry point for the FVG-watch backscan tool."""
    ap = argparse.ArgumentParser(description="Backscan FVG-watch matches over a historical window")
    ap.add_argument("--symbol", default=None, help="One symbol (omit to scan every symbol in the config)")
    ap.add_argument("--tf",     default=None, help="One timeframe (omit to scan every TF configured for the symbol)")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end",   default=None, help="YYYY-MM-DD (default: today)")
    ap.add_argument("--config", default=None, metavar="PATH",
                     help="fvg_watch_params.json path (default: config/scanner/fvg_watch_params.json)")
    ap.add_argument("--csv", default=None, metavar="PATH", help="Also save results to this CSV path")
    args = ap.parse_args(argv)

    end = args.end or datetime.now().strftime("%Y-%m-%d")
    config_path = pathlib.Path(args.config) if args.config else None

    df = run_backscan(config_path, args.symbol, args.tf, args.start, end)
    if df.empty:
        print("No matches.")
        return

    print(df.to_string(index=False))
    print(f"\n{len(df)} match(es)")

    if args.csv:
        out = pathlib.Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"Saved -> {out}")


if __name__ == "__main__":
    main()

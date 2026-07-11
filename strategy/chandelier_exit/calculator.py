#!/usr/bin/env python3
"""Point-in-time chandelier trailing-stop calculator.

Answers "if I set a chandelier stop right now with this ATR period and
multiplier, where would it sit?" -- no position/entry-date required. For
tracking an actual open position's stop as it ratchets over time, use
grid_search.py/chandelier.simulate_chandelier_exit instead (this script only
computes the single latest-bar value, it does not track ratcheting history).

Usage:
    uv run strategy/chandelier_exit/calculator.py --code US.SOXL --tf 30m --period 20 --multiplier 2.0
    uv run strategy/chandelier_exit/calculator.py --code US.SOXL --tf 30m --period 20 --multiplier 2.0 --direction long
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from datetime import datetime, timedelta

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from feeds.fetcher import fetch_klines
from strategy.chandelier_exit.chandelier import current_stop


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Point-in-time chandelier trailing-stop calculator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--code", required=True, help="e.g. US.SOXL")
    p.add_argument("--tf", required=True, help="e.g. 30m, 3m, 15m, 1h, 1d")
    p.add_argument("--period", type=int, default=20, help="ATR / HH-LL lookback bars (default 20)")
    p.add_argument("--multiplier", type=float, default=2.0, help="ATR multiplier (default 2.0)")
    p.add_argument("--direction", choices=["long", "short", "both"], default="both")
    p.add_argument("--lookback-days", type=int, default=30,
                    help="Calendar days of klines to fetch (default 30 -- must cover >= period bars)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    end   = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=args.lookback_days)).strftime("%Y-%m-%d")

    df = fetch_klines(code=args.code, ktype=args.tf, start=start, end=end)
    if df.empty:
        raise SystemExit(f"No klines returned for {args.code} {args.tf} {start}..{end}")

    highs  = df["high"].to_numpy(dtype=float)
    lows   = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    last_time = str(df["time_key"].iloc[-1])

    want_long  = args.direction in ("long", "both")
    want_short = args.direction in ("short", "both")
    long_r  = current_stop(highs, lows, closes, args.period, args.multiplier, "bull") if want_long else None
    short_r = current_stop(highs, lows, closes, args.period, args.multiplier, "bear") if want_short else None

    if (want_long and long_r is None) or (want_short and short_r is None):
        raise SystemExit(
            f"Not enough bars for period={args.period}: got {len(df)} bars "
            f"({start}..{end}). Increase --lookback-days or lower --period."
        )

    ref = long_r or short_r
    print(f"{args.code}  {args.tf}  as of {last_time}")
    print(f"  last close: {ref['price']:.4f}")
    print(f"  ATR({args.period}): {ref['atr']:.4f}")
    print(f"  HighestHigh({args.period}): {ref['hh']:.4f}   LowestLow({args.period}): {ref['ll']:.4f}")
    print(f"  multiplier: {args.multiplier}   offset (ATR*mult): {ref['atr'] * args.multiplier:.4f}")
    print()

    def _dist_str(r: dict) -> str:
        side = "above" if r["stop"] - r["price"] >= 0 else "below"
        return f"({r['dist_pct']:.2f}% {side} last price)"

    if long_r is not None:
        print(f"  LONG  stop = HH - ATR*mult = {long_r['hh']:.4f} - {ref['atr'] * args.multiplier:.4f} "
              f"= {long_r['stop']:.4f}   {_dist_str(long_r)}")
    if short_r is not None:
        print(f"  SHORT stop = LL + ATR*mult = {short_r['ll']:.4f} + {ref['atr'] * args.multiplier:.4f} "
              f"= {short_r['stop']:.4f}   {_dist_str(short_r)}")

    print(
        "\nNote: this is the single latest-bar value, not a ratcheted stop for an "
        "open position -- if you already hold a position, this will only ever "
        "match your real stop on the day you entered; after that your real stop "
        "should never have loosened even if this recomputed value has."
    )


if __name__ == "__main__":
    main()

"""Diagnostic: why wasn't a specific bar detected as CHoCH?

Usage:
    uv run diag_choch.py --symbol US.SNDK --tf 15m --date 2026-05-22 --bar 16:00:00

Fetches data (with warmup), runs detect_bos_choch, and prints a step-by-step
trace of what happened at the target bar.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from datetime import datetime, timedelta

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import numpy as np
import pandas as pd

from strategy.smc.market_structure import (
    find_swings,
    detect_bos_choch,
    _local_trend_array,
    _is_displacement,
    _price_accepted,
)


def _local_trend_label(closes: np.ndarray, j: int, window: int = 20) -> str:
    start = max(0, j - window + 1)
    w = closes[start:j]
    m = len(w)
    if m < 4:
        return "up (too short)"
    third = m // 3
    early = float(w[:third].mean())
    late  = float(w[m - third:].mean())
    return f"{'up' if late > early else 'down'}  (early={early:.4f}, late={late:.4f})"


def diagnose(symbol: str, tf: str, target_date: str, target_time: str,
             max_span: int | None, trend_window: int = 20) -> None:
    from feeds.fetcher import fetch_klines

    # Fetch with generous warmup (30 trading days back)
    dt = datetime.strptime(target_date, "%Y-%m-%d")
    warmup_start = (dt - timedelta(days=45)).strftime("%Y-%m-%d")

    print(f"Fetching {symbol} {tf}  {warmup_start} → {target_date} ...")
    df = fetch_klines(symbol, tf, warmup_start, target_date)
    if df.empty:
        print("ERROR: no data returned — is moomoo OpenD running?")
        return

    print(f"Total bars: {len(df)}  ({df['time_key'].iloc[0]} … {df['time_key'].iloc[-1]})\n")

    target_key = f"{target_date} {target_time}"
    if target_key not in df["time_key"].values:
        print(f"ERROR: bar '{target_key}' not found in data.")
        print("Available times on that date:")
        day_bars = df[df["time_key"].str.startswith(target_date)]
        print(day_bars["time_key"].tolist())
        return

    target_idx = int(df[df["time_key"] == target_key].index[0])
    print(f"Target bar: idx={target_idx}  time_key={target_key}")
    row = df.iloc[target_idx]
    print(f"  OHLC: O={row.open:.4f}  H={row.high:.4f}  L={row.low:.4f}  C={row.close:.4f}")
    print()

    # ── Swing detection ──────────────────────────────────────────────────────
    swings = find_swings(df, lookback=2)
    print(f"Total swings: {len(swings)}")
    # Show swings visible up to + including target bar
    vis_swings = [s for s in swings if s["idx"] <= target_idx]
    print(f"Swings up to target bar: {len(vis_swings)}")
    if vis_swings:
        print("  Last 8 swings before / at target:")
        for s in vis_swings[-8:]:
            print(f"    {s['kind']:5s}  idx={s['idx']:4d}  price={s['price']:10.4f}  time={s['time']}")
    print()

    # ── Local trend at target bar ────────────────────────────────────────────
    closes = df["close"].values.astype(float)
    opens  = df["open"].values.astype(float)
    highs  = df["high"].values.astype(float)
    lows   = df["low"].values.astype(float)

    local_t = _local_trend_array(closes, window=trend_window)
    print(f"Local trend at target bar (idx={target_idx}): '{local_t[target_idx]}'")
    print(f"  {_local_trend_label(closes, target_idx, window=trend_window)}")
    print()

    # ── What reference swing(s) would fire at target bar? ───────────────────
    print("── Checking which reference swings trigger at target bar ──")
    n = len(df)
    for i, sw in enumerate(swings):
        if sw["idx"] != target_idx:
            continue
        # This is a swing at the target bar itself; find reference swing
        if sw["kind"] == "high":
            prev_highs = [s for s in swings[:i] if s["kind"] == "high"]
            if prev_highs:
                prev = prev_highs[-1]
                ref_price = float(highs[prev["idx"]])
                print(f"  Target is a swing {sw['kind']}; reference high idx={prev['idx']} "
                      f"wick={ref_price:.4f}")
        else:
            prev_lows = [s for s in swings[:i] if s["kind"] == "low"]
            if prev_lows:
                prev = prev_lows[-1]
                ref_price = float(lows[prev["idx"]])
                print(f"  Target is a swing {sw['kind']}; reference low idx={prev['idx']} "
                      f"wick={ref_price:.4f}")

    print()
    print("── Checking if target bar BREAKS any prior swing level ──")
    for i, sw in enumerate(swings):
        if sw["idx"] >= target_idx:
            continue
        if sw["kind"] == "high":
            prev_highs_before = [s for s in swings[:i] if s["kind"] == "high"]
            if not prev_highs_before:
                continue
            ph = prev_highs_before[-1]
            ref_high = float(highs[ph["idx"]])
            # does target bar break this level?
            if closes[target_idx] > ref_high:
                span = target_idx - ph["idx"]
                print(f"  BREAK: close={closes[target_idx]:.4f} > ref_high={ref_high:.4f} "
                      f"(from swing idx={ph['idx']}, span={span} bars)")
                if max_span and span > max_span:
                    print(f"    → REJECTED by max_span_bars={max_span} (span {span} > {max_span})")
                    continue
                sig_type = "BOS" if local_t[target_idx] == "up" else "CHoCH"
                print(f"    → classified as {sig_type} (local_t='{local_t[target_idx]}')")
                if sig_type == "CHoCH":
                    displ = _is_displacement(opens, closes, highs, lows, target_idx)
                    acc   = _price_accepted(closes, target_idx, ref_high, "bull", n)
                    print(f"    CHoCH filter:")
                    body = abs(closes[target_idx] - opens[target_idx])
                    total = highs[target_idx] - lows[target_idx]
                    start_d = max(0, target_idx - 5)
                    pb = np.abs(closes[start_d:target_idx] - opens[start_d:target_idx])
                    mb = float(pb.mean()) if len(pb) else 0.0
                    print(f"      _is_displacement: {displ}  "
                          f"(body={body:.4f}, mean_body={mb:.4f}×1.5={mb*1.5:.4f}, "
                          f"body/range={body/total if total else 0:.2f} vs 0.5)")
                    print(f"      _price_accepted:  {acc}  "
                          f"(next 2 closes: {[round(closes[k],4) for k in range(target_idx+1, min(target_idx+3,n))]},"
                          f" ref={ref_high:.4f}, need > ref for bull)")
                    print(f"    → EMITTED: {displ and acc}  (both must be True)")

        elif sw["kind"] == "low":
            prev_lows_before = [s for s in swings[:i] if s["kind"] == "low"]
            if not prev_lows_before:
                continue
            pl = prev_lows_before[-1]
            ref_low = float(lows[pl["idx"]])
            if closes[target_idx] < ref_low:
                span = target_idx - pl["idx"]
                print(f"  BREAK: close={closes[target_idx]:.4f} < ref_low={ref_low:.4f} "
                      f"(from swing idx={pl['idx']}, span={span} bars)")
                if max_span and span > max_span:
                    print(f"    → REJECTED by max_span_bars={max_span} (span {span} > {max_span})")
                    continue
                sig_type = "BOS" if local_t[target_idx] == "down" else "CHoCH"
                print(f"    → classified as {sig_type} (local_t='{local_t[target_idx]}')")
                if sig_type == "CHoCH":
                    displ = _is_displacement(opens, closes, highs, lows, target_idx)
                    acc   = _price_accepted(closes, target_idx, ref_low, "bear", n)
                    print(f"    CHoCH filter:")
                    body = abs(closes[target_idx] - opens[target_idx])
                    total = highs[target_idx] - lows[target_idx]
                    start_d = max(0, target_idx - 5)
                    pb = np.abs(closes[start_d:target_idx] - opens[start_d:target_idx])
                    mb = float(pb.mean()) if len(pb) else 0.0
                    print(f"      _is_displacement: {displ}  "
                          f"(body={body:.4f}, mean_body={mb:.4f}×1.5={mb*1.5:.4f}, "
                          f"body/range={body/total if total else 0:.2f} vs 0.5)")
                    print(f"      _price_accepted:  {acc}  "
                          f"(next 2 closes: {[round(closes[k],4) for k in range(target_idx+1, min(target_idx+3,n))]},"
                          f" ref={ref_low:.4f}, need < ref for bear)")
                    print(f"    → EMITTED: {displ and acc}  (both must be True)")

    print()

    # ── Also check: was the bar the break bar for a higher-up swing scan? ───
    print("── Full detect_bos_choch output (last 10 signals) ──")
    all_sigs = detect_bos_choch(df, max_span_bars=max_span,
                                trend_window=trend_window)
    recent = [s for s in all_sigs if s["idx"] >= target_idx - 10]
    for s in all_sigs[-10:]:
        tag = " ← TARGET" if s["idx"] == target_idx else ""
        print(f"  {s['type']:5s} {s['direction']:4s}  break_idx={s['idx']:4d}  "
              f"from_idx={s.get('from_idx','?'):4}  price={s['price']:.4f}  "
              f"time={df['time_key'].iloc[s['idx']]}{tag}")

    at_target = [s for s in all_sigs if s["idx"] == target_idx]
    print()
    if at_target:
        print(f"Signals AT target bar: {at_target}")
    else:
        print("No signal emitted AT target bar.")

    # ── Show bars around target ──────────────────────────────────────────────
    print()
    print("── Bars around target (±5) ──")
    lo = max(0, target_idx - 5)
    hi = min(n, target_idx + 6)
    slice_df = df.iloc[lo:hi][["time_key", "open", "high", "low", "close"]].copy()
    slice_df["local_t"] = [local_t[i] for i in range(lo, hi)]
    print(slice_df.to_string(index=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="US.SNDK")
    ap.add_argument("--tf",     default="15m")
    ap.add_argument("--date",   default="2026-05-22")
    ap.add_argument("--bar",    default="16:00:00",
                    help="time_key time portion, e.g. 16:00:00")
    ap.add_argument("--max-span", type=int, default=None,
                    help="Override max_span_bars (default: use viewer's _BOS_MAX_SPAN)")
    ap.add_argument("--trend-window", type=int, default=None,
                    help="Override trend_window (default: use _TREND_WINDOW dict)")
    args = ap.parse_args()

    max_span = args.max_span
    if max_span is None:
        _BOS_MAX_SPAN = {"1m": 60, "3m": 20, "5m": 12, "15m": 26, "30m": 13, "1h": 7, "4h": 8, "1d": 5}
        max_span = _BOS_MAX_SPAN.get(args.tf)

    trend_window = args.trend_window
    if trend_window is None:
        _TREND_WINDOW = {"1m": 60, "3m": 20, "5m": 12, "15m": 26, "30m": 13, "1h": 7, "4h": 8, "1d": 5}
        trend_window = _TREND_WINDOW.get(args.tf, 20)

    print(f"max_span_bars={max_span}  trend_window={trend_window}\n")
    diagnose(args.symbol, args.tf, args.date, args.bar, max_span, trend_window)


if __name__ == "__main__":
    main()

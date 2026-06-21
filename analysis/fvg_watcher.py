"""Lightweight FVG-formed watcher — shared by the live signal scanner
(analysis/signal_scanner.py) and the backscan CLI (analysis/fvg_backscan.py).

Detects when a FVG matching hand-tuned params (picked by eyeballing
backtest/fvg_width_sweep.py + backtest/fvg_width_viz.py output) appears for a
(symbol, timeframe) pair. Unlike SignalDetector in signal_scanner.py, this
does NOT check trend/BOS/CHoCH or compute SL/TP/RR — it is purely "a FVG
matching these params just appeared", an alert independent of the full entry
signal pipeline.
"""

from __future__ import annotations

import json
import pathlib

from feeds.fetcher import fetch_klines
from strategy.smc.fvg import build_daily_lvn_profiles, gap_width_pct, gaps_for_combo

_ROOT = pathlib.Path(__file__).parent.parent
_DEFAULT_CONFIG_PATH = _ROOT / "config" / "scanner" / "fvg_watch_params.json"


def load_fvg_watch_config(path: pathlib.Path | None = None) -> dict[str, list[dict]]:
    """Load the per-(symbol, tf) FVG-watch param config.

    Returns {symbol: [{"tf": ..., "min_gap_pct": ..., ...}, ...]}.
    Top-level "_"-prefixed keys (e.g. "_note") are ignored. Returns {} if the
    file does not exist yet.
    """
    p = path or _DEFAULT_CONFIG_PATH
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        raw = json.load(f)
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def scan_symbol_tf(
    symbol: str,
    tf: str,
    params: dict,
    start: str,
    end: str,
    force_refresh: bool = False,
) -> list[dict]:
    """Detect FVGs for (symbol, tf) over [start, end] using the tuned params.

    Returns one dict per matching gap, shaped for db.signals.SignalsDB's
    fvg_watch_signals table (symbol, tf, direction, zone_top, zone_bottom,
    formed_time, filled, params_json) plus a width_pct convenience field.

    Used by both the live scanner (a short rolling window, force_refresh=True
    for freshness) and the backscan CLI (a wide historical window, cached
    klines) — same detection logic, different window/freshness needs.
    """
    klines = fetch_klines(symbol, tf, start, end, force_refresh=force_refresh)
    if klines is None or klines.empty:
        return []

    lvn_profiles = build_daily_lvn_profiles(klines) if params.get("require_lvn_overlap") else None
    gaps = gaps_for_combo(klines, params, lvn_profiles=lvn_profiles)

    params_json = json.dumps(params, sort_keys=True)
    return [
        {
            "symbol":      symbol,
            "tf":          tf,
            "direction":   gap["direction"],
            "zone_top":    gap["top"],
            "zone_bottom": gap["bottom"],
            "formed_time": str(klines.iloc[gap["idx"]]["time_key"]),
            "filled":      gap["filled"],
            "width_pct":   gap_width_pct(gap),
            "params_json": params_json,
        }
        for gap in gaps
    ]

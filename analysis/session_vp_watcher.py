"""Session Value-Area reversal live-scan watcher — shared by the live signal
scanner (analysis/signal_scanner.py).

Detects a FRESH VAL-touch-then-RSI-oversold reversal (same pure logic as
backtest/session_vp_engine.py's backtest loop -- POC/VAL frozen at the
warmup mark, strategy.session_vp.reversal.detect_val_reversal for the
signal) for a (symbol, session) pair, and returns it shaped for
db.signals.SignalsDB's `signals` table with a dynamically computed
recommended SL/TP (see session_vp_engine.py's sl_dist = tp_dist/target_rr
formula, reused verbatim so live signals match what the backtest would have
simulated as an actual trade).
"""

from __future__ import annotations

import json
import pathlib
from datetime import datetime

from backtest.session_vp_engine import ALGO_VERSION, precompute_session_context
from core.time_utils import minutes_since_session_start
from feeds.fetcher import fetch_klines
from strategy.session_vp.profile import compute_value_area
from strategy.session_vp.reversal import detect_val_reversal
from strategy.smc.fvg import compute_volume_profile

_ROOT = pathlib.Path(__file__).parent.parent
_DEFAULT_CONFIG_PATH = _ROOT / "config" / "scanner" / "session_vp_params.json"


def load_session_vp_config(path: pathlib.Path | None = None) -> dict[str, list[dict]]:
    """Load the per-(symbol, session) session_vp watch param config.

    Returns {symbol: [{"session": ..., "warmup_minutes": ..., ...}, ...]}.
    Top-level "_"-prefixed keys (e.g. "_note") are ignored. Returns {} if the
    file does not exist yet.
    """
    p = path or _DEFAULT_CONFIG_PATH
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        raw = json.load(f)
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def scan_symbol_session_vp(
    symbol: str,
    entry_cfg: dict,
    start: str,
    end: str,
    schedule_sessions: dict,
    force_refresh: bool = False,
) -> list[dict]:
    """Detect a fresh session_vp reversal signal for (symbol, entry_cfg["session"]).

    Returns 0 or 1 signal dicts. A signal only fires when the reversal
    confirmation bar is the MOST RECENTLY CLOSED bar in `klines` -- this is
    what prevents re-alerting the same historical signal on every later scan
    cycle (mirrors how SignalDetector.detect() in signal_scanner.py only
    considers zones relative to the latest bar), so no extra in-memory
    dedup state is needed beyond the existing entry_zone-coordinate dedup
    already used by the scanner for every other signal type.
    """
    klines = fetch_klines(symbol, "1m", start, end, force_refresh=force_refresh)
    if klines is None or klines.empty:
        return []
    n = len(klines)

    ctx = precompute_session_context(klines, schedule_sessions)
    session_info  = ctx["session_info"]
    occurrence_id = ctx["occurrence_id"]
    groups        = ctx["groups"]

    last_oid = int(occurrence_id[-1])
    if last_oid < 0:
        return []  # latest bar isn't in any configured session

    idxs = groups[last_oid]
    session_name, elapsed = session_info[idxs[-1]]
    if session_name != entry_cfg["session"]:
        return []
    if elapsed < entry_cfg["warmup_minutes"]:
        return []  # still in the warmup window -- no frozen profile yet

    session_start = idxs[0]
    warmup_end = None
    for idx in idxs:
        if session_info[idx][1] >= entry_cfg["warmup_minutes"]:
            warmup_end = idx
            break
    if warmup_end is None:
        return []  # this occurrence is shorter than the warmup window

    profile_klines = klines.iloc[session_start : warmup_end + 1]
    edges, bin_vols = compute_volume_profile(profile_klines, n_bins=entry_cfg["n_bins"])
    va = compute_value_area(edges, bin_vols, va_pct=entry_cfg["va_pct"])
    poc, val = va["poc"], va["val"]
    if poc <= 0 or val <= 0 or poc <= val:
        return []
    if (poc - val) / val < entry_cfg["min_val_poc_dist_pct"]:
        return []

    signals = detect_val_reversal(
        klines, val,
        rsi_period=entry_cfg["rsi_period"],
        rsi_threshold=entry_cfg["rsi_threshold"],
        start_idx=warmup_end,
        end_idx=n - 1,
        val_proximity_pct=entry_cfg.get("val_proximity_pct", 0.0),
    )
    fresh = [s for s in signals if s["entry_idx"] == n - 1]
    if not fresh:
        return []
    sig = fresh[0]

    entry_price = sig["entry_price"]
    target_rr   = entry_cfg.get("target_rr", 1.0)
    tp          = poc
    tp_dist     = tp - entry_price
    sl_dist     = tp_dist / target_rr
    sl          = entry_price - sl_dist
    if sl_dist <= 0:
        return []

    return [{
        "symbol":            symbol,
        "direction":         "bull",
        "signal_time":       str(klines["time_key"].iloc[sig["entry_idx"]])[:19],
        "trend_tf":          "1m",
        "entry_tf":          "1m",
        "entry_zone_top":    entry_price,
        "entry_zone_bottom": entry_price,
        "sl_price":          sl,
        "tp_price":          tp,
        "rr_ratio":          round(target_rr, 2),
        "bos_price":         None,
        "strategy":          "session_vp",
        "params_json":       json.dumps({**entry_cfg, "poc": poc, "val": val}),
        "algo_version":      ALGO_VERSION,
        "source":            "auto",
        "status":            "open",
        "created_at":        datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
    }]

"""VAL-entry / KD-channel-TP live-scan watcher -- shared by the live signal
scanner (analysis/signal_scanner.py).

Same VAL-touch + RSI-oversold + reversal entry as
strategy/session_vp/reversal.py's detect_val_reversal, but TP is NOT the
session's frozen POC -- it's `up1`, the fast EMA-channel upper band from
strategy/smc/kd_trend.py's compute_kd (`up1 = EMA(High, kd_fast)`), taken
as a snapshot at entry time and held fixed for the trade's life, same
"frozen at entry" contract POC had. Regular-session backtest found this
clearly outperforms the POC-based TP (see
backtest/results/session_vp_v1_review/REVIEW.md section 9): out-of-sample
validation 10/10 candidates positive, smooth (non-spiky) parameter
sensitivity across warmup_minutes/rsi_threshold/kd_fast.

Note: `kd_slow` has NO effect on `up1` (it only feeds `up2`/`mid2`, which
this watcher never uses) -- kept as a config field only for symmetry with
compute_kd's signature, not because it does anything here.

This module stays scanner-only and exploratory -- like vah_soxs_watcher.py,
it is NOT part of strategy/session_vp/ or backtest/session_vp_engine.py
(the frozen-POC TP is what that shared package implements); this is a
distinct TP-source variant kept as its own watcher rather than bolted onto
the POC watcher's code path.
"""

from __future__ import annotations

import json
import pathlib
from datetime import datetime

import numpy as np

from backtest.session_vp_engine import precompute_session_context
from feeds.fetcher import fetch_klines
from strategy.session_vp.profile import compute_value_area
from strategy.session_vp.reversal import detect_val_reversal
from strategy.smc.fvg import compute_volume_profile
from strategy.smc.kd_trend import compute_kd

_ROOT = pathlib.Path(__file__).parent.parent
_DEFAULT_CONFIG_PATH = _ROOT / "config" / "scanner" / "val_kd_tp_params.json"

ALGO_VERSION = "val_kd_tp_exploratory"  # not a git-tagged algo family -- see module docstring


def load_val_kd_tp_config(path: pathlib.Path | None = None) -> dict[str, list[dict]]:
    """Load the per-(symbol, session) VAL+KD-TP watch param config.

    Returns {symbol: [{"session": ..., "warmup_minutes": ..., ...}, ...]}.
    Top-level "_"-prefixed keys are ignored. Returns {} if the file doesn't
    exist yet.
    """
    p = path or _DEFAULT_CONFIG_PATH
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        raw = json.load(f)
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def scan_val_kd_tp(
    symbol: str,
    entry_cfg: dict,
    start: str,
    end: str,
    schedule_sessions: dict,
    force_refresh: bool = False,
) -> list[dict]:
    """Detect a fresh VAL-entry / KD-up1-TP signal.

    Returns 0 or 1 signal dicts, shaped for db.signals.SignalsDB's `signals`
    table (strategy="val_kd_tp"). Fires only when the reversal confirmation
    bar is the most recently closed bar -- same "fresh signal only" dedup
    principle as the other watchers.
    """
    klines = fetch_klines(symbol, "1m", start, end, force_refresh=force_refresh)
    if klines is None or klines.empty:
        return []
    n = len(klines)

    ctx = precompute_session_context(klines, schedule_sessions)
    session_info, occurrence_id, groups = ctx["session_info"], ctx["occurrence_id"], ctx["groups"]

    last_oid = int(occurrence_id[-1])
    if last_oid < 0:
        return []
    idxs = groups[last_oid]
    session_name, elapsed = session_info[idxs[-1]]
    if session_name != entry_cfg["session"]:
        return []
    if elapsed < entry_cfg["warmup_minutes"]:
        return []

    session_start = idxs[0]
    warmup_end = None
    for idx in idxs:
        if session_info[idx][1] >= entry_cfg["warmup_minutes"]:
            warmup_end = idx
            break
    if warmup_end is None:
        return []

    profile_klines = klines.iloc[session_start:warmup_end + 1]
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
        start_idx=warmup_end, end_idx=n - 1,
    )
    fresh = [s for s in signals if s["entry_idx"] == n - 1]
    if not fresh:
        return []
    sig = fresh[0]
    entry_price = sig["entry_price"]

    kd = compute_kd(klines, fast=entry_cfg["kd_fast"], slow=entry_cfg.get("kd_slow", 90))
    up1 = float(kd["up1"].iloc[sig["entry_idx"]])
    if np.isnan(up1) or entry_price >= up1:
        return []  # position check: entry must be below the fast-channel upper band

    target_rr = entry_cfg.get("target_rr", 1.0)
    tp = up1
    tp_dist = tp - entry_price
    sl_dist = tp_dist / target_rr
    sl = entry_price - sl_dist

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
        "strategy":          "val_kd_tp",
        "params_json":       json.dumps({**entry_cfg, "poc": poc, "val": val, "kd_up1": up1}),
        "algo_version":      ALGO_VERSION,
        "source":            "auto",
        "status":            "open",
        "created_at":        datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
    }]

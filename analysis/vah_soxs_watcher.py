"""Cross-asset VAH-short-proxy live-scan watcher -- shared by the live signal
scanner (analysis/signal_scanner.py).

The signal and its SL/TP target come from `signal_symbol`'s (e.g. SOXL) VAH
(Value Area High) + RSI-overbought reversal -- the bear mirror of
strategy/session_vp/reversal.py's VAL long signal -- but SOXL can't be
shorted directly, so the executable trade is a LONG on `exec_symbol` (e.g.
SOXS, the reverse-leveraged ETF on the same underlying). See
backtest/results/session_vp_v1_review/REVIEW.md section 8 for the full
research trail (why this exists, the beta regression, and the ATR-filter
fix for an initial version that produced impossibly good, noise-driven
results). This module stays a scanner-only exploratory watcher -- it is
NOT part of strategy/session_vp/ or backtest/session_vp_engine.py, since
the underlying approach hasn't accumulated enough live/validated samples to
be promoted to that shared, backtest-integrated strategy package.
"""

from __future__ import annotations

import json
import pathlib
from datetime import datetime

import numpy as np
import pandas as pd

from backtest.session_vp_engine import precompute_session_context
from feeds.fetcher import fetch_klines
from strategy.session_vp.profile import compute_value_area
from strategy.session_vp.reversal import compute_rsi
from strategy.smc.fvg import compute_volume_profile

_ROOT = pathlib.Path(__file__).parent.parent
_DEFAULT_CONFIG_PATH = _ROOT / "config" / "scanner" / "vah_soxs_params.json"

ALGO_VERSION = "vah_soxs_proxy_exploratory"  # not a git-tagged algo family -- see module docstring


def load_vah_soxs_config(path: pathlib.Path | None = None) -> dict[str, list[dict]]:
    """Load the per-(exec_symbol, session) cross-asset watch param config.

    Returns {exec_symbol: [{"signal_symbol": ..., "session": ..., ...}, ...]}.
    Top-level "_"-prefixed keys are ignored. Returns {} if the file doesn't
    exist yet.
    """
    p = path or _DEFAULT_CONFIG_PATH
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        raw = json.load(f)
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def compute_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> np.ndarray:
    """Simple-moving-average True Range -- a noise-floor check doesn't need
    Wilder smoothing, just a representative typical-bar-range estimate."""
    prev_close = np.roll(closes, 1)
    prev_close[0] = closes[0]
    tr = np.maximum(highs - lows, np.maximum(np.abs(highs - prev_close), np.abs(lows - prev_close)))
    return pd.Series(tr).rolling(period, min_periods=period).mean().to_numpy()


def _detect_vah_reversal(klines, vah, rsi_period, rsi_threshold, start_idx, end_idx,
                          rsi, lows, highs, closes):
    """Bear mirror of strategy.session_vp.reversal.detect_val_reversal: VAH-touch
    (instead of VAL-touch) + RSI overbought (instead of oversold) + 2-bar
    swing-high break down (instead of swing-low break up), confirmed by
    close[i+1] < low[i] and close[i+1] < vah."""
    n = len(klines)
    stop = n if end_idx is None else min(n, end_idx + 1)
    if stop < start_idx + 2:
        return []
    signals = []
    for i in range(start_idx, stop - 1):
        if highs[i] < vah:
            continue
        if np.isnan(rsi[i]) or rsi[i] <= rsi_threshold:
            continue
        j = i + 1
        if closes[j] < lows[i] and closes[j] < vah:
            signals.append({"entry_idx": j, "entry_price": float(closes[j]),
                             "touch_idx": i, "rsi_at_touch": float(rsi[i])})
    return signals


def scan_vah_soxs(
    exec_symbol: str,
    entry_cfg: dict,
    start: str,
    end: str,
    schedule_sessions: dict,
    force_refresh: bool = False,
) -> list[dict]:
    """Detect a fresh cross-asset VAH-short-proxy signal.

    Returns 0 or 1 signal dicts, shaped for db.signals.SignalsDB's `signals`
    table (strategy="vah_soxs_proxy"). Fires only when the reversal
    confirmation bar on `signal_symbol` is its most recently closed bar --
    same "fresh signal only" dedup principle as session_vp_watcher.py,
    avoiding re-alerting the same historical signal on every later cycle.
    """
    signal_symbol = entry_cfg["signal_symbol"]
    sig_klines = fetch_klines(signal_symbol, "1m", start, end, force_refresh=force_refresh)
    exec_klines = fetch_klines(exec_symbol, "1m", start, end, force_refresh=force_refresh)
    if sig_klines is None or sig_klines.empty or exec_klines is None or exec_klines.empty:
        return []
    n = len(sig_klines)

    ctx = precompute_session_context(sig_klines, schedule_sessions)
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

    profile_klines = sig_klines.iloc[session_start:warmup_end + 1]
    edges, bin_vols = compute_volume_profile(profile_klines, n_bins=entry_cfg["n_bins"])
    va = compute_value_area(edges, bin_vols, va_pct=entry_cfg["va_pct"])
    poc, vah = va["poc"], va["vah"]
    if poc <= 0 or vah <= 0 or vah <= poc:
        return []
    if (vah - poc) / poc < entry_cfg["min_val_poc_dist_pct"]:
        return []

    lows = sig_klines["low"].to_numpy(dtype=float)
    highs = sig_klines["high"].to_numpy(dtype=float)
    closes = sig_klines["close"].to_numpy(dtype=float)
    rsi = compute_rsi(sig_klines["close"], entry_cfg["rsi_period"])

    signals = _detect_vah_reversal(
        sig_klines, vah, entry_cfg["rsi_period"], entry_cfg["rsi_threshold"],
        warmup_end, n - 1, rsi, lows, highs, closes,
    )
    fresh = [s for s in signals if s["entry_idx"] == n - 1]
    if not fresh:
        return []
    sig = fresh[0]

    sig_time = str(sig_klines["time_key"].iloc[sig["entry_idx"]])[:19]
    exec_time_keys = exec_klines["time_key"].astype(str)
    match = exec_klines.index[exec_time_keys == sig_time]
    if len(match) == 0:
        return []  # no aligned exec_symbol bar at this timestamp
    exec_idx = match[0]

    exec_closes = exec_klines["close"].to_numpy(dtype=float)
    exec_highs = exec_klines["high"].to_numpy(dtype=float)
    exec_lows = exec_klines["low"].to_numpy(dtype=float)
    exec_entry_price = exec_closes[exec_idx]

    beta = entry_cfg["beta"]
    signal_move_pct = (poc - sig["entry_price"]) / sig["entry_price"]  # negative: signal_symbol expected to fall
    exec_move_pct = beta * signal_move_pct                              # positive: exec_symbol expected to rise
    exec_tp = exec_entry_price * (1 + exec_move_pct)
    tp_dist = exec_tp - exec_entry_price
    if tp_dist <= 0:
        return []

    atr = compute_atr(exec_highs, exec_lows, exec_closes, period=14)
    atr_here = atr[exec_idx]
    if np.isnan(atr_here) or tp_dist < entry_cfg["min_atr_multiple"] * atr_here:
        return []  # target too small relative to exec_symbol's own noise level

    target_rr = entry_cfg.get("target_rr", 1.0)
    sl_dist = tp_dist / target_rr
    exec_sl = exec_entry_price - sl_dist

    return [{
        "symbol":            exec_symbol,
        "direction":         "bull",
        "signal_time":       sig_time,
        "trend_tf":          "1m",
        "entry_tf":          "1m",
        "entry_zone_top":    exec_entry_price,
        "entry_zone_bottom": exec_entry_price,
        "sl_price":          exec_sl,
        "tp_price":          exec_tp,
        "rr_ratio":          round(target_rr, 2),
        "bos_price":         None,
        "strategy":          "vah_soxs_proxy",
        "params_json":       json.dumps({**entry_cfg, "signal_poc": poc, "signal_vah": vah,
                                          "signal_entry_price": sig["entry_price"]}),
        "algo_version":      ALGO_VERSION,
        "source":            "auto",
        "status":            "open",
        "created_at":        datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
    }]

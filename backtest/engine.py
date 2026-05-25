"""Multi-timeframe SMC backtesting engine.

Entry logic:
  1. HTF: detect trend via consecutive BOS/CHoCH; mark active FVG zones.
  2. LTF: when wick (low for bull / high for bear) touches an HTF FVG at the
     required depth (fvg_entry_depth_pct), proceed to confirmation.
  3. Confirmation (two modes, controlled by require_ltf_confirmation):
       True  — require LTF CHoCH + BOS in trend direction after wick enters zone.
       False — enter immediately once wick reaches fvg_entry_depth_pct.
  4. Enter at close of the triggering bar.
  5. SL = last HTF swing low (bull) / swing high (bear) ± buffer.
  6. TP = nearest HTF swing high (bull) / swing low (bear).
  7. Trade is closed at first SL or TP hit, or after max_bars_in_trade.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import uuid
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from strategy.smc import (
    find_swings,
    detect_bos_choch,
    detect_fvg,
    determine_trend,
    fvg_entry_depth,
    is_displacement_candle,
    check_ltf_confirmation,
    compute_volume_profile,
    fvg_overlaps_lvn,
    compute_kd,
    kd_trend,
)
from backtest.stats import sharpe_ratio, sortino_ratio

def _algo_version() -> str:
    """Return the most recent smc_v* git tag on HEAD, falling back to 'smc_unknown'."""
    import subprocess, pathlib
    try:
        return subprocess.check_output(
            ["git", "describe", "--tags", "--match", "smc_v*", "--abbrev=0"],
            cwd=pathlib.Path(__file__).parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "smc_unknown"


ALGO_VERSION = _algo_version()

_HTF_WINDOW_DEFAULT = 20  # default HTF bars (~5 h at 15 m ≈ one trading day; override via BacktestParams.htf_window_bars)
_WARMUP      = 40   # skip the first N LTF bars while indicators warm up


def _tf_minutes(tf: str) -> int:
    """Parse timeframe string to minutes. '1m'→1, '15m'→15, '1h'→60, '4h'→240."""
    tf = tf.lower().strip()
    if tf.endswith("h"):
        return int(tf[:-1]) * 60
    if tf.endswith("m"):
        return int(tf[:-1])
    raise ValueError(f"Unrecognised TF string: {tf!r}")


def _find_exit(
    lows: np.ndarray,
    highs: np.ndarray,
    closes: np.ndarray,
    from_bar: int,
    sl: float,
    tp: float,
    direction: str,
    max_bars: int,
) -> tuple[int, float, str]:
    """Find the first SL or TP hit after entry using numpy — no Python loop.

    Returns (exit_bar_abs, exit_price, result) where result is
    'win' | 'loss' | 'timeout'.
    """
    end = min(from_bar + max_bars, len(lows))
    lo  = lows[from_bar:end]
    hi  = highs[from_bar:end]

    if direction == "bull":
        sl_mask = lo <= sl
        tp_mask = hi >= tp
    else:
        sl_mask = hi >= sl
        tp_mask = lo <= tp

    first_sl = int(np.argmax(sl_mask)) if sl_mask.any() else max_bars
    first_tp = int(np.argmax(tp_mask)) if tp_mask.any() else max_bars

    if first_sl == max_bars and first_tp == max_bars:
        j = end - 1
        return j, float(closes[j]), "timeout"
    if first_sl <= first_tp:
        return from_bar + first_sl, sl, "loss"
    return from_bar + first_tp, tp, "win"


# ── Parameter set ─────────────────────────────────────────────────────────────

@dataclass
class BacktestParams:
    trend_tf:             str   = "60m"
    entry_tf:             str   = "15m"
    swing_lookback:       int   = 2
    bos_count:            int   = 1      # consecutive BOS to confirm trend
    fvg_min_width_pct:    float = 0.002  # min FVG size as fraction of price
    fvg_entry_depth_pct:  float = 0.10   # 0=edge, 1=far side of zone
    fvg_max_age_bars:     int   = 50    # invalidate FVGs older than this many HTF bars
    displacement_required:    bool  = False  # FVG must be from a displacement candle
    displacement_atr_mult:    float = 1.5   # middle candle range > mult × baseline mean range
    displacement_body_ratio:  float = 0.5   # body / range of middle candle (0=doji, 1=marubozu)
    displacement_lookback:    int   = 5     # candles used to compute baseline mean range
    require_ltf_confirmation: bool  = False  # True = CHoCH+BOS; False = depth-only entry
    require_lvn_overlap:      bool  = False  # FVG must sit in a Low Volume Node
    lvn_threshold:            float = 0.30   # LVN: zone vol < threshold × max bin vol
    sl_buffer_pct:            float = 0.001  # extra % added beyond the swing level
    max_sl_pct:           float = 0.005  # skip trade if SL > this % of entry
    min_rr:               float = 2.0
    htf_window_bars:      int   = 20     # HTF bars for trend/structure (20 × 15 m ≈ 5 h ≈ one trading day)
    allow_short:          bool  = True   # False = long-only (skip bear setups)
    intraday_only:        bool  = False  # True = force-close positions at end of trading day
    kd_sl_fallback:       bool  = False  # use KD slow-channel lo2/up2 as fallback SL/TP anchor
    # Trend method(s) — applied in order; all must agree for a valid trend signal.
    # Supported: "bos_choch", "kd"
    htf_trend_methods:    tuple  = ("bos_choch",)
    # Per-method config dict, keyed by "<method>_<param>".
    # Adaptive mode (default): {"kd_fast": 15, "kd_slow": 60,
    #   "kd_smooth": 3, "kd_min_bars": 3, "kd_atr_threshold": 0.05}
    # Legacy fixed-window:     {"kd_fast": 15, "kd_slow": 60,
    #   "kd_smooth": 0, "kd_window": 10, "kd_atr_threshold": 0.05}
    htf_trend_params:     dict   = field(default_factory=dict)

    def label(self) -> str:
        d    = f"D{self.displacement_atr_mult:.1f}b{self.displacement_body_ratio:.1f}" if self.displacement_required else "d"
        conf = "ltf" if self.require_ltf_confirmation else "raw"
        flags = ""
        if not self.allow_short:
            flags += " Lo"
        if self.intraday_only:
            flags += " ID"
        if self.kd_sl_fallback:
            flags += " kF"
        methods = "+".join(self.htf_trend_methods)
        tp = self.htf_trend_params
        kd_tag = ""
        if "kd" in self.htf_trend_methods:
            at     = tp.get("kd_atr_threshold", 0.0)
            at_tag = f"a{at}" if at else ""
            sm     = tp.get("kd_smooth", 3)
            if sm > 0:
                filt_tag = f"s{sm}m{tp.get('kd_min_bars', 3)}{at_tag}"
            else:
                ft       = tp.get("kd_flat_threshold", 0.0)
                filt_tag = f"w{tp.get('kd_window', 10)}{'f'+str(ft) if ft else ''}{at_tag}"
            kd_tag = (f" kd{tp.get('kd_fast',25)}/{tp.get('kd_slow',90)}"
                      f"{filt_tag}")
        return (
            f"{self.trend_tf}/{self.entry_tf} [{methods}]{kd_tag}"
            f" lb{self.swing_lookback} bos{self.bos_count} w{self.htf_window_bars}"
            f" fvg{self.fvg_min_width_pct:.3f}"
            f" dp{self.fvg_entry_depth_pct:.1f} {d}"
            f" {conf} sl{self.sl_buffer_pct:.3f} msl{self.max_sl_pct:.3f}"
            f" rr{self.min_rr:.1f}{flags}"
        )

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}   # type: ignore[attr-defined]
        d["htf_trend_methods"] = json.dumps(list(self.htf_trend_methods))
        d["htf_trend_params"]  = json.dumps(self.htf_trend_params, sort_keys=True)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "BacktestParams":
        d = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        if "htf_trend_methods" in d and isinstance(d["htf_trend_methods"], str):
            d["htf_trend_methods"] = tuple(json.loads(d["htf_trend_methods"]))
        if "htf_trend_params" in d:
            v = d["htf_trend_params"]
            d["htf_trend_params"] = json.loads(v) if isinstance(v, str) and v else (v or {})
        return cls(**d)


# ── Trade ─────────────────────────────────────────────────────────────────────

@dataclass
class Trade:
    direction:   str    # "bull" | "bear"
    entry_price: float
    sl:          float
    tp:          float
    planned_rr:  float
    entry_time:  str
    exit_time:   str   = ""
    exit_price:  float = 0.0
    result:      str   = ""   # "win" | "loss" | "timeout"
    r_multiple:  float = 0.0  # realised R (positive = profit)
    trade_id:    str   = ""   # set in __post_init__ if not provided
    entry_ltf_bar: int = 0    # absolute LTF bar index at entry (for chart centering)
    fvg_top:     float = 0.0  # HTF FVG price levels that triggered entry
    fvg_bottom:  float = 0.0

    def __post_init__(self) -> None:
        if not self.trade_id:
            # Deterministic ID scoped to algo version — same entry under a different
            # version produces a different ID, preventing cross-version DB collisions.
            key = f"{ALGO_VERSION}:{self.entry_time}:{self.direction}:{self.entry_price:.6f}:{self.sl:.6f}"
            self.trade_id = hashlib.sha256(key.encode()).hexdigest()[:8]


# ── Result ────────────────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    params: BacktestParams
    trades: list[Trade] = field(default_factory=list)

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def n_wins(self) -> int:
        return sum(1 for t in self.trades if t.result == "win")

    @property
    def win_rate(self) -> float:
        return self.n_wins / self.n_trades if self.n_trades else 0.0

    @property
    def total_r(self) -> float:
        return sum(t.r_multiple for t in self.trades)

    @property
    def avg_r(self) -> float:
        return self.total_r / self.n_trades if self.n_trades else 0.0

    @property
    def profit_factor(self) -> float:
        wins  = sum(t.r_multiple for t in self.trades if t.r_multiple > 0)
        losses = sum(-t.r_multiple for t in self.trades if t.r_multiple < 0)
        if losses == 0:
            return float("inf") if wins > 0 else 0.0
        return wins / losses

    @property
    def max_drawdown_r(self) -> float:
        equity = peak = max_dd = 0.0
        for t in self.trades:
            equity += t.r_multiple
            if equity > peak:
                peak = equity
            dd = peak - equity
            if dd > max_dd:
                max_dd = dd
        return max_dd

    @property
    def max_loss_r(self) -> float:
        losses = [-t.r_multiple for t in self.trades if t.r_multiple < 0]
        return max(losses) if losses else 0.0

    @property
    def final_value(self) -> float:
        """Compounding equity from $10,000 initial, risking 1% of current equity per trade."""
        equity = 10_000.0
        for t in self.trades:
            equity += t.r_multiple * (equity * 0.01)
        return round(equity, 2)

    def summary_dict(self) -> dict:
        rs = [t.r_multiple for t in self.trades]
        bull = [t for t in self.trades if t.direction == "bull"]
        bear = [t for t in self.trades if t.direction == "bear"]
        bull_wins = sum(1 for t in bull if t.result == "win")
        bear_wins = sum(1 for t in bear if t.result == "win")
        return {
            "n_trades":        self.n_trades,
            "win_rate":        round(self.win_rate, 3),
            "total_r":         round(self.total_r, 2),
            "avg_r":           round(self.avg_r, 3),
            "profit_factor":   round(self.profit_factor, 2),
            "max_drawdown_r":  round(self.max_drawdown_r, 2),
            "max_loss_r":      round(self.max_loss_r, 2),
            "sharpe":          round(sharpe_ratio(rs), 3),
            "sortino":         round(sortino_ratio(rs), 3),
            "bull_trades":     len(bull),
            "bear_trades":     len(bear),
            "bull_win_rate":   round(bull_wins / len(bull), 3) if bull else 0.0,
            "bear_win_rate":   round(bear_wins / len(bear), 3) if bear else 0.0,
            "bull_total_r":    round(sum(t.r_multiple for t in bull), 2),
            "bear_total_r":    round(sum(t.r_multiple for t in bear), 2),
            "final_value":     self.final_value,
            **self.params.to_dict(),
        }


# ── Engine ────────────────────────────────────────────────────────────────────

def run_backtest(
    htf: pd.DataFrame,
    ltf: pd.DataFrame,
    params: BacktestParams,
    max_bars_in_trade: int = 200,
    rejection_log: list | None = None,
    inspect_window: tuple[str, str] | None = None,
) -> BacktestResult:
    """Run a single SMC multi-TF backtest.

    Args:
        htf:               High-timeframe OHLCV (trend / structure TF)
        ltf:               Low-timeframe OHLCV (entry TF)
        params:            Strategy parameters
        max_bars_in_trade: Force-close after this many LTF bars (timeout)
        rejection_log:     If a list is passed, FVG touch/rejection events are
                           appended to it (opt-in; zero overhead when None).
        inspect_window:    Optional (start_date, end_date) strings "YYYY-MM-DD"
                           to restrict rejection logging to a date range.

    Returns:
        BacktestResult with a list of Trade objects and computed metrics.
    """
    result = BacktestResult(params=params)
    if htf.empty or ltf.empty:
        return result

    # Pre-sort string arrays — np.searchsorted gives O(log n) HTF bar lookup
    htf_times = htf["time_key"].values.astype(str)
    ltf_times = ltf["time_key"].values.astype(str)
    ltf_opens = ltf["open"].values.astype(float)
    ltf_highs = ltf["high"].values.astype(float)
    ltf_lows  = ltf["low"].values.astype(float)
    ltf_cls   = ltf["close"].values.astype(float)
    n_ltf     = len(ltf_times)

    # ── State ─────────────────────────────────────────────────────────────
    active_trade:        Optional[Trade] = None
    active_trade_bar:    int             = -1
    in_fvg_since:        int             = -1   # LTF bar when we first entered FVG
    current_fvg_key:     Optional[tuple] = None  # (bottom, top) — stable across window shifts
    used_fvg_keys:       set             = set() # FVG zones used today; reset each trading day
    _prev_date:          str             = ""    # tracks calendar-day boundary

    # ── Rejection log state (only used when rejection_log is not None) ───────
    _rlog_event:   dict | None = None   # pending event for current FVG zone
    _rlog_depth_ok: bool       = False  # depth threshold reached in this zone

    def _rlog_in_window(t: str) -> bool:
        if inspect_window is None:
            return True
        return inspect_window[0] <= t[:10] <= inspect_window[1]

    def _rlog_finalize(outcome: str, **extra) -> None:
        nonlocal _rlog_event, _rlog_depth_ok
        if _rlog_event is not None:
            rejection_log.append({**_rlog_event, "outcome": outcome, **extra})
        _rlog_event = None
        _rlog_depth_ok = False

    # HTF cache — recomputed only when a new HTF bar closes
    prev_htf_pos:  int             = -1
    htf_view:      pd.DataFrame    = htf.iloc[0:0]
    htf_swings:    list[dict]      = []
    htf_bos:       list[dict]      = []
    htf_fvgs:      list[dict]      = []
    trend:         Optional[str]   = None
    vp_edges:      object          = None   # volume profile edges (np.ndarray | None)
    vp_vols:       object          = None   # volume profile bins  (np.ndarray | None)
    htf_kd_lo2:    float           = float("nan")  # KD slow-channel lower band (SL anchor)
    htf_kd_up2:    float           = float("nan")  # KD slow-channel upper band (SL anchor)

    # Precompute last-bar-of-day index for each LTF bar (intraday_only mode).
    # Groups bars by calendar date (first 10 chars of time_key) and records the
    # absolute index of the last bar on the same date.
    day_end_bars: Optional[np.ndarray] = None
    if params.intraday_only:
        dates = np.array([str(t)[:10] for t in ltf_times])
        day_end_bars = np.empty(n_ltf, dtype=np.int64)
        _, first_occ = np.unique(dates, return_index=True)
        for k in range(len(first_occ)):
            s = first_occ[k]
            e = (first_occ[k + 1] - 1) if k + 1 < len(first_occ) else n_ltf - 1
            day_end_bars[s : e + 1] = e

    # Number of LTF bars that fit in one HTF bar — used to size the confirmation
    # window so LTF structure is evaluated only within the current HTF candle.
    ltf_per_htf = max(1, _tf_minutes(params.trend_tf) // _tf_minutes(params.entry_tf))

    # Pre-compute LTF BOS/CHoCH once — avoids calling detect_bos_choch() on every
    # 1m bar while waiting in an FVG zone (otherwise O(n_bars × window) Python loops).
    # Signals are stored with ABSOLUTE LTF bar indices; bisect gives O(log n) window slicing.
    if params.require_ltf_confirmation:
        _ltf_precomp: list[dict] = sorted(
            detect_bos_choch(ltf, lookback=1, trend_window=ltf_per_htf),
            key=lambda s: s["idx"],
        )
        _ltf_precomp_idxs: list[int] = [s["idx"] for s in _ltf_precomp]
    else:
        _ltf_precomp = []
        _ltf_precomp_idxs = []

    i = _WARMUP
    while i < n_ltf:
        t       = ltf_times[i]
        bar_lo  = ltf_lows[i]
        bar_hi  = ltf_highs[i]
        bar_cls = ltf_cls[i]

        # Reset per-day FVG usage at each new trading day so that fresh sessions
        # can form new setups in the same price zones (prevents cross-day stale blocks).
        _cur_date = t[:10]
        if _cur_date != _prev_date:
            used_fvg_keys.clear()
            _prev_date = _cur_date

        # ── 1. Manage open trade (vectorised exit — no bar-by-bar loop) ───
        if active_trade is not None:
            sl_dist = abs(active_trade.entry_price - active_trade.sl)

            # intraday_only: cap exit search to end of the entry day
            if day_end_bars is not None:
                eod_bar = int(day_end_bars[active_trade.entry_ltf_bar])
                if i > eod_bar:
                    # Entry was on the last bar of the day — close immediately at
                    # that bar's close (avoids crossing into next session).
                    ep  = float(ltf_cls[active_trade.entry_ltf_bar])
                    pnl = (ep - active_trade.entry_price if active_trade.direction == "bull"
                           else active_trade.entry_price - ep)
                    active_trade.exit_price  = ep
                    active_trade.exit_time   = str(ltf_times[active_trade.entry_ltf_bar])
                    active_trade.result      = "timeout"
                    active_trade.r_multiple  = pnl / sl_dist
                    result.trades.append(active_trade)
                    active_trade = None; active_trade_bar = -1
                    in_fvg_since = -1;   current_fvg_key  = None
                    continue  # process bar i normally (may open a new trade)
                effective_max_bars = min(max_bars_in_trade, eod_bar - i + 1)
            else:
                effective_max_bars = max_bars_in_trade

            exit_bar, exit_price, outcome = _find_exit(
                ltf_lows, ltf_highs, ltf_cls,
                from_bar=i,
                sl=active_trade.sl,
                tp=active_trade.tp,
                direction=active_trade.direction,
                max_bars=effective_max_bars,
            )
            active_trade.exit_price = exit_price
            active_trade.result     = outcome
            if outcome == "win":
                active_trade.r_multiple = (
                    (active_trade.tp - active_trade.entry_price)
                    if active_trade.direction == "bull"
                    else (active_trade.entry_price - active_trade.tp)
                ) / sl_dist
            elif outcome == "loss":
                active_trade.r_multiple = -1.0
            else:  # timeout
                pnl = (
                    (exit_price - active_trade.entry_price)
                    if active_trade.direction == "bull"
                    else (active_trade.entry_price - exit_price)
                )
                active_trade.r_multiple = pnl / sl_dist

            active_trade.exit_time = str(ltf_times[exit_bar])
            result.trades.append(active_trade)
            active_trade     = None
            active_trade_bar = -1
            in_fvg_since     = -1
            current_fvg_key  = None
            i = exit_bar + 1   # jump past the exit bar
            continue

        # ── 2. Update HTF analysis (only when a new HTF bar closes) ──────
        # O(log n) lookup: rightmost HTF bar with time_key <= current LTF time
        htf_pos = int(np.searchsorted(htf_times, t, side="right")) - 1
        if htf_pos < 0:
            continue

        if htf_pos != prev_htf_pos:
            htf_start  = max(0, htf_pos + 1 - params.htf_window_bars)
            htf_view   = htf.iloc[htf_start : htf_pos + 1].reset_index(drop=True)
            htf_swings = find_swings(htf_view, params.swing_lookback)
            htf_fvgs   = detect_fvg(htf_view, params.fvg_min_width_pct)
            vp_edges, vp_vols = compute_volume_profile(htf_view)

            htf_bos    = []
            tp         = params.htf_trend_params
            per_method = []

            # Pre-compute full-history slice once (shared by kd trend + SL fallback)
            _need_kd = "kd" in params.htf_trend_methods or params.kd_sl_fallback
            _htf_full = htf.iloc[: htf_pos + 1].reset_index(drop=True) if _need_kd else None

            for method in params.htf_trend_methods:
                if method == "bos_choch":
                    bos = detect_bos_choch(htf_view, params.swing_lookback,
                                           trend_window=params.htf_window_bars)
                    htf_bos = bos
                    per_method.append(determine_trend(bos, params.bos_count))
                elif method == "kd":
                    per_method.append(kd_trend(
                        _htf_full,
                        fast=tp.get("kd_fast", 25),
                        slow=tp.get("kd_slow", 90),
                        window=tp.get("kd_window", 10),
                        flat_threshold=tp.get("kd_flat_threshold", 0.0),
                        atr_threshold=tp.get("kd_atr_threshold", 0.0),
                        atr_period=tp.get("kd_atr_period", 14),
                        smooth=tp.get("kd_smooth", 3),
                        min_bars=tp.get("kd_min_bars", 3),
                    ))

            # All methods must agree on the same direction.
            if per_method and all(t == per_method[0] for t in per_method) and per_method[0] is not None:
                trend = per_method[0]
            else:
                trend = None

            # KD slow-channel boundaries for optional SL/TP fallback
            if _need_kd:
                _kd_df = compute_kd(
                    _htf_full,
                    fast=tp.get("kd_fast", 25),
                    slow=tp.get("kd_slow", 90),
                    atr_period=tp.get("kd_atr_period", 14),
                )
                htf_kd_lo2 = float(_kd_df["lo2"].iloc[-1])
                htf_kd_up2 = float(_kd_df["up2"].iloc[-1])
            else:
                htf_kd_lo2 = htf_kd_up2 = float("nan")

            prev_htf_pos = htf_pos

        if trend is None:
            in_fvg_since    = -1
            current_fvg_key = None
            i += 1
            continue

        # Skip bear setups when long-only mode is active
        if trend == "bear" and not params.allow_short:
            if current_fvg_key is not None:
                in_fvg_since    = -1
                current_fvg_key = None
            i += 1
            continue

        # ── 3. Check if wick has entered an active HTF FVG ───────────────
        # Bull FVG: price pulls back from above — use bar low (wick) to detect touch.
        # Bear FVG: price bounces from below   — use bar high (wick) to detect touch.
        wick = bar_lo if trend == "bull" else bar_hi
        htf_view_last = len(htf_view) - 1

        # Rejection log only: capture counter-trend FVG touches that are
        # silently excluded by the direction filter in the trade path below.
        if rejection_log is not None and _rlog_in_window(t):
            for _g in htf_fvgs:
                if (not _g["filled"]
                        and _g["direction"] != trend
                        and (htf_view_last - _g["idx"]) <= params.fvg_max_age_bars
                        and (wick <= _g["top"] if trend == "bull" else wick >= _g["bottom"])
                        and (round(_g["bottom"], 4), round(_g["top"], 4)) not in used_fvg_keys):
                    rejection_log.append({
                        "touch_time": t,
                        "fvg_bottom": _g["bottom"],
                        "fvg_top":    _g["top"],
                        "direction":  _g["direction"],
                        "trend":      trend,
                        "wick":       round(wick, 4),
                        "depth_time": None,
                        "depth":      None,
                        "outcome":    "direction_mismatch",
                        "detail":     f"FVG {_g['direction']} vs trend {trend}",
                    })

        in_zone = [
            g for g in htf_fvgs
            if not g["filled"]
            and g["direction"] == trend
            and (htf_view_last - g["idx"]) <= params.fvg_max_age_bars
            and (wick <= g["top"] if trend == "bull" else wick >= g["bottom"])
            and (round(g["bottom"], 4), round(g["top"], 4)) not in used_fvg_keys
        ]
        if not in_zone:
            if current_fvg_key is not None:
                if rejection_log is not None:
                    _rlog_finalize("depth_never_reached" if not _rlog_depth_ok else "ltf_confirmation")
                in_fvg_since    = -1
                current_fvg_key = None
            i += 1
            continue

        # Pick the FVG whose midpoint is nearest to the wick price
        fvg     = min(in_zone, key=lambda g: abs(wick - (g["top"] + g["bottom"]) / 2))
        # Identity by price zone (stable even as the HTF window shifts forward)
        fvg_key = (round(fvg["bottom"], 4), round(fvg["top"], 4))
        if fvg_key != current_fvg_key:
            if rejection_log is not None:
                _rlog_finalize("depth_never_reached" if not _rlog_depth_ok else "ltf_confirmation")
                if _rlog_in_window(t):
                    _rlog_event = {
                        "touch_time": t,
                        "fvg_bottom": fvg["bottom"],
                        "fvg_top":    fvg["top"],
                        "direction":  fvg["direction"],
                        "trend":      trend,
                        "wick":       round(wick, 4),
                    }
            current_fvg_key = fvg_key
            in_fvg_since    = i

        # ── 4. Entry depth check (wick-based) ────────────────────────────
        depth = fvg_entry_depth(fvg, wick)
        if depth < params.fvg_entry_depth_pct:
            i += 1
            continue
        if rejection_log is not None and _rlog_event is not None and not _rlog_depth_ok:
            _rlog_depth_ok = True
            _rlog_event["depth_time"] = t
            _rlog_event["depth"]      = round(depth, 3)

        # ── 4a. Over-refilling guard ──────────────────────────────────────
        # Entry close must not punch through the far side of the FVG;
        # if it does, price has over-filled the gap, signalling reversal risk.
        if trend == "bull" and bar_cls < fvg["bottom"]:
            if rejection_log is not None:
                _rlog_finalize("over_refill", detail="close below FVG bottom")
            i += 1
            continue
        if trend == "bear" and bar_cls > fvg["top"]:
            if rejection_log is not None:
                _rlog_finalize("over_refill", detail="close above FVG top")
            i += 1
            continue

        # ── 5a. LVN overlap filter ────────────────────────────────────────
        if params.require_lvn_overlap:
            if not fvg_overlaps_lvn(fvg, vp_edges, vp_vols, params.lvn_threshold):
                if rejection_log is not None:
                    _rlog_finalize("lvn_filter",
                                   detail=f"zone vol >= {params.lvn_threshold:.0%} of max")
                i += 1
                continue

        # ── 5b. Displacement candle filter ────────────────────────────────
        if params.displacement_required:
            if not is_displacement_candle(htf_view, fvg["idx"], params.displacement_atr_mult,
                                           params.displacement_body_ratio, params.displacement_lookback):
                if rejection_log is not None:
                    _rlog_finalize("displacement_filter",
                                   detail=f"middle candle not displacement "
                                          f"(atr_mult={params.displacement_atr_mult}, "
                                          f"body_ratio={params.displacement_body_ratio})")
                i += 1
                continue

        # ── 6. Confirmation ───────────────────────────────────────────────
        if params.require_ltf_confirmation:
            cur_win     = max(0, i - ltf_per_htf)
            lo          = bisect.bisect_left(_ltf_precomp_idxs, cur_win)
            hi          = bisect.bisect_right(_ltf_precomp_idxs, i)
            ltf_bos_sig = [dict(s, idx=s["idx"] - cur_win)
                           for s in _ltf_precomp[lo:hi]]
            anchor      = max(0, in_fvg_since - ltf_per_htf)
            rel_anchor  = max(0, anchor - cur_win)
            if not check_ltf_confirmation(ltf_bos_sig, trend, after_idx=rel_anchor):
                i += 1
                continue

        # ── 7. SL / TP levels ─────────────────────────────────────────────
        # Primary: HTF swing-based anchors.
        # Fallback (kd_sl_fallback=True): KD slow-channel lo2/up2 when no
        # swing is available.  The same fallback is retried at step 8 when
        # the swing SL exists but exceeds max_sl_pct.
        lows_sw  = [s for s in htf_swings if s["kind"] == "low"]
        highs_sw = [s for s in htf_swings if s["kind"] == "high"]

        if trend == "bull":
            sl_candidates = [s for s in lows_sw  if s["price"] < bar_cls]
            tp_candidates = [s for s in highs_sw if s["price"] > bar_cls]

            if sl_candidates:
                sl_price = sl_candidates[-1]["price"] * (1.0 - params.sl_buffer_pct)
            elif (params.kd_sl_fallback
                  and not np.isnan(htf_kd_lo2) and htf_kd_lo2 < bar_cls):
                sl_price = htf_kd_lo2 * (1.0 - params.sl_buffer_pct)
            else:
                if rejection_log is not None:
                    _rlog_finalize("no_sl_tp", detail="no swing low for SL")
                i += 1
                continue

            if tp_candidates:
                tp_price = min(s["price"] for s in tp_candidates)
            elif (params.kd_sl_fallback
                  and not np.isnan(htf_kd_up2) and htf_kd_up2 > bar_cls):
                tp_price = htf_kd_up2
            else:
                if rejection_log is not None:
                    _rlog_finalize("no_sl_tp", detail="no swing high for TP")
                i += 1
                continue

            sl_dist = bar_cls - sl_price
            tp_dist = tp_price - bar_cls

        else:  # bear
            sl_candidates = [s for s in highs_sw if s["price"] > bar_cls]
            tp_candidates = [s for s in lows_sw  if s["price"] < bar_cls]

            if sl_candidates:
                sl_price = sl_candidates[-1]["price"] * (1.0 + params.sl_buffer_pct)
            elif (params.kd_sl_fallback
                  and not np.isnan(htf_kd_up2) and htf_kd_up2 > bar_cls):
                sl_price = htf_kd_up2 * (1.0 + params.sl_buffer_pct)
            else:
                if rejection_log is not None:
                    _rlog_finalize("no_sl_tp", detail="no swing high for SL")
                i += 1
                continue

            if tp_candidates:
                tp_price = max(s["price"] for s in tp_candidates)
            elif (params.kd_sl_fallback
                  and not np.isnan(htf_kd_lo2) and htf_kd_lo2 < bar_cls):
                tp_price = htf_kd_lo2
            else:
                if rejection_log is not None:
                    _rlog_finalize("no_sl_tp", detail="no swing low for TP")
                i += 1
                continue

            sl_dist = sl_price - bar_cls
            tp_dist = bar_cls  - tp_price

        if sl_dist <= 0 or tp_dist <= 0:
            if rejection_log is not None:
                _rlog_finalize("no_sl_tp", detail="sl_dist or tp_dist <= 0")
            i += 1
            continue

        # ── 8. Risk / SL-size filters ─────────────────────────────────────
        # When swing SL is too wide, try KD slow-channel lo2/up2 as a
        # tighter fallback before hard-rejecting the setup.
        if (sl_dist / bar_cls) > params.max_sl_pct and params.kd_sl_fallback:
            if trend == "bull" and not np.isnan(htf_kd_lo2) and htf_kd_lo2 < bar_cls:
                _kd_sl      = htf_kd_lo2 * (1.0 - params.sl_buffer_pct)
                _kd_sl_dist = bar_cls - _kd_sl
                if 0 < _kd_sl_dist / bar_cls <= params.max_sl_pct:
                    sl_price = _kd_sl
                    sl_dist  = _kd_sl_dist
            elif trend == "bear" and not np.isnan(htf_kd_up2) and htf_kd_up2 > bar_cls:
                _kd_sl      = htf_kd_up2 * (1.0 + params.sl_buffer_pct)
                _kd_sl_dist = _kd_sl - bar_cls
                if 0 < _kd_sl_dist / bar_cls <= params.max_sl_pct:
                    sl_price = _kd_sl
                    sl_dist  = _kd_sl_dist

        if (sl_dist / bar_cls) > params.max_sl_pct:
            if rejection_log is not None:
                _rlog_finalize("max_sl_pct",
                               detail=f"sl={sl_dist/bar_cls:.4f} > max={params.max_sl_pct}")
            i += 1
            continue
        rr = tp_dist / sl_dist
        if rr < params.min_rr:
            if rejection_log is not None:
                _rlog_finalize("min_rr", detail=f"rr={rr:.2f} < min={params.min_rr}")
            i += 1
            continue

        # ── 9. Open trade ─────────────────────────────────────────────────
        used_fvg_keys.add(fvg_key)   # prevent re-entry into the same FVG zone
        active_trade = Trade(
            direction     = trend,
            entry_price   = bar_cls,
            sl            = sl_price,
            tp            = tp_price,
            planned_rr    = rr,
            entry_time    = t,
            entry_ltf_bar = i,
            fvg_top       = fvg["top"],
            fvg_bottom    = fvg["bottom"],
        )
        if rejection_log is not None:
            _rlog_finalize("entered", trade_id=active_trade.trade_id)
        active_trade_bar = i
        in_fvg_since     = -1
        current_fvg_key  = None
        i += 1

    return result

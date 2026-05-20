"""Multi-timeframe SMC backtesting engine.

Entry logic:
  1. HTF: detect trend via consecutive BOS/CHoCH; mark active FVG zones.
  2. LTF: when wick (low for bull / high for bear) touches an HTF FVG at the
     required depth, look for a LTF CHoCH + BOS confirmation in trend direction.
     CHoCH + BOS must occur AFTER the wick first enters the FVG.
  3. Enter at close of the confirming bar.
  4. SL = last HTF swing low (bull) / swing high (bear) ± buffer.
  5. TP = nearest HTF swing high (bull) / swing low (bear).
  6. Trade is closed at first SL or TP hit, or after max_bars_in_trade.
"""

from __future__ import annotations

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
)

_HTF_WINDOW  = 200  # HTF bars used for each analysis snapshot (O(n²) cap)
_LTF_WINDOW  = 120  # LTF bars used for swing/BOS detection
_LTF_PRE_FVG = 0    # CHoCH+BOS must start AFTER wick enters FVG
_WARMUP      = 40   # skip the first N LTF bars while indicators warm up


# ── Parameter set ─────────────────────────────────────────────────────────────

@dataclass
class BacktestParams:
    trend_tf:             str   = "60m"
    entry_tf:             str   = "15m"
    swing_lookback:       int   = 2
    bos_count:            int   = 1      # consecutive BOS to confirm trend
    fvg_min_width_pct:    float = 0.002  # min FVG size as fraction of price
    fvg_entry_depth_pct:  float = 0.30   # 0=edge, 1=far side of zone
    displacement_required:    bool  = False  # FVG must be from a displacement candle
    displacement_atr_mult:    float = 1.5   # middle candle range > mult × neighbour max
    displacement_body_ratio:  float = 0.5   # body / range of middle candle (0=doji, 1=marubozu)
    require_lvn_overlap:  bool  = False  # FVG must sit in a Low Volume Node
    lvn_threshold:        float = 0.30   # LVN: zone vol < threshold × max bin vol
    sl_buffer_pct:        float = 0.001  # extra % added beyond the swing level
    max_sl_pct:           float = 0.010  # skip trade if SL > this % of entry
    min_rr:               float = 2.0

    def label(self) -> str:
        d = f"D{self.displacement_atr_mult:.1f}b{self.displacement_body_ratio:.1f}" if self.displacement_required else "d"
        return (
            f"{self.trend_tf}/{self.entry_tf} lb{self.swing_lookback}"
            f" bos{self.bos_count} w{self.fvg_min_width_pct:.3f}"
            f" dp{self.fvg_entry_depth_pct:.1f} {d}"
            f" sl{self.sl_buffer_pct:.3f} msl{self.max_sl_pct:.3f}"
            f" rr{self.min_rr:.1f}"
        )

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}   # type: ignore[attr-defined]


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

    def summary_dict(self) -> dict:
        return {
            "n_trades":        self.n_trades,
            "win_rate":        round(self.win_rate, 3),
            "total_r":         round(self.total_r, 2),
            "avg_r":           round(self.avg_r, 3),
            "profit_factor":   round(self.profit_factor, 2),
            "max_drawdown_r":  round(self.max_drawdown_r, 2),
            "max_loss_r":      round(self.max_loss_r, 2),
            **self.params.to_dict(),
        }


# ── Engine ────────────────────────────────────────────────────────────────────

def run_backtest(
    htf: pd.DataFrame,
    ltf: pd.DataFrame,
    params: BacktestParams,
    max_bars_in_trade: int = 200,
) -> BacktestResult:
    """Run a single SMC multi-TF backtest.

    Args:
        htf:               High-timeframe OHLCV (trend / structure TF)
        ltf:               Low-timeframe OHLCV (entry TF)
        params:            Strategy parameters
        max_bars_in_trade: Force-close after this many LTF bars (timeout)

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

    # HTF cache — recomputed only when a new HTF bar closes
    prev_htf_pos:  int             = -1
    htf_view:      pd.DataFrame    = htf.iloc[0:0]
    htf_swings:    list[dict]      = []
    htf_bos:       list[dict]      = []
    htf_fvgs:      list[dict]      = []
    trend:         Optional[str]   = None
    vp_edges:      object          = None   # volume profile edges (np.ndarray | None)
    vp_vols:       object          = None   # volume profile bins  (np.ndarray | None)

    for i in range(_WARMUP, n_ltf):
        t       = ltf_times[i]
        bar_lo  = ltf_lows[i]
        bar_hi  = ltf_highs[i]
        bar_cls = ltf_cls[i]

        # ── 1. Manage open trade ──────────────────────────────────────────
        if active_trade is not None:
            sl_dist = abs(active_trade.entry_price - active_trade.sl)
            closed  = False

            if active_trade.direction == "bull":
                if bar_lo <= active_trade.sl:
                    active_trade.exit_price = active_trade.sl
                    active_trade.result     = "loss"
                    active_trade.r_multiple = -1.0
                    closed = True
                elif bar_hi >= active_trade.tp:
                    active_trade.exit_price = active_trade.tp
                    active_trade.result     = "win"
                    active_trade.r_multiple = (
                        active_trade.tp - active_trade.entry_price
                    ) / sl_dist
                    closed = True
            else:  # bear
                if bar_hi >= active_trade.sl:
                    active_trade.exit_price = active_trade.sl
                    active_trade.result     = "loss"
                    active_trade.r_multiple = -1.0
                    closed = True
                elif bar_lo <= active_trade.tp:
                    active_trade.exit_price = active_trade.tp
                    active_trade.result     = "win"
                    active_trade.r_multiple = (
                        active_trade.entry_price - active_trade.tp
                    ) / sl_dist
                    closed = True

            if not closed and (i - active_trade_bar) >= max_bars_in_trade:
                active_trade.exit_price = bar_cls
                active_trade.result     = "timeout"
                pnl = (
                    (bar_cls - active_trade.entry_price)
                    if active_trade.direction == "bull"
                    else (active_trade.entry_price - bar_cls)
                )
                active_trade.r_multiple = pnl / sl_dist
                closed = True

            if closed:
                active_trade.exit_time = t
                result.trades.append(active_trade)
                active_trade     = None
                active_trade_bar = -1
                in_fvg_since     = -1
                current_fvg_key  = None
            continue  # one trade at a time — skip setup scanning

        # ── 2. Update HTF analysis (only when a new HTF bar closes) ──────
        # O(log n) lookup: rightmost HTF bar with time_key <= current LTF time
        htf_pos = int(np.searchsorted(htf_times, t, side="right")) - 1
        if htf_pos < 0:
            continue

        if htf_pos != prev_htf_pos:
            htf_start  = max(0, htf_pos + 1 - _HTF_WINDOW)
            htf_view   = htf.iloc[htf_start : htf_pos + 1].reset_index(drop=True)
            htf_swings          = find_swings(htf_view, params.swing_lookback)
            htf_bos             = detect_bos_choch(htf_view, params.swing_lookback)
            htf_fvgs            = detect_fvg(htf_view, params.fvg_min_width_pct)
            trend               = determine_trend(htf_bos, params.bos_count)
            vp_edges, vp_vols   = compute_volume_profile(htf_view)
            prev_htf_pos        = htf_pos

        if trend is None:
            in_fvg_since    = -1
            current_fvg_key = None
            continue

        # ── 3. Check if wick has entered an active HTF FVG ───────────────
        # Bull FVG: price pulls back from above — use bar low (wick) to detect touch.
        # Bear FVG: price bounces from below   — use bar high (wick) to detect touch.
        wick = bar_lo if trend == "bull" else bar_hi
        in_zone = [
            g for g in htf_fvgs
            if not g["filled"]
            and g["direction"] == trend
            and (wick <= g["top"] if trend == "bull" else wick >= g["bottom"])
        ]
        if not in_zone:
            if current_fvg_key is not None:
                in_fvg_since    = -1
                current_fvg_key = None
            continue

        # Pick the FVG whose midpoint is nearest to the wick price
        fvg     = min(in_zone, key=lambda g: abs(wick - (g["top"] + g["bottom"]) / 2))
        # Identity by price zone (stable even as the HTF window shifts forward)
        fvg_key = (round(fvg["bottom"], 4), round(fvg["top"], 4))
        if fvg_key != current_fvg_key:
            current_fvg_key = fvg_key
            in_fvg_since    = i

        # ── 4. Entry depth check (wick-based) ────────────────────────────
        depth = fvg_entry_depth(fvg, wick)
        if depth < params.fvg_entry_depth_pct:
            continue

        # ── 5a. LVN overlap filter ────────────────────────────────────────
        if params.require_lvn_overlap:
            if not fvg_overlaps_lvn(fvg, vp_edges, vp_vols, params.lvn_threshold):
                continue

        # ── 5b. Displacement candle filter ────────────────────────────────
        if params.displacement_required:
            # htf_view is already current (set in step 2)
            if not is_displacement_candle(htf_view, fvg["idx"], params.displacement_atr_mult, params.displacement_body_ratio):
                continue

        # ── 6. LTF CHoCH + BOS confirmation ──────────────────────────────
        # Allow the pattern to start _LTF_PRE_FVG bars before FVG entry —
        # traders spot the LTF reversal as price approaches the zone.
        win_start   = max(0, i - _LTF_WINDOW)
        ltf_window  = ltf.iloc[win_start : i + 1].reset_index(drop=True)
        ltf_bos_sig = detect_bos_choch(ltf_window, lookback=1)
        anchor      = max(0, in_fvg_since - _LTF_PRE_FVG)
        rel_anchor  = max(0, anchor - win_start)
        if not check_ltf_confirmation(ltf_bos_sig, trend, after_idx=rel_anchor):
            continue

        # ── 7. SL / TP levels ─────────────────────────────────────────────
        lows_sw  = [s for s in htf_swings if s["kind"] == "low"]
        highs_sw = [s for s in htf_swings if s["kind"] == "high"]

        if trend == "bull":
            sl_candidates = [s for s in lows_sw  if s["price"] < bar_cls]
            tp_candidates = [s for s in highs_sw if s["price"] > bar_cls]
            if not sl_candidates or not tp_candidates:
                continue
            sl_price = sl_candidates[-1]["price"] * (1.0 - params.sl_buffer_pct)
            tp_price = min(s["price"] for s in tp_candidates)
            sl_dist  = bar_cls - sl_price
            tp_dist  = tp_price - bar_cls
        else:  # bear
            sl_candidates = [s for s in highs_sw if s["price"] > bar_cls]
            tp_candidates = [s for s in lows_sw  if s["price"] < bar_cls]
            if not sl_candidates or not tp_candidates:
                continue
            sl_price = sl_candidates[-1]["price"] * (1.0 + params.sl_buffer_pct)
            tp_price = max(s["price"] for s in tp_candidates)
            sl_dist  = sl_price - bar_cls
            tp_dist  = bar_cls  - tp_price

        if sl_dist <= 0 or tp_dist <= 0:
            continue

        # ── 8. Risk / SL-size filters ─────────────────────────────────────
        if (sl_dist / bar_cls) > params.max_sl_pct:
            continue
        rr = tp_dist / sl_dist
        if rr < params.min_rr:
            continue

        # ── 9. Open trade ─────────────────────────────────────────────────
        active_trade = Trade(
            direction   = trend,
            entry_price = bar_cls,
            sl          = sl_price,
            tp          = tp_price,
            planned_rr  = rr,
            entry_time  = t,
        )
        active_trade_bar = i
        in_fvg_since     = -1
        current_fvg      = None

    return result

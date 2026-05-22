"""BaseStrategy ABC — pluggable strategy interface.

Any strategy that can be plugged into the backtest engine must implement
this interface. The existing SMC functions in strategy/smc/ are accessed
through SMCStrategy, which wraps them behind this ABC.

Minimal contract:
  - detect_zones(htf_df)  → list of zone dicts (FVG, OB, etc.)
  - generate_signals(htf_df, ltf_df, config) → list of signal dicts

Zone dict schema (at minimum):
    {
      "type":      str,     # "fvg" | "ob" | "vp_lvn" | ...
      "direction": str,     # "bull" | "bear"
      "top":       float,
      "bottom":    float,
      "filled":    bool,
      "time":      str,     # ISO timestamp of the zone candle
    }

Signal dict schema (at minimum):
    {
      "direction":   str,    # "bull" | "bear"
      "entry_price": float,
      "sl":          float,
      "tp":          float,
      "planned_rr":  float,
      "entry_time":  str,    # ISO timestamp
      "zone_type":   str,    # which zone type triggered this entry
    }
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class BaseStrategy(ABC):
    """Abstract base for all SMC (and future) backtest strategies.

    Sub-classes receive a config dict at construction. Use it to store
    any strategy-specific parameters instead of adding constructor args.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config: dict[str, Any] = config or {}

    @abstractmethod
    def detect_zones(self, htf_df: pd.DataFrame) -> list[dict]:
        """Identify tradeable zones from the high-timeframe DataFrame.

        Args:
            htf_df: OHLCV DataFrame (high-timeframe, e.g. 4h or 1d).

        Returns:
            List of zone dicts, each with at minimum the keys documented
            in the module docstring.
        """

    @abstractmethod
    def generate_signals(
        self,
        htf_df: pd.DataFrame,
        ltf_df: pd.DataFrame,
    ) -> list[dict]:
        """Generate trade signals from HTF zones + LTF price action.

        Args:
            htf_df: High-timeframe OHLCV.
            ltf_df: Low-timeframe OHLCV (entry timeframe).

        Returns:
            List of signal dicts, each with at minimum the keys documented
            in the module docstring.
        """


class SMCStrategy(BaseStrategy):
    """Wraps the existing strategy/smc/ functions behind the BaseStrategy ABC.

    This is a thin adapter — it delegates to the same detect_fvg, detect_bos_choch,
    determine_trend, etc. functions used by backtest/engine.py.

    Config keys (mirrors BacktestParams fields):
        swing_lookback, bos_count, fvg_min_width_pct, fvg_entry_depth_pct,
        require_ltf_confirmation, sl_buffer_pct, max_sl_pct, min_rr
    """

    def detect_zones(self, htf_df: pd.DataFrame) -> list[dict]:
        from strategy.smc import detect_fvg, find_swings, detect_bos_choch, determine_trend

        min_width = self.config.get("fvg_min_width_pct", 0.002)
        fvgs = detect_fvg(htf_df, min_width)

        bos_count = self.config.get("bos_count", 1)
        lookback  = self.config.get("swing_lookback", 2)
        bos       = detect_bos_choch(htf_df, lookback)
        trend     = determine_trend(bos, bos_count)

        # Tag each zone with metadata useful for downstream filtering
        for z in fvgs:
            z["zone_trend"] = trend
            idx = z.get("idx")
            if idx is not None and idx < len(htf_df):
                z["time"] = str(htf_df["time_key"].iloc[idx])

        return fvgs

    def generate_signals(
        self,
        htf_df: pd.DataFrame,
        ltf_df: pd.DataFrame,
    ) -> list[dict]:
        """Replay the SMC engine logic and return completed signals.

        For full backtest use engine.run_backtest() directly. This method
        is intended for live signal generation or simplified analysis where
        you want signal dicts without trade management overhead.
        """
        from strategy.smc import (
            find_swings, detect_bos_choch, determine_trend, fvg_entry_depth,
        )
        zones = self.detect_zones(htf_df)
        active_zones = [z for z in zones if not z.get("filled") and z.get("zone_trend")]

        min_width   = self.config.get("fvg_min_width_pct", 0.002)
        depth_req   = self.config.get("fvg_entry_depth_pct", 0.10)
        sl_buf      = self.config.get("sl_buffer_pct", 0.001)
        max_sl      = self.config.get("max_sl_pct", 0.005)
        min_rr      = self.config.get("min_rr", 2.0)
        lookback    = self.config.get("swing_lookback", 2)

        swings  = find_swings(htf_df, lookback)
        lows_sw = [s for s in swings if s["kind"] == "low"]
        highs_sw = [s for s in swings if s["kind"] == "high"]

        bos   = detect_bos_choch(htf_df, lookback)
        trend = determine_trend(bos, self.config.get("bos_count", 1))
        if trend is None:
            return []

        signals: list[dict] = []
        for _, bar in ltf_df.iterrows():
            bar_cls = float(bar["close"])
            wick    = float(bar["low"]) if trend == "bull" else float(bar["high"])

            matching = [
                z for z in active_zones
                if z["direction"] == trend
                and (wick <= z["top"] if trend == "bull" else wick >= z["bottom"])
                and fvg_entry_depth(z, wick) >= depth_req
            ]
            if not matching:
                continue

            zone = min(matching, key=lambda z: abs(wick - (z["top"] + z["bottom"]) / 2))

            if trend == "bull":
                sl_cands = [s for s in lows_sw  if s["price"] < bar_cls]
                tp_cands = [s for s in highs_sw if s["price"] > bar_cls]
                if not sl_cands or not tp_cands:
                    continue
                sl = sl_cands[-1]["price"] * (1 - sl_buf)
                tp = min(s["price"] for s in tp_cands)
            else:
                sl_cands = [s for s in highs_sw if s["price"] > bar_cls]
                tp_cands = [s for s in lows_sw  if s["price"] < bar_cls]
                if not sl_cands or not tp_cands:
                    continue
                sl = sl_cands[-1]["price"] * (1 + sl_buf)
                tp = max(s["price"] for s in tp_cands)

            sl_dist = abs(bar_cls - sl)
            tp_dist = abs(tp - bar_cls)
            if sl_dist <= 0 or tp_dist <= 0:
                continue
            if sl_dist / bar_cls > max_sl:
                continue
            rr = tp_dist / sl_dist
            if rr < min_rr:
                continue

            signals.append({
                "direction":   trend,
                "entry_price": bar_cls,
                "sl":          sl,
                "tp":          tp,
                "planned_rr":  round(rr, 2),
                "entry_time":  str(bar["time_key"]),
                "zone_type":   "fvg",
                "zone_top":    zone["top"],
                "zone_bottom": zone["bottom"],
            })

        return signals

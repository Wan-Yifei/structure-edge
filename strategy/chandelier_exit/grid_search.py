"""Grid search over (ATR period, multiplier) chandelier-exit combos.

Reuses backtest.engine.Trade/BacktestResult to aggregate each combo's metrics
(win_rate, total_r, profit_factor, max_drawdown_r, sharpe, sortino, ...) --
wraps each simulated exit as a synthetic Trade so those formulas (already
correct and tested in backtest/engine.py) don't get re-derived here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.engine import BacktestParams, BacktestResult, Trade
from strategy.chandelier_exit.atr import wilder_atr
from strategy.chandelier_exit.chandelier import rolling_extremes, simulate_chandelier_exit
from strategy.chandelier_exit.entries import EntryContext


def _would_have_hit_tp(
    highs: np.ndarray, lows: np.ndarray, direction: str, orig_tp: float,
    scan_start: int, scan_end: int,
) -> bool:
    """Whether price reaches the original engine's TP within [scan_start, scan_end]."""
    if scan_start > scan_end:
        return False
    if direction == "bull":
        return bool(np.any(highs[scan_start:scan_end + 1] >= orig_tp))
    return bool(np.any(lows[scan_start:scan_end + 1] <= orig_tp))


def run_grid_search(
    entries: list[EntryContext], ltf_klines: pd.DataFrame, entry_params: BacktestParams,
    atr_periods: list[int], multipliers: list[float], max_bars_in_trade: int = 200,
) -> pd.DataFrame:
    """Simulate every entry under every (period, multiplier) combo.

    Returns one row per combo with aggregate metrics plus the informational
    (not used for ranking) noise_stopout_rate diagnostic: the fraction of
    trades where the chandelier exit underperformed the original TP but price
    would still have reached that TP had the position stayed open.
    """
    highs  = ltf_klines["high"].to_numpy(dtype=float)
    lows   = ltf_klines["low"].to_numpy(dtype=float)
    closes = ltf_klines["close"].to_numpy(dtype=float)
    times  = ltf_klines["time_key"].to_numpy()
    n_ltf  = len(highs)

    rows: list[dict] = []
    for period in atr_periods:
        atr     = wilder_atr(highs, lows, closes, period)
        hh, ll  = rolling_extremes(highs, lows, period)

        for mult in multipliers:
            trades: list[Trade] = []
            noise_n  = 0
            skipped  = 0

            for e in entries:
                risk_unit = abs(e.entry_price - e.orig_sl)
                if risk_unit <= 0:
                    skipped += 1
                    continue

                res = simulate_chandelier_exit(
                    highs, lows, closes, times, atr, hh, ll,
                    e.entry_ltf_bar, e.entry_price, e.direction,
                    period, mult, risk_unit, max_bars=max_bars_in_trade,
                )
                if res is None:   # insufficient ATR/HH/LL warmup at entry
                    skipped += 1
                    continue

                trades.append(Trade(
                    direction=e.direction, entry_price=e.entry_price,
                    sl=e.orig_sl, tp=e.orig_tp, planned_rr=0.0,
                    entry_time=e.entry_time, exit_time=res.exit_time,
                    exit_price=res.exit_price,
                    result="win" if res.r_multiple > 0 else "loss",
                    r_multiple=res.r_multiple, entry_ltf_bar=e.entry_ltf_bar,
                ))

                tp_r = (
                    (e.orig_tp - e.entry_price) if e.direction == "bull"
                    else (e.entry_price - e.orig_tp)
                ) / risk_unit
                if res.r_multiple < tp_r:
                    scan_end = min(e.entry_ltf_bar + max_bars_in_trade, n_ltf - 1)
                    if _would_have_hit_tp(highs, lows, e.direction, e.orig_tp,
                                           res.exit_bar + 1, scan_end):
                        noise_n += 1

            if not trades:
                continue  # every entry skipped (e.g. all before ATR warmup) for this combo

            combo_result = BacktestResult(params=entry_params, trades=trades)
            row = combo_result.summary_dict()
            row["atr_period"]         = period
            row["multiplier"]         = mult
            row["n_skipped_warmup"]   = skipped
            row["noise_stopout_n"]    = noise_n
            row["noise_stopout_rate"] = round(noise_n / len(trades), 3)
            rows.append(row)

    return pd.DataFrame(rows)

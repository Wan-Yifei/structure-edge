"""Produce a batch of real trade entries by re-running the existing smc_v2
backtest engine, unmodified -- reuses backtest/engine.py's full entry/SL/TP
logic rather than duplicating it. Only the exit method is later swapped out
by chandelier.simulate_chandelier_exit; the entries themselves are identical
to what backtest/run.py would have produced for the same params.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtest.engine import BacktestParams, BacktestResult, run_backtest
from feeds.fetcher import fetch_klines


@dataclass
class EntryContext:
    code:            str
    tf_pair:         tuple[str, str]   # (trend_tf, entry_tf)
    trade_id:        str
    direction:       str               # "bull" | "bear"
    entry_price:     float
    entry_time:      str
    entry_ltf_bar:   int               # positional index into the ltf_df returned alongside
    orig_sl:         float
    orig_tp:         float
    orig_result:     str               # "win" | "loss" | "timeout" from the original engine run
    orig_r_multiple: float
    orig_exit_time:  str


def collect_entries(
    code: str, tf_pair: tuple[str, str], start: str, end: str,
    params: BacktestParams, max_bars_in_trade: int = 200,
) -> tuple[list[EntryContext], BacktestResult, pd.DataFrame]:
    """Fetch klines, run the real engine, and package its trades as EntryContexts.

    Returns (entries, result, ltf_df):
        result  -- the original BacktestResult (static SL/TP baseline;
                   .total_r/.avg_r/.profit_factor/.max_drawdown_r/.summary_dict()
                   are already computed -- reuse them directly, don't recompute).
        ltf_df  -- the SAME DataFrame object `entry_ltf_bar` indexes into.
                   The chandelier simulator must reuse this exact frame, never
                   re-fetch (a re-fetch could shift bar positions).
    """
    htf_tf, ltf_tf = tf_pair
    htf_df = fetch_klines(code=code, ktype=htf_tf, start=start, end=end)
    ltf_df = fetch_klines(code=code, ktype=ltf_tf, start=start, end=end)

    result = run_backtest(htf_df, ltf_df, params, max_bars_in_trade=max_bars_in_trade)

    entries = [
        EntryContext(
            code=code, tf_pair=tf_pair, trade_id=t.trade_id,
            direction=t.direction, entry_price=t.entry_price, entry_time=t.entry_time,
            entry_ltf_bar=t.entry_ltf_bar,
            orig_sl=t.sl, orig_tp=t.tp, orig_result=t.result,
            orig_r_multiple=t.r_multiple, orig_exit_time=t.exit_time,
        )
        for t in result.trades
    ]
    return entries, result, ltf_df

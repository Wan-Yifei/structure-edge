"""Unit tests for backtest/engine.py — BacktestResult metrics and run_backtest."""

import pandas as pd
import numpy as np
import pytest

from backtest.engine import BacktestParams, BacktestResult, Trade, run_backtest


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_trade(result: str, r: float, direction: str = "bull") -> Trade:
    entry = 100.0
    sl    = 99.0
    tp    = 102.0
    return Trade(
        direction=direction, entry_price=entry, sl=sl, tp=tp,
        planned_rr=2.0, entry_time="2025-01-01 09:30:00",
        exit_time="2025-01-01 10:00:00",
        exit_price=tp if result == "win" else sl,
        result=result, r_multiple=r,
    )


def _result_from_trades(trades):
    r = BacktestResult(params=BacktestParams())
    r.trades = trades
    return r


def _trending_klines(n: int, start: float, step: float, tf: str = "60m") -> pd.DataFrame:
    """Monotonically trending OHLCV suitable for HTF structure detection."""
    closes  = [start + i * step for i in range(n)]
    highs   = [c + abs(step) * 0.6 for c in closes]
    lows    = [c - abs(step) * 0.6 for c in closes]
    volumes = [10_000] * n
    times   = pd.date_range("2025-01-02 09:30", periods=n, freq="60min")
    return pd.DataFrame({
        "time_key": times.strftime("%Y-%m-%d %H:%M:%S"),
        "open":     closes, "high": highs, "low": lows,
        "close":    closes, "volume": volumes,
    })


# ── BacktestResult metrics ────────────────────────────────────────────────────

class TestBacktestResultMetrics:
    def test_empty_result(self):
        r = _result_from_trades([])
        assert r.n_trades    == 0
        assert r.win_rate    == 0.0
        assert r.total_r     == 0.0
        assert r.avg_r       == 0.0
        assert r.profit_factor == 0.0
        assert r.max_drawdown_r == 0.0

    def test_all_wins(self):
        trades = [_make_trade("win", 2.0) for _ in range(3)]
        r = _result_from_trades(trades)
        assert r.n_wins    == 3
        assert r.win_rate  == pytest.approx(1.0)
        assert r.total_r   == pytest.approx(6.0)
        assert r.profit_factor == pytest.approx(float("inf"))

    def test_all_losses(self):
        trades = [_make_trade("loss", -1.0) for _ in range(3)]
        r = _result_from_trades(trades)
        assert r.n_wins   == 0
        assert r.win_rate == pytest.approx(0.0)
        assert r.total_r  == pytest.approx(-3.0)
        assert r.profit_factor == pytest.approx(0.0)

    def test_mixed(self):
        trades = [
            _make_trade("win",  2.0),
            _make_trade("loss", -1.0),
            _make_trade("win",  2.0),
        ]
        r = _result_from_trades(trades)
        assert r.win_rate        == pytest.approx(2 / 3)
        assert r.total_r         == pytest.approx(3.0)
        assert r.profit_factor   == pytest.approx(4.0)

    def test_max_drawdown(self):
        trades = [
            _make_trade("win",  2.0),
            _make_trade("loss", -1.0),
            _make_trade("loss", -1.0),
            _make_trade("win",  2.0),
        ]
        r = _result_from_trades(trades)
        # equity: 0 → 2 → 1 → 0 → 2  peak=2, trough=0, dd=2
        assert r.max_drawdown_r == pytest.approx(2.0)

    def test_summary_dict_keys(self):
        r = _result_from_trades([_make_trade("win", 2.0)])
        d = r.summary_dict()
        for key in ("n_trades", "win_rate", "total_r", "avg_r",
                    "profit_factor", "max_drawdown_r", "max_loss_r"):
            assert key in d


# ── run_backtest ──────────────────────────────────────────────────────────────

class TestRunBacktest:
    def test_empty_input_returns_no_trades(self):
        empty = pd.DataFrame(columns=["time_key","open","high","low","close","volume"])
        r = run_backtest(empty, empty, BacktestParams())
        assert r.n_trades == 0

    def test_returns_backtest_result(self):
        htf = _trending_klines(120, start=100.0, step=0.2)
        ltf = _trending_klines(480, start=100.0, step=0.05)
        r = run_backtest(htf, ltf, BacktestParams())
        assert isinstance(r, BacktestResult)

    def test_no_trades_when_rr_too_strict(self):
        htf = _trending_klines(120, start=100.0, step=0.2)
        ltf = _trending_klines(480, start=100.0, step=0.05)
        params = BacktestParams(min_rr=999.0)
        r = run_backtest(htf, ltf, params)
        assert r.n_trades == 0

    def test_trade_fields_populated(self):
        htf = _trending_klines(200, start=50.0, step=0.15)
        ltf = _trending_klines(800, start=50.0, step=0.04)
        params = BacktestParams(min_rr=1.0, max_sl_pct=0.05,
                                fvg_min_width_pct=0.001,
                                fvg_entry_depth_pct=0.05)
        r = run_backtest(htf, ltf, params)
        for t in r.trades:
            assert t.direction in ("bull", "bear")
            assert t.sl > 0
            assert t.tp > 0
            assert t.result in ("win", "loss", "timeout")
            assert isinstance(t.r_multiple, float)

    def test_sl_tp_consistent_with_direction(self):
        htf = _trending_klines(200, start=50.0, step=0.15)
        ltf = _trending_klines(800, start=50.0, step=0.04)
        params = BacktestParams(min_rr=1.0, max_sl_pct=0.05,
                                fvg_min_width_pct=0.001,
                                fvg_entry_depth_pct=0.05)
        r = run_backtest(htf, ltf, params)
        for t in r.trades:
            if t.direction == "bull":
                assert t.sl < t.entry_price < t.tp
            else:
                assert t.tp < t.entry_price < t.sl

    def test_params_label_is_string(self):
        p = BacktestParams()
        assert isinstance(p.label(), str)
        assert len(p.label()) > 0

    def test_params_to_dict_roundtrip(self):
        p = BacktestParams(min_rr=3.0, bos_count=2)
        d = p.to_dict()
        assert d["min_rr"]    == 3.0
        assert d["bos_count"] == 2

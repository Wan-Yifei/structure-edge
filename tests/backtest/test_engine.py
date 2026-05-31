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
                    "profit_factor", "max_drawdown_r", "max_loss_r",
                    "sharpe", "sortino"):
            assert key in d

    def test_summary_dict_sharpe_sortino_types(self):
        trades = [_make_trade("win", 2.0), _make_trade("loss", -1.0),
                  _make_trade("win", 2.0)]
        d = _result_from_trades(trades).summary_dict()
        assert isinstance(d["sharpe"],  float)
        assert isinstance(d["sortino"], float)
        assert d["sharpe"]  > 0
        assert d["sortino"] > 0

    def test_summary_dict_empty_sharpe_zero(self):
        d = _result_from_trades([]).summary_dict()
        assert d["sharpe"]  == 0.0
        assert d["sortino"] == 0.0


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


# ── Gap-fill filter ───────────────────────────────────────────────────────────

def _make_ltf_with_gap(n_pre: int, gap_direction: str, trend: str) -> pd.DataFrame:
    """Build a synthetic 3m LTF kline series with one opening gap.

    gap_direction: "up" (open > prev_close) or "down" (open < prev_close).
    The gap is placed at bar n_pre (0-indexed), preceded by n_pre quiet bars.
    """
    rows = []
    base_price = 100.0
    for i in range(n_pre):
        rows.append({
            "time_key": f"2025-01-02 09:{30 + i * 3:02d}:00",
            "open": base_price, "high": base_price + 0.1,
            "low": base_price - 0.1, "close": base_price, "volume": 1000,
        })
    # Gap bar
    gap_open = base_price * 1.01 if gap_direction == "up" else base_price * 0.99
    rows.append({
        "time_key": f"2025-01-02 09:{30 + n_pre * 3:02d}:00",
        "open": gap_open, "high": gap_open + 0.1,
        "low": gap_open - 0.1, "close": gap_open, "volume": 1000,
    })
    # A few more bars
    for i in range(1, 6):
        p = gap_open + i * 0.05
        rows.append({
            "time_key": f"2025-01-02 09:{30 + (n_pre + i) * 3:02d}:00",
            "open": p, "high": p + 0.1, "low": p - 0.1, "close": p, "volume": 1000,
        })
    return pd.DataFrame(rows)


class TestGapFillFilter:
    """Tests for Step 5c gap-fill filter in run_backtest."""

    def test_default_gap_fill_lookback_is_zero(self):
        p = BacktestParams()
        assert p.gap_fill_lookback == 0
        assert p.gap_fill_min_pct == pytest.approx(0.001)

    def test_gap_fill_params_in_label(self):
        p = BacktestParams(gap_fill_lookback=5)
        assert "gf5" in p.label()

    def test_gap_fill_off_does_not_affect_label(self):
        p = BacktestParams(gap_fill_lookback=0)
        assert "gf" not in p.label()

    def test_gap_fill_params_roundtrip(self):
        p = BacktestParams(gap_fill_lookback=3, gap_fill_min_pct=0.002)
        d = p.to_dict()
        p2 = BacktestParams.from_dict(d)
        assert p2.gap_fill_lookback == 3
        assert p2.gap_fill_min_pct == pytest.approx(0.002)

    def _rejection_log_outcomes(self, htf, ltf, gap_fill_lookback=5) -> list[str]:
        params = BacktestParams(
            min_rr=1.0, max_sl_pct=0.05,
            fvg_min_width_pct=0.001,
            fvg_entry_depth_pct=0.01,
            gap_fill_lookback=gap_fill_lookback,
            gap_fill_min_pct=0.005,
        )
        log: list[dict] = []
        run_backtest(htf, ltf, params, rejection_log=log)
        return [e["outcome"] for e in log]

    def test_gap_fill_filter_disabled_when_lookback_zero(self):
        htf = _trending_klines(200, start=50.0, step=0.15)
        ltf = _trending_klines(800, start=50.0, step=0.04)
        params_no_gf = BacktestParams(
            min_rr=1.0, max_sl_pct=0.05,
            fvg_min_width_pct=0.001, fvg_entry_depth_pct=0.01,
            gap_fill_lookback=0,
        )
        params_gf = BacktestParams(
            min_rr=1.0, max_sl_pct=0.05,
            fvg_min_width_pct=0.001, fvg_entry_depth_pct=0.01,
            gap_fill_lookback=5,
        )
        log_no: list[dict] = []
        log_gf: list[dict] = []
        run_backtest(htf, ltf, params_no_gf, rejection_log=log_no)
        run_backtest(htf, ltf, params_gf, rejection_log=log_gf)
        outcomes_no = [e["outcome"] for e in log_no]
        outcomes_gf = [e["outcome"] for e in log_gf]
        assert "gap_fill_filter" not in outcomes_no
        # With filter enabled and monotonic data (no real gaps), also no gap_fill events
        assert "gap_fill_filter" not in outcomes_gf

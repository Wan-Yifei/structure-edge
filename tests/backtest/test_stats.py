"""Unit tests for backtest/stats.py — Sharpe, Sortino, importance, heatmap."""

import numpy as np
import pandas as pd
import pytest

from backtest.stats import (
    sharpe_ratio,
    sortino_ratio,
    compute_extended_stats,
    parameter_importance,
    time_breakdown,
    time_heatmap_pivot,
    param_heatmap_pivot,
    top_param_pairs,
)
from backtest.engine import BacktestParams, BacktestResult, Trade


# ── Helpers ───────────────────────────────────────────────────────────────────

def _result(rs: list[float]) -> BacktestResult:
    trades = [
        Trade("bull", 100, 98, 104, 2, f"2025-01-{i+1:02d} 09:30",
              f"2025-01-{i+1:02d} 10:00",
              104 if r > 0 else 98, "win" if r > 0 else "loss", r)
        for i, r in enumerate(rs)
    ]
    return BacktestResult(params=BacktestParams(), trades=trades)


def _trades_df(entries: list[tuple]) -> pd.DataFrame:
    """entries: (entry_time_str, r_multiple)"""
    return pd.DataFrame({
        "entry_time": [e[0] for e in entries],
        "r_multiple": [e[1] for e in entries],
    })


# ── sharpe_ratio ──────────────────────────────────────────────────────────────

class TestSharpeRatio:
    def test_empty_returns_zero(self):
        assert sharpe_ratio([]) == 0.0

    def test_single_trade_returns_zero(self):
        assert sharpe_ratio([2.0]) == 0.0

    def test_all_wins_positive(self):
        assert sharpe_ratio([1.0, 2.0, 1.5, 2.5]) > 0

    def test_all_losses_negative(self):
        # Varying losses so std > 0; mean < 0 → Sharpe < 0
        assert sharpe_ratio([-2.0, -1.0, -1.5]) < 0

    def test_zero_std_returns_zero(self):
        # All identical values → std = 0
        assert sharpe_ratio([1.0, 1.0, 1.0]) == 0.0

    def test_higher_mean_higher_sharpe(self):
        s1 = sharpe_ratio([1.0, -0.5, 1.0, -0.5])
        s2 = sharpe_ratio([2.0, -0.5, 2.0, -0.5])
        assert s2 > s1

    def test_accepts_numpy_array(self):
        arr = np.array([1.0, -1.0, 2.0])
        result = sharpe_ratio(arr)
        assert isinstance(result, float)


# ── sortino_ratio ─────────────────────────────────────────────────────────────

class TestSortinoRatio:
    def test_empty_returns_zero(self):
        assert sortino_ratio([]) == 0.0

    def test_single_returns_zero(self):
        assert sortino_ratio([1.0]) == 0.0

    def test_no_losses_returns_inf(self):
        assert sortino_ratio([1.0, 2.0, 0.5]) == float("inf")

    def test_all_losses_negative(self):
        assert sortino_ratio([-1.0, -2.0]) < 0

    def test_sortino_ge_sharpe_when_skewed(self):
        # Sortino ignores upside volatility, so ≥ Sharpe for right-skewed series
        rs = [3.0, 3.0, -1.0, 3.0, -1.0]
        assert sortino_ratio(rs) >= sharpe_ratio(rs)


# ── compute_extended_stats ────────────────────────────────────────────────────

class TestComputeExtendedStats:
    def test_keys_present(self):
        bt = _result([2.0, -1.0, 2.0])
        d = compute_extended_stats(bt)
        assert "sharpe"  in d
        assert "sortino" in d

    def test_values_are_floats(self):
        bt = _result([1.0, -1.0])
        d = compute_extended_stats(bt)
        assert isinstance(d["sharpe"],  float)
        assert isinstance(d["sortino"], float)

    def test_empty_trades(self):
        bt = _result([])
        d = compute_extended_stats(bt)
        assert d["sharpe"]  == 0.0
        assert d["sortino"] == 0.0


# ── parameter_importance ──────────────────────────────────────────────────────

class TestParameterImportance:
    def _df(self):
        return pd.DataFrame({
            "depth":          [0.1, 0.1, 0.5, 0.5, 0.1, 0.5],
            "min_rr":         [1.5, 2.0, 1.5, 2.0, 1.5, 2.0],
            "profit_factor":  [1.2, 1.1, 2.5, 2.4, 1.3, 2.6],
        })

    def test_returns_series(self):
        imp = parameter_importance(self._df(), ["depth", "min_rr"])
        assert isinstance(imp, pd.Series)

    def test_high_impact_param_ranks_first(self):
        imp = parameter_importance(self._df(), ["depth", "min_rr"])
        assert imp.index[0] == "depth"       # depth changes PF by ~1.3x; min_rr barely moves it

    def test_missing_column_skipped(self):
        imp = parameter_importance(self._df(), ["depth", "nonexistent"])
        assert "depth" in imp.index
        assert "nonexistent" not in imp.index

    def test_custom_metric(self):
        df = self._df().rename(columns={"profit_factor": "total_r"})
        imp = parameter_importance(df, ["depth"], metric="total_r")
        assert "depth" in imp.index


# ── time_breakdown ────────────────────────────────────────────────────────────

class TestTimeBreakdown:
    def _df(self):
        return _trades_df([
            ("2025-03-03 09:30", 2.0),   # Mon 09h
            ("2025-03-03 09:45", -1.0),  # Mon 09h
            ("2025-03-04 14:00", 2.0),   # Tue 14h
            ("2025-03-05 09:30", -1.0),  # Wed 09h
        ])

    def test_returns_dataframe(self):
        bd = time_breakdown(self._df())
        assert isinstance(bd, pd.DataFrame)

    def test_expected_columns(self):
        bd = time_breakdown(self._df())
        for col in ("hour", "dow", "n_trades", "avg_r", "win_rate"):
            assert col in bd.columns

    def test_aggregation_correct(self):
        bd = time_breakdown(self._df())
        # Mon 09h row: 2 trades, avg_r = 0.5, win_rate = 0.5
        mon_9h = bd[(bd["dow"] == 0) & (bd["hour"] == 9)]
        assert len(mon_9h) == 1
        assert mon_9h.iloc[0]["n_trades"] == 2
        assert mon_9h.iloc[0]["avg_r"]    == pytest.approx(0.5)
        assert mon_9h.iloc[0]["win_rate"] == pytest.approx(0.5)


# ── time_heatmap_pivot ────────────────────────────────────────────────────────

class TestTimeHeatmapPivot:
    def test_returns_dataframe(self):
        df = _trades_df([
            ("2025-03-03 09:30", 1.0),
            ("2025-03-04 14:00", -1.0),
        ])
        pivot = time_heatmap_pivot(df)
        assert isinstance(pivot, pd.DataFrame)

    def test_rows_are_hours(self):
        df = _trades_df([("2025-03-03 09:30", 1.0), ("2025-03-03 14:00", 1.0)])
        pivot = time_heatmap_pivot(df)
        assert pivot.index.name == "Hour"
        assert set(pivot.index).issubset(set(range(24)))

    def test_columns_are_day_names(self):
        df = _trades_df([("2025-03-03 09:30", 1.0)])   # Monday
        pivot = time_heatmap_pivot(df)
        assert "Mon" in pivot.columns


# ── param_heatmap_pivot ───────────────────────────────────────────────────────

class TestParamHeatmapPivot:
    def _df(self):
        return pd.DataFrame({
            "fvg_depth": [0.1, 0.1, 0.5, 0.5],
            "min_rr":    [1.5, 2.0, 1.5, 2.0],
            "profit_factor": [1.2, 1.1, 2.5, 2.4],
        })

    def test_returns_dataframe(self):
        p = param_heatmap_pivot(self._df(), "fvg_depth", "min_rr")
        assert isinstance(p, pd.DataFrame)

    def test_shape_matches_unique_values(self):
        p = param_heatmap_pivot(self._df(), "fvg_depth", "min_rr")
        assert p.shape == (2, 2)   # 2 unique depths × 2 unique rr values

    def test_cell_values_are_means(self):
        p = param_heatmap_pivot(self._df(), "fvg_depth", "min_rr")
        assert p.loc[0.1, 1.5] == pytest.approx(1.2)
        assert p.loc[0.5, 1.5] == pytest.approx(2.5)


# ── top_param_pairs ───────────────────────────────────────────────────────────

class TestTopParamPairs:
    def _df(self):
        return pd.DataFrame({
            "a": [0.1, 0.1, 0.5, 0.5],
            "b": [1.0, 2.0, 1.0, 2.0],
            "c": [10,  10,  10,  10],       # constant → low interaction
            "profit_factor": [1.0, 1.1, 3.0, 3.2],
        })

    def test_returns_list_of_tuples(self):
        result = top_param_pairs(self._df(), ["a", "b", "c"], top_k=3)
        assert isinstance(result, list)
        assert all(len(t) == 3 for t in result)

    def test_top_k_respected(self):
        result = top_param_pairs(self._df(), ["a", "b", "c"], top_k=2)
        assert len(result) <= 2

    def test_sorted_descending(self):
        result = top_param_pairs(self._df(), ["a", "b", "c"], top_k=3)
        scores = [r[2] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_ab_pair_included_in_results(self):
        result = top_param_pairs(self._df(), ["a", "b", "c"], top_k=3)
        pairs = [frozenset(r[:2]) for r in result]
        # The pair (a, b) drives most of the variance and must appear
        assert frozenset({"a", "b"}) in pairs

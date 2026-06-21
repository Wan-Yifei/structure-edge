"""Unit tests for backtest/fvg_width_select.py — width-floor combo selection."""

import pandas as pd

from backtest.fvg_width_select import select_best_combos, to_watch_params


def _row(code="US.SOXL", tf="15m", n_gaps=0, mean_width_pct=0.0, **kw) -> dict:
    base = {
        "code": code, "tf": tf, "n_gaps": n_gaps, "mean_width_pct": mean_width_pct,
        "total_width_pct": n_gaps * mean_width_pct, "median_width_pct": mean_width_pct,
        "min_gap_pct": 0.001, "require_displacement": False, "require_lvn_overlap": False,
    }
    base.update(kw)
    return base


class TestSelectBestCombos:
    def test_picks_max_n_gaps_among_qualifying_rows(self):
        df = pd.DataFrame([
            _row(n_gaps=100, mean_width_pct=0.001),   # below floor, excluded
            _row(n_gaps=50,  mean_width_pct=0.005),    # qualifies, fewer gaps
            _row(n_gaps=80,  mean_width_pct=0.003),    # qualifies, most gaps -> winner
        ])
        selected = select_best_combos(df, min_mean_width_pct=0.0025)
        assert len(selected) == 1
        assert selected.iloc[0]["n_gaps"] == 80

    def test_ties_broken_by_higher_mean_width(self):
        df = pd.DataFrame([
            _row(n_gaps=50, mean_width_pct=0.003),
            _row(n_gaps=50, mean_width_pct=0.006),  # same n_gaps, wider -> winner
        ])
        selected = select_best_combos(df, min_mean_width_pct=0.0025)
        assert len(selected) == 1
        assert selected.iloc[0]["mean_width_pct"] == 0.006

    def test_group_with_no_qualifying_combo_is_dropped(self):
        df = pd.DataFrame([
            _row(code="US.SOXL", tf="15m", n_gaps=80, mean_width_pct=0.003),
            _row(code="US.SOXL", tf="30m", n_gaps=80, mean_width_pct=0.0001),  # never qualifies
        ])
        selected = select_best_combos(df, min_mean_width_pct=0.0025)
        assert len(selected) == 1
        assert selected.iloc[0]["tf"] == "15m"

    def test_separate_winner_per_code_and_tf(self):
        df = pd.DataFrame([
            _row(code="US.SOXL", tf="15m", n_gaps=80, mean_width_pct=0.003),
            _row(code="US.SOXL", tf="30m", n_gaps=40, mean_width_pct=0.004),
            _row(code="US.SOXS", tf="15m", n_gaps=60, mean_width_pct=0.003),
        ])
        selected = select_best_combos(df, min_mean_width_pct=0.0025)
        assert len(selected) == 3


class TestToWatchParams:
    def test_drops_inapplicable_displacement_and_lvn_keys(self):
        row = pd.Series(_row(require_displacement=False, require_lvn_overlap=False))
        out = to_watch_params(row)
        assert out == {"tf": "15m", "min_gap_pct": 0.001,
                        "require_displacement": False, "require_lvn_overlap": False}

    def test_includes_displacement_params_when_present(self):
        row = pd.Series(_row(
            require_displacement=True, atr_mult=1.5, body_ratio_min=0.5, lookback=5.0,
        ))
        out = to_watch_params(row)
        assert out["require_displacement"] is True
        assert out["atr_mult"] == 1.5
        assert out["lookback"] == 5  # cast to int

    def test_includes_lvn_threshold_when_present(self):
        row = pd.Series(_row(require_lvn_overlap=True, lvn_threshold=0.3))
        out = to_watch_params(row)
        assert out["require_lvn_overlap"] is True
        assert out["lvn_threshold"] == 0.3

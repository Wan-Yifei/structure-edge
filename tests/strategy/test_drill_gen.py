"""Tests for drill_gen -- mostly that a question shows no bar past its decision.

A drill is worthless if any field leaks the future, and the leak would be
invisible: the page would just look unusually easy. So the lookahead checks
are written as the real property -- recompute the field from the truncated
series and demand the same number -- rather than checking an off-by-one.
"""
import numpy as np
import pandas as pd
import pytest

from strategy.drill_gen import (atr, call, flow, label, pivots, structure)


def _ramp(n, start=100.0, step=0.1):
    c = np.array([start + step * i for i in range(n)])
    return c + 0.05, c - 0.05, c          # high, low, close


class TestAtr:
    def test_warmup_is_nan(self):
        h, l, c = _ramp(40)
        a = atr(h, l, c, n=14)
        assert np.isnan(a[:14]).all()
        assert np.isfinite(a[14:]).all()

    def test_constant_range_converges_to_that_range(self):
        h, l, c = _ramp(80, step=0.0)
        a = atr(h, l, c, n=14)
        assert a[-1] == pytest.approx(0.10, abs=1e-6)

    def test_short_series_is_all_nan(self):
        h, l, c = _ramp(10)
        assert np.isnan(atr(h, l, c, n=14)).all()


class TestLabel:
    def test_up_barrier_first_is_a_win(self):
        H = np.array([10.0, 10.0, 13.0])
        L = np.array([10.0, 9.9, 12.0])
        C = np.array([10.0, 10.0, 13.0])
        assert label(H, L, C, 0, a=1.0, up=2.0, dn=1.0, fwd=5) == 1

    def test_down_barrier_first_is_a_loss(self):
        H = np.array([10.0, 10.1, 13.0])
        L = np.array([10.0, 8.5, 12.0])
        C = np.array([10.0, 9.0, 13.0])
        assert label(H, L, C, 0, a=1.0, up=2.0, dn=1.0, fwd=5) == 0

    def test_same_bar_touches_both_counts_as_loss(self):
        """The bar's path is unknown, so the adverse reading is the honest one."""
        H = np.array([10.0, 12.5])
        L = np.array([10.0, 8.5])
        C = np.array([10.0, 10.0])
        assert label(H, L, C, 0, a=1.0, up=2.0, dn=1.0, fwd=5) == 0

    def test_neither_barrier_in_horizon_is_none(self):
        H = np.full(10, 10.05)
        L = np.full(10, 9.95)
        C = np.full(10, 10.0)
        assert label(H, L, C, 0, a=1.0, up=2.0, dn=1.0, fwd=5) is None

    def test_horizon_is_respected(self):
        H = np.array([10.0] + [10.05] * 5 + [99.0])
        L = np.full(7, 9.95)
        C = np.full(7, 10.0)
        assert label(H, L, C, 0, a=1.0, fwd=3) is None
        assert label(H, L, C, 0, a=1.0, fwd=6) == 1


class TestPivots:
    def test_last_k_bars_are_never_confirmed(self):
        """A swing needs k bars on its right, so the newest k cannot qualify --
        the real-time cost of a pivot definition."""
        rng = np.random.default_rng(0)
        H = 100 + rng.normal(0, 1, 80).cumsum()
        L = H - 0.5
        sh, sl = pivots(H, L, 0, 79, k=3)
        assert all(j <= 79 - 3 for j in sh + sl)

    def test_a_clear_peak_is_found(self):
        H = np.array([1, 2, 3, 9, 3, 2, 1], float)
        sh, _ = pivots(H, H - 0.1, 0, 6, k=3)
        assert sh == [3]

    def test_adding_future_bars_cannot_change_past_pivots(self):
        rng = np.random.default_rng(1)
        H = 100 + rng.normal(0, 1, 120).cumsum()
        L = H - 0.4
        a, b = pivots(H, L, 0, 60, k=3)
        a2, b2 = pivots(H[:70], L[:70], 0, 60, k=3)
        assert (a, b) == (a2, b2)


class TestStructure:
    def test_too_few_swings_abstains(self):
        H, L, C = np.full(20, 10.05), np.full(20, 9.95), np.full(20, 10.0)
        sc, why, info = structure(C, H, L, C, 19, a=0.1)
        assert sc == 0 and info["trend"] == "未知"

    def test_rising_series_reads_as_uptrend(self):
        n = 70
        base = np.array([100 + 0.3 * i for i in range(n)])
        wig = np.array([2.2 * np.sin(i / 2.2) for i in range(n)])
        c = base + wig
        sc, why, info = structure(c, c + 0.3, c - 0.3, c, n - 1, a=0.4)
        assert info["trend"] == "上升"
        assert sc > 0

    def test_falling_series_reads_as_downtrend_and_scores_negative(self):
        n = 70
        base = np.array([140 - 0.3 * i for i in range(n)])
        wig = np.array([2.2 * np.sin(i / 2.2) for i in range(n)])
        c = base + wig
        sc, why, info = structure(c, c + 0.3, c - 0.3, c, n - 1, a=0.4)
        assert info["trend"] == "下降"
        assert sc < 0

    def test_reading_ignores_bars_after_the_decision(self):
        rng = np.random.default_rng(3)
        c = 100 + rng.normal(0, 0.4, 200).cumsum()
        H, L = c + 0.3, c - 0.3
        i = 120
        a = structure(c, H, L, c, i, a=0.5)
        shocked = c.copy()
        shocked[i + 1:] += 50.0            # violent, entirely in the future
        b = structure(shocked, shocked + 0.3, shocked - 0.3, shocked, i, a=0.5)
        assert a[0] == b[0]
        assert [t for _, t in a[1]] == [t for _, t in b[1]]

    def test_every_reason_carries_a_signed_weight(self):
        n = 70
        c = np.array([100 + 0.3 * i + 2.2 * np.sin(i / 2.2) for i in range(n)])
        sc, why, _ = structure(c, c + 0.3, c - 0.3, c, n - 1, a=0.4)
        assert why and all(isinstance(v, int) for v, _ in why)
        assert sc == sum(v for v, _ in why)


class TestCall:
    def test_thresholds(self):
        assert call(2) == "看涨" and call(5) == "看涨"
        assert call(1) == "中性" and call(0) == "中性"
        assert call(-1) == "看跌" and call(-4) == "看跌"


class TestFlow:
    def test_window_ends_at_the_decision_bar(self):
        buy = np.array([0.0] * 10 + [1000.0])
        sell = np.array([1000.0] * 10 + [0.0])
        assert flow(buy, sell, 10, 1) == 1.0          # only the last bar
        assert flow(buy, sell, 9, 1) == -1.0          # only bar 9

    def test_future_bars_cannot_move_the_answer(self):
        rng = np.random.default_rng(5)
        buy = rng.uniform(200, 900, 60)
        sell = rng.uniform(200, 900, 60)
        i = 40
        for n in (1, 5, 15):
            before = flow(buy, sell, i, n)
            b2, s2 = buy.copy(), sell.copy()
            b2[i + 1:] = 1e9                 # future stuffed with buying
            assert flow(b2, s2, i, n) == before

    def test_thin_tape_abstains_rather_than_reporting_noise(self):
        buy = np.full(20, 5.0)
        sell = np.full(20, 5.0)
        assert flow(buy, sell, 10, 1) is None

    def test_longer_window_smooths_a_single_bar_flip(self):
        buy = np.concatenate([np.full(14, 100.0), [5000.0]])
        sell = np.concatenate([np.full(14, 900.0), [50.0]])
        one, five = flow(buy, sell, 14, 1), flow(buy, sell, 14, 5)
        # The newest bar is overwhelmingly bought; the four before it were
        # sold. The 1m reading takes the flip at face value, the 5m keeps
        # the preceding bars in view and lands far short of it.
        assert one == pytest.approx(0.980, abs=0.01)
        assert five == pytest.approx(0.193, abs=0.01)
        assert abs(five) < abs(one)

    def test_balanced_tape_is_zero(self):
        buy = np.full(20, 500.0)
        sell = np.full(20, 500.0)
        assert flow(buy, sell, 10, 5) == 0.0


class TestHeatmap:
    """The grid is the only field built from a second-resolution source, so
    its two aggregations -- sum across price levels, average across seconds --
    are easy to get backwards and invisible once drawn."""

    @staticmethod
    def _ladder(n_sec=600, levels=(100.0, 100.5, 101.0), size=1000.0):
        t0 = pd.Timestamp("2026-10-14 10:00:00")
        rows = []
        for s in range(n_sec):
            for p in levels:
                rows.append((t0 + pd.Timedelta(seconds=s), p, size, "B"))
        return pd.DataFrame(rows, columns=["ts", "price", "size", "side"])

    def test_empty_or_missing_input_returns_none(self):
        from strategy.drill_gen import heatmap
        t = pd.Timestamp("2026-10-14 10:10:00")
        assert heatmap(None, t, 99.0, 102.0) is None
        empty = pd.DataFrame(columns=["ts", "price", "size", "side"])
        assert heatmap(empty, t, 99.0, 102.0) is None

    def test_window_excludes_anything_after_the_decision(self):
        from strategy.drill_gen import heatmap
        lad = self._ladder()
        t = pd.Timestamp("2026-10-14 10:05:00")
        a = heatmap(lad, t, 99.0, 102.0)
        future = lad.copy()
        future.loc[future.ts > t, "size"] = 9e6
        b = heatmap(future, t, 99.0, 102.0)
        assert a["d"] == b["d"] and a["top"] == b["top"]

    def test_levels_in_one_bin_sum_while_seconds_average(self):
        """Three levels of 1000 in one bin, held every second, is a band of
        3000 -- not 1000 (averaged across levels) and not 3000 x seconds."""
        from strategy.drill_gen import heatmap
        # 99-102 over 48 bins is 0.0625 a bin, so all three share one.
        lad = self._ladder(levels=(100.0, 100.01, 100.02), size=1000.0)
        t = pd.Timestamp("2026-10-14 10:10:00")
        hm = heatmap(lad, t, 99.0, 102.0)
        assert hm["top"] == pytest.approx(3000, rel=0.02)

    def test_grid_shape_matches_the_declared_dimensions(self):
        import base64
        from strategy.drill_gen import heatmap
        lad = self._ladder()
        hm = heatmap(lad, pd.Timestamp("2026-10-14 10:10:00"), 99.0, 102.0)
        assert len(base64.b64decode(hm["d"])) == hm["nb"] * hm["nc"]
        assert hm["nc"] == 30 * 60 // hm["sec"]

    def test_a_wall_lands_in_its_own_price_row(self):
        import base64
        from strategy.drill_gen import heatmap
        lad = self._ladder(levels=(100.0, 101.0), size=100.0)
        lad.loc[lad.price == 101.0, "size"] = 50000.0
        lo, hi = 99.0, 102.0
        hm = heatmap(lad, pd.Timestamp("2026-10-14 10:10:00"), lo, hi)
        g = np.frombuffer(base64.b64decode(hm["d"]), np.uint8) \
              .reshape(hm["nb"], hm["nc"])
        hot = int((101.0 - lo) / (hi - lo) * hm["nb"])
        assert g[hot].max() == 255
        quiet = int((100.0 - lo) / (hi - lo) * hm["nb"])
        assert g[quiet].max() < 20

    def test_a_price_range_with_no_rows_returns_none(self):
        from strategy.drill_gen import heatmap
        lad = self._ladder()
        t = pd.Timestamp("2026-10-14 10:10:00")
        assert heatmap(lad, t, 200.0, 210.0) is None

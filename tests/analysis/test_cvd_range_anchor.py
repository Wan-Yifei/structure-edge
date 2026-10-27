"""CVD accumulates over the Range window instead of always resetting daily.

The Range control already scoped the session volume profile; CVD now anchors on
the same setting, so a level read off the profile and the CVD under it describe
the same stretch of tape. Blocks are counted back from the newest trading day,
which is what keeps the most recent block a full N days however much history
happens to be loaded.

These drive the real _draw_cvd on a constructed viewer and read _cvd_arr back.
"""

import os
from datetime import datetime, timedelta

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6.QtWidgets")

from PyQt6.QtWidgets import QApplication   # noqa: E402

CM = 5                       # 5-minute bars
BARS_PER_DAY = 2
DAY0 = datetime(2026, 9, 21, 10, 0)   # 10:00, well inside one trading day


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app
    app.processEvents()


@pytest.fixture(scope="module")
def viewer(qapp):
    from analysis.trade_viewer_qt import TradeViewerQt

    v = TradeViewerQt()
    v.resize(1400, 900)
    v.show()
    for _ in range(30):
        qapp.processEvents()
    yield v
    v.close()
    qapp.processEvents()


def _build(n_days: int, per_bar_delta: float = 100.0):
    """n_days trading days, BARS_PER_DAY bars each, constant +delta per bar."""
    from core.time_utils import candle_start

    rows, buckets = [], {}
    for d in range(n_days):
        for b in range(BARS_PER_DAY):
            bar_end = DAY0 + timedelta(days=d, minutes=CM * (b + 1))
            ts = bar_end.strftime("%Y-%m-%d %H:%M")
            rows.append((ts, 100.0, 101.0, 99.0, 100.0, 1000.0))
            bk = candle_start(bar_end - timedelta(minutes=CM), CM)
            buckets[bk] = {100.0: {"buy": per_bar_delta, "sell": 0.0,
                                   "neutral": 0.0}}
    kl = pd.DataFrame(
        rows, columns=["time_key", "open", "high", "low", "close", "volume"])
    return kl, buckets


def _run(v, n_days, range_val):
    for val in ("1d", "2d", "3d", "7d"):
        v._ind_checks[f"range_{val}"].setChecked(val == range_val)
    kl, buckets = _build(n_days)
    v._draw_cvd(kl, buckets, CM)
    return list(v._cvd_arr)


class TestDailyDefault:
    def test_one_day_range_resets_every_day(self, viewer):
        cvd = _run(viewer, 4, "1d")
        # each day: 100, 200 then back to 100
        assert cvd == [100, 200, 100, 200, 100, 200, 100, 200]

    def test_peak_never_exceeds_one_day_of_flow(self, viewer):
        assert max(_run(viewer, 5, "1d")) == 100 * BARS_PER_DAY


class TestWiderWindows:
    def test_two_day_range_accumulates_across_a_pair(self, viewer):
        cvd = _run(viewer, 4, "2d")
        # days 1-2 are one block, days 3-4 the next
        assert cvd == [100, 200, 300, 400, 100, 200, 300, 400]

    def test_three_day_range(self, viewer):
        cvd = _run(viewer, 3, "3d")
        assert cvd == [100, 200, 300, 400, 500, 600]

    def test_week_range_holds_five_sessions(self, viewer):
        cvd = _run(viewer, 5, "7d")
        assert cvd[-1] == 100 * BARS_PER_DAY * 5
        assert cvd == sorted(cvd), "no reset inside the block"


class TestBlocksCountBackFromTheNewest:
    def test_the_newest_block_is_always_full(self, viewer):
        """5 days at 2D: the last two days must be one block, so the leftover
        odd day lands at the far end, not the near end."""
        cvd = _run(viewer, 5, "2d")
        assert cvd[-4:] == [100, 200, 300, 400], "last two days accumulate"
        assert cvd[:2] == [100, 200], "the odd day out is the oldest"

    def test_adding_older_history_does_not_move_the_recent_boundary(self, viewer):
        """Counting forward from the oldest day would shift every boundary
        whenever the lookback changed."""
        tail4 = _run(viewer, 4, "2d")[-4:]
        tail5 = _run(viewer, 5, "2d")[-4:]
        tail6 = _run(viewer, 6, "2d")[-4:]
        assert tail4 == tail5 == tail6

    def test_range_wider_than_the_data_is_one_block(self, viewer):
        cvd = _run(viewer, 2, "7d")
        assert cvd == [100, 200, 300, 400]


class TestSignAndShape:
    def test_negative_flow_accumulates_downward(self, viewer):
        from core.time_utils import candle_start

        rows, buckets = [], {}
        for d in range(2):
            for b in range(BARS_PER_DAY):
                bar_end = DAY0 + timedelta(days=d, minutes=CM * (b + 1))
                rows.append((bar_end.strftime("%Y-%m-%d %H:%M"),
                             100.0, 101.0, 99.0, 100.0, 1000.0))
                bk = candle_start(bar_end - timedelta(minutes=CM), CM)
                buckets[bk] = {100.0: {"buy": 0.0, "sell": 150.0, "neutral": 0.0}}
        kl = pd.DataFrame(rows, columns=["time_key", "open", "high", "low",
                                         "close", "volume"])
        viewer._ind_checks["range_2d"].setChecked(True)
        viewer._draw_cvd(kl, buckets, CM)
        assert list(viewer._cvd_arr) == [-150, -300, -450, -600]

    def test_bars_without_tick_coverage_are_not_drawn(self, viewer):
        """A bar with no ticks has no cumulative delta, and saying 0 is
        not the same as saying nothing.

        This used to hold the line flat at the running total, which on a
        real frame was most of the panel: the chart keeps every kline
        that was fetched while the Range control loads ticks for one to
        five days, so a 3,980-bar frame with one day of ticks had 2,881
        points -- 72% -- sitting at exactly 0. Read as "buying and
        selling are balanced", meant "nothing is known", and the climb
        out of it into the first real value drew a vertical stroke that
        set the panel's whole Y scale.
        """
        import math

        from core.time_utils import candle_start

        kl, buckets = _build(2)
        bar_end = DAY0 + timedelta(days=0, minutes=CM * 1)
        del buckets[candle_start(bar_end - timedelta(minutes=CM), CM)]
        viewer._ind_checks["range_2d"].setChecked(True)
        viewer._draw_cvd(kl, buckets, CM)
        got = list(viewer._cvd_arr)
        assert math.isnan(got[0]), f"没有逐笔的那根仍然画了值：{got[0]}"
        assert got[1:] == [100, 200, 300], \
            f"有逐笔的那几根被带歪了：{got[1:]}"


class TestSharedWithTheProfile:
    def test_both_read_the_same_range_map(self):
        """A second copy of the day counts would let the two drift apart."""
        from analysis.trade_viewer_qt import RANGE_DAYS
        assert RANGE_DAYS == {"1d": 1, "2d": 2, "3d": 3, "7d": 5}

    def test_cvd_block_matches_the_profile_window(self, viewer):
        """The profile trims to the last N trading days; CVD's newest block
        must start on the same bar, or the two panels describe different
        stretches of tape."""
        from analysis.trade_viewer_qt import apply_profile_range, _trading_day

        kl, buckets = _build(5)
        for range_val in ("1d", "2d", "3d"):
            for val in ("1d", "2d", "3d", "7d"):
                viewer._ind_checks[f"range_{val}"].setChecked(val == range_val)
            viewer._draw_cvd(kl, buckets, CM)
            cvd = list(viewer._cvd_arr)
            trimmed = apply_profile_range(kl, range_val, CM)
            first_day = _trading_day(trimmed["time_key"].iloc[0], CM)
            days = [_trading_day(tk, CM) for tk in kl["time_key"]]
            start_idx = days.index(first_day)
            assert cvd[start_idx] == 100, (
                f"{range_val}: CVD should restart at the profile's first bar")
            assert cvd[start_idx - 1] != 0 or start_idx == 0


class TestThePanelFollowsTheView:
    """CVD is an unbounded running sum, so the Y window has to follow the
    X window or the curve walks out of the panel.

    It was fitted at load and on each new bar, never on pan or zoom. A
    session's CVD climbs hundreds of thousands of shares, so a few
    minutes of panning left a window tens of thousands tall somewhere the
    curve no longer was -- the panel read as empty, or showed a single
    vertical stroke where the curve crossed it on its way past.
    """

    def _long_climb(self, v):
        for val in ("1d", "2d", "3d", "7d"):
            v._ind_checks[f"range_{val}"].setChecked(val == "7d")
        kl, buckets = _build(5)
        v._draw_cvd(kl, buckets, CM)
        return kl

    def test_a_pan_refits_the_panel(self, viewer):
        kl = self._long_climb(viewer)
        n = len(kl)
        viewer._plot_c.setXRange(0, n - 1, padding=0)
        viewer._plot_cvd.vb.viewRange()           # settle
        viewer._plot_c.setXRange(n - 3, n - 1, padding=0)
        lo, hi = viewer._plot_cvd.vb.viewRange()[1]
        vis = [x for x in list(viewer._cvd_arr)[n - 3:] if x == x]
        assert lo <= min(vis) and hi >= max(vis), \
            f"窗口 [{lo:.0f},{hi:.0f}] 装不下可见的 {min(vis)}..{max(vis)}"
        assert hi - lo < viewer._cvd_arr.max(), \
            "缩放之后 Y 轴还是整条曲线的高度，等于没跟着看"

    def test_a_window_with_no_coverage_is_left_alone(self, viewer):
        """All-NaN in view: nanmin of that is NaN and setYRange(NaN)
        blanks the panel, so the range must simply not move."""
        from core.time_utils import candle_start

        for val in ("1d", "2d", "3d", "7d"):
            viewer._ind_checks[f"range_{val}"].setChecked(val == "7d")
        kl, buckets = _build(3)
        for b in list(buckets)[:BARS_PER_DAY]:     # day one loses its ticks
            del buckets[b]
        viewer._draw_cvd(kl, buckets, CM)
        viewer._plot_cvd.setYRange(-7.0, 11.0, padding=0)
        viewer._autorange_cumulative_subplots(0, BARS_PER_DAY - 1)
        lo, hi = viewer._plot_cvd.vb.viewRange()[1]
        assert (round(lo, 6), round(hi, 6)) == (-7.0, 11.0), \
            f"全是 NaN 的窗口把 Y 轴改成了 [{lo},{hi}]"

    def test_a_window_straddling_the_coverage_edge_still_fits(self, viewer):
        """The common case, and the one the all-NaN guard does not cover.

        numpy's min() propagates NaN, so a window holding both covered
        and uncovered bars -- which is most windows, since the coverage
        edge sits inside the loaded klines -- yields NaN and blanks the
        panel. Only nanmin/nanmax get the real extent.
        """
        for val in ("1d", "2d", "3d", "7d"):
            viewer._ind_checks[f"range_{val}"].setChecked(val == "7d")
        kl, buckets = _build(3)
        for b in list(buckets)[:BARS_PER_DAY]:     # day one loses its ticks
            del buckets[b]
        viewer._draw_cvd(kl, buckets, CM)
        n = len(kl)
        viewer._autorange_cumulative_subplots(0, n - 1)
        lo, hi = viewer._plot_cvd.vb.viewRange()[1]
        assert lo == lo and hi == hi, f"Y 轴被 NaN 污染：[{lo},{hi}]"
        vis = [x for x in list(viewer._cvd_arr) if x == x]
        assert lo <= min(vis) and hi >= max(vis), \
            f"窗口 [{lo:.0f},{hi:.0f}] 装不下 {min(vis)}..{max(vis)}"

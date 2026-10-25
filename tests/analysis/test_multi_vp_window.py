"""The multi-session profile window draws what it claims to.

A Qt window that opens without raising proves nothing -- the failure to
catch is an empty plot, or a value area that does not belong to the day
it is drawn over. So these count what reached the plot and check the
levels against the session they came from.

Klines are synthetic and fetch_klines is replaced, so the tests neither
touch the network nor depend on which days happen to be cached.
"""
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app
    app.processEvents()


def _session(day, base, n=390, amp=1.5):
    """One RTH session of 1m bars with a drift and a volume hump, so the
    profile has a real peak rather than a flat distribution.

    `amp` varies the day's range. Sessions of IDENTICAL range would make
    the per-session and shared grids agree by accident, and the control
    that proves the shared grid is doing anything would pass vacuously."""
    t = pd.date_range(f"{day} 09:31", periods=n, freq="1min")
    i = np.arange(n, dtype=float)
    c = base + np.sin(i / 40) * amp + i * 0.002
    vol = 1000 + 4000 * np.exp(-((i - n / 2) ** 2) / (2 * (n / 8) ** 2))
    return pd.DataFrame(
        {"time_key": t.strftime("%Y-%m-%d %H:%M:%S"),
         "open": c, "high": c + 0.25, "low": c - 0.25, "close": c,
         "volume": vol})


@pytest.fixture
def win(qapp, monkeypatch):
    days = ["2026-10-19", "2026-10-20", "2026-10-21"]
    frames = [_session(d, 100 + k * 2, amp=1.0 + k * 0.8)
              for k, d in enumerate(days)]
    import feeds.fetcher as fetcher
    monkeypatch.setattr(fetcher, "fetch_klines",
                        lambda *a, **k: pd.concat(frames, ignore_index=True))
    from analysis.multi_vp_window import MultiVPWindow
    w = MultiVPWindow()
    w._days.blockSignals(True)
    w._days.setValue(3)
    w._days.blockSignals(False)
    w.reload()
    yield w, days
    w.close()


def kinds(w):
    out = {}
    for it in w._items:
        out[type(it).__name__] = out.get(type(it).__name__, 0) + 1
    return out


class TestItDraws:
    def test_one_profile_per_session(self, win):
        w, days = win
        assert kinds(w).get("BarGraphItem", 0) == len(days)

    def test_levels_and_path_and_joins(self, win):
        """3 level lines + 1 price path a session, 2 joins a gap. Counting
        them is how an empty or half-drawn panel gets caught."""
        w, days = win
        want = len(days) * 4 + (len(days) - 1) * 2
        assert kinds(w).get("PlotCurveItem", 0) == want

    def test_joins_can_be_turned_off(self, win):
        w, days = win
        w._join.setChecked(False)
        w.reload()
        assert kinds(w).get("PlotCurveItem", 0) == len(days) * 4

    def test_a_redraw_does_not_pile_items_up(self, win):
        """reload() has to clear first; without that the plot accumulates a
        fresh copy on every parameter change and slows to a halt."""
        w, _ = win
        before = len(w._items)
        w.reload()
        w.reload()
        assert len(w._items) == before


class TestLevelsBelongToTheirSession:
    def test_va_is_ordered_and_inside_the_day(self, win):
        w, days = win
        from analysis.trade_viewer_qt import _compute_poc_vah_val
        from strategy.smc.fvg import compute_volume_profile
        for day, g in w._sessions():
            edges, vols = compute_volume_profile(g, n_bins=w._bins.value())
            c = (edges[:-1] + edges[1:]) / 2
            poc, vah, val = _compute_poc_vah_val(c, vols,
                                                 w._va.value() / 100)
            assert val <= poc <= vah
            assert float(g.low.min()) - 0.01 <= val
            assert vah <= float(g.high.max()) + 0.01

    def test_a_wider_value_area_contains_the_narrower_one(self, win):
        """70% to 90% can only grow. If it does not, the expansion is
        anchored somewhere it should not be."""
        w, _ = win
        from analysis.trade_viewer_qt import _compute_poc_vah_val
        from strategy.smc.fvg import compute_volume_profile
        day, g = w._sessions()[0]
        edges, vols = compute_volume_profile(g, n_bins=w._bins.value())
        c = (edges[:-1] + edges[1:]) / 2
        _, v70, l70 = _compute_poc_vah_val(c, vols, 0.70)
        _, v90, l90 = _compute_poc_vah_val(c, vols, 0.90)
        assert v90 >= v70 and l90 <= l70

    def test_each_session_keeps_its_own_x_slab(self, win):
        """The levels are segments, not full-width lines -- a session's VA
        must not be drawn across its neighbours."""
        w, days = win
        spans = [(min(it.xData), max(it.xData)) for it in w._items
                 if type(it).__name__ == "PlotCurveItem"
                 and it.xData is not None and len(it.xData) == 2
                 and it.yData[0] == it.yData[1]]
        assert spans, "no horizontal level segments were drawn"
        for lo, hi in spans:
            assert hi - lo < 1.0, f"a level line spans a whole slab: {lo}-{hi}"


class TestEmptyInput:
    def test_no_data_says_so_instead_of_raising(self, qapp, monkeypatch):
        import feeds.fetcher as fetcher
        monkeypatch.setattr(fetcher, "fetch_klines",
                            lambda *a, **k: pd.DataFrame())
        from analysis.multi_vp_window import MultiVPWindow
        w = MultiVPWindow()
        w.reload()
        assert "no sessions" in w._status.text()
        assert not w._items
        w.close()

    def test_a_stub_session_is_skipped(self, qapp, monkeypatch):
        """A holiday half-hour is not a session; its one-bar value area
        would drag the price scale over everything else."""
        import feeds.fetcher as fetcher
        full = _session("2026-10-20", 100)
        stub = _session("2026-10-21", 140, n=20)
        monkeypatch.setattr(
            fetcher, "fetch_klines",
            lambda *a, **k: pd.concat([full, stub], ignore_index=True))
        from analysis.multi_vp_window import MultiVPWindow
        w = MultiVPWindow()
        w.reload()
        assert [d for d, _ in w._sessions()] == ["2026-10-20"]
        w.close()


class TestTradingDayBoundary:
    """20:00 ET starts the next trading day, as everywhere else in the
    project (_trading_day_of_bucket, and the bar CVD resets on).

    Only visible with extended hours shown: under RTH the session sits
    inside one calendar date either way, so the default view cannot catch
    a calendar-date grouping.
    """

    @staticmethod
    def _overnight():
        """Evening of the 20th through the next afternoon -- one trading
        day by the project's rule, two calendar dates."""
        ev = pd.date_range("2026-10-20 20:00", periods=120, freq="1min")
        ng = pd.date_range("2026-10-21 04:00", periods=120, freq="1min")
        rt = pd.date_range("2026-10-21 09:31", periods=200, freq="1min")
        t = ev.append(ng).append(rt)
        c = 100 + np.linspace(0, 2, len(t))
        return pd.DataFrame(
            {"time_key": t.strftime("%Y-%m-%d %H:%M:%S"),
             "open": c, "high": c + 0.2, "low": c - 0.2, "close": c,
             "volume": np.full(len(t), 1000.0)})

    def test_the_overnight_session_is_not_split(self, qapp, monkeypatch):
        import feeds.fetcher as fetcher
        monkeypatch.setattr(fetcher, "fetch_klines",
                            lambda *a, **k: self._overnight())
        from analysis.multi_vp_window import MultiVPWindow
        w = MultiVPWindow()
        w._rth.setChecked(False)
        w.reload()
        days = [d for d, _ in w._sessions()]
        assert days == ["2026-10-21"], (
            f"20:00 之后应归入次日，实得 {days}")
        n = len(w._sessions()[0][1])
        assert n == 440, f"该交易日应有 440 根，实得 {n}"
        w.close()

    def test_rth_view_is_unaffected(self, qapp, monkeypatch):
        """The default view must not move: 09:30-16:00 is the same session
        under either rule, and this is the case that was already right."""
        import feeds.fetcher as fetcher
        monkeypatch.setattr(fetcher, "fetch_klines",
                            lambda *a, **k: self._overnight())
        from analysis.multi_vp_window import MultiVPWindow
        w = MultiVPWindow()
        w._rth.setChecked(True)
        w.reload()
        days = [d for d, _ in w._sessions()]
        assert days == ["2026-10-21"]
        w.close()


class TestSharedGrid:
    def test_every_session_gets_the_same_bin_width(self, win):
        """What the shared grid is for. Sessions differ in range, so a
        per-session grid gives each a different bin width and the same
        bin index means a different price in each slab."""
        w, _ = win
        from strategy.smc.fvg import compute_volume_profile
        import numpy as np
        sess = w._sessions()
        p_lo = min(float(g.low.min()) for _, g in sess)
        p_hi = max(float(g.high.max()) for _, g in sess)
        grid = np.linspace(p_lo, p_hi, w._bins.value() + 1)
        widths = set()
        for _, g in sess:
            e, _v = compute_volume_profile(g, n_bins=w._bins.value(),
                                           edges=grid)
            widths.add(round(float(e[1] - e[0]), 9))
        assert len(widths) == 1, f"共享网格下档宽应一致，实得 {widths}"

    def test_per_session_grids_differ(self, win):
        """The control: without a shared grid the widths must differ, or
        the previous test proves nothing."""
        w, _ = win
        from strategy.smc.fvg import compute_volume_profile
        widths = set()
        for _, g in w._sessions():
            e, _v = compute_volume_profile(g, n_bins=w._bins.value())
            widths.add(round(float(e[1] - e[0]), 9))
        assert len(widths) > 1

    def test_a_supplied_grid_keeps_the_volume(self, win):
        """Bars outside the grid are clipped into the end bins, not
        dropped -- otherwise a session near the edge of the window would
        quietly lose volume and its POC would move."""
        from strategy.smc.fvg import compute_volume_profile
        import numpy as np
        w, _ = win
        _, g = w._sessions()[0]
        # A grid that stops short of the session on both sides.
        lo, hi = float(g.low.min()), float(g.high.max())
        mid = (lo + hi) / 2
        grid = np.linspace(mid - 0.2, mid + 0.2, 21)
        _e, v = compute_volume_profile(g, n_bins=20, edges=grid)
        assert abs(v.sum() - float(g.volume.sum())) < 1.0

    def test_turning_it_off_still_draws(self, win):
        w, days = win
        w._shared.setChecked(False)
        w.reload()
        assert kinds(w).get("BarGraphItem", 0) == len(days)


class TestLevelLabels:
    def test_three_labels_a_session_plus_the_day_tag(self, win):
        w, days = win
        assert kinds(w).get("TextItem", 0) == len(days) * 4

    def test_they_can_be_turned_off(self, win):
        w, days = win
        w._labels.setChecked(False)
        w.reload()
        assert kinds(w).get("TextItem", 0) == len(days)

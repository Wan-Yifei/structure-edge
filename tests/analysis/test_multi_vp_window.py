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


def _session(day, base, n=390):
    """One RTH session of 1m bars with a drift and a volume hump, so the
    profile has a real peak rather than a flat distribution."""
    t = pd.date_range(f"{day} 09:31", periods=n, freq="1min")
    i = np.arange(n, dtype=float)
    c = base + np.sin(i / 40) * 1.5 + i * 0.002
    vol = 1000 + 4000 * np.exp(-((i - n / 2) ** 2) / (2 * (n / 8) ** 2))
    return pd.DataFrame(
        {"time_key": t.strftime("%Y-%m-%d %H:%M:%S"),
         "open": c, "high": c + 0.25, "low": c - 0.25, "close": c,
         "volume": vol})


@pytest.fixture
def win(qapp, monkeypatch):
    days = ["2026-10-19", "2026-10-20", "2026-10-21"]
    frames = [_session(d, 100 + k * 2) for k, d in enumerate(days)]
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
        assert spans, "没有找到水平的价位线段"
        for lo, hi in spans:
            assert hi - lo < 1.0, f"价位线跨越了一整格：{lo}–{hi}"


class TestEmptyInput:
    def test_no_data_says_so_instead_of_raising(self, qapp, monkeypatch):
        import feeds.fetcher as fetcher
        monkeypatch.setattr(fetcher, "fetch_klines",
                            lambda *a, **k: pd.DataFrame())
        from analysis.multi_vp_window import MultiVPWindow
        w = MultiVPWindow()
        w.reload()
        assert "没有" in w._status.text()
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

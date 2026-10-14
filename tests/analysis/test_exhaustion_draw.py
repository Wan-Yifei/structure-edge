"""The exhaustion marks as they reach the main chart.

The maths has its own tests; this covers the part that puts marks on screen,
which had none. Three things can go wrong between a correct mark and a
correct chart, and each is cheap to get wrong silently: drawing on a
timeframe where the signature is invisible, drawing nothing because the tick
minute and the bar label disagree by one, and leaving items behind on a
redraw so they pile up every second the chart refreshes.
"""

import os
from datetime import datetime, timedelta

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6.QtWidgets")

from PyQt6.QtWidgets import QApplication        # noqa: E402

DAY = "2026-10-13"


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app
    app.processEvents()


@pytest.fixture(scope="module")
def viewer(qapp):
    from analysis.trade_viewer_qt import TradeViewerQt
    v = TradeViewerQt()
    v.resize(1200, 800)
    for _ in range(10):
        qapp.processEvents()
    yield v
    v.close()
    qapp.processEvents()


def klines():
    """A falling regular session, so RSI(6) is far under the gate."""
    t0 = datetime.strptime(f"{DAY} 09:31", "%Y-%m-%d %H:%M")
    rows = []
    for i in range(120):
        c = 200 - i * 0.5
        rows.append({"time_key": (t0 + timedelta(minutes=i)
                                  ).strftime("%Y-%m-%d %H:%M:%S"),
                     "open": c + 0.2, "high": c + 0.3,
                     "low": c - 0.3, "close": c, "volume": 10_000})
    return pd.DataFrame(rows)


def fake_marks(*minutes):
    """A mark landing in each given TICK minute (bar label is one later)."""
    out = []
    for hhmm in minutes:
        out.append({"ts": datetime.strptime(f"{DAY} {hhmm}:40",
                                            "%Y-%m-%d %H:%M:%S"),
                    "burst": 30_000.0, "burst_x": 48.0,
                    "lull": 1_200.0, "rsi": 8.2})
    return out


@pytest.fixture
def armed(viewer, monkeypatch):
    """Viewer with the tick pass stubbed out, so only drawing is exercised."""
    import analysis.trade_viewer_qt as tv
    monkeypatch.setattr(tv, "load_raw_ticks", lambda *a, **k: [{"stub": 1}])
    monkeypatch.setattr(tv, "compute_exhaustion_marks",
                        lambda rows, rsi_map, **k: fake_marks("09:40", "10:15"))
    viewer._exh_cache.clear()
    viewer._clear_exhaust_items()
    return viewer


class TestItAppears:
    def test_marks_land_on_the_main_chart(self, armed):
        armed._tf_combo.setCurrentText("1m")
        armed._draw_exhaust(klines(), "US.SOXL")
        assert len(armed._exhaust_items) > 0
        for item in armed._exhaust_items:
            assert item in armed._plot_c.items

    def test_one_scatter_and_one_label_per_mark(self, armed):
        armed._tf_combo.setCurrentText("1m")
        armed._draw_exhaust(klines(), "US.SOXL")
        import pyqtgraph as pg
        scatters = [i for i in armed._exhaust_items
                    if isinstance(i, pg.ScatterPlotItem)]
        labels = [i for i in armed._exhaust_items
                  if isinstance(i, pg.TextItem)]
        assert len(scatters) == 1          # one item carrying both points
        assert len(labels) == 2

    def test_the_mark_sits_below_the_bar(self, armed):
        armed._tf_combo.setCurrentText("1m")
        k = klines()
        armed._draw_exhaust(k, "US.SOXL")
        import pyqtgraph as pg
        sc = next(i for i in armed._exhaust_items
                  if isinstance(i, pg.ScatterPlotItem))
        pts = sc.getData()
        lows = {round(v, 4) for v in k["low"]}
        for y in pts[1]:
            assert not any(abs(y - lo) < 1e-9 for lo in lows)
            assert y < max(lows)


class TestItStaysOff:
    def test_nothing_is_drawn_on_a_coarser_timeframe(self, armed):
        """The burst and the collapse live inside one minute."""
        armed._tf_combo.setCurrentText("5m")
        armed._draw_exhaust(klines(), "US.SOXL")
        assert armed._exhaust_items == []

    def test_a_session_with_no_marks_draws_nothing(self, viewer, monkeypatch):
        import analysis.trade_viewer_qt as tv
        monkeypatch.setattr(tv, "load_raw_ticks", lambda *a, **k: [])
        monkeypatch.setattr(tv, "compute_exhaustion_marks",
                            lambda *a, **k: [])
        viewer._exh_cache.clear()
        viewer._clear_exhaust_items()
        viewer._tf_combo.setCurrentText("1m")
        viewer._draw_exhaust(klines(), "US.SOXL")
        assert viewer._exhaust_items == []


class TestRedraw:
    def test_items_do_not_accumulate(self, armed):
        """The chart redraws on a one-second timer."""
        armed._tf_combo.setCurrentText("1m")
        armed._draw_exhaust(klines(), "US.SOXL")
        first = len(armed._exhaust_items)
        for _ in range(3):
            armed._clear_exhaust_items()
            armed._draw_exhaust(klines(), "US.SOXL")
        assert len(armed._exhaust_items) == first

    def test_clearing_removes_them_from_the_plot(self, armed):
        armed._tf_combo.setCurrentText("1m")
        armed._draw_exhaust(klines(), "US.SOXL")
        items = list(armed._exhaust_items)
        armed._clear_exhaust_items()
        assert armed._exhaust_items == []
        for item in items:
            assert item not in armed._plot_c.items

    def test_a_finished_day_is_computed_once(self, viewer, monkeypatch):
        """Three quarters of a million ticks must not be re-read each tick."""
        import analysis.trade_viewer_qt as tv
        calls = []
        monkeypatch.setattr(tv, "load_raw_ticks",
                            lambda *a, **k: calls.append(a) or [{"stub": 1}])
        monkeypatch.setattr(tv, "compute_exhaustion_marks",
                            lambda *a, **k: fake_marks("09:40"))
        viewer._exh_cache.clear()
        viewer._tf_combo.setCurrentText("1m")
        for _ in range(4):
            viewer._clear_exhaust_items()
            viewer._draw_exhaust(klines(), "US.SOXL")
        assert len(calls) == 1

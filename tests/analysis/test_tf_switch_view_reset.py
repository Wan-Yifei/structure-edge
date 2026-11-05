"""Switching timeframe re-fits the view, even if a repaint lands first.

The chart auto-fits once per new chart: _render compares a key against
_last_chart_key and, when it differs, schedules _reset_view for the bar
count it just drew. That key used to be read off the toolbar.

The toolbar moves first. Picking a timeframe changes the combo
immediately, and the bars stay the old ones until the fetch lands a second
or two later -- during which the redraw timer repaints, in Live mode,
every second or two. That repaint carried the NEW timeframe in its key
while drawing the OLD bars, so it banked the reset against them; the paint
that finally carried the new bars found its key already recorded and left
the view alone. Reported 2026-11-05 as "switching timeframe does not come
back home -- I have to press Home to see the chart at all", and it is that
bad because the bar count changes with the timeframe (390 at 1m, 78 at
5m): the stale range, and the stale limits _reset_view sets alongside it,
can leave the new bars off-screen and unreachable by scrolling.

The key now names the bars in hand, recorded by _on_data_ready from the
fetch that delivered them.
"""
from __future__ import annotations

import os

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6.QtWidgets")

from PyQt6.QtWidgets import QApplication        # noqa: E402

import analysis.trade_viewer_qt as tv           # noqa: E402

CODE = "US.SOXL"


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def viewer(qapp):
    v = tv.TradeViewerQt()
    v.resize(900, 600)
    for _ in range(3):
        qapp.processEvents()
    yield v
    v.close()


def _frame(n: int, freq: str) -> pd.DataFrame:
    t = pd.date_range("2026-11-05 09:31", periods=n, freq=freq)
    return pd.DataFrame({
        "time_key": t.strftime("%Y-%m-%d %H:%M:%S"),
        "open": 100.0, "close": 100.4,
        "high": 101.0, "low": 100.0,
        "volume": 1000.0}, index=range(n))


@pytest.fixture
def fits(viewer, monkeypatch):
    """Bar counts the view was fitted to, newest last."""
    seen: list[int] = []
    monkeypatch.setattr(type(viewer), "_reset_view",
                        lambda self, n, init_bars=150: seen.append(n))
    return seen


def _deliver(viewer, qapp, tf: str, n: int, freq: str,
             code: str = CODE, historical: bool = False) -> None:
    """A finished fetch, handed over the way the worker hands it over.

    Through _on_data_ready rather than by setting _klines directly: that is
    where the bars' identity is recorded, so a test that sets it by hand
    cannot tell whether anything records it at all.
    """
    frame = _frame(n, freq)
    viewer._on_data_ready({
        "new_tick_days": {},
        "klines":        frame,
        "warmup":        frame,
        "smc_signals":   [],
        "fvg_gaps":      [],
        "ob_blocks":     [],
        "ticks":         {},
        "candle_mins":   1,
        "historical":    historical,
        "date_str":      "",
        "tf":            tf,
        "code":          code,
    })
    for _ in range(3):
        qapp.processEvents()          # the reset is deferred by singleShot


class TestSwitchingTimeframe:

    def test_the_new_bars_are_fitted(self, viewer, qapp, fits):
        viewer._code_edit.setText(CODE)
        _deliver(viewer, qapp, "1m", 40, "1min")
        fits.clear()

        viewer._tf_combo.setCurrentText("5m")
        _deliver(viewer, qapp, "5m", 8, "5min")

        assert fits, "the view was never fitted to the new timeframe"
        assert fits[-1] == 8, f"fitted to {fits[-1]} bars, not the 8 drawn"

    def test_a_repaint_while_the_fetch_runs_does_not_consume_it(
            self, viewer, qapp, fits):
        """The reported case. The redraw timer repaints the old bars in the
        second or two between the combo moving and the bars arriving."""
        viewer._code_edit.setText(CODE)
        _deliver(viewer, qapp, "1m", 40, "1min")
        fits.clear()

        viewer._tf_combo.setCurrentText("5m")
        viewer._redraw()                      # old bars, new toolbar
        for _ in range(3):
            qapp.processEvents()
        _deliver(viewer, qapp, "5m", 8, "5min")

        assert fits, "a repaint before the data arrived ate the view reset"
        assert fits[-1] == 8

    def test_leaving_historical_for_live_refits(self, viewer, qapp, fits):
        """Symbol, timeframe and date can all agree across that switch while
        the bar count does not: Historical loads a multi-day lookback, Live
        one session. Without the mode in the key the two frames are
        indistinguishable, and the stale repaint eats the reset again."""
        viewer._code_edit.setText(CODE)
        _deliver(viewer, qapp, "5m", 160, "5min", historical=True)
        fits.clear()

        viewer._redraw()                      # the old long frame, repainted
        for _ in range(3):
            qapp.processEvents()
        _deliver(viewer, qapp, "5m", 78, "5min", historical=False)

        assert fits, "going live kept the view fitted to the lookback"
        assert fits[-1] == 78

    def test_switching_symbols_is_the_same_story(self, viewer, qapp, fits):
        viewer._code_edit.setText(CODE)
        _deliver(viewer, qapp, "1m", 40, "1min")
        fits.clear()

        viewer._code_edit.setText("US.SOXS")
        viewer._redraw()
        for _ in range(3):
            qapp.processEvents()
        _deliver(viewer, qapp, "1m", 25, "1min", code="US.SOXS")

        assert fits and fits[-1] == 25


class TestWhenItMustNotFit:

    def test_a_plain_repaint_leaves_the_view_alone(self, viewer, qapp, fits):
        """Every live repaint fitting the view would fight the user's zoom."""
        viewer._code_edit.setText(CODE)
        _deliver(viewer, qapp, "1m", 40, "1min")
        fits.clear()

        viewer._redraw()
        for _ in range(3):
            qapp.processEvents()

        assert fits == []

    def test_live_bars_arriving_do_not_refit(self, viewer, qapp, fits):
        """Same chart, one bar longer -- the key is unchanged on purpose."""
        viewer._code_edit.setText(CODE)
        _deliver(viewer, qapp, "1m", 40, "1min")
        fits.clear()

        _deliver(viewer, qapp, "1m", 41, "1min")

        assert fits == []

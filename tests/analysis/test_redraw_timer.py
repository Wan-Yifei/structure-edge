"""Repainting runs on its own clock, separately from fetching.

A live cycle used to do two unrelated things on one timer: one
get_cur_kline, and a full _render. But the tape does not come from that
call -- ticks are pushed into _live_ticks by the TICKER handler and
ticks.db is kept by the collector, so everything the tape-driven
overlays need is already in hand between fetches. There are now two
timers, and repainting costs no API calls at all.

The trap this is mostly here to guard is the tick mapping. _render
overwrites self._ticks with the MERGED mapping, and its merge skips a
live bucket that is already present:

    for bk, pd_ in self._live_ticks.items():
        if bk in buckets:
            continue

So handing self._ticks back to _render freezes the newest bucket -- the
only one that is still moving, and the entire point of repainting. The
pristine mapping is kept as self._db_ticks, and
test_feeding_the_merged_mapping_back_freezes_it pins that the trap is
real rather than imagined.
"""

import os
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6.QtWidgets")

from PyQt6.QtWidgets import QApplication        # noqa: E402

import analysis.trade_viewer_qt as tv           # noqa: E402
from core.time_utils import candle_start        # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(scope="module")
def viewer(qapp):
    v = tv.TradeViewerQt()
    v.resize(1000, 700)
    v.show()
    for _ in range(5):
        qapp.processEvents()
    yield v
    v.close()


N = 20
PRICE = 100.5


def frame():
    """A frame whose last bar's bucket exists only in _live_ticks.

    That is the live case: ticks.db lags the tape by a bucket or two, so
    the newest bucket is the pushed one.
    """
    t = pd.date_range("2026-10-23 09:31", periods=N, freq="1min")
    kl = pd.DataFrame({
        "time_key": t.strftime("%Y-%m-%d %H:%M:%S"),
        "open": 100.0, "close": 100.4,
        "high": 101.0, "low": 100.0,
        "volume": 1000.0}, index=range(N))
    keys = [candle_start(x.to_pydatetime() - timedelta(minutes=1), 1)
            for x in t]
    db = {k: {PRICE: {"buy": 100, "sell": 40, "neutral": 5}}
          for k in keys[:-1]}
    return kl, db, keys[-1]


@pytest.fixture(autouse=True)
def live(viewer):
    viewer._mode_combo.setCurrentText("Live")
    viewer._live_ticks.clear()
    viewer._flow_cache = None
    viewer._prev_frame = None
    viewer._profile_cache = {}
    viewer._bk_sig = None
    viewer._tick_seq = 0
    viewer._drawn_tick_seq = -1
    yield


def arm(viewer, buy=200):
    kl, db, last_bk = frame()
    viewer._klines = kl
    viewer._db_ticks = db
    viewer._live_ticks[last_bk][PRICE]["buy"] += buy
    return kl, db, last_bk


class TestTheNewestBucketKeepsMoving:
    def test_a_second_redraw_sees_new_live_ticks(self, viewer):
        kl, db, bk = arm(viewer)
        viewer._redraw()
        before = viewer._ticks[bk][PRICE]["buy"]
        viewer._live_ticks[bk][PRICE]["buy"] += 500
        viewer._redraw()
        assert viewer._ticks[bk][PRICE]["buy"] == before + 500, \
            "the newest bucket stopped updating after the first repaint"

    def test_feeding_the_merged_mapping_back_freezes_it(self, viewer):
        """Why _db_ticks exists. Not a bug being pinned -- a trap."""
        kl, db, bk = arm(viewer)
        viewer._redraw()
        before = viewer._ticks[bk][PRICE]["buy"]
        viewer._live_ticks[bk][PRICE]["buy"] += 500
        viewer._render(kl, viewer._ticks)            # the wrong source
        assert viewer._ticks[bk][PRICE]["buy"] == before

    def test_db_ticks_is_not_what_render_leaves_behind(self, viewer):
        kl, db, bk = arm(viewer)
        viewer._redraw()
        assert viewer._db_ticks is db
        assert viewer._ticks is not db
        assert bk in viewer._ticks and bk not in db


class TestAnIdleTapeCostsNothing:
    def test_no_new_prints_means_no_repaint(self, viewer, monkeypatch):
        arm(viewer)
        viewer._redraw()
        calls = []
        real = type(viewer)._render
        monkeypatch.setattr(type(viewer), "_render",
                            lambda self, k, t: (calls.append(1),
                                                real(self, k, t))[1])
        viewer._on_redraw_tick()
        assert calls == [], "repainted an unchanged picture"

    def test_a_new_print_means_a_repaint(self, viewer, monkeypatch):
        _, _, bk = arm(viewer)
        viewer._redraw()
        calls = []
        real = type(viewer)._render
        monkeypatch.setattr(type(viewer), "_render",
                            lambda self, k, t: (calls.append(1),
                                                real(self, k, t))[1])
        viewer._tick_seq += 1                 # what the handler does
        viewer._on_redraw_tick()
        assert len(calls) == 1

    def test_render_records_what_it_drew(self, viewer):
        arm(viewer)
        viewer._tick_seq = 7
        viewer._redraw()
        assert viewer._drawn_tick_seq == 7

    def test_a_print_arriving_mid_paint_is_not_lost(self, viewer):
        """_render stamps the sequence it STARTED with, so a print that
        lands while it paints still triggers the next repaint."""
        arm(viewer)
        viewer._tick_seq = 3
        viewer._redraw()
        viewer._tick_seq = 4                  # arrived after the stamp
        assert viewer._drawn_tick_seq != viewer._tick_seq

    def test_nothing_to_draw_is_harmless(self, viewer):
        viewer._klines = None
        viewer._on_redraw_tick()
        viewer._redraw()


class TestTheControl:
    def test_the_floor_is_half_a_second(self, viewer):
        """A render is ~220ms; 0.3s would be a 73% duty cycle."""
        assert viewer._redraw_spin.minimum() == pytest.approx(0.5)
        assert viewer._redraw_spin.singleStep() == pytest.approx(0.5)

    def test_changing_it_retimes_a_running_timer(self, viewer):
        viewer._redraw_timer.start(5000)
        viewer._redraw_spin.setValue(2.0)
        assert viewer._redraw_timer.interval() == 2000
        viewer._redraw_timer.stop()

    def test_changing_it_does_not_start_a_stopped_timer(self, viewer):
        viewer._redraw_timer.stop()
        viewer._redraw_spin.setValue(1.5)
        assert not viewer._redraw_timer.isActive()

    def test_it_is_a_separate_timer_from_the_fetch(self, viewer):
        assert viewer._redraw_timer is not viewer._refresh_timer


class TestTheCounterTheWholeThingRestsOn:
    """_tick_seq is what tells the timer there is anything to paint.

    If it never moved, the chart would simply never repaint between
    fetches -- the feature silently doing nothing. A mutation that
    removed the bump passed every other test in this file, which is why
    the ingest was lifted out of the subscription handler: so it can be
    called without a live feed.
    """

    @staticmethod
    def _frame(rows):
        return pd.DataFrame(
            [{"time": t, "price": p, "volume": v, "ticker_direction": d}
             for t, p, v, d in rows])

    def test_a_pushed_frame_bumps_the_counter(self, viewer):
        before = viewer._tick_seq
        viewer._ingest_ticks(self._frame([
            ("2026-10-23 09:31:02", 100.5, 300, "BUY")]), 1)
        assert viewer._tick_seq == before + 1

    def test_once_per_frame_not_once_per_row(self, viewer):
        before = viewer._tick_seq
        viewer._ingest_ticks(self._frame([
            ("2026-10-23 09:31:02", 100.5, 300, "BUY"),
            ("2026-10-23 09:31:03", 100.6, 100, "SELL"),
            ("2026-10-23 09:31:04", 100.6, 50, "NEUTRAL")]), 1)
        assert viewer._tick_seq == before + 1

    def test_the_volumes_land_in_the_right_bucket_and_side(self, viewer):
        viewer._ingest_ticks(self._frame([
            ("2026-10-23 09:31:02", 100.5, 300, "BUY"),
            ("2026-10-23 09:31:07", 100.5, 120, "SELL"),
            ("2026-10-23 09:31:09", 100.5, 40, "NOTHING")]), 1)
        bk = candle_start(
            pd.Timestamp("2026-10-23 09:31:02").to_pydatetime(), 1)
        lvl = viewer._live_ticks[bk][100.5]
        assert lvl["buy"] == 300 and lvl["sell"] == 120
        assert lvl["neutral"] == 40, "未知方向应计入 neutral"

    def test_a_push_makes_the_next_tick_repaint(self, viewer, monkeypatch):
        """End to end: a pushed frame, then the timer repaints."""
        arm(viewer)
        viewer._redraw()
        calls = []
        real = type(viewer)._render
        monkeypatch.setattr(type(viewer), "_render",
                            lambda self, k, t: (calls.append(1),
                                                real(self, k, t))[1])
        viewer._on_redraw_tick()
        assert calls == []
        viewer._ingest_ticks(self._frame([
            ("2026-10-23 09:45:02", 100.5, 300, "BUY")]), 1)
        viewer._on_redraw_tick()
        assert len(calls) == 1

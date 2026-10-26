"""The session in progress: shown early, marked, and refreshed alone.

A reload was measured at 9922ms, of which 9913ms is the fetch and 4.7ms
the binning. So three things follow, and each is pinned here:

  a 30s refresh must not be a reload -- it refetches the newest trading
  day only, and the rest comes from the cache;
  a control that changes only the drawing (bins, VA%, join) must not
  fetch at all, which the old wiring did on every keystroke;
  the refresh button must still really go back to the feed.

The live session is also let in at 15 bars rather than 60, and drawn so
it cannot be read as finished: a value area built on fifteen minutes
will keep moving all day, and that is exactly the comparison this panel
invites.

fetch_klines is replaced and counted, so "did not fetch" is an assertion
rather than a hope.
"""
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

pytest.importorskip("PyQt6.QtWidgets")


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app
    app.processEvents()


def _session(day, base, n=390, start="09:31", amp=1.5):
    t = pd.date_range(f"{day} {start}", periods=n, freq="1min")
    i = np.arange(n, dtype=float)
    c = base + np.sin(i / 40) * amp + i * 0.002
    vol = 1000 + 4000 * np.exp(-((i - n / 2) ** 2) / (2 * (n / 8) ** 2))
    return pd.DataFrame(
        {"time_key": t.strftime("%Y-%m-%d %H:%M:%S"),
         "open": c, "high": c + 0.25, "low": c - 0.25, "close": c,
         "volume": vol})


class _Call(tuple):
    """A recorded fetch: still (start, end), plus what it asked for.

    A tuple subclass so every assertion written against the old log --
    `start, end = calls[0]`, `calls == []` -- keeps working unchanged.
    """
    fresh = False


@pytest.fixture
def live(qapp, monkeypatch):
    """Three finished sessions plus a short one dated 'today'."""
    from analysis.multi_vp_window import MultiVPWindow

    today = MultiVPWindow._today()
    past = [str((pd.Timestamp(today) - pd.Timedelta(days=d)).date())
            for d in (5, 4, 3)]
    frames = [_session(d, 100 + k * 2, amp=1.0 + k * 0.8)
              for k, d in enumerate(past)]
    calls = []

    def fake(code, tf, start, end, *a, **k):
        c = _Call((start, end))
        c.fresh = bool(k.get("force_refresh"))
        calls.append(c)
        return pd.concat(frames + [TODAY[0]], ignore_index=True)

    TODAY = [_session(today, 108, n=20)]       # 20 bars so far
    import feeds.fetcher as fetcher
    monkeypatch.setattr(fetcher, "fetch_klines", fake)

    w = MultiVPWindow()
    w._live_timer.stop()                       # drive it by hand
    w.show()                                   # the tick skips a hidden panel
    w._days.blockSignals(True); w._days.setValue(4)
    w._days.blockSignals(False)
    w.reload()
    yield w, today, past, calls, TODAY
    w.close()


class TestTheBoundary:
    def test_today_rolls_at_20_00(self, qapp, monkeypatch):
        """19:59 stays put, 20:00 belongs to the next trading day --
        the same rule the viewer's CVD resets on."""
        from analysis import multi_vp_window as m
        for stamp, want in [("2026-10-22 19:59", "2026-10-22"),
                            ("2026-10-22 20:00", "2026-10-23"),
                            ("2026-10-22 09:30", "2026-10-22"),
                            ("2026-10-23 03:00", "2026-10-23")]:
            monkeypatch.setattr(m.pd.Timestamp, "now",
                                staticmethod(lambda s=stamp: pd.Timestamp(s)))
            assert m.MultiVPWindow._today() == want, stamp


class TestTheLiveSessionIsLetInEarly:
    def test_twenty_bars_is_enough_for_today(self, live):
        w, today, past, _, _ = live
        assert [d for d, _ in w._sessions()][-1] == today
        assert w._live_day == today

    def test_but_not_for_a_finished_day(self, qapp, monkeypatch):
        """A completed session with a handful of bars is bad data, and
        the 60-bar floor still applies to it."""
        from analysis.multi_vp_window import MultiVPWindow
        today = MultiVPWindow._today()
        stub = str((pd.Timestamp(today) - pd.Timedelta(days=4)).date())
        good = str((pd.Timestamp(today) - pd.Timedelta(days=3)).date())
        import feeds.fetcher as fetcher
        monkeypatch.setattr(
            fetcher, "fetch_klines",
            lambda *a, **k: pd.concat(
                [_session(stub, 100, n=20), _session(good, 101)],
                ignore_index=True))
        w = MultiVPWindow()
        w._live_timer.stop()
        try:
            assert [d for d, _ in w._sessions()] == [good]
        finally:
            w.close()

    def test_the_floor_for_today_is_the_constant(self, qapp, monkeypatch):
        from analysis.multi_vp_window import MultiVPWindow, LIVE_MIN_BARS
        today = MultiVPWindow._today()
        import feeds.fetcher as fetcher
        monkeypatch.setattr(
            fetcher, "fetch_klines",
            lambda *a, **k: _session(today, 100, n=LIVE_MIN_BARS - 1))
        w = MultiVPWindow()
        w._live_timer.stop()
        try:
            assert w._sessions() == []
        finally:
            w.close()


class TestItIsDrawnAsUnfinished:
    def test_the_tick_carries_a_mark(self, live):
        w, today, past, _, _ = live
        ax = w._plot.getPlotItem().getAxis("bottom")
        labels = [t[1] for t in ax._tickLevels[0]]
        assert labels[-1].endswith("·"), labels
        assert not any(l.endswith("·") for l in labels[:-1]), labels

    def test_the_status_says_what_the_mark_means(self, live):
        w, *_ = live
        assert "in progress" in w._status.text()

    def test_a_closed_window_has_no_mark(self, qapp, monkeypatch):
        from analysis.multi_vp_window import MultiVPWindow
        today = MultiVPWindow._today()
        past = [str((pd.Timestamp(today) - pd.Timedelta(days=d)).date())
                for d in (5, 4, 3)]
        import feeds.fetcher as fetcher
        monkeypatch.setattr(
            fetcher, "fetch_klines",
            lambda *a, **k: pd.concat([_session(d, 100) for d in past],
                                      ignore_index=True))
        w = MultiVPWindow()
        w._live_timer.stop()
        try:
            w.reload()
            assert w._live_day is None
            assert "in progress" not in w._status.text()
        finally:
            w.close()


class TestTheFetchIsNotRepeated:
    def test_a_drawing_control_does_not_fetch(self, live):
        """bins / VA% / join change nothing about the data, and the old
        wiring paid 9.9s for each one."""
        w, _, _, calls, _ = live
        n = len(calls)
        w._bins.blockSignals(True); w._bins.setValue(w._bins.value() + 10)
        w._bins.blockSignals(False)
        w.reload()
        assert len(calls) == n, f"refetched {len(calls) - n} time(s)"

    def test_the_live_tick_asks_for_a_few_days(self, live):
        w, _, _, calls, _ = live
        calls.clear()
        w.refresh_live()
        assert len(calls) == 1
        start, end = calls[0]
        span = (pd.Timestamp(end) - pd.Timestamp(start)).days
        assert span <= 5, f"a live tick pulled {span} days"

    def test_the_live_tick_picks_up_new_bars(self, live):
        w, today, _, _, TODAY = live
        before = len(dict(w._sessions())[today])
        TODAY[0] = _session(today, 108, n=60)
        w.refresh_live()
        after = len(dict(w._sessions())[today])
        assert after > before, f"{before} -> {after}"

    def test_the_refresh_button_really_refetches(self, live):
        w, _, _, calls, _ = live
        calls.clear()
        w.reload_all()
        assert len(calls) == 1
        start, end = calls[0]
        assert (pd.Timestamp(end) - pd.Timestamp(start)).days > 5

    def test_changing_rth_drops_the_cache(self, live):
        """The cache key carries it: the same day filtered differently is
        a different session."""
        w, _, _, calls, _ = live
        calls.clear()
        w._rth.blockSignals(True)
        w._rth.setChecked(not w._rth.isChecked())
        w._rth.blockSignals(False)
        w.reload()
        assert len(calls) == 1, "an RTH change must go back to the feed"

    def test_more_days_than_cached_refetches(self, live):
        w, _, _, calls, _ = live
        calls.clear()
        w._days.blockSignals(True); w._days.setValue(w._days.value() + 3)
        w._days.blockSignals(False)
        w.reload()
        assert len(calls) == 1

    def test_only_the_live_slab_is_dashed(self, live):
        """The mark that is actually on the profile, not beside it.

        Join VA is turned off first: those connectors are dashed too, so
        with them on "exactly one dashed curve" would not mean what it
        says.
        """
        from PyQt6.QtCore import Qt
        import pyqtgraph as pg

        w, *_ = live
        w._join.blockSignals(True); w._join.setChecked(False)
        w._join.blockSignals(False)
        w.reload()

        curves = [it for it in w._items if isinstance(it, pg.PlotCurveItem)]
        styles = [c.opts["pen"].style() for c in curves]
        assert len(curves) >= 2, f"expected one outline per session, got {styles}"
        assert styles[-1] == Qt.PenStyle.DashLine, styles
        assert all(s == Qt.PenStyle.SolidLine for s in styles[:-1]), styles


    def test_a_hidden_panel_does_not_fetch(self, live):
        """The fetch blocks the GUI thread, which the main chart shares.
        A panel nobody is looking at must not stall it every half
        minute."""
        w, _, _, calls, _ = live
        w.hide()
        calls.clear()
        w.refresh_live()
        assert calls == [], "a hidden panel went to the feed"
        w.show()
        w.refresh_live()
        assert len(calls) == 1, "and it must resume once shown"


class TestRthIsOffByDefault:
    """The switch exists because extended hours are thin and stretch the
    value area. It defaults OFF anyway, because with it on the session in
    progress cannot appear until about fifteen minutes after the open --
    so for most of the day the panel stops at the last closed session and
    looks like stale data.

    That is exactly how it was first read: at 00:56 on a Monday the panel
    ended at the previous Friday and the question was "why is it three
    days behind".
    """

    def test_the_default(self, qapp, monkeypatch):
        from analysis.multi_vp_window import MultiVPWindow
        import feeds.fetcher as fetcher
        monkeypatch.setattr(fetcher, "fetch_klines",
                            lambda *a, **k: _session("2026-10-20", 100))
        w = MultiVPWindow()
        w._live_timer.stop()
        try:
            assert w._rth.isChecked() is False
        finally:
            w.close()

    def test_an_overnight_only_session_shows_with_rth_off(self, qapp,
                                                          monkeypatch):
        """Today, a few hours before the open: bars exist, none of them
        in regular hours."""
        from analysis.multi_vp_window import MultiVPWindow
        today = MultiVPWindow._today()
        past = [str((pd.Timestamp(today) - pd.Timedelta(days=d)).date())
                for d in (5, 4, 3)]
        # Overnight belongs to `today` under the 20:00 boundary, so it is
        # stamped on the previous evening.
        prev_eve = str((pd.Timestamp(today) - pd.Timedelta(days=1)).date())
        frames = [_session(d, 100) for d in past]
        frames.append(_session(prev_eve, 108, n=200, start="20:01"))
        import feeds.fetcher as fetcher
        monkeypatch.setattr(fetcher, "fetch_klines",
                            lambda *a, **k: pd.concat(frames,
                                                      ignore_index=True))
        w = MultiVPWindow()
        w._live_timer.stop()
        w._days.blockSignals(True); w._days.setValue(4)
        w._days.blockSignals(False)
        try:
            assert w._rth.isChecked() is False
            days = [d for d, _ in w._sessions()]
            assert days[-1] == today, f"RTH off 时今天应当出现: {days}"
            assert w._live_day == today

            w._rth.blockSignals(True); w._rth.setChecked(True)
            w._rth.blockSignals(False)
            days = [d for d, _ in w._sessions()]
            assert today not in days, (
                f"RTH on 时今天没有盘中数据，不该出现: {days}")
            assert w._live_day is None
        finally:
            w.close()

class TestTheTickActuallyGetsNewBars:
    """The panel refreshed every 30s and showed the same bars all day.

    Measured on the real cache: SOXL 1m held 53 bars for 2026-10-26,
    the newest stamped 00:52, while the market was trading five dollars
    higher. Every tick since had been a cache hit.

    The store is not wrong to do that -- `missing_days` counts today as
    present as soon as it holds one bar, which is what keeps a backtest
    off the API -- so the fix is at this call site, and these pin it
    there rather than in the store.
    """

    def test_the_tick_asks_for_fresh_data(self, live):
        w, _, _, calls, _ = live
        calls.clear()
        w.refresh_live()
        assert len(calls) == 1
        assert calls[0].fresh is True, (
            "a live tick that honours the cache replays the morning's bars")

    def test_a_reload_that_does_fetch_does_not_force(self, live):
        """Only the tail is worth paying for: the cache is right about
        every finished day behind it, and forcing would throw away a
        9.9s fetch to re-learn what it already knows.

        Days is raised first because a reload the cache can serve does
        not fetch at all -- pinned elsewhere in this file."""
        w, _, _, calls, _ = live
        w._days.blockSignals(True); w._days.setValue(12)
        w._days.blockSignals(False)
        calls.clear()
        w.reload()
        assert len(calls) == 1
        assert calls[0].fresh is False

    def test_the_tick_stays_narrow_when_days_outruns_the_feed(self, live):
        """Days above what the feed can serve left the cache permanently
        short, and the 'cannot serve this' branch then widened every
        single tick to the whole window -- a 26-day fetch every 30s, on
        the thread the main chart draws on."""
        w, _, _, calls, _ = live
        w._days.blockSignals(True); w._days.setValue(12)
        w._days.blockSignals(False)
        calls.clear()
        w.refresh_live()
        assert len(calls) == 1
        start, end = calls[0]
        assert (pd.Timestamp(end) - pd.Timestamp(start)).days == 2, \
            f"the tick widened to {start}..{end}"

    def test_the_status_shows_the_newest_bar(self, live):
        """'refreshed every 30s' is a promise about the timer, not about
        the data: a stopped feed reads identically. The bar's own clock
        is what makes a freeze visible."""
        w, _, _, _, TODAY = live
        last = pd.to_datetime(TODAY[0].time_key.iloc[-1])
        assert last.strftime("%H:%M") in w._status.text(), w._status.text()

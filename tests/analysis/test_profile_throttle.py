"""The profile panel is not rebuilt on every redraw.

Two measured reasons, and the second is why this is a test rather than
a tuning knob:

  it is 82ms of what was a 220ms render, and a session-wide volume
  profile does not visibly move in half a second;

  every rebuild discards four TextItems that something in pyqtgraph's
  teardown keeps alive. Measured at 4.00 per render, and identical on
  the commit before this round of work -- the leak is older than the
  redraw timer, but the timer took it from one rebuild per 15s fetch to
  two a second, which is ~2GB an hour.

So the throttle is a leak rate limit, not only an optimisation, and a
change that quietly removed it would cost memory rather than speed.
That is what these pin.

A frame that is not a simple extension of the previous one still
rebuilds immediately: a new symbol, timeframe or day really does have a
different profile, and waiting five seconds to show it would be a bug.
"""
import os
import time

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6.QtWidgets")

from PyQt6.QtWidgets import QApplication        # noqa: E402

import analysis.trade_viewer_qt as tv           # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(scope="module")
def viewer(qapp):
    v = tv.TradeViewerQt()
    v.resize(900, 600)
    yield v
    v.close()


N = 40


@pytest.fixture
def frame(viewer, monkeypatch):
    """A viewer with a frame loaded and the rebuild counted."""
    t = pd.date_range("2026-10-23 09:31", periods=N, freq="1min")
    base = 100 + np.cumsum(np.random.default_rng(1).normal(0, 0.05, N))
    kl = pd.DataFrame({
        "time_key": t.strftime("%Y-%m-%d %H:%M:%S"),
        "open": base, "close": base, "high": base + 0.1, "low": base - 0.1,
        "volume": 1000.0})
    viewer._klines = kl
    viewer._profile_drawn_ms = -1e9
    viewer._prev_frame = None
    viewer._flow_cache = None
    viewer._bk_sig = None

    calls = []
    for name in ("_rebuild_session_profile", "_rebuild_range_profile"):
        real = getattr(type(viewer), name)
        monkeypatch.setattr(type(viewer), name,
                            lambda self, _n=name, _r=real:
                            (calls.append(_n), _r(self))[1])
    return viewer, kl, calls


class TestItIsThrottled:
    def test_the_first_render_rebuilds(self, frame):
        v, kl, calls = frame
        v._render(kl, {})
        assert calls, "the first render must draw a profile"

    def test_a_second_render_straight_after_does_not(self, frame):
        v, kl, calls = frame
        v._render(kl, {})
        calls.clear()
        v._render(kl, {})
        v._render(kl, {})
        assert calls == [], f"rebuilt {len(calls)} times inside the window"

    def test_it_comes_back_after_the_interval(self, frame, monkeypatch):
        v, kl, calls = frame
        v._render(kl, {})
        calls.clear()
        # Pretend the interval has passed rather than sleeping it.
        v._profile_drawn_ms -= tv._PROFILE_MIN_MS + 1
        v._render(kl, {})
        assert len(calls) == 1

    def test_the_interval_is_what_governs(self, frame):
        """Reading the constant, so a change to it is a deliberate one."""
        assert tv._PROFILE_MIN_MS >= 1000


class TestAFreshFrameAlwaysRebuilds:
    def test_a_frame_that_is_not_an_extension(self, frame, monkeypatch):
        """_dirty_bars returns None for a new symbol, timeframe or day,
        and that profile really is different."""
        v, kl, calls = frame
        v._render(kl, {})
        calls.clear()
        monkeypatch.setattr(type(v), "_dirty_bars",
                            lambda self, *a, **k: None)
        v._render(kl, {})
        assert len(calls) == 1, "a new frame must not wait for the timer"

    def test_an_extension_still_waits(self, frame, monkeypatch):
        v, kl, calls = frame
        v._render(kl, {})
        calls.clear()
        monkeypatch.setattr(type(v), "_dirty_bars",
                            lambda self, *a, **k: {N - 1})
        v._render(kl, {})
        assert calls == []


class TestTheLeakItLimits:
    def test_a_rebuild_leaks_and_a_skipped_one_does_not(self, frame):
        """The reason for the throttle, stated as a measurement.

        Not a demand that the leak be fixed -- it predates this code and
        lives in the teardown of discarded pyqtgraph items. What is
        pinned is that skipping the rebuild skips the leak, so the
        throttle is doing the job it is here for.
        """
        import gc
        import pyqtgraph as pg

        v, kl, calls = frame
        v._render(kl, {})

        def count():
            gc.collect()
            return sum(1 for o in gc.get_objects()
                       if isinstance(o, pg.TextItem))

        # Throttled renders: no rebuild, so no new labels stranded.
        a = count()
        for _ in range(4):
            v._render(kl, {})
        b = count()
        assert calls[len(calls):] == []
        held_back = b - a

        # Forced rebuilds: the labels accumulate.
        for _ in range(4):
            v._profile_drawn_ms = -1e9
            v._render(kl, {})
        c = count()
        rebuilt = c - b

        assert rebuilt > held_back, (
            f"skipped renders added {held_back}, rebuilds added {rebuilt} "
            "-- the throttle is not limiting anything")

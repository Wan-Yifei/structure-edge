"""Walls from a dead fetch must not stay on the chart forever.

_on_option_fetch_error used to log and retitle an 8pt label while leaving
self._option_wall_data in place -- and _render repaints from that cache on
every tick. OpenD dropping out, the chain rate limit, a symbol with no
options, or 0DTE ticked on a day nothing expires all land in that handler,
and each left the last good levels drawn with nothing saying how old they
were.

Everything here drives the REAL entry point, w._render(), rather than
_draw_option_walls: the bug was never that the drawing code is broken, it is
that the cache survives, and only _render shows that. The fixture also
asserts the indicator is on BY DEFAULT, so these cannot pass against a
configuration the user never runs.
"""
import os

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6.QtWidgets")

from PyQt6.QtWidgets import QApplication        # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app
    app.processEvents()


@pytest.fixture(scope="module")
def _viewer(qapp):
    """One window for the whole module.

    Building a TradeViewerQt per test opens an OpenD context and a set of
    timers each time; ten of them in one process hung the run outright. The
    poll timer is stopped so a live background fetch cannot overwrite the
    state these tests are asserting on.
    """
    from analysis.trade_viewer_qt import TradeViewerQt
    w = TradeViewerQt()
    # Not a test setting: this is the shipped default, and an overlay that
    # only works when a test turns it on is the failure mode this guards.
    assert w._ind("option_walls"), "option_walls is off by default -- rewrite"
    w._option_timer.stop()
    # Construction kicks off one poll before the timer is stopped. Drain it
    # HERE, once: dropping the last reference to a running QThread aborts
    # the process, and waiting per-test instead cost ~50 s on a file whose
    # assertions take 2 s.
    f = w._option_fetcher
    if f is not None and f.isRunning():
        f.wait(5000)
    yield w

    # Close the OpenD contexts the window opened, not just the widget. Their
    # network threads are non-daemon: the tests themselves finish in ~2 s,
    # but the process then sits there forever waiting on them, which looks
    # exactly like a hung test run.
    for attr in ("_option_ctx", "_ctx"):
        ctx = getattr(w, attr, None)
        if ctx is not None:
            try:
                ctx.close()
            except Exception:                       # noqa: BLE001
                pass
            setattr(w, attr, None)
    w.close()
    qapp.processEvents()


@pytest.fixture
def win(_viewer):
    """Reset the option state the shared window carries between tests.

    The fetcher needs care: constructing the window kicks off one poll, and
    dropping the last reference to a QThread that is still running aborts
    the process outright ("QThread: Destroyed while thread is still
    running") -- it killed the whole file before any test reported.
    """
    f = _viewer._option_fetcher
    assert f is None or not f.isRunning(), (
        "a fetcher thread is still running; the module fixture should have "
        "drained it, and nulling the reference here would abort the process")
    _viewer._option_fetcher = None
    _viewer._option_wall_data = None
    _viewer._option_fail_streak = 0
    _viewer._option_code = ""
    _viewer._clear_option_wall_items()
    _viewer._option_status_lbl.setText("")
    return _viewer


def _klines(n: int = 20) -> pd.DataFrame:
    idx = pd.date_range("2026-11-20 09:30", periods=n, freq="1min")
    return pd.DataFrame({
        "time_key": idx, "open": 100.0, "high": 101.0,
        "low": 99.0, "close": 100.5, "volume": 1000.0,
    })


def _cache(coverage=None) -> dict:
    from datetime import datetime
    return {
        "call_walls": [{"strike": 105.0, "gex": 9e5}, {"strike": 106.0, "gex": 5e5}],
        "put_walls":  [{"strike": 95.0, "gex": -9e5}, {"strike": 94.0, "gex": -5e5}],
        "zero_gamma": 100.0, "net_gex": -1e6, "coverage": coverage,
        "expiries": ["2026-11-27"], "spot": 100.0,
        "computed_at": datetime.now(), "large_trades": [], "volumes": {},
    }


def _drawn(w) -> int:
    return len(w._option_wall_items)


class _Sig:
    def connect(self, *_a, **_k):
        pass


class _StubFetcher:
    """Stands in for OptionChainFetcher so no thread and no socket appear."""

    def __init__(self, *_a, **_k):
        self.ready = _Sig()
        self.error = _Sig()

    def start(self):
        pass

    def isRunning(self):
        return False


def _switch_symbol(win, monkeypatch, code: str):
    """Run _trigger_option_fetch's symbol-change block, hermetically.

    Two guards sit in front of it: it opens a real OpenD context, and it
    bails out if a previous poll is still in flight. Left alone, this test
    races a live background fetch on a machine where OpenD is up, and takes
    the opposite early return on one where it is down -- wrong in both
    directions for reasons that have nothing to do with the code under test.
    """
    import analysis.trade_viewer_qt as tv
    monkeypatch.setattr(tv, "OptionChainFetcher", _StubFetcher)
    monkeypatch.setattr(win, "_ensure_option_ctx", lambda: None)
    win._option_ctx = object()          # non-None: get past the context guard
    win._option_fetcher = None          # nothing in flight
    win._code_edit.setText(code)
    win._trigger_option_fetch()


class TestTheWallsGoAwayWhenTheDataDoes:

    def test_the_cache_is_what_puts_them_on_screen(self, win):
        """The ruler for every other test here: without it, 'they are gone'
        and 'they were never drawn' are the same observation."""
        k = _klines()
        win._render(k, None)
        assert _drawn(win) == 0, "nothing cached, yet something was drawn"
        win._option_wall_data = _cache()
        win._render(k, None)
        assert _drawn(win) > 0, "a populated cache drew nothing"

    def test_one_failure_keeps_them(self, win):
        """A single refusal is usually the rate limit, which clears itself.
        Dropping on the first error makes the lines blink once per poll."""
        k = _klines()
        win._option_wall_data = _cache()
        win._render(k, None)
        before = _drawn(win)

        win._on_option_fetch_error("transient")
        win._render(k, None)

        assert _drawn(win) == before
        assert win._option_wall_data is not None
        assert "stale 1/" in win._option_status_lbl.text()

    def test_they_are_dropped_once_the_grace_runs_out(self, win):
        from analysis.trade_viewer_qt import _OPTION_STALE_POLLS
        k = _klines()
        win._option_wall_data = _cache()
        win._render(k, None)
        assert _drawn(win) > 0

        for _ in range(_OPTION_STALE_POLLS + 1):
            win._on_option_fetch_error("still down")
        win._render(k, None)

        # BOTH, deliberately. Clearing only the items looks fixed until the
        # next _render repaints them from the surviving cache -- which is a
        # near relative of the bug being fixed; clearing only the cache
        # leaves the current frame's lines up. One assertion catches one.
        assert win._option_wall_data is None
        assert _drawn(win) == 0
        assert win._option_status_lbl.text() == "options: no data"

    def test_a_good_poll_resets_the_grace(self, win):
        k = _klines()
        win._option_wall_data = _cache()
        win._on_option_fetch_error("blip")
        win._on_option_fetch_error("blip")
        win._on_option_fetch_ready(_cache())
        win._on_option_fetch_error("blip")
        win._on_option_fetch_error("blip")
        win._render(k, None)

        assert win._option_wall_data is not None, "streak was not reset"
        assert _drawn(win) > 0


class TestSwitchingSymbolDropsTheOldWalls:

    def test_the_cache_goes_not_just_the_items(self, win, monkeypatch):
        """Clearing the drawn items alone was not enough: _render repaints
        from the cache, so the old symbol's walls landed on the new chart."""
        k = _klines()
        win._option_code = "US.SOXL"
        win._option_wall_data = _cache()
        win._render(k, None)
        assert _drawn(win) > 0

        _switch_symbol(win, monkeypatch, "US.QQQ")
        win._render(k, None)

        assert win._option_wall_data is None
        assert _drawn(win) == 0

    def test_the_grace_resets_with_the_symbol(self, win, monkeypatch):
        """Inheriting the streak would drop the new symbol's walls on its
        first hiccup, having spent the grace on the previous one."""
        win._option_code = "US.SOXL"
        win._on_option_fetch_error("x")
        win._on_option_fetch_error("x")
        assert win._option_fail_streak == 2

        _switch_symbol(win, monkeypatch, "US.QQQ")

        assert win._option_fail_streak == 0
        assert win._option_code == "US.QQQ", "the symbol block did not run"


class TestThePartialBannerRidesAlong:

    def test_a_partial_fetch_is_announced_on_the_chart(self, win):
        cov = {"expiries_requested": ["a", "b", "c", "d"], "expiries_ok": ["a", "b"],
               "batches_ok": 5, "batches_total": 5}
        k = _klines()
        win._option_wall_data = _cache(coverage=cov)
        win._render(k, None)

        texts = [getattr(i, "toPlainText", lambda: "")()
                 for i in win._option_wall_items]
        assert any("PARTIAL" in t for t in texts), texts

    def test_a_full_fetch_says_nothing(self, win):
        cov = {"expiries_requested": ["a"], "expiries_ok": ["a"],
               "batches_ok": 1, "batches_total": 1}
        k = _klines()
        win._option_wall_data = _cache(coverage=cov)
        win._render(k, None)

        texts = [getattr(i, "toPlainText", lambda: "")()
                 for i in win._option_wall_items]
        assert not any("PARTIAL" in t for t in texts), texts

    def test_the_banner_does_not_widen_the_label_columns(self, win):
        """The wall labels share one column width, set by the widest of them.
        Letting the banner into that list would push the leftmost column off
        the view -- so it goes into _option_wall_items only."""
        cov = {"expiries_requested": ["a", "b", "c", "d"], "expiries_ok": ["a"],
               "batches_ok": 1, "batches_total": 1}
        k = _klines()

        win._option_wall_data = _cache()
        win._render(k, None)
        plain = [lbl.pos().x() for _, lbl in win._option_wall_labels]

        win._option_wall_data = _cache(coverage=cov)
        win._render(k, None)
        with_banner = [lbl.pos().x() for _, lbl in win._option_wall_labels]

        assert plain == with_banner, (plain, with_banner)

    def test_the_banner_is_cleared_with_everything_else(self, win):
        cov = {"expiries_requested": ["a", "b"], "expiries_ok": ["a"],
               "batches_ok": 1, "batches_total": 1}
        k = _klines()
        win._option_wall_data = _cache(coverage=cov)
        win._render(k, None)
        assert _drawn(win) > 0

        win._clear_option_wall_items()
        assert _drawn(win) == 0

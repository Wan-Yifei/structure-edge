"""A symbol whose reconcile failed to launch has to be retried.

The cron claimed the minute before the loop:

    cur_min = (...)
    if cur_min == self._last_opt_reconcile_minute: return
    self._last_opt_reconcile_minute = cur_min
    for code in codes: ...

and it matches one minute a day. So a symbol whose Popen raised inside that
minute was not merely skipped -- it was skipped for the whole session, with
no further attempt. The per-symbol try added in the third review narrowed
"one failure loses every later symbol" to "one failure loses itself", which
is better and still permanent.

The claim is now per symbol and per day: a launched symbol is recorded
AFTER Popen returns, so it is not relaunched, while a failed one comes back
on the next tick -- still inside the matching minute.
"""
from __future__ import annotations

import pathlib
import sys
from datetime import datetime

import pytest

ROOT = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import analysis.scheduler as mod                                 # noqa: E402

CRON = "50 16 * * 1-5"
#: A Tuesday, a full session, inside the cron's minute.
AT = datetime(2026, 11, 24, 16, 50, 3)


class _App:
    """Only what _check_option_reconcile_cron touches."""

    def __init__(self, codes=("US.SOXL", "US.QQQ", "US.TQQQ")):
        self.cfg = {"option_window": {"enabled": True,
                                      "reconcile_cron": CRON,
                                      "codes": list(codes)}}
        self._DEFAULT_OPTION = mod.SchedulerApp._DEFAULT_OPTION
        self._option_codes = mod.SchedulerApp._option_codes
        self._last_opt_reconcile_minute = None
        self._opt_reconciled_on = (None, set())
        self._opt_reconcile_proc = None
        self.logged: list[str] = []

    def _log(self, msg):
        self.logged.append(msg)

    def _tail_opt_reconcile(self, proc=None, code=""):
        pass

    @property
    def _cron_field_match(self):
        return mod.SchedulerApp._cron_field_match

    def check(self, et=AT):
        mod.SchedulerApp._check_option_reconcile_cron(self, et)

    def assert_no_cron_error(self):
        errs = [m for m in self.logged if "Reconcile cron error" in m]
        assert not errs, errs


class _Popen:
    """Records launches; raises for the symbols named in `fail`."""

    def __init__(self, fail=()):
        self.fail = set(fail)
        self.launched: list[str] = []

    def __call__(self, cmd, **kw):
        code = cmd[cmd.index("--code") + 1]
        self.launched.append(code)
        if code in self.fail:
            raise OSError(24, "too many open files")

        class _P:
            stdout = None
            returncode = None

            def poll(self):
                return None
        return _P()


@pytest.fixture(autouse=True)
def _no_threads(monkeypatch):
    class _T:
        def __init__(self, *a, **k):
            pass

        def start(self):
            pass
    monkeypatch.setattr(mod.threading, "Thread", _T)


class TestAFailedLaunchComesBack:

    def test_the_failed_symbol_is_retried_on_the_next_tick(self, monkeypatch):
        popen = _Popen(fail={"US.QQQ"})
        monkeypatch.setattr(mod.subprocess, "Popen", popen)
        app = _App()

        app.check()
        assert popen.launched == ["US.SOXL", "US.QQQ", "US.TQQQ"]

        # One second later, still 16:50.
        app.check(AT.replace(second=4))
        app.assert_no_cron_error()
        assert popen.launched[3:] == ["US.QQQ"], (
            "the failed symbol was not retried, or a succeeded one was "
            f"relaunched: {popen.launched}")

    def test_a_symbol_that_launched_is_not_launched_twice(self, monkeypatch):
        popen = _Popen()
        monkeypatch.setattr(mod.subprocess, "Popen", popen)
        app = _App()
        for s in range(3, 10):
            app.check(AT.replace(second=s))
        app.assert_no_cron_error()
        assert popen.launched == ["US.SOXL", "US.QQQ", "US.TQQQ"], (
            "the cron matches every second of its minute; without a claim "
            f"this is 7 rounds of 3: {popen.launched}")

    def test_the_retry_stops_once_it_succeeds(self, monkeypatch):
        """Not an unbounded retry: it is bounded by the symbol succeeding
        and, failing that, by the cron's minute ending."""
        popen = _Popen(fail={"US.QQQ"})
        monkeypatch.setattr(mod.subprocess, "Popen", popen)
        app = _App()
        app.check(AT.replace(second=3))
        popen.fail.clear()
        app.check(AT.replace(second=4))
        app.check(AT.replace(second=5))
        app.assert_no_cron_error()
        assert popen.launched == ["US.SOXL", "US.QQQ", "US.TQQQ", "US.QQQ"]

    def test_the_cron_matched_line_is_not_printed_once_a_second(self,
                                                                monkeypatch):
        monkeypatch.setattr(mod.subprocess, "Popen", _Popen())
        app = _App()
        for s in range(3, 20):
            app.check(AT.replace(second=s))
        matched = [m for m in app.logged if "Reconcile cron matched" in m]
        assert len(matched) == 1, matched

    def test_the_line_names_only_what_is_still_pending(self, monkeypatch):
        popen = _Popen(fail={"US.TQQQ"})
        monkeypatch.setattr(mod.subprocess, "Popen", popen)
        app = _App()
        app.check(AT.replace(second=3))
        app.check(AT.replace(second=4))
        matched = [m for m in app.logged if "Reconcile cron matched" in m]
        assert len(matched) == 2
        assert "US.TQQQ" in matched[1] and "US.SOXL" not in matched[1], (
            "the second line must name the retry, not the whole set: "
            + matched[1])


class TestTheClaimIsPerDay:

    def test_a_new_day_starts_over(self, monkeypatch):
        popen = _Popen()
        monkeypatch.setattr(mod.subprocess, "Popen", popen)
        app = _App(codes=("US.SOXL",))
        app.check(AT)
        assert popen.launched == ["US.SOXL"]
        # The next session, same wall-clock minute.
        app.check(AT.replace(day=25))
        app.assert_no_cron_error()
        assert popen.launched == ["US.SOXL", "US.SOXL"]

    def test_the_same_day_is_not_repeated_after_a_non_matching_tick(
            self, monkeypatch):
        """A tick outside the minute must not clear the day's record --
        that would make every later matching minute a fresh round."""
        popen = _Popen()
        monkeypatch.setattr(mod.subprocess, "Popen", popen)
        app = _App(codes=("US.SOXL",))
        app.check(AT)
        app.check(AT.replace(hour=12))          # cron does not match
        app.check(AT.replace(second=30))        # back inside the minute
        app.assert_no_cron_error()
        assert popen.launched == ["US.SOXL"]

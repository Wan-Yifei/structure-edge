"""How the scheduler supervises the option collector.

Four behaviours that the stock collectors have had for a while and the
option path did not:

B5  Each tail thread reads its process off `self`, so two processes started
    in the same tick both get tailed by whoever `self` points at last --
    certain, not merely possible, and for the reconcile its output IS the
    deliverable.
B6  No backoff. "tick" and "ob" both pass through _collector_backoff_ok; a
    refused option collector relaunched on the next tick, about once a
    second.
B8  Quitting killed the stock collector and left the option one to the job
    object, which hard-kills it -- skipping CTRL_BREAK, so the 200
    subscriptions stay booked against the account quota.

Built from explicit stubs rather than a real SchedulerApp: constructing one
opens a Tk window, a tray icon and an OpenD connection.
"""
from __future__ import annotations

import pytest

import analysis.scheduler as mod


class _Proc:
    """A Popen stand-in whose stdout is a fixed list of lines."""

    def __init__(self, pid, lines=(), alive=True, rc=0):
        self.pid = pid
        self.stdout = list(lines)
        self._alive = alive
        self.returncode = rc

    def poll(self):
        return None if self._alive else self.returncode


class _Tailer:
    """Enough of the app to call the two tail methods."""

    def __init__(self):
        self.lines: list[str] = []
        self._opt_procs: dict = {}
        self._opt_reconcile_proc = None

    def after(self, _delay, fn, *a):
        fn(*a)

    def _log(self, msg):
        self.lines.append(msg)


class TestEachTailThreadReadsItsOwnProcess:

    def test_the_reconcile_tail_follows_the_process_it_was_given(self):
        app = _Tailer()
        first = _Proc(101, ["first symbol\n"])
        second = _Proc(102, ["second symbol\n"])
        # The race, made deterministic: the attribute has already moved on
        # to the second process by the time the first thread runs.
        app._opt_reconcile_proc = second
        mod.SchedulerApp._tail_opt_reconcile(app, first)
        assert any("first symbol" in l for l in app.lines), app.lines
        assert not any("second symbol" in l for l in app.lines), app.lines

    def test_the_collector_tail_follows_the_process_it_was_given(self):
        app = _Tailer()
        app._opt_procs = {"US.QQQ": _Proc(102, ["wrong one\n"])}
        mod.SchedulerApp._tail_opt_proc(app, _Proc(101, ["right one\n"]))
        assert any("right one" in l for l in app.lines), app.lines
        assert not any("wrong one" in l for l in app.lines), app.lines

    @pytest.mark.parametrize("name", ["_tail_opt_proc", "_tail_opt_reconcile"])
    def test_it_falls_back_to_the_attribute(self, name):
        """The single-process call sites pass nothing, and must keep
        working -- otherwise this change breaks the case it is not for."""
        app = _Tailer()
        proc = _Proc(101, ["from the attribute\n"])
        app._opt_procs = {"US.SOXL": proc}
        app._opt_reconcile_proc = proc
        getattr(mod.SchedulerApp, name)(app)
        assert any("from the attribute" in l for l in app.lines), app.lines

    @pytest.mark.parametrize("name", ["_tail_opt_proc", "_tail_opt_reconcile"])
    def test_the_call_sites_pass_the_process(self, name):
        """A parameter nobody passes is the same as no parameter. Read off
        the source, because the launch paths need a live Popen to run."""
        import inspect
        src = inspect.getsource(mod.SchedulerApp)
        assert f"target=self.{name}" in src
        i = src.index(f"target=self.{name}")
        window = src[i:i + 220]
        assert "args=(" in window, window


class _Supervised:
    """Enough of the app for _check_option_window's relaunch branch."""

    def __init__(self, rc=None, alive=False, codes=("US.SOXL",)):
        self.cfg = {"option_window": {"enabled": True, "start": "09:15",
                                      "end": "16:25", "codes": list(codes)}}
        self._DEFAULT_OPTION = mod.SchedulerApp._DEFAULT_OPTION
        self._POST_CLOSE_PAD_MIN = mod.SchedulerApp._POST_CLOSE_PAD_MIN
        self._collector_running = True
        self._opt_procs = ({} if rc is None else
                           {c: _Proc(9, alive=alive, rc=rc) for c in codes})
        self.opt_status_lbl = None
        self._launched_at = {}
        self.launches: list[str] = []
        self.logged: list[str] = []

    # Real implementations, because they are the subject.
    _hhmm = staticmethod(mod.SchedulerApp._hhmm)
    _option_window_active = mod.SchedulerApp._option_window_active
    _collector_backoff_ok = mod.SchedulerApp._collector_backoff_ok
    _died_young = mod.SchedulerApp._died_young

    _option_codes = staticmethod(mod.SchedulerApp._option_codes)

    def _launch_opt(self, code):
        self.launches.append(code)

    def _kill_opt(self, codes=None):
        for c in (list(codes) if codes is not None else list(self._opt_procs)):
            self._opt_procs.pop(c, None)

    # The REAL one. The first version of this file stubbed it, and the stub
    # recorded the failure without arming the backoff -- so the suppression
    # test failed against correct code, because the thing it measured had
    # been replaced by the measurement.
    _collector_failed = mod.SchedulerApp._collector_failed

    def _log(self, msg):
        self.logged.append(msg)

    @property
    def failures(self):
        """What the real _collector_failed recorded, read off its own state."""
        return [(k, v) for k, v in
                getattr(self, "_relaunch_tries", {}).items()]


def _et(day, hhmm):
    from datetime import datetime
    from zoneinfo import ZoneInfo
    return datetime.strptime(f"{day} {hhmm}", "%Y-%m-%d %H:%M").replace(
        tzinfo=ZoneInfo("America/New_York"))


ORDINARY = "2026-11-25"


class TestTheOptionPathHasABackoff:
    """A refusal used to become a process every tick, about once a second."""

    def test_a_refusal_is_recorded_as_a_failure(self):
        app = _Supervised(rc=mod.single_instance.EXIT_ALREADY_RUNNING)
        mod.SchedulerApp._check_option_window(app, _et(ORDINARY, "10:00"))
        assert app._relaunch_tries.get("opt:US.SOXL") == 1, app._relaunch_tries
        assert any("already holds" in l for l in app.logged), app.logged
        assert app.launches == []

    def test_dying_young_is_recorded_as_a_failure(self):
        app = _Supervised(rc=1)
        app._launched_at["opt:US.SOXL"] = __import__("time").monotonic()
        mod.SchedulerApp._check_option_window(app, _et(ORDINARY, "10:00"))
        assert app._relaunch_tries.get("opt:US.SOXL") == 1, app._relaunch_tries
        assert any("did not survive" in l for l in app.logged), app.logged

    def test_a_long_lived_crash_is_restarted_at_once(self):
        """The control: a process that ran for an hour and then died is a
        crash worth restarting now, not a loop to back off from."""
        app = _Supervised(rc=1)
        app._launched_at["opt:US.SOXL"] = __import__("time").monotonic() - 3600
        mod.SchedulerApp._check_option_window(app, _et(ORDINARY, "10:00"))
        assert not getattr(app, "_relaunch_tries", {}), app._relaunch_tries
        assert app.launches == ["US.SOXL"]

    def test_the_backoff_actually_suppresses_the_next_launch(self):
        """Recording a failure is only half of it -- the wait has to be
        honoured, which is what the relaunch loop was missing."""
        app = _Supervised(rc=mod.single_instance.EXIT_ALREADY_RUNNING)
        mod.SchedulerApp._check_option_window(app, _et(ORDINARY, "10:00"))
        for _ in range(5):          # five more ticks, one second apart
            mod.SchedulerApp._check_option_window(app, _et(ORDINARY, "10:00"))
        assert app.launches == [], (
            f"{app.launches} launches while backed off -- the refusal loop "
            f"is still a loop")

    def test_a_fresh_start_with_no_history_is_not_blocked(self):
        """Without this the backoff test above passes against a path that
        never launches at all."""
        app = _Supervised()
        mod.SchedulerApp._check_option_window(app, _et(ORDINARY, "10:00"))
        assert app.launches == ["US.SOXL"]


class TestQuittingStopsTheOptionCollector:

    def test_quit_app_calls_kill_opt(self):
        """Left to the job object it is hard-killed, which skips
        CTRL_BREAK and therefore skips unsubscribe -- 200 contracts stay
        booked against the account quota."""
        killed = []

        class _Quit:
            _tray = None
            _tick_job = None
            def _kill_collector(self):
                killed.append("tick")
            def _kill_opt(self):
                killed.append("opt")
            def destroy(self):
                killed.append("destroy")

        mod.SchedulerApp._quit_app(_Quit())
        assert "opt" in killed, killed
        # Before destroy, or the window is gone while the child is still up.
        assert killed.index("opt") < killed.index("destroy"), killed

"""A failed collector is retried, a refused one is not, and neither spins.

One stale collector turned into a process every eight seconds for as long as
anyone was watching. The scheduler's watchdog saw a subprocess that had
exited and relaunched it; the subprocess exited again, because another copy
still held the single-instance lock; and the only thing setting the pace was
the lock's own eight-second retry window.

Three shapes have to be told apart, and the difference is NOT crash versus
refusal -- it is whether a restart could possibly help:

  * refused (EXIT_ALREADY_RUNNING): something else holds the lock. Another
    start earns another refusal. Say whose PID it is and stop.
  * died during startup: whatever killed it will kill the next one too, at
    one process per tick. Back off.
  * died after running: a genuine mid-session crash, and the session still
    needs a collector. Restart at once -- this is the case the watchdog
    exists for, and it must not be slowed down by the other two.

The exit code is readable from the scheduler even though Popen holds a uv
trampoline rather than the collector itself: measured, the trampoline passes
the child's code through unchanged.

These drive the real SchedulerApp methods against a stand-in process, with
no Tk and no subprocess, because what is being tested is the decision.
"""

import pathlib
import types

import pytest

from analysis import scheduler as sch
from core import single_instance

REFUSED = single_instance.EXIT_ALREADY_RUNNING


class _Exited:
    """A Popen that has already exited with `rc`."""

    def __init__(self, rc):
        self.returncode = rc
        self.pid = 4212

    def poll(self):
        return self.returncode


@pytest.fixture
def app(monkeypatch):
    """A SchedulerApp reduced to the parts the watchdog touches."""
    monkeypatch.setattr(sch, "session_status", lambda *a, **k: "ACTIVE")
    a = types.SimpleNamespace()
    a.log, a.launched = [], []
    a._log = a.log.append
    a._proc = None
    a._ob_proc = None
    a.cfg = {"order_book_enabled": True}
    a._launch_tick = lambda: a.launched.append("tick")
    a._launch_ob = lambda: a.launched.append("ob")
    a._launched_at = {}
    a._relaunch_wait = {}
    a._relaunch_tries = {}
    for name in ("_collector_backoff_ok", "_collector_failed",
                 "_collector_started", "_died_young",
                 "_watchdog_collectors"):
        setattr(a, name, types.MethodType(getattr(sch.SchedulerApp, name), a))
    return a


def _run(a, rc, *, age, ticks=40, ob=False):
    """Feed the watchdog a collector that keeps exiting with `rc`.

    A launch is what produces the process, not the tick -- the harness used
    to refill it every tick, which hid the retry path entirely: the watchdog
    only looks at a collector it has none of when it is deciding whether a
    backoff has expired.
    """
    now = sch.time_mod.monotonic()
    started = now if age == "young" else now - 3600
    attr = "_ob_proc" if ob else "_proc"
    key = "ob" if ob else "tick"

    def make(which, which_attr):
        def launch():
            # Every launch is recorded, including the collector this case is
            # not about: a branch that starts both is the defect W6 names,
            # and a stub that drops the other one hides it.
            a.launched.append(which)
            if which == key:
                setattr(a, which_attr, _Exited(rc))
                a._launched_at[which] = started
        return launch

    a._launch_tick = make("tick", "_proc")
    a._launch_ob = make("ob", "_ob_proc")

    a._launch_ob() if ob else a._launch_tick()   # the session start
    a.launched.clear()             # only count what the watchdog decides
    for _ in range(ticks):
        a._watchdog_collectors(None)


class TestARefusalIsNotACrash:
    """The shape that ran for hours. Relaunching at once cannot help -- but
    never relaunching is its own bug: clearing the process that holds the
    lock has to be enough to get the collector back, without waiting for the
    next session boundary. So: rarely, not never."""

    def test_it_is_relaunched_rarely_not_every_tick(self, app):
        _run(app, REFUSED, age="young")
        assert len(app.launched) <= 5, (
            f"relaunched {len(app.launched)} times in 40 ticks against a "
            f"held lock; the shipped bug was one a tick")
        assert app.launched, (
            "never relaunched at all: clearing the lock holder would then "
            "leave this collector down for the rest of the session")

    def test_the_log_says_a_lock_is_held(self, app):
        _run(app, REFUSED, age="young", ticks=2)
        assert any("already holds" in ln for ln in app.log), app.log[:3]

    def test_it_does_not_claim_the_process_died(self, app):
        """The two causes need different words: one is fixed by stopping
        another process, the other by reading that process's output."""
        _run(app, REFUSED, age="young", ticks=2)
        assert not any("did not survive" in ln for ln in app.log), app.log[:3]

    def test_the_order_book_collector_too(self, app):
        _run(app, REFUSED, age="young", ob=True)
        assert len(app.launched) <= 5, app.launched


class TestAStartupCrashBacksOff:
    """Dying two seconds in, once a second, is the same loop wearing a
    different log line."""

    def test_it_slows_down_instead_of_spinning(self, app):
        _run(app, 1, age="young")
        assert len(app.launched) <= 5, (
            f"relaunched {len(app.launched)} times in 40 ticks against a "
            f"collector that cannot get through its own startup")

    def test_the_log_says_it_died_rather_than_was_refused(self, app):
        _run(app, 1, age="young", ticks=2)
        assert any("did not survive" in ln for ln in app.log), app.log[:3]
        assert not any("already holds" in ln for ln in app.log), app.log[:3]

    def test_the_wait_grows(self, app):
        _run(app, 1, age="young", ticks=2)
        first = app._relaunch_wait["tick"]
        _run(app, 1, age="young", ticks=100)
        assert app._relaunch_wait["tick"] > first

    def test_the_wait_is_capped(self, app):
        """Sampled where it is SET, not where it ends up.

        The stored wait counts down one per tick, so reading it after a run
        gives whatever is left of the current countdown -- a small number
        whether or not there is a cap. The peak is the quantity with the
        meaning, and reaching one over five minutes needs seven failures,
        which is about 500 ticks of waiting through the first six.
        """
        peaks = []
        real = app._collector_failed

        def spy(key, label, name, reason):
            real(key, label, name, reason)
            peaks.append(app._relaunch_wait[key])

        app._collector_failed = spy
        _run(app, 1, age="young", ticks=4000)
        assert peaks, "no failures recorded; this test measured nothing"
        assert max(peaks) <= 300, (
            f"backoff reached {max(peaks)}s; an unbounded one is its own "
            f"failure -- the collector stops coming back within any "
            f"session you would notice")


class TestAMidSessionCrashStillRestartsAtOnce:
    """The case the watchdog was written for. If the other two slowed this
    one down, the fix would cost more than the bug."""

    def test_it_relaunches_every_tick(self, app):
        _run(app, 1, age="old", ticks=40)
        assert len(app.launched) == 40, (
            f"only {len(app.launched)} restarts; a collector that was up "
            f"and died should come back immediately")

    def test_no_backoff_is_recorded(self, app):
        _run(app, 1, age="old", ticks=10)
        assert "tick" not in app._relaunch_wait


class TestTheBranchesAreSeparate:
    """Both branches used to call the launcher that starts BOTH collectors,
    so one dead collector produced two launches a round."""

    def test_a_dead_tick_collector_does_not_start_the_order_book_one(self, app):
        _run(app, 1, age="old", ticks=5)
        assert set(app.launched) == {"tick"}, app.launched

    def test_a_dead_order_book_collector_does_not_start_the_tick_one(self, app):
        _run(app, 1, age="old", ticks=5, ob=True)
        assert set(app.launched) == {"ob"}, app.launched


class TestTheExitCodeIsOurs:
    """Not 2. Everything else that exits 2 here -- argparse on a bad flag,
    CPython on a missing script -- is also a do-not-retry case, so 2 would
    have behaved correctly while meaning nothing in particular."""

    def test_the_collectors_use_the_shared_constant(self):
        """Read beside the scheduler that was imported, not from a path
        written out here: an absolute path would pin this file to one
        checkout and quietly test the wrong copy everywhere else."""
        root = pathlib.Path(sch.__file__).parent
        for name in ("tick_collector.py", "order_book_collector.py"):
            src = (root / name).read_text(encoding="utf-8")
            assert "EXIT_ALREADY_RUNNING" in src, (
                f"{name} does not use the shared exit code; the watchdog "
                f"reads that number to tell a refusal from a crash")

    def test_it_is_not_a_code_something_else_returns(self):
        assert REFUSED not in (0, 1, 2), (
            "0 is success, 1 is an unhandled exception or a terminate() on "
            "Windows, 2 is argparse and CPython's own startup errors")


class TestSpawningIsNotSurviving:
    """The regression that got through every test above and showed up in
    production: the backoff grew in the harness and never grew on the machine.

    `_launch_tick` cleared the escalation as soon as Popen returned. But a
    collector refused by the lock takes about eight seconds to discover that
    and exit, so every round reset the counter and the log read "(attempt 1)"
    forever -- a loop at eight seconds instead of one, which is still a loop.

    The tests above could not see it, because their harness replaces
    `_launch_tick` with a stub. So these drive the real one, with only the
    subprocess call stood in for.
    """

    @pytest.fixture
    def launcher(self, app, monkeypatch):
        class _Alive:
            returncode = None
            pid = 9001

            def poll(self):
                return None

        monkeypatch.setattr(sch.subprocess, "Popen",
                            lambda *a, **k: _Alive())
        monkeypatch.setattr(sch, "_kill_with_me", lambda pid: None)
        monkeypatch.setattr(sch.threading, "Thread",
                            lambda **k: types.SimpleNamespace(
                                start=lambda: None))
        app.timeout_var = types.SimpleNamespace(get=lambda: "10")
        app._tail_proc = lambda: None
        app._tail_ob_proc = lambda: None
        for name in ("_launch_tick", "_launch_ob"):
            setattr(app, name,
                    types.MethodType(getattr(sch.SchedulerApp, name), app))
        return app

    def test_a_launch_does_not_clear_an_outstanding_backoff(self, launcher):
        a = launcher
        a._relaunch_tries = {"tick": 3}
        a._relaunch_wait = {"tick": 64}
        a._launch_tick()
        assert a._relaunch_tries.get("tick") == 3, (
            "the attempt count was reset by the act of spawning; the next "
            "failure then starts over at attempt 1 and the wait never grows")

    def test_the_order_book_launcher_too(self, launcher):
        a = launcher
        a._ob_proc = None
        a._relaunch_tries = {"ob": 3}
        a._launch_ob()
        assert a._relaunch_tries.get("ob") == 3, a._relaunch_tries

    def test_surviving_is_what_clears_it(self, app):
        """The other half: something must clear it, or a collector that
        recovers carries a five-minute wait for the rest of the session."""
        app._relaunch_tries = {"tick": 3}
        app._relaunch_wait = {"tick": 64}
        app._proc = _Exited(None)
        app._proc.returncode = None
        app._proc.poll = lambda: None
        app._launched_at["tick"] = sch.time_mod.monotonic() - 3600
        app._watchdog_collectors(None)
        assert "tick" not in app._relaunch_tries, (
            "a collector that has been up for an hour is working; its "
            "escalation has to be forgotten or the next crash waits 300s")

    def test_surviving_clears_it_on_the_order_book_side_too(self, app):
        """Written because a mutation survived: removing this branch left
        every test green, so the order book collector could have carried a
        five-minute wait for a whole session with nothing to show for it."""
        app._relaunch_tries = {"ob": 3}
        app._relaunch_wait = {"ob": 64}
        app._ob_proc = _Exited(None)
        app._ob_proc.returncode = None
        app._ob_proc.poll = lambda: None
        app._launched_at["ob"] = sch.time_mod.monotonic() - 3600
        app._watchdog_collectors(None)
        assert "ob" not in app._relaunch_tries, app._relaunch_tries

    def test_a_young_process_does_not_clear_it_yet(self, app):
        """Alive for two seconds proves nothing -- that is exactly how long
        a refused collector lives before it exits."""
        app._relaunch_tries = {"tick": 3}
        app._proc = _Exited(None)
        app._proc.returncode = None
        app._proc.poll = lambda: None
        app._launched_at["tick"] = sch.time_mod.monotonic()
        app._watchdog_collectors(None)
        assert app._relaunch_tries.get("tick") == 3, app._relaunch_tries


class TestTheLockHolderIsNotTheProcessWeStarted:
    """A trap worth a test of its own: .venv/Scripts/python.exe is a uv
    trampoline that execs the real interpreter, so Popen's pid and the pid
    in the lock file are one level apart. Code that compares them to decide
    "is this collector mine?" always answers no -- including about the one
    it started a second ago."""

    def test_holder_is_documented_as_the_inner_pid(self):
        doc = single_instance.holder.__doc__ or ""
        assert "trampoline" in doc.lower(), (
            "holder() returns the collector's own pid, not the one Popen "
            "holds; that has to be said where someone will read it")

    def test_holder_reports_nothing_for_a_free_lock(self, tmp_path):
        assert single_instance.holder("nobody_holds_this",
                                      lock_dir=tmp_path) is None

    def test_holder_reports_the_pid_for_a_held_lock(self, tmp_path):
        import os

        fh = single_instance.acquire("held", lock_dir=tmp_path)
        try:
            assert single_instance.holder("held",
                                          lock_dir=tmp_path) == os.getpid()
        finally:
            fh.close()

    def test_holder_takes_nothing_it_does_not_give_back(self, tmp_path):
        """It has to try the lock to answer, so it must leave it free.

        Mutation-checked and UNDEFENDED: deleting holder()'s `fh.close()`
        leaves this green, because CPython's refcounting closes the handle
        when the function returns and that releases the lock anyway. The
        explicit close is still right -- the release should not depend on
        which interpreter is running it -- but nothing here can tell the
        difference, and a test that cannot fail should say so rather than
        look like cover.
        """
        single_instance.holder("probe_me", lock_dir=tmp_path)
        fh = single_instance.acquire("probe_me", lock_dir=tmp_path)
        fh.close()

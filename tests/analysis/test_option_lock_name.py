"""The scheduler must ask about the lock the collector actually takes.

The refusal path looked up `single_instance.holder(f"option_collector_{code}")`
while the collector acquired `f"option_tick_collector_{code.replace('.','_')}"`
-- a different prefix and an unreplaced dot. holder() therefore returned
None every time, so the message named a lock file that does not exist and
degraded to "another process already holds the option_collector_US.SOXL
lock. ... Stop that process and the next attempt will take" with no PID in
it. The one number the operator is told to act on was the one the mismatch
removed.

tick_collector and order_book_collector use the same convention correctly,
which is how the option path looked fine by inspection.

This matters on the first live session specifically: the documented manual
start (scripts/launch_option_collector.bat) and the 09:15 scheduler are
certain to collide over the lock, so the refusal branch is the one being
walked, not a corner case.
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from analysis import option_tick_collector as col                # noqa: E402
from analysis import scheduler as sch                            # noqa: E402
from core import single_instance                                 # noqa: E402


class TestThereIsOneSpelling:

    def test_the_scheduler_imports_the_collectors_function(self):
        """Not two literals that agree. POST_CLOSE_PAD_MIN is shared the
        same way, and that precedent is why this one was noticed."""
        assert sch._opt_lock_name is col.lock_name

    def test_the_dot_becomes_an_underscore(self):
        assert col.lock_name("US.SOXL") == "option_tick_collector_US_SOXL"

    def test_a_class_share_keeps_both_dots_replaced(self):
        assert col.lock_name("US.BRK.B") == "option_tick_collector_US_BRK_B"


class TestTheNameMatchesWhatIsOnDisk:
    """An end-to-end check: take the lock under the collector's name, then
    ask under the scheduler's. The mismatch made the second answer None
    while the first held the lock."""

    def test_holder_finds_the_pid_through_the_schedulers_name(self, tmp_path,
                                                              monkeypatch):
        import os
        monkeypatch.setattr(single_instance, "LOCK_DIR", tmp_path)
        code = "US.SOXL"
        lock = single_instance.acquire(col.lock_name(code), wait_s=0.1)
        try:
            # The scheduler's spelling, resolved the way the refusal path
            # resolves it.
            assert single_instance.holder(sch._opt_lock_name(code)) == os.getpid()
        finally:
            lock.close()

    def test_the_old_spelling_would_have_found_nothing(self, tmp_path,
                                                       monkeypatch):
        """The control. Without it, the test above passes for any name at
        all that both sides happen to share -- including a wrong one."""
        monkeypatch.setattr(single_instance, "LOCK_DIR", tmp_path)
        code = "US.SOXL"
        lock = single_instance.acquire(col.lock_name(code), wait_s=0.1)
        try:
            assert single_instance.holder(f"option_collector_{code}") is None
        finally:
            lock.close()


class TestTheRefusalMessageCarriesThePid:
    """The message is the deliverable. holder() returning a PID is only
    useful if the string the operator reads contains it."""

    def test_the_pid_is_in_the_line(self, tmp_path, monkeypatch):
        import os

        monkeypatch.setattr(single_instance, "LOCK_DIR", tmp_path)

        class _App:
            def __init__(self):
                self.logged: list[str] = []
                self._collector_tries = {}
                self._collector_next_try = {}

            def _log(self, msg):
                self.logged.append(msg)

        app = _App()
        app._collector_failed = sch.SchedulerApp._collector_failed.__get__(app)

        lock = single_instance.acquire(col.lock_name("US.SOXL"), wait_s=0.1)
        try:
            app._collector_failed("opt:US.SOXL", "Option collector (US.SOXL)",
                                  sch._opt_lock_name("US.SOXL"), "refused")
        finally:
            lock.close()

        line = next(m for m in app.logged if "already holds" in m)
        assert f"PID {os.getpid()}" in line, (
            "the refusal said 'another process' -- the operator is told to "
            "stop a process the message cannot identify: " + line)

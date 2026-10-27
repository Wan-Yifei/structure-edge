"""A second copy of a collector must not start.

The failure this guards against is silent: two collectors write the
same SQLite file, queue behind each other's write lock, and since the
insert runs on the SDK's push callback thread, a 300ms cadence turns
into a 9.5-second callback. Nothing crashes -- the data just arrives
late and twice.

The important case is across PROCESSES, so the central test spawns a
real one. An in-process check would pass on a lock implementation that
only tracks state in a module-level variable.
"""
import pathlib
import subprocess
import sys
import textwrap
import time

import pytest

ROOT = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from core import single_instance                                  # noqa: E402


HOLDER = """
import os, pathlib, sys, time
sys.path.insert(0, {root!r})
from core import single_instance
fh = single_instance.acquire("probe", lock_dir=pathlib.Path({d!r}))
print("HELD", os.getpid(), flush=True)
time.sleep(30)
"""


@pytest.fixture
def lockdir(tmp_path):
    return tmp_path / "locks"


def _holder_proc(lockdir):
    """Start a process holding the lock; return (proc, the pid holding it).

    Not proc.pid: on Windows the venv's python.exe is a shim that execs
    the real interpreter as a child, and it is the CHILD that takes the
    lock. The same shape is why a `tasklist` of this project shows every
    collector twice. So the holder reports its own pid.
    """
    src = HOLDER.format(root=str(ROOT), d=str(lockdir))
    p = subprocess.Popen([sys.executable, "-c", textwrap.dedent(src)],
                         stdout=subprocess.PIPE, text=True)
    first = p.stdout.readline().split()
    assert first and first[0] == "HELD", "子进程没拿到锁"
    return p, int(first[1])


class TestTheLockIsHeldAcrossProcesses:
    def test_a_second_process_is_refused(self, lockdir):
        p, pid = _holder_proc(lockdir)
        try:
            with pytest.raises(single_instance.AlreadyRunning):
                single_instance.acquire("probe", lock_dir=lockdir)
        finally:
            p.kill()
            p.wait(timeout=10)

    def test_the_error_names_the_holder(self, lockdir):
        p, pid = _holder_proc(lockdir)
        try:
            with pytest.raises(single_instance.AlreadyRunning) as e:
                single_instance.acquire("probe", lock_dir=lockdir)
            assert e.value.holder == pid, \
                f"报的持有者是 {e.value.holder}，实际是 {pid}"
            assert str(pid) in str(e.value)
        finally:
            p.kill()
            p.wait(timeout=10)

    def test_killing_the_holder_releases_it(self, lockdir):
        """The point of locking a file descriptor instead of writing a
        pid file: a process that is killed leaves nothing to clean up,
        and the next start is not blocked by a dead predecessor."""
        p, pid = _holder_proc(lockdir)
        p.kill()
        p.wait(timeout=10)
        fh = single_instance.acquire("probe", wait_s=5.0, lock_dir=lockdir)
        fh.close()

    def test_a_different_name_is_a_different_lock(self, lockdir):
        """tick_collector and order_book_collector run side by side."""
        p, pid = _holder_proc(lockdir)
        try:
            fh = single_instance.acquire("other", lock_dir=lockdir)
            fh.close()
        finally:
            p.kill()
            p.wait(timeout=10)


class TestWaiting:
    def test_it_waits_out_a_departing_predecessor(self, lockdir):
        """The scheduler kills a collector and starts the next one at
        once; failing on a half-second overlap would leave nothing
        running at all."""
        p, pid = _holder_proc(lockdir)
        import threading
        threading.Timer(1.0, p.kill).start()
        t0 = time.monotonic()
        fh = single_instance.acquire("probe", wait_s=10.0, lock_dir=lockdir)
        took = time.monotonic() - t0
        fh.close()
        p.wait(timeout=10)
        assert took >= 0.5, "根本没等就拿到了锁 —— 前一个还在跑吗？"

    def test_without_waiting_it_fails_at_once(self, lockdir):
        p, pid = _holder_proc(lockdir)
        try:
            t0 = time.monotonic()
            with pytest.raises(single_instance.AlreadyRunning):
                single_instance.acquire("probe", lock_dir=lockdir)
            assert time.monotonic() - t0 < 1.0, "wait_s=0 不该等"
        finally:
            p.kill()
            p.wait(timeout=10)


class TestReleasing:
    def test_closing_the_handle_frees_the_name(self, lockdir):
        fh = single_instance.acquire("probe", lock_dir=lockdir)
        fh.close()
        fh2 = single_instance.acquire("probe", lock_dir=lockdir)
        fh2.close()

    def test_an_unwritable_pid_file_does_not_stop_the_lock(self, lockdir):
        """The pid is for the error message. Losing it must not stop a
        collector from starting."""
        lockdir.mkdir(parents=True, exist_ok=True)
        (lockdir / "probe.pid").mkdir()      # a directory: write fails
        fh = single_instance.acquire("probe", lock_dir=lockdir)
        fh.close()

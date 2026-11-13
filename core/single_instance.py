"""One running copy of a long-lived process, enforced by the OS.

A scheduler only knows the collectors IT started: `_launch_collector`
guards on `self._proc.poll()`, which says nothing about a collector left
behind by a previous scheduler that crashed or was closed without
running `_kill_collector`. Two order-book collectors then write the same
SQLite file, queue behind each other's write lock, and -- because the
insert runs on the SDK's push callback thread -- turn a 300ms cadence
into a 9.5-second callback. Found exactly that way on 2026-10-27, with a
pair of collectors from two days earlier still running.

An advisory lock on an open file, NOT a pid file: the kernel releases it
when the holder exits however it exits, so a killed process never leaves
a stale lock behind. A pid file would, and recovering from that means
guessing whether a recorded pid is still the same program.

The pid is written alongside, in a separate file, for the error message
only -- it is a hint, never the thing being tested.
"""
from __future__ import annotations

import os
import pathlib
import time

LOCK_DIR = pathlib.Path(__file__).resolve().parent.parent / "db" / "locks"


# What a process exits with when it found the lock held. The scheduler
# reads this to tell a refusal from a crash: a crash is worth restarting and
# a refusal is not -- restarting it only produces another refusal, which is
# how one stale collector turned into a process every eight seconds.
#
# Not 2. That works, because everything else that exits 2 here (argparse on
# a bad flag, CPython on a missing script) is also a do-not-retry case, but
# it would be a number that happens to behave rather than one that says what
# it means.
EXIT_ALREADY_RUNNING = 3


class AlreadyRunning(RuntimeError):
    """Another process holds the lock for this name."""

    def __init__(self, name: str, holder: int | None):
        self.name = name
        self.holder = holder
        who = f"PID {holder}" if holder else "another process"
        super().__init__(
            f"{name} is already running ({who}). Refusing to start a "
            f"second copy: two of these write the same database and "
            f"queue behind each other's write lock."
        )


def _try_lock(fh) -> bool:
    """Take an exclusive, non-blocking lock on the first byte."""
    try:
        if os.name == "nt":
            import msvcrt
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False


def _holder(pid_path: pathlib.Path) -> int | None:
    try:
        return int(pid_path.read_text().strip())
    except (OSError, ValueError):
        return None


def holder(name: str, lock_dir: pathlib.Path | None = None) -> int | None:
    """The PID holding `name`, or None if nothing does. Takes no lock.

    A second opinion for the scheduler, independent of any exit code: it
    tries the lock and releases it immediately, so a free lock reads as
    free even when the pid file is stale.

    THE PID THIS RETURNS IS THE COLLECTOR ITSELF, not the process the
    scheduler started. Those differ: `.venv/Scripts/python.exe` is a uv
    trampoline that execs the real interpreter, so Popen holds the
    trampoline's PID and the lock holds the one underneath it. Comparing
    this against `proc.pid` will say "not mine" about a collector the
    scheduler started seconds ago. Compare process trees or command lines,
    never these two numbers.
    """
    d = pathlib.Path(lock_dir) if lock_dir else LOCK_DIR
    lock_path = d / f"{name}.lock"
    if not lock_path.exists():
        return None
    try:
        fh = open(lock_path, "a+b")
    except OSError:
        return None
    try:
        # Closing the handle is what releases it, on either platform, so
        # taking the lock here and dropping it in `finally` leaves nothing
        # held either way.
        if _try_lock(fh):
            return None
    finally:
        fh.close()
    return _holder(d / f"{name}.pid")


def acquire(name: str, wait_s: float = 0.0,
            lock_dir: pathlib.Path | None = None):
    """Hold the lock for `name`, or raise AlreadyRunning.

    Keep the returned handle for the life of the process -- closing it,
    or letting it be garbage collected, releases the lock.

    `wait_s` retries for that long before giving up. A scheduler that
    restarts a collector may still be inside the old process's exit, and
    failing on a half-second overlap would leave nothing running at all.
    """
    d = pathlib.Path(lock_dir) if lock_dir else LOCK_DIR
    d.mkdir(parents=True, exist_ok=True)
    lock_path = d / f"{name}.lock"
    pid_path = d / f"{name}.pid"

    deadline = time.monotonic() + max(wait_s, 0.0)
    while True:
        fh = open(lock_path, "a+b")
        if _try_lock(fh):
            try:
                pid_path.write_text(str(os.getpid()))
            except OSError:
                pass            # diagnostics only; never fail on this
            return fh
        fh.close()
        if time.monotonic() >= deadline:
            raise AlreadyRunning(name, _holder(pid_path))
        time.sleep(0.25)

"""Make the test process exit when the tests are done.

The suite itself takes about a minute. What used to cost ten was that
pytest finished, wrote its report, and then the interpreter sat there:
25 test files construct TradeViewerQt, and each OpenD context carries a
CallbackExecutor whose worker thread is NOT a daemon. Python joins every
non-daemon thread at shutdown, so the process hung with "2071 passed"
already on screen. Mistaking that for a slow or stuck suite cost three
separate waits in one afternoon.

The fix is the SDK's own switch, set before any context exists. The hard
exit below stays as a backstop and should now never fire -- if it does,
something new is leaking and it says what.
"""
import faulthandler
import os
import sys
import threading
import traceback

import pytest

# Before any OpenQuoteContext is built, which means at import time: the
# flag is read in CallbackExecutor.__init__ (moomoo/common/callback_executor
# .py:23), so a thread created before this runs stays non-daemon for good.
#
# It is the callback executor, NOT the network manager. NetManager already
# sets daemon=True on its own thread (network_manager.py:185 -- that is the
# setDaemon deprecation warning the suite prints), so the first version of
# this file blamed the wrong thread and tried to close the wrong object.
# The real source is open_context_base._close_callback_executor(), which
# under _auto_reconnect builds a REPLACEMENT executor before closing the
# old one, leaving one unowned thread behind after the last ctx.close().
try:
    from moomoo.common.sys_config import SysConfig
    SysConfig.set_all_thread_daemon(True)
except Exception:                                           # noqa: BLE001
    # The SDK is optional for the non-GUI half of the suite; the backstop
    # below still covers it.
    pass


def pytest_sessionfinish(session, exitstatus):
    """Stash the status for pytest_unconfigure, which is not given one."""
    session.config._opt_exitstatus = int(exitstatus)


def _lingering() -> list[threading.Thread]:
    """Non-daemon threads that will hold the interpreter open."""
    me = threading.current_thread()
    return [t for t in threading.enumerate()
            if t is not me and t.is_alive() and not t.daemon]


def _describe(t: threading.Thread) -> str:
    """Enough to recognise the leak next time.

    "Thread-77 (run)" says nothing. The frame it is parked in names the
    module, which is how the callback executor was identified at all.
    """
    where = ""
    frame = sys._current_frames().get(t.ident)
    if frame is not None:
        stack = traceback.extract_stack(frame)
        if stack:
            f = stack[-1]
            where = f"  at {f.filename}:{f.lineno} in {f.name}"
    target = getattr(t, "_target", None)
    name = getattr(target, "__qualname__", None) or repr(target)
    return f"{t.name}  target={name}{where}"


def pytest_unconfigure(config):
    """Name whatever is still running, then leave if it would block us.

    unconfigure, NOT sessionfinish. The terminal reporter writes the
    "N passed" summary from a sessionfinish HOOKWRAPPER, and the half of a
    wrapper after its yield runs after every plain implementation -- even
    one marked trylast. Both were tried: the run reached 100% and then died
    with no result line, which is worse than the hang it replaced.

    Reaching the exit at all now means a NEW leak, since the daemon switch
    above covers the known one. That is why it describes the threads rather
    than just counting them.
    """
    left = _lingering()
    if not left:
        return

    # Not 0. Several paths reach unconfigure without passing through
    # sessionfinish (a crash during configure, a UsageError, a conftest
    # import failure), and defaulting to success there would report a
    # failed run as a passing one.
    exitstatus = getattr(config, "_opt_exitstatus", 3)

    for t in left:
        print(f"[conftest] still alive: {_describe(t)}",
              file=sys.stderr, flush=True)
    print(f"[conftest] {len(left)} non-daemon thread(s) would hold the "
          f"interpreter open. The known one (moomoo's CallbackExecutor) is "
          f"covered by set_all_thread_daemon at the top of this file, so "
          f"this is something new. Exiting with status {exitstatus}; the "
          f"test report is complete.", file=sys.stderr, flush=True)

    sys.stderr.flush()
    sys.stdout.flush()
    # Tearing the process down under a thread parked in queue.get() raises
    # a Windows access violation, and faulthandler then prints every stack
    # it can find -- a dump that says nothing while looking exactly like a
    # real crash.
    faulthandler.disable()
    # os._exit, not sys.exit: sys.exit unwinds into the same threading
    # shutdown join that is the problem. NOTE: this skips atexit handlers,
    # so `coverage run -m pytest` would lose its .coverage file. Nothing in
    # this repo uses it today; if that changes, fix the leak instead of
    # reaching here.
    os._exit(exitstatus)

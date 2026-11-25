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
import warnings

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


# ── keep the test suite off the live OpenD ────────────────────────────────
#
# Constructing TradeViewerQt opened a REAL quote context and pulled a REAL
# option chain. viewer_config defaults `option_walls` to True and __init__
# calls _on_option_indicator_toggle() unconditionally, so each of the ~25
# test files that builds a viewer spent calls against the account-wide
# 10-per-30s option-chain limit. Measured, before this:
#
#     Warning: chain fetch failed for 2026-11-27:
#     获取期权链频率太高，请求失败，每30秒最多10次。
#
# That is the same budget the collectors' contract selection depends on, so
# a test run could make a live session pick a short chain -- and it makes
# any quota figure read off this machine untrustworthy.
#
# ONE method is patched: _ensure_option_ctx. _trigger_option_fetch already
# returns when _option_ctx is None (trade_viewer_qt.py:5300), so stubbing
# the context is enough to stop the network without silencing the fetch
# logic that two test files drive on purpose. An earlier version also
# stubbed _trigger_option_fetch and _opend_listening, which broke three
# tests: the probe's own test, and two that call the fetch with the context
# already stubbed on the instance.
#
# In pytest_configure, not an autouse fixture: a module-scoped `viewer`
# fixture is instantiated BEFORE any function-scoped fixture, so the patch
# arrived after the viewer had already been built and gone to the network.
_REAL_ENSURE_OPTION_CTX = None

# Spellings of localhost a test could reach OpenD by.
_OPEND_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "0.0.0.0"})

# Duplicated deliberately. 11111 is a literal in ten modules with no shared
# constant to import, and conftest must not import project code to decide
# whether to guard it -- an import failure here is the exact hole this
# guard exists to cover.
_OPEND_PORT = 11111

# An absent one of these means no GUI, which is a legitimate environment:
# the non-GUI half of the suite still runs. Anything else failing to import
# is a broken tree and must not be mistaken for "no PyQt6 here".
_OPTIONAL_GUI_MODULES = frozenset({"PyQt6", "pyqtgraph", "matplotlib"})


def pytest_addoption(parser):
    parser.addoption(
        "--allow-opend", action="store_true", default=False,
        help="let tests open real OpenD connections and fetch option chains "
             "(spends the account-wide 10-per-30s chain budget)")


def _install_opend_guard():
    """Make a connection to OpenD fail the test that opened it.

    The stub below covers ONE entrance. Measured, the rest are not covered
    at all: trade_viewer_qt._connect_opend (no listening probe, so a test
    that passes `args` hangs forever), liq_hm_window, scheduler,
    signal_scanner, the collectors, and fetcher.fetch_klines on a cache
    miss. Today no test reaches them -- that is discipline, not structure,
    and discipline is what a new test file silently breaks.

    This is the structure. Measured cost before installing it: the default
    run makes 5 socket connections, none of them to OpenD (they are the
    probe's own tests, against port 1, a throwaway listener, and
    10.255.255.1), so nothing legitimate is caught. With --allow-opend the
    same instrumentation counted 111 connections from 56 tests.
    """
    import socket

    real_connect = socket.socket.connect

    def guarded(self, address, *a, **k):
        try:
            host, port = address[0], address[1]
        except Exception:                                   # noqa: BLE001
            return real_connect(self, address, *a, **k)
        if port == _OPEND_PORT and str(host) in _OPEND_HOSTS:
            raise RuntimeError(
                f"a test connected to OpenD at {host}:{port}. The suite "
                f"shares one account-wide option-chain budget with the "
                f"live collectors, so a run that talks to OpenD can make a "
                f"real session pick a short chain. Stub the context (see "
                f"conftest's _ensure_option_ctx patch) or pass "
                f"--allow-opend if you really mean to.")
        return real_connect(self, address, *a, **k)

    socket.socket.connect = guarded


def pytest_configure(config):
    global _REAL_ENSURE_OPTION_CTX
    if config.getoption("--allow-opend"):
        return
    _install_opend_guard()
    try:
        from analysis import trade_viewer_qt as tvq
    except ImportError as exc:
        # NOT `except Exception: return`. That swallowed every import-time
        # failure and left the suite talking to the live OpenD with nothing
        # on screen to say so: simulating one (sys.modules entry set to
        # None before configure) took two test files to 71 passed, green,
        # with 48 connections to OpenD and 24 option-API calls. The guard
        # above now catches that case regardless, but a missing stub is
        # still worth a word, and anything that is not a missing module is
        # worth a crash.
        missing = (getattr(exc, "name", "") or "").split(".")[0]
        if missing not in _OPTIONAL_GUI_MODULES:
            raise
        warnings.warn(
            f"conftest could not import analysis.trade_viewer_qt ({exc}); "
            f"the option-context stub is NOT installed. The OpenD socket "
            f"guard is.", RuntimeWarning, stacklevel=1)
        return
    _REAL_ENSURE_OPTION_CTX = tvq.TradeViewerQt._ensure_option_ctx
    tvq.TradeViewerQt._ensure_option_ctx = lambda self, *a, **k: None


@pytest.fixture()
def real_ensure_option_ctx(monkeypatch):
    """Put the real _ensure_option_ctx back, for the tests OF that method.

    pytest_configure stubs it suite-wide to keep the chain budget intact.
    A test whose subject IS the probe-then-connect logic has to opt back
    in, explicitly, rather than the suite-wide stub being loosened for
    everyone -- that loosening is what put the network back under 25 files.
    """
    from analysis import trade_viewer_qt as tvq
    # Falls back to the live attribute instead of skipping. Under
    # --allow-opend nothing was patched, so the class already carries the
    # real method -- and skipping there removed the only four tests of the
    # probe-then-connect logic from precisely the mode someone uses when
    # they are debugging real OpenD behaviour. These four stub
    # OpenQuoteContext and _opend_listening themselves, so they reach no
    # network either way (verified: 0 connections to OpenD under
    # --allow-opend with this fallback).
    real = _REAL_ENSURE_OPTION_CTX or tvq.TradeViewerQt._ensure_option_ctx
    monkeypatch.setattr(tvq.TradeViewerQt, "_ensure_option_ctx", real)
    return real

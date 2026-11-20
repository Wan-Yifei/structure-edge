"""Stop the moomoo SDK's own log file from wedging the push thread.

WHAT GOES WRONG
---------------
The SDK logs to one hard-coded path -- `%APPDATA%\\com.moomoo.OpenD\\Log\\
py_<date>.log` -- through a `TimedRotatingFileHandler(when='H',
interval=24)`. `FTLog` is a singleton per PROCESS, not per machine, so
every process of ours that opens a quote context installs its own
rotating handler on that same file.

A rotating handler decides a rollover is due from the file's age. Once
the file is a day old, every emit tries to rename it -- and on Windows
the rename fails while another process holds it open:

    PermissionError: [WinError 32] The process cannot access the file
    because it is being used by another process:
    'py_2026_11_15.log' -> 'py_2026_11_15.log.2026-11-15_12'

The failure does not stop at a dropped log line. `logging` prints the
whole "--- Logging error ---" traceback, and the SDK logs from inside
its single push-callback thread (`callback_executor.py`), so that thread
is now spending its time formatting tracebacks. Measured on 2026-11-17:

    Push callback use too much time: ... time=6.632s

Everything queued behind it is late, including K-line pushes. Minute
bars went missing from the live frame, and because the candle chart
plots by INDEX, two non-adjacent bars drawn side by side look like a
price gap -- one that moomoo's own chart did not show, and that healed
itself when the next full fetch refilled the missing minutes.

WHAT THIS DOES
--------------
Detaches the SDK's file handler in THIS process. We keep our own logs;
the SDK's debug file is not read by anything here, and nothing about it
is worth a stalled push thread. The console handler is left alone, so
SDK warnings still reach our own captured stdout.

Call it once, before opening any context:

    from core.sdk_log import quiet_sdk_file_log
    quiet_sdk_file_log()

It is safe to call more than once, safe before the SDK is imported, and
never raises -- a process must not fail to start because a log tweak did.
"""
from __future__ import annotations

import logging

_done = False


def quiet_sdk_file_log() -> bool:
    """Detach the SDK's rotating file handler. True if one was removed.

    Returns rather than logs so a caller can say so in its own log, and
    so the test can tell "removed" from "there was nothing to remove"
    without reading handler internals.
    """
    global _done
    removed = False
    try:
        from moomoo.common.ft_logger import FTLog
    except Exception:                                   # noqa: BLE001
        # SDK not installed or not importable: nothing to quieten, and
        # certainly nothing worth raising over.
        _done = True
        return False

    try:
        log = FTLog()
        handler = getattr(log, "fileHandler", None)
        if handler is not None:
            # Remove it from the logger first, then close it, so a
            # concurrent emit cannot find a closed handler still attached.
            try:
                log.file_logger.removeHandler(handler)
                removed = True
            except Exception:                           # noqa: BLE001
                pass
            try:
                handler.close()
            except Exception:                           # noqa: BLE001
                pass
            # The singleton checks `hasattr(self, 'fileHandler')` before
            # building one, so leaving the attribute in place is what
            # stops a later FTLog() call from installing a fresh handler
            # on the same file. Do NOT delete it.
        # Belt and braces: even with no handler attached, a CRITICAL from
        # the SDK would propagate nowhere useful and cost nothing.
        log.file_logger.setLevel(logging.CRITICAL)
    except Exception:                                   # noqa: BLE001
        pass

    # A failure inside logging must not print a traceback from the push
    # thread. This is process-wide and affects our own handlers too --
    # accepted deliberately: a logging failure should cost a line, not a
    # stalled callback thread.
    logging.raiseExceptions = False
    _done = True
    return removed


def already_quiet() -> bool:
    """Whether `quiet_sdk_file_log` has run in this process."""
    return _done

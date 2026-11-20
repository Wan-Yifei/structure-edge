"""The SDK's rotating log handler has to be gone, and the bug it causes
has to be reproducible -- otherwise this is a fix for a story.

The failure is specific: two processes hold one rotating file handler on
the same path, the file is old enough that a rollover is due, and on
Windows the rename fails because the other handle is open. `logging`
then prints a full traceback from whatever thread emitted -- which in
the SDK is the single push-callback thread.

So the first test REPRODUCES the rename failure with two handlers on one
file, rather than asserting that the SDK is configured a certain way.
If the platform ever stops failing that rename, that test says so and
the rest of this file can go.
"""

import logging
import logging.handlers
import os
import time

import pytest


def _aged_log(tmp_path, hours: float):
    """A log file whose mtime is `hours` old, so a rollover is due."""
    p = tmp_path / "py_old.log"
    p.write_text("x\n", encoding="utf-8")
    old = time.time() - hours * 3600
    os.utime(p, (old, old))
    return p


class TestTheFailureIsReal:
    """Does a second handler on an aged file actually break rollover?"""

    def test_two_handlers_on_one_aged_file_fail_to_rotate(self, tmp_path):
        p = _aged_log(tmp_path, hours=48)
        holder = logging.handlers.TimedRotatingFileHandler(
            str(p), when="H", interval=24, backupCount=20)
        rotator = logging.handlers.TimedRotatingFileHandler(
            str(p), when="H", interval=24, backupCount=20)
        try:
            assert rotator.shouldRollover(
                logging.LogRecord("t", logging.WARNING, __file__, 1,
                                  "m", None, None)), (
                "the fixture file is not old enough for a rollover to be "
                "due, so this test would pass without testing anything")
            if os.name != "nt":
                pytest.skip("POSIX renames an open file happily; this "
                            "failure mode is Windows-only")
            with pytest.raises(PermissionError) as e:
                rotator.doRollover()
            assert "being used by another process" in str(e.value) \
                or e.value.winerror == 32
        finally:
            for h in (rotator, holder):
                try:
                    h.close()
                except Exception:
                    pass


class TestTheHandlerIsDetached:

    def test_it_removes_the_sdk_file_handler(self):
        pytest.importorskip("moomoo.common.ft_logger")
        from moomoo.common.ft_logger import FTLog

        from core.sdk_log import quiet_sdk_file_log

        log = FTLog()
        # FTLog is a singleton and another test may already have cleaned
        # it, so put a handler back before asserting one is taken away.
        if getattr(log, "fileHandler", None) not in log.file_logger.handlers:
            log.file_logger.addHandler(log.fileHandler)

        assert log.fileHandler in log.file_logger.handlers
        assert quiet_sdk_file_log() is True
        assert log.fileHandler not in log.file_logger.handlers, (
            "the SDK's rotating handler is still attached, so the push "
            "thread can still be wedged by a failed rollover")

    def test_the_attribute_stays_so_a_new_handler_is_not_built(self):
        """FTLog() checks `hasattr(self, 'fileHandler')` before creating
        one. Deleting the attribute would make the next FTLog() call --
        and there is one per context -- install a fresh handler on the
        same file, which is the whole problem back again."""
        pytest.importorskip("moomoo.common.ft_logger")
        from moomoo.common.ft_logger import FTLog

        from core.sdk_log import quiet_sdk_file_log

        quiet_sdk_file_log()
        assert hasattr(FTLog(), "fileHandler")
        FTLog()          # a second context would do this
        assert FTLog().fileHandler not in FTLog().file_logger.handlers

    def test_calling_it_twice_is_safe(self):
        from core.sdk_log import quiet_sdk_file_log
        quiet_sdk_file_log()
        quiet_sdk_file_log()          # must not raise

    def test_logging_no_longer_prints_tracebacks(self):
        """The traceback printing is what cost the 6.6 seconds, not the
        rename itself."""
        from core.sdk_log import quiet_sdk_file_log
        quiet_sdk_file_log()
        assert logging.raiseExceptions is False

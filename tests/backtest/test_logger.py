"""Unit tests for backtest/logger.py — multiprocessing-safe logging helpers."""

import logging
import logging.handlers
import multiprocessing
import pathlib
import time

import pytest

from backtest.logger import (
    make_listener,
    worker_init,
    get_logger,
    setup_main_logger,
)


# ── make_listener ─────────────────────────────────────────────────────────────

class TestMakeListener:
    def test_returns_queue_and_listener(self, tmp_path):
        q, listener = make_listener(tmp_path / "test.log")
        assert isinstance(q, multiprocessing.queues.Queue)
        assert isinstance(listener, logging.handlers.QueueListener)
        listener.start()
        listener.stop()

    def test_log_file_created(self, tmp_path):
        log_path = tmp_path / "run.log"
        q, listener = make_listener(log_path)
        listener.start()
        # Send a record through the queue
        handler = logging.handlers.QueueHandler(q)
        record = logging.LogRecord(
            name="test", level=logging.INFO,
            pathname="", lineno=0,
            msg="hello log", args=(), exc_info=None,
        )
        record.__dict__["tag"] = "test"
        handler.emit(record)
        time.sleep(0.15)   # let listener flush
        listener.stop()
        assert log_path.exists()
        assert "hello log" in log_path.read_text(encoding="utf-8")

    def test_parent_dir_created(self, tmp_path):
        nested = tmp_path / "deep" / "nested" / "run.log"
        q, listener = make_listener(nested)
        listener.start()
        listener.stop()
        assert nested.parent.exists()


# ── worker_init ───────────────────────────────────────────────────────────────

class TestWorkerInit:
    def test_installs_queue_handler(self):
        q = multiprocessing.Queue(-1)
        root_before = list(logging.getLogger().handlers)
        worker_init(q)
        root = logging.getLogger()
        queue_handlers = [h for h in root.handlers
                          if isinstance(h, logging.handlers.QueueHandler)]
        assert len(queue_handlers) == 1
        # Cleanup: restore root handlers
        logging.getLogger().handlers = root_before

    def test_clears_existing_handlers(self):
        q = multiprocessing.Queue(-1)
        root = logging.getLogger()
        root.addHandler(logging.StreamHandler())   # add a dummy handler first
        worker_init(q)
        # After init, only the QueueHandler should remain
        assert all(isinstance(h, logging.handlers.QueueHandler)
                   for h in root.handlers)
        root.handlers = []   # cleanup


# ── get_logger ────────────────────────────────────────────────────────────────

class TestGetLogger:
    def test_returns_logger_adapter(self):
        log = get_logger("test_tag")
        assert isinstance(log, logging.LoggerAdapter)

    def test_tag_in_extra(self):
        log = get_logger("W01 4h/15m")
        assert log.extra.get("tag") == "W01 4h/15m"

    def test_logger_name_is_backtest(self):
        log = get_logger("main")
        assert log.logger.name == "backtest"

    def test_usable_as_standard_logger(self):
        setup_main_logger()
        log = get_logger("unit_test")
        # Should not raise
        log.info("test message %s", 42)
        log.debug("debug message")
        log.warning("warn message")


# ── setup_main_logger ─────────────────────────────────────────────────────────

class TestSetupMainLogger:
    def test_adds_stream_handler(self):
        root = logging.getLogger()
        original = list(root.handlers)
        root.handlers.clear()
        setup_main_logger()
        stream_handlers = [h for h in root.handlers
                           if isinstance(h, logging.StreamHandler)]
        assert len(stream_handlers) >= 1
        root.handlers = original   # restore

    def test_idempotent(self):
        root = logging.getLogger()
        original = list(root.handlers)
        root.handlers.clear()
        setup_main_logger()
        count_after_first = len(root.handlers)
        setup_main_logger()   # second call
        assert len(root.handlers) == count_after_first
        root.handlers = original

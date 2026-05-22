"""Multiprocessing-safe logging for the backtest system.

ProcessPoolExecutor on Windows uses spawn mode — each worker is an independent
process with its own file handles. Multiple workers writing to the same log file
causes interleaved / truncated output.

Solution: workers enqueue LogRecord objects via QueueHandler; a single
QueueListener in the main process performs all I/O.

    Worker-0 ──→ ┐
    Worker-1 ──→ ├──→ multiprocessing.Queue ──→ QueueListener ──→ run.log
    Worker-N ──→ ┘                                             ──→ stderr

Usage in main process:
    q, listener = make_listener(log_path)
    listener.start()
    # ... spawn workers with initializer=worker_init, initargs=(q,) ...
    listener.stop()

Usage inside a worker process (called automatically via initializer):
    log = get_logger("W03 4h/15m lb2")
    log.info("FVG detected at %.2f", price)
"""

from __future__ import annotations

import logging
import logging.handlers
import multiprocessing
import pathlib
from typing import Optional

LOG_FORMAT = "%(asctime)s [%(tag)-26s] %(levelname)-7s %(message)s"
DATE_FORMAT = "%H:%M:%S"

# Module-level queue — set by worker_init() in each worker process
_worker_queue: Optional[multiprocessing.Queue] = None


def make_listener(
    log_path: str | pathlib.Path,
    level: int = logging.DEBUG,
) -> tuple[multiprocessing.Queue, logging.handlers.QueueListener]:
    """Create a Queue + QueueListener. Call in the main process before spawning workers.

    Returns (queue, listener). Call listener.start() before submitting work,
    and listener.stop() after all futures complete.
    """
    q: multiprocessing.Queue = multiprocessing.Queue(-1)

    log_path = pathlib.Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))

    listener = logging.handlers.QueueListener(
        q, file_handler, stream_handler,
        respect_handler_level=True,
    )
    return q, listener


def worker_init(log_queue: multiprocessing.Queue, level: int = logging.INFO) -> None:
    """ProcessPoolExecutor initializer — called once per worker process on spawn.

    Installs a QueueHandler so all logging in the worker is forwarded to the
    main process listener. Safe to call multiple times (idempotent).
    """
    global _worker_queue
    _worker_queue = log_queue

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(logging.handlers.QueueHandler(log_queue))
    root.setLevel(level)


def get_logger(tag: str) -> logging.LoggerAdapter:
    """Return a LoggerAdapter pre-tagged with [tag]. Use exactly like logging.Logger.

    Tag format examples:
        "W03 4h/15m lb2 bos1"   — backtest worker
        "reviewer a1b2c3d4"      — trade reviewer session
        "main"                   — main process / orchestrator
    """
    logger = logging.getLogger("backtest")
    return logging.LoggerAdapter(logger, {"tag": tag})


def setup_main_logger(level: int = logging.INFO) -> None:
    """Configure basic logging for the main process (no queue, no workers).

    Call this when running single-process code (tests, CLI tools) that
    uses get_logger() but does not spin up workers.
    """
    root = logging.getLogger()
    if root.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    root.addHandler(handler)
    root.setLevel(level)

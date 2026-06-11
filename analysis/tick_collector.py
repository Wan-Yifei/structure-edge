"""Real-time tick collector — subscribes to TICKER feed and writes to DuckDB.

Usage:
    uv run analysis/tick_collector.py [--config config/schedule.json] [--host HOST] [--port PORT]

The script reads target codes from the schedule config and subscribes to each.
Press Ctrl-C to stop.
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import signal
import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
import threading
import time
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_DEFAULT_CONFIG = pathlib.Path(__file__).parent.parent / "config" / "schedule.json"
_DEFAULT_DB     = pathlib.Path(__file__).parent.parent / "db" / "ticks.db"


# ── Tick handler ───────────────────────────────────────────────────────────────

def _make_handler(store, state: dict):
    """Return a TickerHandlerBase subclass that writes to *store*.

    *state* is a shared dict with keys:
        last_tick_time  float | None   — time.time() of most recent tick batch
        first_tick_done bool           — whether the first-tick log was emitted
    """
    from moomoo import TickerHandlerBase

    class _Handler(TickerHandlerBase):
        def on_recv_rsp(self, rsp_pb):
            ret, data = super().on_recv_rsp(rsp_pb)
            if ret != 0 or data is None or data.empty:
                return ret, data

            rows = []
            for _, row in data.iterrows():
                direction = str(row.get("ticker_direction", "NEUTRAL")).upper()
                if direction not in ("BUY", "SELL"):
                    direction = "NEUTRAL"
                rows.append({
                    "code":      row["code"],
                    "ts":        row["time"],
                    "price":     float(row["price"]),
                    "volume":    int(row["volume"]),
                    "direction": direction,
                })
            n = store.insert_ticks(rows)
            if n:
                state["last_tick_time"] = time.time()
                state["session_count"]  = state.get("session_count", 0) + n
                if not state["first_tick_done"]:
                    state["first_tick_done"] = True
                    last = rows[-1]
                    log.info(
                        "DATA RECEIVED  first tick: %s  ts=%s  price=%.4f  vol=%d  dir=%s",
                        last["code"], last["ts"], last["price"], last["volume"], last["direction"],
                    )
                else:
                    last = rows[-1]
                    log.debug(
                        "%s  %s  price=%.4f  vol=%d  dir=%s",
                        last["code"], last["ts"], last["price"], last["volume"], last["direction"],
                    )
            return ret, data

    return _Handler


def _watchdog(state: dict, timeout_minutes: int, stop_event: threading.Event,
              ctx, codes):
    """Warn and auto-reconnect when no tick received for longer than *timeout_minutes*."""
    from moomoo import SubType, RET_OK, Session

    timeout_sec    = timeout_minutes * 60
    check_interval = min(60, timeout_sec)   # check every 60 s (or timeout if shorter)
    last_log_time  = time.time()
    reconnecting   = False

    while not stop_event.wait(check_interval):
        now = time.time()

        # Periodic count log (once per timeout window)
        if now - last_log_time >= timeout_sec:
            log.info("This session: %d ticks", state.get("session_count", 0))
            last_log_time = now

        last = state["last_tick_time"]
        if last is None:
            continue

        elapsed = now - last
        if elapsed > timeout_sec:
            if not reconnecting:
                log.warning(
                    "NO DATA  %.0f min since last tick — re-subscribing",
                    elapsed / 60,
                )
                reconnecting = True
                try:
                    ctx.unsubscribe(codes, [SubType.TICKER])
                except Exception:
                    pass
                try:
                    ret, msg = ctx.subscribe(
                        codes, [SubType.TICKER],
                        subscribe_push=True, extended_time=True, session=Session.ALL,
                    )
                    if ret == RET_OK:
                        log.info("Re-subscribed OK — waiting for first tick")
                        state["last_tick_time"] = None   # reset so warning doesn't repeat
                        reconnecting = False
                    else:
                        log.error("Re-subscribe failed: %s — will retry next cycle", msg)
                except Exception as exc:
                    log.error("Re-subscribe error: %s — will retry next cycle", exc)
        else:
            reconnecting = False             # data resumed, clear flag


# ── main ───────────────────────────────────────────────────────────────────────

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Real-time tick collector — streams TICKER feed to DuckDB.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run analysis/tick_collector.py
  uv run analysis/tick_collector.py --config config/schedule.json --host 127.0.0.1 --port 11111
  uv run analysis/tick_collector.py --codes US.AAPL US.TSLA
        """,
    )
    p.add_argument("--config", default=str(_DEFAULT_CONFIG),
                   help="Path to schedule.json (default: config/schedule.json)")
    p.add_argument("--db", default=str(_DEFAULT_DB),
                   help="Path to DuckDB file (default: store/ticks.duckdb)")
    p.add_argument("--codes", nargs="*",
                   help="Override target codes from config (e.g. US.AAPL US.TSLA)")
    p.add_argument("--timeout", type=int, default=5,
                   help="Warn if no tick received for this many minutes (default: 5)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=11111)
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    # ── resolve target codes and timeout ──────────────────────────────────
    timeout_minutes = args.timeout
    if args.codes:
        codes = args.codes
    else:
        cfg_path = pathlib.Path(args.config)
        if not cfg_path.exists():
            log.error("Config not found: %s", cfg_path)
            sys.exit(1)
        with open(cfg_path, encoding='utf-8') as f:
            cfg = json.load(f)
        codes           = cfg.get("targets", [])
        timeout_minutes = cfg.get("data_timeout_minutes", timeout_minutes)

    if not codes:
        log.error("No target codes — set 'targets' in config or pass --codes")
        sys.exit(1)

    log.info("Targets: %s", ", ".join(codes))
    log.info("DB: %s", args.db)
    log.info("No-data warning after: %d min", timeout_minutes)

    # ── open store (retry if another process briefly holds the WAL lock) ───
    from feeds.tick_store import TickStore
    for _attempt in range(10):
        try:
            store = TickStore(args.db)
            break
        except Exception as _exc:
            if _attempt < 9:
                log.warning("DB locked, retrying in 3 s… (%s)", _exc)
                time.sleep(3)
            else:
                log.error("Cannot open DB after 10 attempts: %s", _exc)
                sys.exit(1)

    # ── shared state for handler ↔ watchdog ───────────────────────────────
    state = {"last_tick_time": None, "first_tick_done": False, "session_count": 0}

    # ── open moomoo quote context ──────────────────────────────────────────
    from moomoo import OpenQuoteContext, SubType, RET_OK, Session

    ctx          = OpenQuoteContext(host=args.host, port=args.port)
    HandlerClass = _make_handler(store, state)
    ctx.set_handler(HandlerClass())

    ret, msg = ctx.subscribe(codes, [SubType.TICKER], subscribe_push=True,
                             extended_time=True, session=Session.ALL)
    if ret != RET_OK:
        log.error("Subscribe failed: %s", msg)
        ctx.close()
        store.close()
        sys.exit(1)

    log.info("Subscribed — collecting ticks. Press Ctrl-C to stop.")

    # ── watchdog thread ────────────────────────────────────────────────────
    stop_event = threading.Event()
    threading.Thread(
        target=_watchdog, args=(state, timeout_minutes, stop_event, ctx, codes),
        daemon=True,
    ).start()

    # ── run until interrupted ──────────────────────────────────────────────
    stop = False

    def _handle_signal(sig, frame):
        nonlocal stop
        stop = True
        stop_event.set()
        log.info("Stopping…")

    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        while not stop:
            time.sleep(1)
    finally:
        ctx.close()
        log.info("Session ended.")
        store.close()


if __name__ == "__main__":
    main()

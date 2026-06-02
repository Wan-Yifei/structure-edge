"""Order book collector — subscribes to ORDER_BOOK push and writes to SQLite.

Stores resting limit order depth (bid/ask levels) as time-series snapshots.
This data feeds the Liquidity Heatmap in trade_viewer_qt.py — distinct from
the tick collector which records executed trades.

Usage:
    uv run analysis/order_book_collector.py
    uv run analysis/order_book_collector.py --codes US.AAPL US.TSLA
    uv run analysis/order_book_collector.py --config config/schedule.json
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
_DEFAULT_DB     = pathlib.Path(__file__).parent.parent / "db" / "order_book.db"


def _parse_side(items) -> list[tuple[float, int]]:
    """Parse bid or ask levels from push data into (price, volume) tuples.

    Push items may be dicts {"price": ..., "volume": ...} or sequences [price, volume, ...].
    """
    result = []
    for item in items:
        try:
            if isinstance(item, dict):
                result.append((float(item["price"]), int(item["volume"])))
            else:
                result.append((float(item[0]), int(item[1])))
        except (KeyError, IndexError, TypeError, ValueError):
            pass
    return result


# ── Order book handler ─────────────────────────────────────────────────────────

_MIN_WRITE_INTERVAL = 2.0  # seconds — minimum gap between DB writes per code


def _make_handler(store, state: dict):
    """Return an OrderBookHandlerBase subclass that writes snapshots to *store*.

    *state* keys:
        last_update_time  float | None   — time.time() of most recent snapshot
        first_update_done bool           — whether first-update log was emitted
        session_count     int            — total rows inserted this session
    Writes are rate-limited to _MIN_WRITE_INTERVAL seconds per code to prevent
    WAL runaway growth on high-frequency ORDER_BOOK pushes.
    """
    from moomoo import OrderBookHandlerBase, RET_OK

    last_write: dict[str, float] = {}  # code -> time.time() of last DB write

    class _Handler(OrderBookHandlerBase):
        def on_recv_rsp(self, rsp_pb):
            ret, data = super().on_recv_rsp(rsp_pb)
            if ret != RET_OK or data is None:
                return ret, data

            code = data.get("code", "")
            bids = _parse_side(data.get("Bid", []))
            asks = _parse_side(data.get("Ask", []))
            if not bids and not asks:
                return ret, data

            now = time.time()
            if now - last_write.get(code, 0.0) < _MIN_WRITE_INTERVAL:
                return ret, data  # skip — too soon since last write for this code

            ts = datetime.now()
            n  = store.insert_snapshot(code, ts, bids, asks)
            if n:
                last_write[code] = now
                state["last_update_time"] = now
                state["session_count"]    = state.get("session_count", 0) + n
                if not state["first_update_done"]:
                    state["first_update_done"] = True
                    log.info(
                        "DATA RECEIVED  first snapshot: %s  ts=%s  %d bid + %d ask levels",
                        code, ts.strftime("%H:%M:%S"), len(bids), len(asks),
                    )
                else:
                    log.debug(
                        "%s  %s  %d bid + %d ask levels",
                        code, ts.strftime("%H:%M:%S.%f")[:-3], len(bids), len(asks),
                    )
            return ret, data

    return _Handler


_CHECKPOINT_INTERVAL = 10  # run PASSIVE checkpoint every N watchdog ticks


def _watchdog(state: dict, timeout_minutes: int, stop_event: threading.Event,
              ctx=None, codes: list[str] | None = None, store=None):
    """Warn when no update received for *timeout_minutes*; re-subscribe if possible.

    Also runs a PASSIVE WAL checkpoint every _CHECKPOINT_INTERVAL ticks to keep
    the WAL file from growing unboundedly under high write rates.
    """
    from moomoo import SubType
    timeout_sec   = timeout_minutes * 60
    warned        = False
    tick_count    = 0
    while not stop_event.wait(timeout_sec):
        tick_count += 1
        session = state.get("session_count", 0)
        log.info("This session: %d rows", session)

        if store is not None:
            try:
                deleted = store.prune(keep=1000)
                if deleted:
                    log.info("Pruned %d old rows (keeping ≤1000 per code)", deleted)
            except Exception as exc:
                log.warning("Prune failed: %s", exc)

        if store is not None and tick_count % _CHECKPOINT_INTERVAL == 0:
            try:
                store._con.execute("PRAGMA wal_checkpoint(PASSIVE)")
                log.info("WAL checkpoint (PASSIVE) done")
            except Exception as exc:
                log.warning("WAL checkpoint failed: %s", exc)

        last = state["last_update_time"]
        if last is None:
            continue
        elapsed = time.time() - last
        if elapsed > timeout_sec:
            if not warned:
                log.warning(
                    "NO DATA  %.0f min since last order book update — attempting re-subscribe",
                    elapsed / 60,
                )
                warned = True
                if ctx is not None and codes:
                    try:
                        ret, msg = ctx.subscribe(codes, [SubType.ORDER_BOOK],
                                                 subscribe_push=True)
                        if ret == 0:
                            log.info("Re-subscribed OK: %s", ", ".join(codes))
                            warned = False  # reset so next timeout triggers another warning+retry
                        else:
                            log.error("Re-subscribe failed: %s", msg)
                    except Exception as exc:
                        log.error("Re-subscribe error: %s", exc)
        else:
            warned = False


# ── main ───────────────────────────────────────────────────────────────────────

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Order book snapshot collector — subscribes to ORDER_BOOK push.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run analysis/order_book_collector.py
  uv run analysis/order_book_collector.py --codes US.AAPL US.TSLA
  uv run analysis/order_book_collector.py --config config/schedule.json
        """,
    )
    p.add_argument("--config",  default=str(_DEFAULT_CONFIG),
                   help="Path to schedule.json (default: config/schedule.json)")
    p.add_argument("--db",      default=str(_DEFAULT_DB),
                   help="Path to SQLite DB (default: db/order_book.db)")
    p.add_argument("--codes",   nargs="*",
                   help="Override target codes (e.g. US.AAPL US.TSLA)")
    p.add_argument("--timeout", type=int, default=5,
                   help="Warn if no update for this many minutes (default: 5)")
    p.add_argument("--host",    default="127.0.0.1")
    p.add_argument("--port",    type=int, default=11111)
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    if args.codes:
        codes           = args.codes
        timeout_minutes = args.timeout
    else:
        cfg_path = pathlib.Path(args.config)
        if not cfg_path.exists():
            log.error("Config not found: %s", cfg_path)
            sys.exit(1)
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        codes           = cfg.get("targets", [])
        timeout_minutes = cfg.get("data_timeout_minutes", args.timeout)

    if not codes:
        log.error("No target codes — set 'targets' in config or pass --codes")
        sys.exit(1)

    log.info("Targets: %s", ", ".join(codes))
    log.info("DB: %s", args.db)
    log.info("No-data warning after: %d min", timeout_minutes)

    from feeds.order_book_store import OrderBookStore
    store = OrderBookStore(args.db)
    log.info("DB ready: %s", args.db)

    state = {"last_update_time": None, "first_update_done": False, "session_count": 0}

    from moomoo import OpenQuoteContext, SubType, RET_OK

    ctx          = OpenQuoteContext(host=args.host, port=args.port)
    HandlerClass = _make_handler(store, state)
    ctx.set_handler(HandlerClass())

    ret, msg = ctx.subscribe(codes, [SubType.ORDER_BOOK], subscribe_push=True)
    if ret != RET_OK:
        log.error("Subscribe failed: %s", msg)
        ctx.close()
        store.close()
        sys.exit(1)

    log.info("Subscribed — collecting order book snapshots. Press Ctrl-C to stop.")

    stop_event = threading.Event()
    threading.Thread(
        target=_watchdog,
        args=(state, timeout_minutes, stop_event, ctx, codes, store),
        daemon=True,
    ).start()

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

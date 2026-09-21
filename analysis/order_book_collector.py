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
from zoneinfo import ZoneInfo

_ET = ZoneInfo("America/New_York")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_DEFAULT_CONFIG = pathlib.Path(__file__).parent.parent / "config" / "schedule.json"
_DEFAULT_DB     = pathlib.Path(__file__).parent.parent / "db" / "order_book.db"


# ── Order book handler ─────────────────────────────────────────────────────────

# The 2s-per-code write throttle that used to live here is gone. It existed
# because the row-per-price-level schema turned one push into ~120 inserts, so
# an unthrottled feed wrote ~800 rows/s and the WAL grew without bound. It cost
# 93% of the book updates the feed delivered (measured, SOXL 2026-09-21: 2,008
# pushes in 300s, 143 kept) and every one of those was a distinct book -- the
# same run found zero repeats among 2,007 consecutive comparisons.
#
# One row per snapshot makes the same feed ~7 rows/s, so there is nothing left
# to throttle away.
_SIDE_STALE_SECS = 10.0  # drop a cached side once it's gone this long without a
                          # fresh (non-empty) push. The push protocol omits a
                          # side when it's simply *unchanged*, which is the
                          # normal case the merge cache below exists for -- but
                          # if a side goes dark far longer than that (feed
                          # hiccup, subscription issue), re-writing the same
                          # increasingly-old snapshot forever turns one stale
                          # push into an indefinitely-persisting phantom price
                          # level (reported: a frozen bid band sitting for
                          # minutes in the Liquidity Heatmap, priced well above
                          # the live ask -- structurally impossible for a real
                          # resting order, since it would just cross and fill).

# Retention (see _watchdog's prune call) -- same "give this one symbol more
# history" intent as tick_collector.py's _RETENTION_EXEMPT, but ob_snapshots
# is pruned by row count, not age, so an unbounded exemption (like ticks.db's)
# isn't appropriate here.
#
# The counts are DERIVED from an hours target rather than written down, because
# a bare count silently changes meaning whenever the store rate changes. That
# already happened once: 60_000 was chosen when the collector throttled writes
# to one snapshot per 2s and meant 33 hours of SOXL history; removing the
# throttle raised the rate ~12x and the same constant quietly became 2.7 hours.
#
# _SNAPSHOTS_PER_SEC is the *regular-session* rate, which is the densest one
# (the feed's ~300ms cadence is fixed, but it bursts more per tick when the
# market is active -- 2.25 pushes/tick at the open against 1.0 overnight).
# Sizing on the densest rate means the hours target is a floor across every
# session, at the cost of some disk in the quiet ones.
#
# Disk: ~2,093 B per snapshot on disk including the index (measured), so SOXL's
# 24h works out to ~1.1 GB and each additional code to ~0.3 GB.
_SNAPSHOTS_PER_SEC = 6.2                       # measured, regular session
_RETENTION_HOURS_DEFAULT = 6.5                 # one regular session
_RETENTION_HOURS_OVERRIDE = {"US.SOXL": 24.0}  # a full 20:00->20:00 cycle


def _keep_for_hours(hours: float) -> int:
    return int(hours * 3600 * _SNAPSHOTS_PER_SEC)


_RETENTION_KEEP_DEFAULT = _keep_for_hours(_RETENTION_HOURS_DEFAULT)
_RETENTION_KEEP_OVERRIDE = {c: _keep_for_hours(h)
                            for c, h in _RETENTION_HOURS_OVERRIDE.items()}


def _make_handler(store, state: dict):
    """Return an OrderBookHandlerBase subclass that writes snapshots to *store*.

    *state* keys:
        last_update_time  float | None   — time.time() of most recent snapshot
        first_update_done bool           — whether first-update log was emitted
        session_count     int            — total rows inserted this session
    Every push that yields a complete book is written. There is no write
    throttle any more -- see the note where _MIN_WRITE_INTERVAL used to be.

    Folding partial pushes into complete two-sided snapshots is
    feeds.order_book_merge.OrderBookMerger's job -- see it for why a push
    cannot be trusted to carry both sides, and why a crossed book has to be
    attributed to one side or dropped. That logic is shared with the
    Liquidity Heatmap, which consumes the same feed directly; a second copy
    would drift and re-earn the bugs those rules were written for.
    """
    from moomoo import OrderBookHandlerBase, RET_OK
    from feeds.order_book_merge import OrderBookMerger

    merger = OrderBookMerger(stale_secs=_SIDE_STALE_SECS, log=log)

    class _Handler(OrderBookHandlerBase):
        def on_recv_rsp(self, rsp_pb):
            ret, data = super().on_recv_rsp(rsp_pb)
            if ret != RET_OK or data is None:
                return ret, data

            code = data.get("code", "")
            now  = time.time()
            merged = merger.merge(code, data.get("Bid", []), data.get("Ask", []), now)
            if merged is None:
                return ret, data
            bids, asks = merged

            ts = datetime.now(_ET).replace(tzinfo=None)
            n  = store.insert_snapshot(code, ts, bids, asks)
            if n:
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
              ctx_holder: list, codes: list[str],
              host: str, port: int, handler_class,
              store=None):
    """Warn and reconnect when no OB update received for *timeout_minutes*.

    Two-tier recovery (mirrors tick_collector):
      1. Re-subscribe on the existing OpenQuoteContext.
      2. If that fails, close the dead ctx and rebuild a fresh one.

    Also runs a PASSIVE WAL checkpoint every _CHECKPOINT_INTERVAL ticks.
    """
    from moomoo import OpenQuoteContext, SubType

    timeout_sec      = timeout_minutes * 60
    check_interval   = min(60, timeout_sec)
    last_log_time    = time.time()
    last_resub_time: float | None = None
    tick_count       = 0

    def _try_subscribe(ctx) -> bool:
        try:
            ctx.unsubscribe(codes, [SubType.ORDER_BOOK])
        except Exception:
            pass
        try:
            ret, msg = ctx.subscribe(codes, [SubType.ORDER_BOOK], subscribe_push=True)
            if ret == 0:
                return True
            log.warning("Re-subscribe on existing ctx failed: %s", msg)
        except Exception as exc:
            log.warning("Re-subscribe on existing ctx error: %s", exc)
        return False

    def _rebuild_ctx() -> bool:
        try:
            ctx_holder[0].close()
        except Exception:
            pass
        try:
            new_ctx = OpenQuoteContext(host=host, port=port)
            new_ctx.set_handler(handler_class())
            ret, msg = new_ctx.subscribe(codes, [SubType.ORDER_BOOK], subscribe_push=True)
            if ret == 0:
                ctx_holder[0] = new_ctx
                log.info("Rebuilt context and re-subscribed OK")
                return True
            log.error("New ctx subscribe failed: %s — will retry", msg)
            new_ctx.close()
        except Exception as exc:
            log.error("Context rebuild failed: %s — will retry", exc)
        return False

    while not stop_event.wait(check_interval):
        tick_count += 1
        now = time.time()

        if now - last_log_time >= timeout_sec:
            log.info("This session: %d rows", state.get("session_count", 0))
            last_log_time = now

        if store is not None:
            try:
                override_codes = [c for c in codes if c in _RETENTION_KEEP_OVERRIDE]
                default_codes  = [c for c in codes if c not in _RETENTION_KEEP_OVERRIDE]
                deleted = 0
                for c in override_codes:
                    deleted += store.prune(keep=_RETENTION_KEEP_OVERRIDE[c], codes=[c])
                if default_codes:
                    deleted += store.prune(keep=_RETENTION_KEEP_DEFAULT, codes=default_codes)
                if deleted:
                    log.info("Pruned %d old snapshots (keeping <=%d per code, %s)",
                             deleted, _RETENTION_KEEP_DEFAULT,
                             ", ".join(f"{c}<={k}" for c, k in _RETENTION_KEEP_OVERRIDE.items()))
            except Exception as exc:
                log.warning("Prune failed: %s", exc)

        if store is not None and tick_count % _CHECKPOINT_INTERVAL == 0:
            try:
                store._con.execute("PRAGMA wal_checkpoint(PASSIVE)")
                log.info("WAL checkpoint (PASSIVE) done")
            except Exception as exc:
                log.warning("WAL checkpoint failed: %s", exc)

        last = state["last_update_time"]
        if last is not None:
            elapsed = now - last
        elif last_resub_time is not None:
            elapsed = now - last_resub_time
        else:
            continue

        if elapsed <= timeout_sec:
            continue

        log.warning("NO DATA  %.0f min since last OB update — reconnecting", elapsed / 60)
        last_resub_time = now
        state["last_update_time"] = None

        if _try_subscribe(ctx_holder[0]):
            log.info("Re-subscribed on existing context OK")
        else:
            _rebuild_ctx()


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

    from moomoo import OpenQuoteContext, SubType, RET_OK, set_futu_debug_model

    # moomoo's own internal SDK logger defaults to DEBUG-level file logging,
    # writing every single push message to a shared, date-named log file
    # under %APPDATA%\com.moomoo.OpenD\Log -- the *same* file for every
    # moomoo-connected process on the machine (this collector, tick_collector,
    # the main viewer, etc). Its TimedRotatingFileHandler tries to os.rename()
    # that file when the rotation window rolls over; with multiple processes
    # all holding it open for concurrent writes, that rename intermittently
    # collides with Windows' exclusive lock and fails with PermissionError
    # (WinError 32) -- once triggered it re-fires on every subsequent push
    # until something releases the lock, flooding this collector's own log.
    # Dropping the SDK's file/console log level to WARNING (its documented
    # "debug model" toggle) stops it from writing these routine per-push
    # acknowledgements at all, which avoids hitting that race in practice.
    set_futu_debug_model(False)

    HandlerClass = _make_handler(store, state)
    ctx          = OpenQuoteContext(host=args.host, port=args.port)
    ctx.set_handler(HandlerClass())

    ret, msg = ctx.subscribe(codes, [SubType.ORDER_BOOK], subscribe_push=True)
    if ret != RET_OK:
        log.error("Subscribe failed: %s", msg)
        ctx.close()
        store.close()
        sys.exit(1)

    log.info("Subscribed — collecting order book snapshots. Press Ctrl-C to stop.")

    ctx_holder = [ctx]   # mutable so watchdog can replace a dead ctx
    stop_event = threading.Event()
    threading.Thread(
        target=_watchdog,
        args=(state, timeout_minutes, stop_event,
              ctx_holder, codes, args.host, args.port, HandlerClass, store),
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
        ctx_holder[0].close()
        log.info("Session ended.")
        store.close()


if __name__ == "__main__":
    main()

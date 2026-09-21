"""ORDER_BOOK push-rate probe — measures, writes nothing.

Answers the question `analysis/order_book_collector.py` throws away: how often
does moomoo actually push ORDER_BOOK? The collector rate-limits DB writes to
_MIN_WRITE_INTERVAL (2.0 s per code) to stop the WAL growing unboundedly, so
`order_book.db` cannot answer it -- every gap measured there is floored at 2 s
by that limit, not by the feed.

This subscribes with the same SubType.ORDER_BOOK / subscribe_push=True the
collector uses and times every on_recv_rsp callback, then reports what the
collector's throttle would have dropped. Nothing is written to any database.

Run it alongside the collector (a second subscription to the same codes counts
against the OpenD quota -- if the subscribe fails with a quota error, stop the
collector first, or probe fewer codes with --codes).

Usage:
    uv run analysis/ob_push_probe.py --seconds 300
    uv run analysis/ob_push_probe.py --codes US.SOXL --seconds 600
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics as st
import sys
import threading
import time
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

_DEFAULT_CONFIG = pathlib.Path(__file__).parent.parent / "config" / "schedule.json"
_THROTTLE = 2.0   # mirrors order_book_collector._MIN_WRITE_INTERVAL


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Measure the real ORDER_BOOK push rate. Writes nothing.")
    p.add_argument("--config", default=str(_DEFAULT_CONFIG),
                   help="Path to schedule.json (default: config/schedule.json)")
    p.add_argument("--codes", nargs="*",
                   help="Override target codes (e.g. US.SOXL US.SOXS)")
    p.add_argument("--seconds", type=int, default=300,
                   help="How long to sample before reporting (default: 300)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=11111)
    return p.parse_args(argv)


def _pct(xs: list[float], q: float) -> float:
    """Percentile of a sorted list, nearest-rank."""
    if not xs:
        return float("nan")
    return xs[min(len(xs) - 1, int(q / 100.0 * len(xs)))]


def main(argv=None) -> None:
    args = _parse_args(argv)

    if args.codes:
        codes = args.codes
    else:
        cfg_path = pathlib.Path(args.config)
        if not cfg_path.exists():
            print(f"Config not found: {cfg_path}", file=sys.stderr)
            sys.exit(1)
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        # Same shape order_book_collector.py reads: a flat list of code
        # strings, so the probe measures exactly what the collector sees.
        codes = cfg.get("targets", [])
    if not codes:
        print("No codes to probe.", file=sys.stderr)
        sys.exit(1)

    from moomoo import OpenQuoteContext, OrderBookHandlerBase, RET_OK, SubType
    try:
        from moomoo import set_futu_debug_model
        set_futu_debug_model(False)   # same log-rotation race the collector hits
    except Exception:
        pass

    lock = threading.Lock()
    first: dict[str, float] = {}
    last: dict[str, float] = {}
    gaps: dict[str, list[float]] = defaultdict(list)
    n_push: dict[str, int] = defaultdict(int)
    n_bid_only: dict[str, int] = defaultdict(int)
    n_ask_only: dict[str, int] = defaultdict(int)
    n_both: dict[str, int] = defaultdict(int)
    n_kept: dict[str, int] = defaultdict(int)      # would survive the 2 s throttle
    last_kept: dict[str, float] = {}

    class _Probe(OrderBookHandlerBase):
        def on_recv_rsp(self, rsp_pb):
            ret, data = super().on_recv_rsp(rsp_pb)
            if ret != RET_OK or data is None:
                return ret, data
            now = time.time()
            code = data.get("code", "")
            bids, asks = data.get("Bid", []), data.get("Ask", [])
            with lock:
                n_push[code] += 1
                if bids and asks:
                    n_both[code] += 1
                elif bids:
                    n_bid_only[code] += 1
                elif asks:
                    n_ask_only[code] += 1
                if code in last:
                    gaps[code].append(now - last[code])
                else:
                    first[code] = now
                last[code] = now
                # What the collector would actually have persisted.
                if now - last_kept.get(code, 0.0) >= _THROTTLE:
                    n_kept[code] += 1
                    last_kept[code] = now
            return ret, data

    ctx = OpenQuoteContext(host=args.host, port=args.port)
    ctx.set_handler(_Probe())
    ret, msg = ctx.subscribe(codes, [SubType.ORDER_BOOK], subscribe_push=True)
    if ret != RET_OK:
        print(f"Subscribe failed: {msg}", file=sys.stderr)
        print("(a second ORDER_BOOK subscription counts against the OpenD quota -- "
              "stop the collector or probe fewer codes with --codes)", file=sys.stderr)
        ctx.close()
        sys.exit(1)

    print(f"Probing {len(codes)} code(s) for {args.seconds}s — writing nothing. "
          f"Ctrl-C to stop early.\n  {', '.join(codes)}", flush=True)
    t0 = time.time()
    try:
        while time.time() - t0 < args.seconds:
            time.sleep(min(15, args.seconds))
            with lock:
                tot = sum(n_push.values())
            print(f"  [{time.time()-t0:5.0f}s] {tot:,} pushes so far", flush=True)
    except KeyboardInterrupt:
        print("\ninterrupted — reporting what was collected", flush=True)
    finally:
        ctx.close()

    elapsed = time.time() - t0
    with lock:
        active = sorted(n_push, key=lambda c: -n_push[c])

    print(f"\n{'='*78}\nORDER_BOOK push rate — {elapsed:.0f}s sample, "
          f"collector throttle = {_THROTTLE}s/code\n{'='*78}")
    if not active:
        print("No pushes received. Market closed, OpenD down, or the codes are "
              "not subscribable. Nothing can be concluded from this run.")
        return

    print(f"{'code':10s} {'pushes':>7s} {'/min':>7s} | {'min':>7s} {'p10':>7s} "
          f"{'med':>7s} {'p90':>7s} | {'kept':>6s} {'dropped':>8s}")
    for code in active:
        g = sorted(gaps[code])
        rate = n_push[code] / max(elapsed, 1e-9) * 60
        kept, total = n_kept[code], n_push[code]
        drop_pct = (total - kept) / total * 100 if total else 0.0
        print(f"{code:10s} {total:7,} {rate:7.1f} | "
              f"{(g[0] if g else float('nan')):7.3f} {_pct(g,10):7.3f} "
              f"{(st.median(g) if g else float('nan')):7.3f} {_pct(g,90):7.3f} | "
              f"{kept:6,} {drop_pct:7.1f}%")

    print(f"\n{'code':10s} {'both sides':>11s} {'bid-only':>10s} {'ask-only':>10s}")
    for code in active:
        print(f"{code:10s} {n_both[code]:11,} {n_bid_only[code]:10,} {n_ask_only[code]:10,}")

    all_gaps = sorted(g for c in active for g in gaps[c])
    if all_gaps:
        med = st.median(all_gaps)
        print(f"\nVERDICT")
        print(f"  median inter-push gap across all codes: {med:.3f}s")
        if med >= _THROTTLE:
            print(f"  -> Pushes arrive no faster than the {_THROTTLE}s throttle. The "
                  f"throttle costs nothing;\n     a push-driven LiqHm would not refresh "
                  f"faster, and its Col(s) could go to {_THROTTLE:.0f}s.")
        else:
            worst = max(active, key=lambda c: (n_push[c] - n_kept[c]) / max(n_push[c], 1))
            wp = (n_push[worst] - n_kept[worst]) / max(n_push[worst], 1) * 100
            print(f"  -> Pushes are FASTER than the throttle. It is discarding book "
                  f"updates\n     (worst code {worst}: {wp:.0f}% dropped). Lowering "
                  f"_MIN_WRITE_INTERVAL or driving\n     LiqHm from the push would show "
                  f"more. Watch WAL growth if you lower it.")


if __name__ == "__main__":
    main()

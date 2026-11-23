"""Collect the option tape and the greeks needed to sign it. SOXL first.

Usage:
    uv run analysis/option_tick_collector.py [--code US.SOXL] [--pct 30]
                                             [--greeks-secs 5] [--dry-run]

WHY A COLLECTOR AND NOT A QUERY
-------------------------------
`get_rt_ticker` returns at most 1000 prints per contract. On QQQ's busiest
0DTE strike that is 25 minutes of tape; on an active open it would be less.
The option tape cannot be fetched after the fact, so it is either collected
live or it does not exist.

WHAT IT IS FOR
--------------
A signed delta flow -- the option-side equivalent of CVD. For every print:

    side x contracts x delta-at-that-moment x 100

Measured before building this: option contracts do accept TICKER and
ORDER_BOOK subscriptions; `ticker_direction` resolves to BUY or SELL on
86-94% of prints; and the flow is genuinely one-sided (net/total ran from
-6% to +49% across four QQQ contracts, with mirrored prints only 1-6%), so
the signing is not cancelling itself out by construction.

WHAT IT CANNOT TELL YOU
-----------------------
Whether a customer was opening or closing. For DEALER inventory that does
not matter -- a customer selling to close and a customer selling to open
both leave the dealer long the same contract -- but it does mean the figure
is "net delta-weighted aggressor flow", not a position. The error term is
customer-to-customer crosses, where no dealer inventory changes at all and
this still counts the trade. That is a property of the measure, not a bug
to fix later.

SUBSCRIPTION QUOTA
------------------
200 option contracts, account-wide (`option_remain_quota`). A full QQQ 0DTE
chain is 303, so the set has to be chosen. Chosen by MONEYNESS rather than
by count: gamma and volume both concentrate near the money, and a fixed
count would behave differently on a $31 underlying than on a $600 one.
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import signal
import sys
import threading
import time
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from core import single_instance
from core.sdk_log import quiet_sdk_file_log
from feeds.option_tick_store import OptionTickStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_DEFAULT_DB = pathlib.Path(__file__).parent.parent / "db" / "option_ticks.db"
_QUOTA = 200                 # option_remain_quota, account-wide
_RETENTION_DAYS = 60
_CHECKPOINT_EVERY = 10       # watchdog ticks between WAL checkpoints
_MAX_CHAINS = 8              # get_option_chain is limited to 10 per 30 s

_ET = ZoneInfo("America/New_York")
# US equity options trade 09:30-16:00, with late prints to about 16:15. The
# window is widened a little either side so the first sweep predates the
# open. Outside it the sampler sleeps: left running overnight the 300 s
# heartbeat alone writes ~41,000 greeks rows a night, every one of them
# carrying a `spot` that is yesterday's close.
_SESSION_OPEN_ET = time(9, 20)
_SESSION_CLOSE_ET = time(16, 20)


def in_session(now=None) -> bool:
    now = now or datetime.now(_ET)
    if now.weekday() >= 5:
        return False
    return _SESSION_OPEN_ET <= now.time() <= _SESSION_CLOSE_ET

# A greeks row is written only when something actually moved. Writing all 200
# contracts every 5 s would be 936k rows a session, about 75 MB/day and
# 2.2 GB a month, nearly all of it identical to the row before it. Deep
# out-of-the-money contracts barely change at all.
_GREEK_EPS = {"delta": 1e-4, "gamma": 1e-5, "iv": 1e-3}
_GREEK_HEARTBEAT_SECS = 300  # write anyway this often, so "unchanged" and
                             # "not sampled" stay distinguishable


# ── contract selection ─────────────────────────────────────────────────────

def _score_contracts(ctx, rows: list[dict]) -> tuple[list[dict], int]:
    """Attach |delta| x OI to each contract. Returns (scored, unscored_count).

    A contract the broker has no greeks for scores 0 and sinks to the
    bottom rather than being dropped: zero-filled greeks are a reporting
    gap, not evidence the contract is quiet, and a few of them do trade.
    """
    from moomoo import RET_OK

    by_code = {r["code"]: r for r in rows}
    codes = list(by_code)
    unscored = 0
    for i in range(0, len(codes), 200):
        ret, snap = ctx.get_market_snapshot(codes[i:i + 200])
        if ret != RET_OK or snap is None or snap.empty:
            log.warning("scoring snapshot failed: %s", str(snap)[:100])
            continue
        for _, r in snap.iterrows():
            tgt = by_code.get(r["code"])
            if tgt is None:
                continue
            d, oi = _f(r.get("option_delta")), _i(r.get("option_open_interest"))
            if d is None or oi is None or (d == 0 and oi == 0):
                unscored += 1
                tgt["_score"] = 0.0
            else:
                tgt["_score"] = abs(d) * max(oi, 0)
    for r in rows:
        r.setdefault("_score", 0.0)
    return rows, unscored


def pick_contracts(ctx, code: str, pct: float, quota: int = _QUOTA,
                   days: int = 45) -> tuple[list[dict], dict]:
    """Nearest expiry, strikes within +/-pct% of spot, capped at *quota*.

    Returns (contracts, info). Each contract dict carries code/strike/
    opt_type/expiry so the store can record what the codes mean.

    Picking by moneyness, not by a fixed strike count: strike SPACING varies
    by underlying ($1 on SOXL at $31, $5 on QQQ at $608), so a count means a
    different price window on each name, while a percentage does not.
    """
    import pandas as pd
    from moomoo import RET_OK
    from strategy.option.gex import get_near_expiry_dates

    ret, snap = ctx.get_market_snapshot([code])
    if ret != RET_OK or snap is None or snap.empty:
        raise RuntimeError(f"spot fetch failed for {code}: {snap}")
    spot = float(snap["last_price"].iloc[0])

    expiries = get_near_expiry_dates(ctx, code, days)[:_MAX_CHAINS]
    if not expiries:
        raise RuntimeError(f"no expiries within {days} days for {code}")

    # Across expiries, not just the nearest. Measured on 2026-11-20: the
    # same-day expiry carried 53% of the CONTRACTS traded but only 16% of
    # the delta-weighted flow (far out-of-the-money lottery tickets have
    # almost no delta), while the following Friday carried 47% and the
    # monthly another 22%. Collecting one expiry reached 35-43% of the
    # chain's delta-weighted flow while leaving 81% of the subscription
    # quota unused.
    lo, hi = spot * (1 - pct / 100), spot * (1 + pct / 100)
    rows = []
    for expiry in expiries:
        ret, chain = ctx.get_option_chain(code, start=expiry, end=expiry)
        if ret != RET_OK or chain is None or chain.empty:
            log.warning("chain fetch failed for %s: %s", expiry, str(chain)[:100])
            continue
        for _, r in chain.iterrows():
            k = r.get("strike_price")
            if k is None or not (lo <= float(k) <= hi):
                continue
            t = str(r.get("option_type", "")).upper()
            rows.append({
                "code": r["code"], "underlying": code, "expiry": expiry,
                "strike": float(k),
                "opt_type": "CALL" if "CALL" in t else ("PUT" if "PUT" in t else t),
            })
    if not rows:
        raise RuntimeError(f"no strikes inside +/-{pct}% of {spot:.2f}")

    # Rank by |delta| x OI, which is what the indicator actually sums -- not
    # by distance from spot. Distance ranks a near-dated 20-delta strike
    # above a monthly 60-delta one purely because it is closer, and the
    # second contributes three times as much signed delta per contract.
    # |delta|xOI ranked at Spearman 0.785 against the realised
    # delta-weighted flow; open interest alone managed 0.613.
    scored, unscored = _score_contracts(ctx, rows)
    scored.sort(key=lambda r: -r["_score"])
    dropped = max(0, len(scored) - quota)
    rows = scored[:quota]
    for r in rows:
        r.pop("_score", None)
    rows.sort(key=lambda r: (r["expiry"], r["strike"], r["opt_type"]))

    # `last_price` is the REGULAR-HOURS price. Started before the open it is
    # yesterday's close, and the window is centred on a price the market has
    # already left: SOXL closed at 30.88 on 2026-11-20 and was trading 34.75
    # overnight, 12.5% away. The extended fields are read here only to WARN
    # -- using one as the centre would be worse, because during regular
    # hours they hold the PREVIOUS session's values and would drag the
    # window backwards.
    ext = {}
    for field in ("overnight_price", "after_price", "pre_price"):
        v = _f(snap.get(field, pd.Series([None])).iloc[0]) if field in snap else None
        if v:
            ext[field] = v
    outside = {k: v for k, v in ext.items() if not (lo <= v <= hi)}

    per_expiry: dict = {}
    for r in rows:
        per_expiry[r["expiry"]] = per_expiry.get(r["expiry"], 0) + 1
    info = {"spot": spot, "expiries": expiries, "per_expiry": per_expiry,
            "candidates": len(scored), "unscored": unscored,
            "window": (lo, hi), "dropped_for_quota": dropped,
            "extended": ext, "extended_outside": outside}
    return rows, info


# ── tick handler ───────────────────────────────────────────────────────────

def _make_handler(store: OptionTickStore, state: dict):
    from moomoo import TickerHandlerBase

    class _Handler(TickerHandlerBase):
        def on_recv_rsp(self, rsp_pb):
            # The SDK swallows whatever is raised in here. Without this,
            # a single bad push can stop tick ingestion for the rest of the
            # session with nothing in the log to say so -- the collector
            # would look healthy and simply stop writing.
            try:
                return self._handle(rsp_pb)
            except Exception as exc:                            # noqa: BLE001
                state["handler_errors"] = state.get("handler_errors", 0) + 1
                if state["handler_errors"] <= 5:
                    log.exception("tick handler raised: %s", exc)
                return 0, None

        def _handle(self, rsp_pb):
            ret, data = super().on_recv_rsp(rsp_pb)
            if ret != 0 or data is None or data.empty:
                return ret, data

            # recvTime/timestamp never reach `data` -- TickerHandlerBase
            # builds its frame from a col_list naming neither. Same records,
            # same order. Guarded: a malformed push must cost ticks, not the
            # session.
            svr = []
            try:
                for rec in rsp_pb.s2c.tickerList:
                    svr.append((rec.recvTime, rec.timestamp))
            except Exception:                                   # noqa: BLE001
                svr = []
            if len(svr) != len(data):
                svr = []

            rows = []
            for i, (_, row) in enumerate(data.iterrows()):
                direction = str(row.get("ticker_direction", "NEUTRAL")).upper()
                if direction not in ("BUY", "SELL"):
                    direction = "NEUTRAL"
                recv_ts, feed_ts = svr[i] if svr else (None, None)
                rows.append({
                    "code": row["code"], "ts": row["time"],
                    "price": float(row["price"]), "volume": int(row["volume"]),
                    "direction": direction, "ttype": row.get("type"),
                    "seq": row.get("sequence"),  # -1 fallback applied in the store
                    "recv_ts": recv_ts, "feed_ts": feed_ts,
                })

            n = store.insert_ticks(rows)
            state["offered"] = state.get("offered", 0) + len(rows)
            state["dropped"] = state.get("dropped", 0) + len(rows) - n
            if n:
                state["last_tick_time"] = time.time()
                state["ticks"] = state.get("ticks", 0) + n
                if not state.get("first_tick_done"):
                    state["first_tick_done"] = True
                    last = rows[-1]
                    log.info("DATA RECEIVED  %s  ts=%s  px=%.4f  vol=%d  %s",
                             last["code"], last["ts"], last["price"],
                             last["volume"], last["direction"])
            return ret, data

    return _Handler


# ── greeks sampler ─────────────────────────────────────────────────────────

def _changed(prev: dict | None, cur: dict) -> bool:
    if prev is None:
        return True
    if prev.get("oi") != cur.get("oi"):
        return True
    for k, eps in _GREEK_EPS.items():
        a, b = prev.get(k), cur.get(k)
        if a is None or b is None:
            if a is not b:
                return True
            continue
        if abs(a - b) > eps:
            return True
    return False


def sample_greeks(ctx, codes: list[str], underlying: str, store: OptionTickStore,
                  last: dict, state: dict) -> int:
    """One snapshot sweep; write only the contracts that moved.

    `last` is the per-code cache of what was written, mutated in place.
    """
    from moomoo import RET_OK
    import pandas as pd

    # Exchange time, explicitly -- the tick table's `ts` is the exchange's
    # own stamp, and joining a tick to its greeks row only works if both
    # sides mean the same clock. This machine happens to be set to ET, so
    # datetime.now() agreed by luck; a machine in another zone would offset
    # the two tables by hours with nothing in the data to show it.
    now = datetime.now(_ET)
    now_s = now.strftime("%Y-%m-%d %H:%M:%S.%f")

    ret, u = ctx.get_market_snapshot([underlying])
    spot = float(u["last_price"].iloc[0]) if ret == RET_OK and not u.empty else None

    out = []
    for i in range(0, len(codes), 200):
        ret, snap = ctx.get_market_snapshot(codes[i:i + 200])
        if ret != RET_OK or snap is None or snap.empty:
            state["snap_fail"] = state.get("snap_fail", 0) + 1
            log.warning("greeks snapshot failed: %s", str(snap)[:120])
            continue
        for _, r in snap.iterrows():
            d = _f(r.get("option_delta"))
            g = _f(r.get("option_gamma"))
            v = _f(r.get("option_implied_volatility"))
            # All three exactly zero is the feed's way of saying it has no
            # greeks for this contract, not a measurement. Keeping the zero
            # would make the indicator multiply real prints by nothing: a
            # deep in-the-money put with a true delta near -0.97 arrives
            # here as 0.0 and would contribute exactly as much as a contract
            # that never traded.
            missing = (d == 0.0 and g == 0.0 and v == 0.0)
            cur = {
                "delta": None if missing else d,
                "gamma": None if missing else g,
                "iv": None if missing else v,
                "oi": _i(r.get("option_open_interest")),
                "nogreeks": 1 if missing else 0,
            }
            code = r["code"]
            prev = last.get(code)
            stale = (prev is not None
                     and now.timestamp() - prev["_at"] < _GREEK_HEARTBEAT_SECS)
            if stale and not _changed(prev, cur):
                continue
            out.append({**cur, "code": code, "ts": now_s, "spot": spot,
                        "upd": str(r.get("update_time") or "")})
            last[code] = {**cur, "_at": now.timestamp()}

    n = store.insert_greeks(out)
    state["greeks"] = state.get("greeks", 0) + n
    state["greek_sweeps"] = state.get("greek_sweeps", 0) + 1
    return n


def reconcile(ctx, codes: list[str], store: OptionTickStore,
              session: str | None = None) -> dict:  # noqa: C901
    """Compare what we stored against the broker's own daily volume.

    The one check that can catch under-collection while it still matters.
    Nothing else can: the tape is not re-fetchable, so a session missed is
    a session gone, and the failure mode -- a subscription that quietly
    stops delivering, or a throughput ceiling we have not hit yet -- leaves
    the process looking perfectly healthy.

    Run it after the close, when the snapshot's `volume` is the contract's
    settled total for the day. Run before the close it will read low for
    the honest reason that the day is not over, which is why the result
    records the time it was taken.
    """
    from moomoo import RET_OK

    session = session or datetime.now(_ET).strftime("%Y-%m-%d")
    ours = store.our_volume_for(session, codes)

    feed: dict[str, int | None] = {}
    wrong_day = 0
    for i in range(0, len(codes), 200):
        ret, snap = ctx.get_market_snapshot(codes[i:i + 200])
        if ret != RET_OK or snap is None or snap.empty:
            log.warning("reconcile snapshot failed: %s", str(snap)[:100])
            continue
        for _, r in snap.iterrows():
            # `volume` is the LAST SESSION's total, not the session being
            # asked about. Run on a Sunday against today's date it reports
            # Friday's 251,263 contracts against an empty table and calls
            # the result 0% coverage -- an alarming number about nothing.
            # The contract's own update_time says which day the figure is
            # for; when they disagree there is no comparison to make, and
            # "unknown" is the honest answer.
            upd = str(r.get("update_time") or "")[:10]
            if upd and upd != session:
                feed[r["code"]] = None
                wrong_day += 1
            else:
                feed[r["code"]] = _i(r.get("volume"))

    rows = []
    for c in codes:
        v, n = ours.get(c, (0, 0))
        rows.append({"code": c, "feed_vol": feed.get(c),
                     "our_vol": v, "our_ticks": n})
    store.record_coverage(session, rows)
    summary = store.coverage_summary(session)
    summary["session"] = session

    # Worst offenders first: a contract the feed says traded and we have
    # nothing for is the shape a dropped subscription makes.
    missed = sorted(
        (r for r in rows if r["feed_vol"] and r["our_vol"] < r["feed_vol"]),
        key=lambda r: r["feed_vol"] - r["our_vol"], reverse=True)[:5]
    summary["worst"] = [(r["code"], r["our_vol"], r["feed_vol"]) for r in missed]
    summary["caught_nothing"] = sum(
        1 for r in rows if r["feed_vol"] and r["our_vol"] == 0)
    summary["wrong_day"] = wrong_day
    return summary


def _f(v):
    try:
        f = float(v)
        return None if f != f else f
    except (TypeError, ValueError):
        return None


def _i(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


# ── main ───────────────────────────────────────────────────────────────────

def _log_reconcile(ctx, codes: list[str], store: OptionTickStore,
                   session: str | None = None) -> dict | None:
    """Reconcile and say the result out loud. Never fatal."""
    try:
        s = reconcile(ctx, codes, store, session)
    except Exception as exc:                                    # noqa: BLE001
        log.exception("reconcile failed: %s", exc)
        return None
    if s.get("wrong_day"):
        log.warning(
            "coverage %s: NOT COMPARABLE -- the feed's volume for %d of %d "
            "contracts is stamped a different day, so there is nothing to "
            "check against. Run this after the close of the session being "
            "reconciled.", s["session"], s["wrong_day"], s["contracts"])
        return s
    pct = f"{s['ratio'] * 100:.1f}%" if s["ratio"] is not None else "n/a"
    log.info("coverage %s: %s of the feed's volume (%s of %s contracts over "
             "%d prints, %d contracts with no feed figure)",
             s["session"], pct, f"{s['our_vol']:,}", f"{s['feed_vol']:,}",
             s["our_ticks"], s["feed_vol_unknown"])
    if s["caught_nothing"]:
        log.warning("%d contracts traded per the feed and we caught NOTHING "
                    "-- that is the shape of a dropped subscription",
                    s["caught_nothing"])
    for code, ours, theirs in s["worst"]:
        log.warning("  short on %s: %s of %s", code, f"{ours:,}", f"{theirs:,}")
    return s


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Option tape + greeks collector")
    p.add_argument("--code", default="US.SOXL", help="underlying (default US.SOXL)")
    # 80, not 30: at +/-30% the selection reached 35-43% of the chain's
    # delta-weighted flow, and +/-50% already covers 97.8% of a single
    # expiry's volume even on a -31.8% day. The window is a pre-filter now;
    # the quota is what actually bounds the set.
    p.add_argument("--pct", type=float, default=80.0,
                   help="pre-filter: strikes within +/-PCT%% of spot (default 80)")
    p.add_argument("--days", type=int, default=45,
                   help="include expiries up to DAYS out (default 45)")
    p.add_argument("--greeks-secs", type=float, default=15.0,
                   help="seconds between greeks sweeps (default 15)")
    p.add_argument("--db", default=str(_DEFAULT_DB))
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=11111)
    p.add_argument("--always", action="store_true",
                   help="sample greeks outside 09:20-16:20 ET too (testing)")
    p.add_argument("--reconcile-only", action="store_true",
                   help="reconcile a session already collected, then exit")
    p.add_argument("--session", default=None,
                   help="trading day for --reconcile-only (default today ET)")
    p.add_argument("--dry-run", action="store_true",
                   help="pick contracts, report, and exit without subscribing")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    quiet_sdk_file_log()

    from moomoo import OpenQuoteContext, RET_OK, SubType

    ctx = OpenQuoteContext(host=args.host, port=args.port)
    try:
        if args.reconcile_only:
            # Before pick_contracts, not after: selection costs one
            # get_option_chain per expiry against a 10-per-30s budget, and
            # reconciling needs none of it. Reconcile the contracts already
            # RECORDED for that session, not the ones today's selection
            # would choose -- a set that changed between sessions would
            # otherwise hide exactly the contracts that went uncollected.
            store = OptionTickStore(args.db)
            codes = sorted({c["code"] for c in store.contracts_for(args.code)})
            session = args.session or datetime.now(_ET).strftime("%Y-%m-%d")
            if not codes:
                log.error("no contracts recorded for %s in %s", args.code, args.db)
                return 5
            log.info("reconciling %s over %d recorded contracts", session, len(codes))
            s = _log_reconcile(ctx, codes, store, session)
            store.close()
            return 0 if s else 5

        contracts, info = pick_contracts(ctx, args.code, args.pct, days=args.days)
        log.info("%s spot $%.2f  window $%.2f-$%.2f  expiries %s",
                 args.code, info["spot"], *info["window"], info["expiries"])
        log.info("selected %d of %d candidates (%d dropped for the %d quota, "
                 "%d had no greeks to score)",
                 len(contracts), info["candidates"], info["dropped_for_quota"],
                 _QUOTA, info["unscored"])
        log.info("per expiry: %s",
                 "  ".join(f"{k}={v}" for k, v in sorted(info["per_expiry"].items())))
        if info["extended"]:
            log.info("extended-hours prices: %s",
                     "  ".join(f"{k}={v:.2f}" for k, v in info["extended"].items()))
        if info["extended_outside"]:
            # Deliberately does not claim which of these is current -- that
            # is exactly what the API does not say. pre_price is live only
            # pre-market, overnight_price only overnight, and each holds the
            # PREVIOUS session's value the rest of the time. The judgement
            # is the operator's; the warning's job is to put the numbers in
            # front of them before a session's worth of ticks is collected
            # on the wrong strikes.
            log.warning(
                "spot $%.2f is the REGULAR-HOURS price. Outside the window: %s. "
                "Each of those is live only in its own session and stale "
                "otherwise -- if one of them is the current market, restart "
                "after the open or raise --pct.",
                info["spot"],
                ", ".join(f"{k} ${v:.2f}" for k, v in info["extended_outside"].items()))
        if not contracts:
            log.error("nothing selected -- widen --pct")
            return 2
        if args.dry_run:
            strikes = sorted({c["strike"] for c in contracts})
            log.info("strikes %.2f .. %.2f (%d)", strikes[0], strikes[-1], len(strikes))
            log.info("dry run -- not subscribing")
            return 0

        # Keep the handle alive for the whole process: the lock is released
        # by closing it, not by a release() call.
        name = f"option_tick_collector_{args.code.replace('.', '_')}"
        try:
            lock = single_instance.acquire(name, wait_s=8.0)
        except single_instance.AlreadyRunning as exc:
            log.error("%s", exc)
            return single_instance.EXIT_ALREADY_RUNNING

        store = OptionTickStore(args.db)
        store.upsert_contracts(contracts)
        codes = [c["code"] for c in contracts]

        state: dict = {"last_tick_time": None, "first_tick_done": False}
        ctx.set_handler(_make_handler(store, state))
        ret, msg = ctx.subscribe(codes, [SubType.TICKER])
        if ret != RET_OK:
            log.error("subscribe failed: %s", msg)
            return 4
        log.info("subscribed TICKER on %d contracts", len(codes))

        stop = threading.Event()

        def _bye(*_a):
            stop.set()
        signal.signal(signal.SIGINT, _bye)
        signal.signal(signal.SIGTERM, _bye)

        started = datetime.now().isoformat(sep=" ")
        last_greeks: dict = {}
        sweeps = 0
        try:
            idle_logged = False
            while not stop.is_set():
                if not args.always and not in_session():
                    if not idle_logged:
                        log.info("outside 09:20-16:20 ET -- sampler idle "
                                 "(TICKER stays subscribed)")
                        idle_logged = True
                    stop.wait(60)
                    continue
                idle_logged = False
                sample_greeks(ctx, codes, args.code, store, last_greeks, state)
                sweeps += 1
                if sweeps % _CHECKPOINT_EVERY == 0:
                    store.checkpoint()
                    t, g = store.session_counts(started)
                    log.info("session: %d ticks (%d dropped dup), %d greeks rows "
                             "over %d sweeps", t, state.get("dropped", 0), g, sweeps)
                stop.wait(args.greeks_secs)
        finally:
            log.info("stopping -- unsubscribing")
            try:
                ctx.unsubscribe(codes, [SubType.TICKER])
            except Exception:                                   # noqa: BLE001
                pass
            t, g = store.session_counts(started)
            log.info("session totals: %d ticks, %d greeks rows, %d snapshot "
                     "failures, %d handler errors", t, g,
                     state.get("snap_fail", 0), state.get("handler_errors", 0))
            _log_reconcile(ctx, codes, store)
            pt, pg = store.prune_before(_RETENTION_DAYS)
            if pt or pg:
                log.info("pruned %d ticks / %d greeks older than %d days "
                         "(the FILE does not shrink -- SQLite reuses the pages)",
                         pt, pg, _RETENTION_DAYS)
            store.checkpoint()
            store.close()
            try:
                lock.close()
            except Exception:                                   # noqa: BLE001
                pass
        return 0
    finally:
        ctx.close()


if __name__ == "__main__":
    raise SystemExit(main())

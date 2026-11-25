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
import time as time_mod
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from core import single_instance
from core.market_calendar import (
    is_market_holiday, is_trading_day, market_close_et)
from core.sdk_log import quiet_sdk_file_log
from core.sdk_patches import patch_close_reason
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
# Telemetry every ~30 min at the default 15 s sweep: the subscription count,
# the quota fields and the greeks-lag distribution are what a first live
# session exists to measure, and none of them can be read with the market shut.
_TELEMETRY_EVERY = 120  # overridden to 2 only in the smoke test below
_MAX_CHAINS = 8              # expiries to collect -- NOT chain calls

# Span of one get_option_chain call, in days. The API documents a 30-day
# limit on start..end, so a 45-day window is two calls -- against a budget
# of ten per thirty seconds that is account-wide, and therefore shared by
# every underlying being collected at once.
#
# Chunked by SPAN, not by count: eight SOXL weeklies reach 49 days apart
# while eight QQQ dailies span about eleven, so fixed-size slices would ask
# for a range the API refuses for one and waste a call on the other.
_CHAIN_SPAN_DAYS = 28

_ET = ZoneInfo("America/New_York")
# US equity options trade 09:30-16:00, with late prints to about 16:15. The
# window is widened a little either side so the first sweep predates the
# open. Outside it the sampler sleeps: left running overnight the 300 s
# heartbeat alone writes ~41,000 greeks rows a night, every one of them
# carrying a `spot` that is yesterday's close.
_SESSION_OPEN_ET = time(9, 20)
# Minutes past the close to keep sampling. The last greeks of the day are
# worth having, and the close itself is not a clean edge.
#
# Public, because the scheduler derives its own window from it. The two used
# to be independent literals kept in step by a test -- which worked, but a
# test is a guard and this is a definition.
POST_CLOSE_PAD_MIN = 20
_POST_CLOSE_PAD_MIN = POST_CLOSE_PAD_MIN      # retained for older callers


def _session_end_et(day=None) -> time:
    """When greeks sampling stops: the exchange close plus the pad.

    Its own function so the log line and in_session cannot disagree about
    the window. They did: the log said a fixed "09:20-16:20" while
    in_session derived 13:20 on a half session.
    """
    day = day or datetime.now(_ET).date()
    close = datetime.combine(day, market_close_et(day))
    return (close + timedelta(minutes=POST_CLOSE_PAD_MIN)).time()


def in_session(now=None) -> bool:
    """Should greeks be sampled right now.

    Full closures are the reason this is not a weekday test. A holiday used
    to read as an ordinary session: the collector started, raised its "NO
    PRINTS in 3 minutes" alarm against a shut exchange, and wrote a
    session's worth of greeks rows carrying the previous day's spot.

    The end of the window is DERIVED from the exchange close rather than
    fixed at 16:20, so a 13:00 half session stops at 13:20 instead of
    sampling three dead hours. 2026-11-27 is one.
    """
    now = now or datetime.now(_ET)
    if not is_trading_day(now.date()):
        return False
    return _SESSION_OPEN_ET <= now.time() <= _session_end_et(now.date())

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


def _chunk_by_span(expiries: list[str], span_days: int) -> list[list[str]]:
    """Group sorted expiry dates into runs no wider than `span_days`."""
    out: list[list[str]] = []
    for e in expiries:
        # Skipped, not raised. The SDK substitutes a single space for a
        # missing strikeTime (quote_query.py:3576 -- `optional` in the
        # proto, unlike the distance field beside it) and
        # get_near_expiry_dates appends it unchecked on its `dist` branch.
        # The per-expiry loop this replaced asked OpenD for `start=' '`,
        # got nothing back and carried on one expiry short; raising here
        # would turn that into no session at all, since contracts are
        # picked once per run.
        try:
            when = date.fromisoformat(e)
        except ValueError:
            log.warning("skipping malformed expiry %r from the ladder", e)
            continue
        if out and (when - date.fromisoformat(out[-1][0])).days <= span_days:
            out[-1].append(e)
        else:
            out.append([e])
    return out


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
    off_range = 0
    rows = []
    want = set(expiries)
    # Separate from `rows`: a chain can answer perfectly for an expiry that
    # happens to have no strike inside the +/-pct band, and comparing
    # against the filtered rows reported that as a MISSING chain. Measured
    # once on a synthetic ladder: "covered 5 of 6, MISSING ['2026-12-11']"
    # while the chain had answered for it.
    answered: set[str] = set()
    # Ranges, not one call per expiry -- see _CHAIN_SPAN_DAYS for why the
    # per-expiry version was expensive enough to matter.
    for chunk in _chunk_by_span(expiries, _CHAIN_SPAN_DAYS):
        ret, chain = ctx.get_option_chain(code, start=chunk[0], end=chunk[-1])
        if ret != RET_OK or chain is None or chain.empty:
            log.warning("chain fetch failed for %s..%s: %s",
                        chunk[0], chunk[-1], str(chain)[:100])
            continue
        for _, r in chain.iterrows():
            # Expiry FIRST, band second. `answered` has to record that the
            # chain replied for this expiry at all, and reading it after
            # the band filter made it a count of in-band rows instead --
            # which reported a perfectly good chain as MISSING whenever an
            # expiry had no strike inside +/-pct%.
            #
            # From the ROW, not a loop variable: a range answers for every
            # expiry inside it, including ones this run did not ask for, so
            # anything outside the requested set is dropped rather than
            # quietly widening the selection past _MAX_CHAINS.
            expiry = str(r.get("strike_time") or "")[:10]
            if expiry not in want:
                off_range += 1
                continue
            answered.add(expiry)
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
        # The band is the usual reason, but not the only one: if every row
        # carried an expiry outside `want` the band is innocent and the
        # ladder is the suspect, and a message about strike prices would
        # send the next reader to the wrong place.
        raise RuntimeError(
            f"no strikes inside +/-{pct}% of {spot:.2f} "
            f"({off_range} rows dropped for being outside the requested "
            f"expiries)")
    if answered != want:
        # Compared against what the CHAIN returned, not against the rows
        # that survived the band filter. Named rather than counted: a
        # missing expiry is gone for the whole session, because contracts
        # are picked once per run.
        log.warning("chain covered %d of %d expiries; MISSING %s",
                    len(answered), len(want), sorted(want - answered))

    # Rank by |delta| x OI, which is what the indicator actually sums -- not
    # by distance from spot. Distance ranks a near-dated 20-delta strike
    # above a monthly 60-delta one purely because it is closer, and the
    # second contributes three times as much signed delta per contract.
    # |delta|xOI ranked at Spearman 0.785 against the realised
    # delta-weighted flow; open interest alone managed 0.613.
    scored, unscored = _score_contracts(ctx, rows)
    scored.sort(key=lambda r: -r["_score"])
    dropped = max(0, len(scored) - quota)
    # Captured HERE, before the pop two lines down and before `rows` is
    # re-sorted by strike: after that the scores are gone from the dicts and
    # nothing in the process ever sees them again. They come from a chain
    # snapshot that is not re-fetchable, so a selection not recorded now can
    # never be graded -- see the option_candidates table.
    picked = {r["code"] for r in scored[:quota]}
    candidates = [{"code": r["code"], "rank": i + 1, "score": r.get("_score"),
                   "expiry": r.get("expiry"), "strike": r.get("strike"),
                   "opt_type": r.get("opt_type"),
                   "picked": r["code"] in picked}
                  for i, r in enumerate(scored)]
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
            "scored": candidates,
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
                # Four values, not three. moomoo's TickerDirect is
                # BUY / SELL / NEUTRAL / NONE('N/A'), and the last two are
                # different facts: NEUTRAL means the print landed between
                # the bid and the ask, 'N/A' means the feed does not know.
                # Folding 'N/A' into NEUTRAL -- which this did -- throws
                # away the only thing that separates "classified as
                # mid-market" from "unclassified", and the indicator that
                # drops NEUTRAL would then be dropping both without being
                # able to report how much of each.
                direction = str(row.get("ticker_direction", "")).upper()
                if direction not in ("BUY", "SELL", "NEUTRAL"):
                    direction = "N/A"
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
                # time_mod, not time: `from datetime import ... time` for the
                # session gate shadows the module, so time.time() raises
                # AttributeError here, inside the handler, where it is caught.
                #
                # It does NOT lose ticks -- insert_ticks() above has already
                # committed them, verified by replaying a real four-print
                # burst against both versions: 4 rows stored either way. What
                # it loses is every SIGN OF LIFE: state["ticks"] stays 0,
                # last_tick_time stays None, and DATA RECEIVED never prints.
                #
                # That is the worse failure for an operator. The runbook says
                # "no DATA RECEIVED within a minute of the open -> the feed is
                # not connected -> restart", so this bug would have had
                # someone restarting a perfectly healthy collector, and any
                # future zombie check keyed on last_tick_time would restart it
                # forever. Never ran in testing: the market was shut.
                state["last_tick_time"] = time_mod.time()
                state["ticks"] = state.get("ticks", 0) + n
                if not state.get("first_tick_done"):
                    state["first_tick_done"] = True
                    last = rows[-1]
                    log.info("DATA RECEIVED  %s  ts=%s  px=%.4f  vol=%d  %s",
                             last["code"], last["ts"], last["price"],
                             last["volume"], last["direction"])
            return ret, data

    # An INSTANCE, not the class. HandlerContext.set_handler decides with
    # isinstance(handler, TickerHandlerBase), so a class falls through every
    # branch and the call returns RET_ERROR. Returning the class cost the
    # first 22 minutes of 2026-11-23: subscribe() reported RET_OK, OpenD
    # pushed, and the SDK had no handler registered for the proto, so every
    # print was dropped before any of this code ran -- which is why the
    # counters read 0 ticks, 0 dropped AND 0 handler_errors at once.
    return _Handler()


# How long into the session a silent tape is still plausibly just a quiet
# market. SOXL's options did 656 prints a minute at the 2026-11-23 open and
# 69 at the median, so three minutes of nothing while the market is open is
# not quiet, it is broken.
_SILENT_MINUTES = 3


def check_not_blind(ctx, codes: list[str], store, state: dict, args) -> bool:
    """Say so, loudly, if the session is open and nothing has arrived.

    The failure this exists for is silent by construction: subscribe()
    returns RET_OK, greeks keep sampling, telemetry keeps printing, and the
    only symptom is a table that is not growing. Nothing read
    last_tick_time, so on 2026-11-23 the symptom went unread for nine and a
    half minutes -- which is the entire coverage gap for that session.

    Fires once. Re-subscribing in a loop against an OpenD that is refusing
    is how a stale process elsewhere in this repo turned into a restart
    every eight seconds.
    """
    if state.get("blind_checked") or state.get("ticks", 0) > 0:
        return False
    mins = (time_mod.time() - state.get("t0", time_mod.time())) / 60.0
    if mins < _SILENT_MINUTES:
        return False
    state["blind_checked"] = True
    log.error("NO PRINTS in %.1f minutes with the session open. The feed is "
              "not reaching the handler. Re-subscribing and backfilling "
              "once; if this line repeats tomorrow, check set_handler's "
              "return value before anything else.", mins)
    try:
        from moomoo import SubType
        ctx.unsubscribe(codes, [SubType.TICKER])
        ret, msg = ctx.subscribe(codes, [SubType.TICKER])
        log.error("re-subscribe returned %s %s", ret, msg)
    except Exception as exc:                                    # noqa: BLE001
        log.exception("re-subscribe failed: %s", exc)
    if not getattr(args, "no_backfill", False):
        try:
            backfill_after_subscribe(ctx, codes, store, state)
        except Exception as exc:                                # noqa: BLE001
            log.exception("recovery backfill failed: %s", exc)
    return True


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
            # None counts as missing too. Written as `d == 0.0 and ...` alone,
            # a non-numeric field that _f() turned into None takes the False
            # branch and stores delta=None with nogreeks=0 -- the flag lying
            # in precisely the case it exists for.
            missing = all(x is None or x == 0.0 for x in (d, g, v))
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
            last[code] = {**cur, "_at": now.timestamp(),
                          "_upd": str(r.get("update_time") or "")}

    n = store.insert_greeks(out)
    state["greeks"] = state.get("greeks", 0) + n
    state["greek_sweeps"] = state.get("greek_sweeps", 0) + 1
    return n


def _feed_volumes(ctx, codes: list[str],
                  session: str) -> tuple[dict[str, int | None], int, dict]:
    """The broker's own daily volume per contract, for *session*.

    Split out of reconcile so the band pass can reuse it rather than
    restate the wrong-day rule, which is the subtle part: `volume` is the
    LAST SESSION's total, not the session being asked about. Run on a
    Sunday against today's date it reports Friday's 251,263 contracts
    against an empty table and calls the result 0% coverage -- an alarming
    number about nothing. The contract's own update_time says which day the
    figure is for; when they disagree there is no comparison to make, and
    "unknown" is the honest answer.

    Returns (volume by code, how many were stamped another day, the other
    fields the same response already carried). The third value is additive --
    the two existing callers ignore it -- and exists because delta and open
    interest arrive in this very response: reading them here is free, and
    asking for them separately would double the call count of the band pass.
    """
    from moomoo import RET_OK

    feed: dict[str, int | None] = {}
    wrong_day = 0
    extra: dict[str, dict] = {}
    for i in range(0, len(codes), 200):
        ret, snap = ctx.get_market_snapshot(codes[i:i + 200])
        if ret != RET_OK or snap is None or snap.empty:
            log.warning("volume snapshot failed: %s", str(snap)[:100])
            continue
        for _, r in snap.iterrows():
            upd = str(r.get("update_time") or "")[:10]
            if upd and upd != session:
                feed[r["code"]] = None
                wrong_day += 1
            else:
                feed[r["code"]] = _i(r.get("volume"))
            # Kept even for a wrong-day row: the VOLUME is what that rule
            # invalidates, and a delta read is not a claim about any
            # session's trading.
            extra[r["code"]] = {
                "delta_close": _f(r.get("option_delta")),
                "oi_close": _i(r.get("option_open_interest"))}
    return feed, wrong_day, extra


def _band_coverage(ctx, store: OptionTickStore, session: str,
                   underlying: str | None, our_vol: int) -> dict | None:
    """Our volume against the whole candidate BAND, not just what we took.

    The headline ratio divides by the feed's volume for the contracts we
    SUBSCRIBED to, which is our own selection scoring itself. It cannot see
    a bad selection, and worse, it improves when the selection gets worse:
    fewer contracts means the survivors are the active ones, so the ratio
    walks toward 100% while real coverage falls. On 2026-11-24 the
    subscribed set reconciled at 100.0% while holding 62.4% of the
    candidate band's volume -- and that gap is about to be the normal case,
    because an account-wide quota divided between two underlyings is a
    smaller selection on both.

    Needs the recorded candidate list, so it answers None for any session
    collected before option_candidates existed, and for one where the
    selection was never written.

    Returns {contracts, feed_vol, ratio, unknown, missed} where `missed`
    names the biggest-volume contracts the ranking passed over -- the
    diagnostic that found a single unsubscribed SOXL strike carrying 13% of
    a day's band volume.
    """
    if underlying is None:
        return None
    cands = store.candidates_on(session, underlying)
    if not cands:
        return None

    codes = [c["code"] for c in cands]
    feed, _wrong, extra = _feed_volumes(ctx, codes, session)
    known = [(c, feed.get(c["code"])) for c in cands
             if feed.get(c["code"]) is not None]
    band_vol = sum(v for _c, v in known)
    picked_vol = sum(v for c, v in known if c["picked"])
    missed = sorted(((v, c["code"], c["rank"]) for c, v in known
                     if not c["picked"]), reverse=True)[:5]

    # Written here because here is where the data is: the snapshot above has
    # already been paid for, and before this the per-contract figures were
    # summed and discarded. See option_band_volume for what the pairing with
    # option_candidates buys.
    #
    # Its own try: _log_reconcile wraps this whole call and returns None on
    # any exception, so a failed write would take the coverage report down
    # with it -- and the report is the part somebody reads tonight.
    try:
        n = store.record_band_volume(session, [
            {"code": c["code"],
             # Absent from `feed` entirely (the snapshot never answered) and
             # present-but-None (stamped another day) are both unknown; 0 is
             # a contract that did not trade. Do not collapse them.
             "feed_vol": feed.get(c["code"]),
             **(extra.get(c["code"]) or {})}
            for c in cands])
        log.info("recorded %d band volumes for %s", n, underlying)
    except Exception as exc:                                    # noqa: BLE001
        log.warning("could not record band volumes: %s", exc)

    return {
        "contracts": len(cands),
        "picked": sum(1 for c in cands if c["picked"]),
        "feed_vol": band_vol,
        # What the SELECTION reached, before any question of whether the
        # subscriptions delivered. picked_vol/band_vol grades the ranking;
        # our_vol/picked_vol grades the subscriptions; the two were previously
        # tangled in one number that could not separate them.
        "picked_feed_vol": picked_vol,
        "selection_ratio": (picked_vol / band_vol) if band_vol else None,
        "unknown": len(cands) - len(known),
        "ratio": (our_vol / band_vol) if band_vol else None,
        "missed": [(code, vol, rank) for vol, code, rank in missed],
    }


def reconcile(ctx, codes: list[str], store: OptionTickStore,
              session: str | None = None,
              underlying: str | None = None) -> dict:  # noqa: C901
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
    underlying = underlying or _underlying_of(codes)

    feed, wrong_day, _extra = _feed_volumes(ctx, codes, session)

    firsts = store.first_ts_for(session, codes)
    rows = []
    for c in codes:
        v, n = ours.get(c, (0, 0))
        rows.append({"code": c, "feed_vol": feed.get(c),
                     "our_vol": v, "our_ticks": n,
                     "first_ts": firsts.get(c)})
    store.record_coverage(session, rows)
    # Summed from `rows`, NOT read back with coverage_summary(session).
    # `rows` is already scoped to this underlying; the table is not, and
    # reading it back mixed both symbols into one ratio -- QQQ's true 25%
    # printed as 62.5% -- with the value depending on which of the two
    # reconciles happened to write first. The numbers are already here.
    feed_known = [r for r in rows if r["feed_vol"] is not None]
    feed_total = sum(r["feed_vol"] for r in feed_known)
    our_total = sum(r["our_vol"] for r in rows)
    summary = {
        "contracts": len(rows),
        "feed_vol": feed_total,
        "our_vol": our_total,
        "our_ticks": sum(r["our_ticks"] for r in rows),
        "feed_vol_unknown": len(rows) - len(feed_known),
        "ratio": (our_total / feed_total) if feed_total else None,
    }
    summary["session"] = session
    summary["underlying"] = underlying
    # When the tape actually starts. The ratio above divides by the feed's
    # figure for the WHOLE session, so a collector that went live late
    # reports the time before it was subscribed as volume it lost. Carried
    # into the log line rather than left for a reader to work out -- on
    # 2026-11-23 nobody could have, because nothing stored said so.
    summary["live_from"] = min(firsts.values()) if firsts else None

    # Worst offenders first: a contract the feed says traded and we have
    # nothing for is the shape a dropped subscription makes.
    missed = sorted(
        (r for r in rows if r["feed_vol"] and r["our_vol"] < r["feed_vol"]),
        key=lambda r: r["feed_vol"] - r["our_vol"], reverse=True)[:5]
    summary["worst"] = [(r["code"], r["our_vol"], r["feed_vol"]) for r in missed]
    summary["caught_nothing"] = sum(
        1 for r in rows if r["feed_vol"] and r["our_vol"] == 0)
    summary["wrong_day"] = wrong_day
    # Scored against the band we chose FROM, not the set we chose. See
    # _band_coverage for why the ratio above cannot stand on its own.
    summary["band"] = _band_coverage(ctx, store, session, underlying,
                                     our_total)
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

def backfill_after_subscribe(ctx, codes: list[str], store, state: dict,
                             batch: int = 20, pause_s: float = 1.0) -> int:
    """Fetch the prints that happened before we subscribed.

    Subscribing replays nothing -- it pushes each contract's last trade and
    then only new ones -- so without this, starting at 09:39 means the
    session simply has no 09:30-09:39. get_rt_ticker returns up to 1000 of
    OpenD's cached prints for a contract this context is subscribed to,
    which is the same data the push would have delivered.

    Inserted through the same store call as live prints, so the UNIQUE key
    does the de-duplication: anything the push already delivered comes back
    as a duplicate and is counted, not stored twice.

    Written to be harmless if a frequency limit exists and is hit: batches
    with a pause, and the FIRST refusal stops the whole thing. Retrying
    into a limiter is how one stale collector became a process every eight
    seconds elsewhere in this repo.
    """
    from moomoo import RET_OK

    got = dropped = refused = 0
    for i in range(0, len(codes), batch):
        for code in codes[i:i + batch]:
            try:
                ret, df = ctx.get_rt_ticker(code, 1000)
            except Exception as exc:                        # noqa: BLE001
                log.warning("backfill raised on %s: %s", code, exc)
                refused += 1
                break
            if ret != RET_OK or df is None or df.empty:
                if ret != RET_OK:
                    log.warning("backfill refused for %s: %s", code, str(df)[:80])
                    refused += 1
                    break
                continue
            rows = []
            for _, r in df.iterrows():
                direction = str(r.get("ticker_direction", "")).upper()
                if direction not in ("BUY", "SELL", "NEUTRAL"):
                    direction = "N/A"
                rows.append({
                    "code": r["code"], "ts": r["time"],
                    "price": float(r["price"]), "volume": int(r["volume"]),
                    "direction": direction, "ttype": r.get("type"),
                    "seq": r.get("sequence"),
                    "recv_ts": None, "feed_ts": None,
                })
            n = store.insert_ticks(rows)
            got += n
            dropped += len(rows) - n
        if refused:
            break
        if i + batch < len(codes):
            wait_interruptible(state["stop"], pause_s)
    log.info("backfill: %d new prints, %d already had, %d contracts refused",
             got, dropped, refused)
    state["backfilled"] = got
    return got


def wait_interruptible(stop: threading.Event, secs: float,
                       slice_s: float = 1.0) -> None:
    """Sleep, but stay able to notice a signal while doing it.

    On Windows a signal handler only runs between bytecodes in the main
    thread, and Event.wait() blocks below that level -- the handler cannot
    run until the wait returns on its own. A single wait(60) therefore
    ignores CTRL_BREAK for up to a minute.

    That is not theoretical. The supervisor allows 10 s before falling back
    to terminate(), so a 15 s sampling wait -- the default -- already
    outlasts it, and the whole cleanup path (unsubscribe, reconcile, prune,
    checkpoint) was being skipped on every scheduled stop. It passed an
    earlier check only because that one happened to run with a 5 s interval.

    Waiting in short slices returns to Python between each, which is where
    a pending signal gets handled.
    """
    deadline = time_mod.monotonic() + secs
    while not stop.is_set():
        left = deadline - time_mod.monotonic()
        if left <= 0:
            return
        stop.wait(min(slice_s, left))


def verify_subscription(ctx, codes: list[str]) -> int | None:
    """How many of *codes* OpenD lists as subscribed, read right now.

    subscribe() returning RET_OK does not mean the subscriptions landed --
    the first 22 minutes of 2026-11-23 are the recorded case, and the
    re-subscribe path below exists for it. log_telemetry already carries
    this same comparison, but it first runs when `sweeps` reaches
    _TELEMETRY_EVERY, about 30 minutes in at the default sweep, and a
    shortfall at the open is worth knowing at the open.

    It matters more now than it did with one underlying: with the
    account-wide quota divided between two collectors, a short subscription
    is the expected shape of asking for more contracts than are left, and
    nothing else reports it. The quota fields cannot: option_used_quota is
    not a contract count (see log_telemetry), so the subscription LIST is
    the only honest read of what was actually taken.

    Returns the count, or None when the read itself failed -- which is not
    the same as zero and must not be logged as one.
    """
    from moomoo import RET_OK
    try:
        ret, sub = ctx.query_subscription()
        if ret != RET_OK or not isinstance(sub, dict):
            log.warning("subscription read failed, cannot verify: %s",
                        str(sub)[:100])
            return None
        lst = sub.get("sub_list") or {}
        live = len({c for v in lst.values() for c in (v or [])
                    if isinstance(c, str)} & set(codes))
    except Exception as exc:                                    # noqa: BLE001
        log.warning("subscription read failed, cannot verify: %s", exc)
        return None

    if live < len(codes):
        log.warning(
            "only %d of %d contracts are in the subscription list right "
            "after subscribing -- %d short. With the account-wide quota "
            "divided between underlyings, this is what asking for more "
            "than is left looks like; lower this collector's --quota or "
            "the other one's.", live, len(codes), len(codes) - live)
    else:
        log.info("subscription verified: %d of %d contracts listed",
                 live, len(codes))
    return live


def log_telemetry(ctx, codes: list[str], last_greeks: dict, state: dict) -> None:
    """The four things a first live session has to answer. Never fatal.

    Each of these is a number that cannot be obtained with the market shut,
    and each one is a parameter something downstream needs:

      subscription count   whether subscribing one symbol's chain quietly
                           evicts part of it. A Sunday probe saw 210 SOXL
                           contracts fall to 60 after 30 QQQ ones were
                           added, but unsubscribe_all did not take effect
                           in the same run and the figure decayed on its
                           own, so it could equally have been a stale read
                           at the close. Only a live session separates them.
      quota fields         option_used_quota is NOT a contract count --
                           50 contracts read 1, 400 read 2, and adding 120
                           already-subscribed ones took it from 4 DOWN to 3.
                           Logged to learn the unit, never used as a gate.
      greeks staleness     now - update_time, which sets how much a tick's
                           delta can be wrong. The after-hours snapshot only
                           gives the ordering, not the magnitudes.
      flow rate            ticks per minute, which sets the zombie-detection
                           threshold a supervisor would need.
    """
    from moomoo import RET_OK
    try:
        ret, sub = ctx.query_subscription()
        if ret == RET_OK:
            lst = (sub.get("sub_list") or {}) if isinstance(sub, dict) else {}
            tick_subs = {c for v in lst.values() for c in (v or [])
                         if isinstance(c, str)}
            # Unconditional. `if tick_subs else None` printed "? of 200"
            # for an EMPTY subscription list -- the loudest possible signal,
            # every contract gone, which is the 210->60 shape this line
            # exists to catch -- and suppressed the shortfall warning below
            # with it. A read that failed raises and is caught; an empty
            # list is an answer, not a missing one.
            live = len(tick_subs & set(codes))
            # total_used/remain are the ACCOUNT-WIDE subscription quota,
            # a different budget from option_used/option_remain. own_used is
            # this connection's share, so the difference is everyone else's
            # -- which on a two-underlying day separates "the other
            # collector took the quota" from "this is a stale read".
            #
            # NOT conn_sub_list. query_subscription builds a fresh dict of
            # eight keys and only ITERATES conn_sub_list to derive own_used
            # (open_quote_context.py:1199-1228), so reading it back gave a
            # constant zero dressed up as a measurement.
            total = sub.get("total_used")
            mine = sub.get("own_used")
            theirs = (total - mine) if (total is not None
                                        and mine is not None) else None
            log.info("telemetry: subscribed %s of %d requested  "
                     "option_used=%s option_remain=%s  "
                     "total_used=%s remain=%s  ours=%s others=%s",
                     live if live is not None else "?", len(codes),
                     sub.get("option_used_quota"),
                     sub.get("option_remain_quota"),
                     total, sub.get("remain"), mine,
                     theirs if theirs is not None else "?")
            # NOT a warning on theirs > 0. The tick and order-book
            # collectors connect to this same OpenD (127.0.0.1:11111) and
            # their subscriptions count toward total_used but not toward
            # this connection's own_used, so `theirs` is non-zero in every
            # normal session -- and the two option collectors each count the
            # other's 200 contracts. Warning on it would fire ~28 times a
            # day saying nothing, in the one subsystem whose log the first
            # live session asks the operator to read.
            #
            # The unit of total_used is also still unknown: whether a
            # TICKER subscription on an option contract costs one slot or
            # some other number is not in the proto, so `others` is a
            # number to watch, not one to act on. It stays on the INFO line.
            if theirs is not None and theirs < 0:
                log.warning("telemetry: own_used (%s) exceeds total_used "
                            "(%s), which cannot happen in one response -- "
                            "treat these quota figures as unreliable",
                            mine, total)
            if live < len(codes):
                log.warning("telemetry: %d of %d requested contracts are NOT "
                            "in the subscription list", len(codes) - live,
                            len(codes))
    except Exception as exc:                                    # noqa: BLE001
        log.warning("telemetry: subscription read failed: %s", exc)

    now = datetime.now(_ET)
    lags = []
    for v in last_greeks.values():
        upd = v.get("_upd")
        if not upd:
            continue
        try:
            t = datetime.strptime(str(upd)[:19], "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
        lags.append((now.replace(tzinfo=None) - t).total_seconds() / 60.0)
    if lags:
        lags.sort()
        def _q(p):
            return lags[min(len(lags) - 1, int(p * len(lags)))]
        # BIASED HIGH, by up to one heartbeat (300 s). This reads the
        # write-cache, which only updates when a row is written, so a
        # contract whose greeks are genuinely one minute old can read six.
        # The bias falls on the ACTIVE contracts, which is the end that
        # matters. Live, it is a direction indicator only; the unbiased
        # figure is option_greeks.ts - upd over the rows that actually
        # changed, computed after the close -- see doc/option_first_session.md.
        log.info("telemetry: greeks lag min (biased high, <=1 heartbeat)  "
                 "p50=%.1f p90=%.1f p99=%.1f max=%.1f (n=%d)",
                 _q(0.5), _q(0.9), _q(0.99), lags[-1], len(lags))

    mins = max((time_mod.time() - state.get("t0", time_mod.time())) / 60.0, 1e-9)
    log.info("telemetry: %d ticks over %.1f min = %.1f/min  "
             "(dup %d, snap_fail %d, handler_err %d)",
             state.get("ticks", 0), mins, state.get("ticks", 0) / mins,
             state.get("dropped", 0), state.get("snap_fail", 0),
             state.get("handler_errors", 0))


def _underlying_of(codes: list[str]) -> str | None:
    """The symbol a list of option codes belongs to, or None if mixed.

    Only for LABELLING a log line. An option code is
    <market>.<symbol><yymmdd><C|P><strike>, so the symbol is the leading
    letters after the market prefix. Scoping is done by the caller, which
    knows the symbol for certain; this exists so a reconcile started by
    hand still names what it reconciled rather than printing a number that
    could belong to either collector.
    """
    seen = set()
    for c in codes:
        market, dot, rest = c.partition(".")
        if not dot:
            return None
        sym = ""
        for ch in rest:
            if not ch.isalpha():
                break
            sym += ch
        if not sym:
            return None
        seen.add(f"{market}.{sym}")
        if len(seen) > 1:
            return None
    return next(iter(seen), None)


def calibration_advice(store: OptionTickStore, session: str,
                       underlying: str, summary: dict) -> list[str]:
    """What the session says about this underlying's parameters.

    PROPOSES ONLY. It returns lines to log and writes nothing, because a
    single session is not enough to tune anything: one contract carried 19.5%
    of a day's delta-weighted flow on 2026-11-24, so a parameter fitted to
    one day is fitted to that day's accident. Ranking changes, per-expiry slot
    caps and the quota split between underlyings all need the 8-10 sessions
    the band dataset is being collected for.

    What CAN be read from one session is the difference between a parameter
    that is wrong and one that is merely unproven, and only where the reading
    is a FACT rather than an estimate:

      * asked vs listed is a fact. If OpenD listed fewer contracts than were
        requested, the account-wide quota was over-committed and this code's
        share is too high -- no amount of further sessions makes that more
        true.
      * candidates vs slots is a fact about whether the pre-filter is even
        binding. With 2,288 candidates for 60 slots the band cannot be too
        narrow in any way that matters, whatever the volume did. Measured on
        2026-11-24: the strikes between QQQ's +/-15% band and +/-40% held
        0.74% of the day's raw volume and 0.01% of its delta-weighted flow,
        so a band this loose is nowhere near the constraint.
      * a shortfall with a full subscription list is NOT a quota problem.
        Saying so is the point: the two look identical in the coverage
        numbers and call for opposite responses.
    """
    out: list[str] = []
    cands = store.candidates_on(session, underlying)
    chk = store.subscription_check_on(session, underlying)
    band = summary.get("band") or {}

    picked = sum(1 for c in cands if c["picked"])
    if cands and not picked:
        # Unreachable from today's caller -- main returns before recording
        # when pick_contracts selects nothing -- and pinned anyway, because
        # the arithmetic below had `or 1` and that fabricated the HEALTHIEST
        # possible reading out of the loudest possible failure: 2,288
        # candidates divided by a made-up single slot printed as "2288x, not
        # binding". Same mistake as reporting a failed read as zero, which
        # the quota branch below is careful about.
        out.append(f"  band      {len(cands):,} candidates and NOTHING was "
                   f"selected -- the ratio below is not computable and the "
                   f"session collected no tape")
    elif cands:
        ratio = len(cands) / picked
        out.append(f"  band      {len(cands):,} candidates for {picked} slots "
                   f"({ratio:.0f}x)")
        if ratio < 2:
            out.append(f"  -> the pre-filter is binding at {ratio:.1f}x; "
                       f"widen this code's pct so the quota, not the band, "
                       f"chooses")
    else:
        out.append("  band      no candidate list recorded for this session")

    if chk is None:
        out.append("  quota     no subscription check recorded "
                   "(session predates it, or selection never finished)")
    elif chk["listed"] is None:
        out.append(f"  quota     asked {chk['asked']}, the subscription read "
                   f"failed -- unknown, not zero")
    elif chk["listed"] < chk["asked"]:
        out.append(f"  quota     asked {chk['asked']}, listed "
                   f"{chk['listed']} -- {chk['asked'] - chk['listed']} short")
        out.append(f"  -> the account-wide quota was over-committed; lower "
                   f"this code's share to {chk['listed']} or lower another "
                   f"code's")
    else:
        out.append(f"  quota     asked {chk['asked']}, listed "
                   f"{chk['listed']} -- the split held")
        if summary.get("caught_nothing"):
            out.append(f"  -> {summary['caught_nothing']} contracts traded "
                       f"and we caught nothing, with the full list "
                       f"subscribed: NOT a quota problem, so lowering it "
                       f"would change nothing. A subscription stopped "
                       f"delivering")

    if band.get("ratio") is not None:
        sel = band.get("selection_ratio")
        out.append(
            f"  coverage  band {band['ratio'] * 100:.1f}%"
            + (f", of which selection {sel * 100:.1f}%" if sel is not None
               else ""))

    if not any(line.lstrip().startswith("->") for line in out):
        out.append("  -> no change proposed")
    return out


def _log_reconcile(ctx, codes: list[str], store: OptionTickStore,
                   session: str | None = None,
                   underlying: str | None = None) -> dict | None:
    """Reconcile and say the result out loud. Never fatal."""
    try:
        s = reconcile(ctx, codes, store, session, underlying)
    except Exception as exc:                                    # noqa: BLE001
        log.exception("reconcile failed: %s", exc)
        return None
    if s.get("wrong_day"):
        log.warning(
            "coverage %s: NOT COMPARABLE -- the feed's volume for %d of %d "
            "contracts is stamped a different day, so there is nothing to "
            "check against. Run this after the close of the session being "
            "reconciled.", f"{s.get('underlying') or '?'} {s['session']}",
            s["wrong_day"], s["contracts"])
        return s
    pct = f"{s['ratio'] * 100:.1f}%" if s["ratio"] is not None else "n/a"
    # "WHOLE-SESSION", and the time we went live, because the ratio divides
    # by the feed's figure for the entire day. If the collector started
    # late, this number counts the time before it was subscribed as data it
    # lost -- which is exactly how 2026-11-23 read 78.6% (279,295 / 355,460,
    # read back from option_coverage) while every contract whose whole tape
    # fell inside the live window matched the feed exactly.
    live = s.get("live_from") or "unknown"
    # The symbol is IN the line. Two collectors reconcile in the same
    # minute and both lines used to read "coverage 2026-11-24: ...", which
    # is how a figure for one symbol gets quoted as the day's coverage.
    log.info("coverage %s %s: %s of the feed's WHOLE-SESSION volume (%s of "
             "%s contracts over %d prints, %d contracts with no feed "
             "figure) -- our tape starts %s; anything the feed counted "
             "before that is in this ratio too",
             s.get("underlying") or "?", s["session"], pct,
             f"{s['our_vol']:,}", f"{s['feed_vol']:,}",
             s["our_ticks"], s["feed_vol_unknown"], live)
    band = s.get("band")
    if band is None:
        log.info("band coverage: not available -- no scored candidate list "
                 "was recorded for this session, so the only denominator "
                 "is our own selection")
    else:
        bpct = (f"{band['ratio'] * 100:.1f}%"
                if band["ratio"] is not None else "n/a")
        # The number that can actually fall when the selection gets worse.
        # Read it, not the one above: the one above rises.
        log.info("band coverage %s %s: %s of the CANDIDATE BAND's volume "
                 "(%s of %s contracts over the %d we subscribed to, %d with "
                 "no feed figure)",
                 s.get("underlying") or "?", s["session"], bpct,
                 f"{s['our_vol']:,}", f"{band['feed_vol']:,}",
                 band["picked"], band["unknown"])
        # The decomposition. One number cannot say whether a low band
        # coverage is a bad selection or a dropped subscription, and the two
        # call for opposite responses.
        if band["selection_ratio"] is not None:
            got = (s["our_vol"] / band["picked_feed_vol"]
                   if band["picked_feed_vol"] else None)
            log.info("  of which: the selection reached %.1f%% of the band, "
                     "and the subscriptions delivered %s of that",
                     band["selection_ratio"] * 100,
                     f"{got * 100:.1f}%" if got is not None else "n/a")
        for code, vol, rank in band["missed"]:
            log.warning("  passed over %s: %s contracts, ranked %d of %d",
                        code, f"{vol:,}", rank, band["contracts"])
    if s["caught_nothing"]:
        log.warning("%d contracts traded per the feed and we caught NOTHING "
                    "-- that is the shape of a dropped subscription",
                    s["caught_nothing"])
    for code, ours, theirs in s["worst"]:
        log.warning("  short on %s: %s of %s", code, f"{ours:,}", f"{theirs:,}")
    # Last, and in its own guard: advice is the least important thing in this
    # function and must never be the reason the coverage report is lost.
    if s.get("underlying"):
        try:
            # Computed before the header is printed: logging the header first
            # leaves a heading with nothing under it when this raises.
            advice = calibration_advice(store, s["session"],
                                        s["underlying"], s)
        except Exception as exc:                                # noqa: BLE001
            log.warning("calibration advice failed: %s", exc)
        else:
            log.info("calibration %s %s:", s["underlying"], s["session"])
            for line in advice:
                log.info("%s", line)
    return s


def lock_name(code: str) -> str:
    """The single-instance lock name for one underlying's collector.

    Exported because the scheduler has to ask single_instance.holder() the
    SAME name to report the PID holding it, and it was asking a different
    one -- "option_collector_US.SOXL" against this process's
    "option_tick_collector_US_SOXL". holder() therefore always returned
    None, and the refusal message named a lock file that does not exist
    while telling the operator to stop a process it could not identify.
    One function, two callers, the way POST_CLOSE_PAD_MIN is shared.
    """
    return f"option_tick_collector_{code.replace('.', '_')}"


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Option tape + greeks collector")
    p.add_argument("--code", default="US.SOXL", help="underlying (default US.SOXL)")
    # 80, not 30: at +/-30% the selection reached 35-43% of the chain's
    # delta-weighted flow, and +/-50% already covers 97.8% of a single
    # expiry's volume even on a -31.8% day. The window is a pre-filter now;
    # the quota is what actually bounds the set.
    p.add_argument("--pct", type=float, default=80.0,
                   help="pre-filter: strikes within +/-PCT%% of spot (default 80)")
    # The quota is ACCOUNT-WIDE, so collecting two underlyings at once means
    # dividing it -- the scheduler passes each collector its share. Left at
    # the default, one collector asks for the whole account budget, which is
    # correct for one underlying and wrong for two.
    p.add_argument("--quota", type=int, default=_QUOTA,
                   help=f"subscribe at most QUOTA contracts (default "
                        f"{_QUOTA}, the WHOLE account-wide limit -- divide "
                        f"it when collecting more than one underlying)")
    p.add_argument("--no-backfill", action="store_true",
                   help="skip the get_rt_ticker catch-up after subscribing")
    p.add_argument("--days", type=int, default=45,
                   help="include expiries up to DAYS out (default 45)")
    p.add_argument("--greeks-secs", type=float, default=15.0,
                   help="seconds between greeks sweeps (default 15)")
    p.add_argument("--db", default=str(_DEFAULT_DB))
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=11111)
    p.add_argument("--always", action="store_true",
                   help="ignore the session window and the holiday calendar "
                        "(testing)")
    p.add_argument("--reconcile-only", action="store_true",
                   help="reconcile a session already collected, then exit")
    p.add_argument("--session", default=None,
                   help="trading day for --reconcile-only (default today ET)")
    p.add_argument("--dry-run", action="store_true",
                   help="pick contracts, report, and exit without subscribing")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    # SQLite's = is case sensitive on TEXT and pick_contracts stores
    # args.code verbatim as `underlying`, so `--code us.soxl` writes rows
    # that subscribed_on("US.SOXL") cannot see. The scheduler's own
    # normaliser upper-cases, so this only ever bit a hand-typed run --
    # which is exactly what the first live session is.
    args.code = args.code.strip().upper()
    quiet_sdk_file_log()
    patch_close_reason()

    # Before the context, the lock and pick_contracts -- not inside the
    # sampling loop. in_session guards only the loop, so a hand-started
    # collector (scripts/launch_option_collector.bat is the documented
    # first-session move) still took the single-instance lock, spent
    # get_option_chain calls against an account-wide 10-per-30s budget and
    # subscribed 200 contracts before idling all day on a shut exchange.
    #
    # --reconcile-only is exempt: reconciling LAST Wednesday's tape on a
    # holiday is a reasonable thing to ask for, and it touches no quota.
    if not args.reconcile_only and not args.always:
        today = datetime.now(_ET).date()
        if not is_trading_day(today):
            log.error("%s is not a trading session (%s) -- refusing to start. "
                      "Pass --always to collect anyway.", today,
                      "exchange holiday" if is_market_holiday(today)
                      else "weekend")
            return 7

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
            session = args.session or datetime.now(_ET).strftime("%Y-%m-%d")
            # What was subscribed THAT DAY, not every contract ever recorded.
            # option_contracts is cumulative and never pruned, so within a
            # week it holds several days' rolling selections. Scoring today's
            # tape against contracts today never subscribed to puts every one
            # of them in `caught_nothing` -- the single detector for a dropped
            # subscription, drowned in false positives and then ignored.
            # Filtered by underlying. The error message below has always
            # named args.code while the query ignored it, and with a second
            # symbol in the same database that mismatch stops being cosmetic:
            # every contract of the OTHER symbol scores as uncollected under
            # this one's name.
            codes = store.subscribed_on(session, args.code)
            # Unconditional, not only on the empty branch. A PARTIAL orphan
            # -- 200 subscribed, 50 with no contract row -- returns 150
            # codes and looks exactly like a quiet symbol, which is the
            # failure the filter's INNER JOIN can create and the one thing
            # the first version of this fix did not cover.
            if (orphans := store.orphans_on(session)):
                # NOT filtered by underlying, and it cannot be: an orphan is
                # a subscription with no contract row, and the contract row
                # is the only place the underlying is written. So the count
                # is session-wide and may belong to the OTHER symbol -- said
                # here, because "short by at most that many" is false for an
                # orphan of a symbol this reconcile never looked at.
                log.warning(
                    "%d subscription rows on %s have no option_contracts "
                    "row and are in NO reconcile. Which symbol they belong "
                    "to is unknowable (the contract row is where the "
                    "underlying is recorded), so this count is session-wide "
                    "-- %s's own total is short by at most that many",
                    orphans, session, args.code)
            if not codes:
                present = store.underlyings_on(session)
                log.error("nothing recorded as subscribed for %s on %s in %s "
                          "-- was the collector running that day? "
                          "(that session does hold: %s)",
                          args.code, session, args.db,
                          ", ".join(f"{k}={v}"
                                    for k, v in sorted(present.items()))
                          or "nothing at all")
                store.close()
                return 5
            log.info("reconciling %s for %s over %d recorded contracts",
                     session, args.code, len(codes))
            s = _log_reconcile(ctx, codes, store, session, args.code)
            store.close()
            return 0 if s else 5

        # The lock goes BEFORE contract selection, not after. Selection
        # spends one get_option_chain per expiry against a 10-per-30s
        # budget, so two copies racing both burned six to eight of them
        # before one was refused -- and the winner could then fail on the
        # limit the loser had helped exhaust.
        #
        # --dry-run is exempt: it writes nothing and subscribes to nothing,
        # and the point of it is to inspect the selection WHILE a collector
        # is running. Taking the lock here would have made the one case it
        # is for the one case it refuses.
        lock = None
        name = lock_name(args.code)
        try:
            if not args.dry_run:
                lock = single_instance.acquire(name, wait_s=8.0)
        except single_instance.AlreadyRunning as exc:
            log.error("%s", exc)
            return single_instance.EXIT_ALREADY_RUNNING

        contracts, info = pick_contracts(ctx, args.code, args.pct,
                                         quota=args.quota, days=args.days)
        log.info("%s spot $%.2f  window $%.2f-$%.2f  expiries %s",
                 args.code, info["spot"], *info["window"], info["expiries"])
        log.info("selected %d of %d candidates (%d dropped for the %d quota, "
                 "%d had no greeks to score)",
                 len(contracts), info["candidates"], info["dropped_for_quota"],
                 args.quota, info["unscored"])
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

        store = OptionTickStore(args.db)
        store.upsert_contracts(contracts)
        session_day = datetime.now(_ET).strftime("%Y-%m-%d")
        store.record_subscriptions(session_day,
                                   [c["code"] for c in contracts])
        # The ones NOT taken, with the score that passed them over. Without
        # this the reconcile can only ever score us against our own
        # selection, where a narrower selection reads as better coverage.
        n_cand = store.record_candidates(session_day, args.code,
                                         info["scored"])
        log.info("recorded %d scored candidates (%d picked)",
                 n_cand, sum(1 for c in info["scored"] if c["picked"]))
        codes = [c["code"] for c in contracts]

        state: dict = {"last_tick_time": None, "first_tick_done": False,
                      "t0": time_mod.time()}
        # Checked, because the failure is otherwise completely silent:
        # set_handler returns RET_ERROR for anything isinstance() rejects and
        # the subscribe below still succeeds, so the collector runs all day,
        # samples greeks, logs healthy telemetry, and stores not one tick.
        if ctx.set_handler(_make_handler(store, state)) != RET_OK:
            log.error("set_handler refused the tick handler -- no print would "
                      "ever reach the store; refusing to run")
            return 6
        ret, msg = ctx.subscribe(codes, [SubType.TICKER])
        if ret != RET_OK:
            log.error("subscribe failed: %s", msg)
            return 4
        log.info("subscribed TICKER on %d contracts", len(codes))
        # Read back immediately rather than waiting for the first telemetry
        # sweep: a quota shortfall shows up here, and it is the one failure
        # a divided quota introduces. Recorded, not just logged -- the next
        # morning's parameter advice reads it, and nothing else in the
        # database can answer it (see option_subscription_check).
        listed = verify_subscription(ctx, codes)
        try:
            store.record_subscription_check(session_day, args.code,
                                            len(codes), listed)
        except Exception as exc:                                # noqa: BLE001
            log.warning("could not record the subscription check: %s", exc)

        stop = threading.Event()
        state["stop"] = stop
        # Immediately, not on a timer: every second between the session
        # opening and this call is tape that only get_rt_ticker can still
        # reach. Harmless before the open, when there is nothing to fetch.
        if not args.no_backfill:
            backfill_after_subscribe(ctx, codes, store, state)

        def _bye(*_a):
            stop.set()
        # SIGBREAK as well as SIGTERM, because on Windows SIGTERM is not
        # what a parent actually sends: Popen.terminate() calls
        # TerminateProcess, which runs no handler and no `finally` -- the
        # unsubscribe, the reconcile, the prune and the WAL checkpoint below
        # would all be dead code under a supervisor that stops it that way.
        # CTRL_BREAK_EVENT, to a process started in its own process group,
        # does arrive here. (The repo's two existing collectors register
        # only SIGTERM and have the same dead path.)
        signal.signal(signal.SIGINT, _bye)
        signal.signal(signal.SIGTERM, _bye)
        if hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, _bye)

        started = datetime.now(_ET).strftime("%Y-%m-%d %H:%M:%S.%f")
        last_greeks: dict = {}
        sweeps = 0
        try:
            idle_logged = False
            while not stop.is_set():
                if not args.always and not in_session():
                    if not idle_logged:
                        # The window is DERIVED, so state the one in force
                        # today. A fixed "09:20-16:20" reported a window
                        # that is wrong on every half session -- and a
                        # second copy of this window is what let the
                        # holiday gap live as long as it did.
                        log.info("outside %s-%s ET -- sampler idle "
                                 "(TICKER stays subscribed)",
                                 _SESSION_OPEN_ET.strftime("%H:%M"),
                                 _session_end_et().strftime("%H:%M"))
                        idle_logged = True
                    wait_interruptible(stop, 60)
                    continue
                idle_logged = False
                # Before the sampling, so a blind session is caught on the
                # first sweep that can see it rather than one interval later.
                check_not_blind(ctx, codes, store, state, args)
                sample_greeks(ctx, codes, args.code, store, last_greeks, state)
                sweeps += 1
                if sweeps % _CHECKPOINT_EVERY == 0:
                    store.checkpoint()
                    t, g = store.session_counts(started)
                    log.info("session: %d ticks (%d dropped dup), %d greeks rows "
                             "over %d sweeps", t, state.get("dropped", 0), g, sweeps)
                if sweeps % _TELEMETRY_EVERY == 0:
                    log_telemetry(ctx, codes, last_greeks, state)
                wait_interruptible(stop, args.greeks_secs)
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
            _log_reconcile(ctx, codes, store, None, args.code)
            pt, pg = store.prune_before(_RETENTION_DAYS)
            if pt or pg:
                log.info("pruned %d ticks / %d greeks older than %d days "
                         "(the FILE does not shrink -- SQLite reuses the pages)",
                         pt, pg, _RETENTION_DAYS)
            store.checkpoint()
            store.close()
            try:
                if lock is not None:
                    lock.close()
            except Exception:                                   # noqa: BLE001
                pass
        return 0
    finally:
        ctx.close()


if __name__ == "__main__":
    raise SystemExit(main())

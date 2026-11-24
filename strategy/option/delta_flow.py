"""Signed option delta flow from the option tape.

The quantity: for every option print whose side is determinable, the delta
exposure that changed hands, in SHARES of the underlying.

    aggressor_delta_shares = side_sign * delta * volume * 100

WHOSE DELTA THIS IS. The aggressor's -- whoever crossed the spread. Not
"the customer's", not "the dealer's" and not "the option seller's". Three
framings get conflated here and only one of them is what this computes:

    the AGGRESSOR's delta     whoever lifted the offer or hit the bid   <-- this
    the DEALER's delta        the market maker's inventory             = negation*
    the OPTION SELLER's delta the short-option side                    neither

The third is the trap. "Option seller" is not "passive side" and not
"dealer": a customer who hits the bid to sell a call is both the option
seller AND the aggressor, and this series records that print as bearish --
correct for the customer, while the dealer picked up the long call. So
"seller's view" coincides with "aggressor's view" on one print and opposes
it on the next. It is not a stable framing and is not offered.

* The negation holds only under "the dealer is the sole counterparty", which
  is an assumption, not a fact: market makers also take liquidity and
  customers also rest passive orders. Nothing stored can separate them --
  the option book is not recorded -- so this module does not pretend to,
  and the name says `aggressor`, which is what the tape actually reports.
  Note also that a dealer's held delta is not the quantity they must trade:
  hedging a long delta means SELLING the underlying.

`side_sign` is +1 for a BUY print and -1 for a SELL print. `delta` is the
contract's delta -- positive for a call, negative for a put -- so the sign
of the product is already the direction of the aggressor's underlying
exposure, with no separate call/put branch:

    aggressor BUYs a call   +1 * (+0.55) =  bullish
    aggressor BUYs a put    +1 * (-0.45) =  bearish
    aggressor SELLs a call  -1 * (+0.55) =  bearish
    aggressor SELLs a put   -1 * (-0.45) =  bullish

OPEN VERSUS CLOSE IS NOT KNOWN. A BUY print may be opening a long or
closing a short. For a FLOW series that does not matter -- the buyer's delta
moves the same way either way -- but it means nothing here supports a claim
about accumulated positioning. The field that would settle it, open
interest, is frozen overnight.

NEUTRAL and 'N/A' are both DROPPED, and they are different things: per the
moomoo quote docs NEUTRAL means the print landed between the bid and the
ask, 'N/A' covers the rest -- a missing field, a non-int, or the feed's own
unknown value. attach_delta() counts them separately so a session can
report which it lost.

READ `unknown` AS "THE COLLECTOR COULD NOT SAY", NOT AS "THERE WERE NONE".
The collector folded 'N/A' into NEUTRAL until 2026-11-23; the fix landed in
the file at 11:55 that day but the process running the session had started
at 09:39, so the WHOLE of 2026-11-23 is folded and its ~7.8% is the two
combined, unsplittable after the fact. Any row written by a collector older
than that fix reports `unknown == 0` because the information was destroyed
at write time, which is indistinguishable here from a clean session.

Dropping them is a real cost, not a technicality: that bucket is
off-exchange flow carrying genuine direction. The alternative -- inventing
a classification rule -- was rejected because no rule can be CHECKED here;
unlike the equity tape, no option order book is stored to classify against.

THE WEAKEST LINK is upstream of all of this: moomoo documents what
ticker_direction MEANS (BUY = lifting the offer, SELL = hitting the bid,
NEUTRAL = between them, derived from NBBO time & sales) but does not
publish the comparison rule it applies. Everything below is only as good as
that classification.

WHAT THIS IS NOT. This measures delta that traded, not dealer gamma
positioning, and it says nothing about whether any of it was hedged. It is
a flow series, like CVD, not a level like GEX. It has NOT been shown to
predict anything; it is built to be looked at.
"""
from __future__ import annotations

import sqlite3
from datetime import date, timedelta

import pandas as pd

from core.time_utils import candle_start


def _bar_starts(ts: pd.Series, minutes: int) -> pd.Series:
    """candle_start() over a column, without the per-row Python call.

    Vectorised because the obvious loop cost 140 ms per refresh on one
    session of prints and the panel refreshes on a timer -- and the loop was
    itself introduced while fixing something else, which is how a correctness
    fix turns into a performance bug nobody attributes.

    Branch for branch the same as core.time_utils.candle_start, NOT
    Series.dt.floor(): floor() is epoch-aligned while candle_start() aligns
    within the hour, and the two agree only while 60 is divisible by the
    candle size. The viewer's bar_start_keys() carries the identical warning
    for the identical reason; these two have to agree or no option bar lands
    on a chart bar, and the panel goes blank with no error.
    """
    ts = pd.to_datetime(ts, errors="coerce")
    if minutes >= 1440:
        return ts.dt.normalize()
    if minutes > 60:
        total = ts.dt.hour * 60 + ts.dt.minute
        floored = (total // minutes) * minutes
        return (ts.dt.normalize()
                + pd.to_timedelta(floored, unit="m"))
    return (ts.dt.floor("h")
            + pd.to_timedelta((ts.dt.minute // minutes) * minutes, unit="m"))

# Shares per option contract. US equity options are 100x except after a
# corporate action; SOXL's 2026-11-05 3:1 split adjusted the underlying,
# not the multiplier.
CONTRACT_MULTIPLIER = 100

# How stale a greeks row may be and still be used for a print. The sampler
# writes on change plus a 300 s heartbeat, so a quiet contract's newest row
# can legitimately be five minutes old while its delta has not moved. Beyond
# this the row is not evidence about this print any more.
MAX_GREEKS_AGE_S = 900

# The same question asked of the FEED's timestamp rather than ours. These
# are not the same measurement and on 2026-11-23 they disagreed by over 15
# minutes on 21.8% of greeks rows: our sample can be two seconds old while
# the quote inside it is twenty minutes old.
#
# Rows past this are USED but counted in report["stale_quote"], not dropped
# -- dropping them moved that day's session net by -11.8% (+1,135,206 ->
# +1,001,508 shares), which is a large enough swing that the choice belongs
# to the caller rather than to a constant in this file.
MAX_QUOTE_AGE_S = 300

_SIDE_SIGN = {"BUY": 1, "SELL": -1}


def trading_day_of(ts: pd.Timestamp) -> date:
    """Trading day under the 20:00 ET boundary.

    Deliberately identical to _trading_day_of_bucket() in trade_viewer_qt:
    this series is meant to be read beside that module's CVD, and two
    indicators that reset on different boundaries cannot be compared at all
    in the hours between those boundaries. Weekends are NOT rolled forward
    here, unlike strategy/journal/confirmations.trading_day -- the market is
    shut from Friday 20:00 to Sunday 20:00, so no print can land there, and
    matching the viewer matters more than handling an impossible case.
    """
    d = ts.date()
    return d + timedelta(days=1) if ts.hour >= 20 else d


def load_ticks(db: str, underlying: str, start: str, end: str) -> pd.DataFrame:
    """Every option print for one underlying in [start, end)."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return pd.read_sql_query(
            """
            SELECT t.code, t.ts, t.price, t.volume, t.direction,
                   c.expiry, c.strike, c.opt_type
            FROM option_ticks t
            JOIN option_contracts c ON c.code = t.code
            WHERE c.underlying = ? AND t.ts >= ? AND t.ts < ?
            ORDER BY t.ts
            """,
            con, params=(underlying, start, end),
            parse_dates={"ts": {"format": "ISO8601"}},
        )
    finally:
        con.close()


def load_greeks(db: str, underlying: str, start: str, end: str) -> pd.DataFrame:
    """Greeks samples for one underlying, widened at the start.

    CALL THIS AFTER load_ticks, NEVER BEFORE. Each opens its own connection,
    so under WAL the two see snapshots taken a few hundred ms apart while the
    collector writes. Ticks first means the greeks snapshot is the newer one
    and every print can find its sample; reversed, the prints written in
    between arrive with no greeks behind them and are silently dropped as
    `no_greeks`.

    The window is widened backwards by MAX_GREEKS_AGE_S because a print at
    `start` is legitimately priced off a sample taken before it -- querying
    the same window as the ticks would leave the first minutes of every
    session unmatched for exactly the contracts that are quiet.
    """
    lo = (pd.Timestamp(start) - pd.Timedelta(seconds=MAX_GREEKS_AGE_S)
          ).strftime("%Y-%m-%d %H:%M:%S.%f")
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return pd.read_sql_query(
            """
            SELECT g.code, g.ts, g.delta, g.spot, g.upd
            FROM option_greeks g
            JOIN option_contracts c ON c.code = g.code
            WHERE c.underlying = ? AND g.ts >= ? AND g.ts < ?
              AND g.nogreeks = 0 AND g.delta IS NOT NULL
              AND g.upd IS NOT NULL AND g.upd <> ''
            ORDER BY g.ts
            """,
            con, params=(underlying, lo, end),
            parse_dates={"ts": {"format": "ISO8601"},
                         "upd": {"format": "ISO8601"}},
        )
    finally:
        con.close()


def attach_delta(ticks: pd.DataFrame, greeks: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """As-of join each print to the newest greeks sample at or before it.

    Backwards only, never forwards. A sample taken AFTER a print knows
    where the underlying went, so using it would feed the series information
    the print did not have -- the same one-minute lookahead that invalidated
    the drill light on 2026-11-14, and the reason that rule is stated here
    rather than left to the caller.

    The cost is real and is reported rather than hidden: prints that arrived
    through the subscription's same-day backfill can predate the first
    sample of their contract entirely, and those are unmatched. On
    2026-11-23 the collector was restarted at 09:39 and the backfill carried
    prints from 09:30, so this is not a hypothetical.

    Returns (frame, report). The frame holds only matched, signed prints.
    """
    report = {
        "prints": len(ticks),
        # Counted apart, because they are different losses: NEUTRAL was
        # classified as mid-market, 'N/A' was not classified at all.
        # `unknown == 0` does NOT mean there were none -- see the module
        # docstring: a collector older than the 2026-11-23 fix wrote both as
        # NEUTRAL, and that includes the whole of 2026-11-23 itself.
        "neutral": 0, "unknown": 0,
        "no_greeks": 0, "quote_ahead": 0, "delta_sign_violation": 0,
        "stale_quote": 0, "used": 0,
    }
    if ticks.empty:
        return ticks.assign(delta=[], signed_delta=[]), report

    signed = ticks[ticks["direction"].isin(_SIDE_SIGN)].copy()
    dropped = ticks[~ticks["direction"].isin(_SIDE_SIGN)]
    report["neutral"] = int((dropped["direction"] == "NEUTRAL").sum())
    report["unknown"] = len(dropped) - report["neutral"]
    if signed.empty or greeks.empty:
        report["no_greeks"] = len(signed)
        return signed.assign(delta=pd.NA, signed_delta=pd.NA), report

    merged = pd.merge_asof(
        signed.sort_values("ts"),
        greeks.sort_values("ts").rename(columns={"ts": "greeks_ts"}),
        left_on="ts", right_on="greeks_ts", by="code",
        direction="backward",
        tolerance=pd.Timedelta(seconds=MAX_GREEKS_AGE_S),
    )

    unmatched = merged["delta"].isna()
    report["no_greeks"] = int(unmatched.sum())
    merged = merged[~unmatched].copy()

    merged["greeks_age_s"] = (
        merged["ts"] - merged["greeks_ts"]).dt.total_seconds()
    # The feed's own timestamp, which is what the number actually describes.
    # greeks_age_s measures when WE sampled; this measures how old the quote
    # already was when we did. On 2026-11-23 those differed by more than 15
    # minutes on 21.8% of greeks rows.
    merged["quote_age_s"] = (merged["ts"] - merged["upd"]).dt.total_seconds()

    # Guard 1: the feed's clock can run ahead of ours. 260 greeks rows on
    # 2026-11-23 carried an `upd` LATER than our own sample time, by up to
    # 218 s. None of them matched a print that day, but a match is a genuine
    # lookahead -- the quote would postdate the trade it is pricing -- and
    # "it did not happen today" is not a guard.
    ahead = merged["quote_age_s"] < 0
    report["quote_ahead"] = int(ahead.sum())
    merged = merged[~ahead].copy()

    # Guard 2: a call's delta is positive and a put's is negative. Violations
    # are dropped rather than signed, because the formula would turn one into
    # flow pointing the wrong way with nothing on screen to say so. Zero
    # violations in 15,848 greeks rows on 2026-11-23 -- one symbol, one
    # session, one broker, which is why this is an assertion and not a
    # comment.
    bad = ((merged["opt_type"] == "CALL") & (merged["delta"] < 0)
           | (merged["opt_type"] == "PUT") & (merged["delta"] > 0))
    report["delta_sign_violation"] = int(bad.sum())
    merged = merged[~bad].copy()

    report["stale_quote"] = int((merged["quote_age_s"] > MAX_QUOTE_AGE_S).sum())
    merged["side_sign"] = merged["direction"].map(_SIDE_SIGN).astype("int8")
    merged["signed_delta"] = (merged["side_sign"] * merged["delta"]
                              * merged["volume"] * CONTRACT_MULTIPLIER)
    report["used"] = len(merged)
    return merged, report


def by_bar(flow: pd.DataFrame, minutes: int = 1) -> pd.DataFrame:
    """Per-bar net delta, and the running total within each trading day.

    Both modes come out of one call because they are one series read two
    ways, which is how DeepGamma presents it: `net` is the bar-by-bar
    reading, `cum` the session-cumulative one. `cum` restarts at each
    trading day rather than running across the whole frame -- a cumulative
    series with no reset is dominated by whichever day happened to be first.
    """
    # `day` included: an empty frame and a populated one have to carry the
    # same columns, or every caller that touches bars["day"] works until the
    # first symbol with no option tape -- which is the COMMON case here, not
    # the edge one.
    cols = ["bar", "net", "cum", "buy", "sell", "prints", "contracts", "day"]
    if flow.empty:
        return pd.DataFrame(columns=cols)

    # attach_delta's no-greeks path returns signed_delta = NA rather than
    # raising. Summed, NA becomes 0, so that frame would render as a full
    # session of prints with exactly zero net flow -- a plausible-looking
    # chart of nothing. Refuse it instead of drawing it.
    f = flow[flow["signed_delta"].notna()].copy()
    if f.empty:
        return pd.DataFrame(columns=cols)

    # candle_start(), NOT Series.dt.floor(). floor() is epoch-aligned while
    # candle_start() aligns within the hour, and the two agree only while 60
    # is divisible by the candle size. Every size in the viewer's
    # TIMEFRAME_MAP satisfies that today, so floor() is correct right up
    # until a 45-minute timeframe is added -- at which point these bars land
    # on keys the chart never produces and the whole row silently goes
    # blank. bar_start_keys() in trade_viewer_qt carries the same warning
    # for the same reason, and these two have to agree or nothing aligns.
    f["bar"] = _bar_starts(f["ts"], minutes)
    f["day"] = f["bar"].map(trading_day_of)
    # Columns, not a second groupby whose row order is assumed to line up
    # with the first one's. Both would sort by the same keys today, so the
    # bug would be invisible until a pandas release changed a default -- and
    # it would show up as buy/sell attached to the wrong bars, which no
    # assertion on totals would catch.
    f["_pos"] = f["signed_delta"].clip(lower=0)
    f["_neg"] = f["signed_delta"].clip(upper=0)

    out = f.groupby(["day", "bar"], as_index=False).agg(
        net=("signed_delta", "sum"),
        buy=("_pos", "sum"),
        sell=("_neg", "sum"),
        prints=("signed_delta", "size"),
        contracts=("volume", "sum"),
    )
    out["cum"] = out.groupby("day")["net"].cumsum()
    return out[cols].sort_values("bar").reset_index(drop=True)

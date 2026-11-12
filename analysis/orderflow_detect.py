"""Order flow detection algorithms — iceberg, spoofing, and absorption.

Pure Python + NumPy, no Qt dependency.  Imported by trade_viewer_qt.py and
directly testable without a display.
"""

from __future__ import annotations

import bisect as _bisect
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np


# ── Level-change attribution ───────────────────────────────────────────────
#
# Every detector below this point asks the same unasked question: when the
# displayed volume at a price level falls, where did it go? The answer is
# either that someone traded against it or that whoever posted it took it
# back, and those mean opposite things. detect_icebergs never distinguished
# them -- it read any drop-then-refresh as replenishment, which is why it
# flags 92% of eligible levels and why shuffling its input's time order
# RAISES the count (383 real against 423 shuffled over 14 windows).
# detect_spoofs has the same hole and papers over it with a proxy
# (_spread_moved_through) that only asks whether the quote moved.
#
# attribute_level_flow() answers it from the tape instead. For each pair of
# consecutive books it matches trades to the level they hit and reports
# THREE QUANTITIES, never a verdict:
#
#     traded     volume the tape executed at that level
#     added      volume that appeared without a trade explaining it
#     cancelled  volume that left without a trade explaining it
#
# from the identity  dv = added - cancelled - traded, which has one degree of
# freedom too many: only (added - cancelled) is observable, so a level that
# simultaneously gained and lost shows the net. That is the minimum
# assumption available and it is stated rather than hidden.
#
# WHAT THIS IS GOOD FOR, AND WHAT IT IS NOT. An audit of the first version
# established that the trade-to-level link is real: misalign the tape against
# the book and the number of levels with any trade against them collapses
# from 2414 to about 500. It also established three limits that are part of
# the instrument and not of the market:
#
#   * `traded` is nearly unidentifiable in aggregate. Summed over a 20-minute
#     window the unexplained term of the identity runs to 115M shares against
#     a 5.5M-share tape, so at this snapshot resolution `traded` is a 0.9%
#     correction. Per interval it is sound; as a window total it is noise.
#
#   * `added` and `cancelled` are TOTAL VARIATION, not flow. They sum a
#     signed quantity in two sign buckets, so they grow with how often the
#     book is sampled: thinning the same window by a stride of 32 takes
#     (added+cancelled)/traded from 127 to 34. doc/ORDER_FLOW_GUIDE.md
#     appendix A.2 already banned accumulating per-snapshot differences for
#     exactly this reason, and this module repeated the mistake. Report
#     monotone quantities instead -- peak_displayed, net_change, the tape as
#     the denominator -- and never a per-episode sum of these two.
#
#   * The criterion `traded > max_displayed` DOES NOT WORK. Against a null
#     that circularly shifts the tape (preserving every print's size,
#     direction and clustering, destroying only the alignment to the book) it
#     reached p >= 0.18 in all eight windows tested, and in two of them the
#     null's median exceeded the real figure. Step 3 cannot be built on it.

#: Trade prices that match no displayed level are EXCLUDED, never snapped to
#: the nearest one. A sub-penny print at 24.1234 is off-exchange flow that
#: never touched this book; rounding it onto 24.12 would credit a level with
#: an execution that did not happen there, which is precisely the error that
#: would manufacture iceberg evidence. 45.5% of this tape by volume is
#: sub-penny, so the error would not be a rounding detail.
_PRICE_DECIMALS = 4


@dataclass
class LevelEpisode:
    """One appearance of one price level, with its flow split.

    An episode ends when the level is cancelled out of the book, or when the
    feed pauses long enough that the next book cannot be compared with this
    one. Scrolling out of the reported depth does NOT end it: the orders are
    still resting there, and closing the episode would hand the level a fresh
    and smaller max_displayed every time price wandered past.

    traded / added / cancelled are sums over the episode's intervals. Only
    `traded` means what it says at that scale -- see the note above on why
    the other two are total variation. For a size to threshold on, use
    max_displayed and net_change, which do not move with the sampling rate.
    """
    side: str
    price: float
    t_first: datetime
    t_last: datetime
    n_books: int            # books in which the level was present
    max_displayed: float
    first_displayed: float
    last_displayed: float
    traded: float
    traded_at_touch: float  # the part of `traded` that hit rank 0 of its side
    added: float
    cancelled: float
    n_occluded: int         # intervals where the level was out of reported depth

    @property
    def net_change(self) -> float:
        """Displayed size at the end less at the start. Sampling-rate free."""
        return self.last_displayed - self.first_displayed


@dataclass
class FlowAudit:
    """Where every share of tape volume went. The buckets partition it.

    attributed is the only bucket the decomposition uses; the rest record
    what had to be set aside and why, so the coverage is a number rather
    than an assumption.
    """
    tick_vol_total: float = 0.0
    attributed: float = 0.0
    outside_books: float = 0.0   # before the first book or after the last
    in_gap: float = 0.0          # inside an interval the feed paused through
    neutral: float = 0.0         # the feed gave no side
    off_grid: float = 0.0        # price matched no displayed level at all
    wrong_side: float = 0.0      # matched only the side that cannot be hit
    #: Not a bucket of the partition -- a subset of `neutral`, counted so the
    #: caller can see the size of what it gave up. The feed leaves the
    #: majority of this tape NEUTRAL and about 57% of that volume prints at
    #: sub-penny prices, which only off-exchange price improvement produces;
    #: those never touched this book and excluding them is exact. But 15.8%
    #: of the whole tape is NEUTRAL volume sitting squarely ON a displayed
    #: level -- the same order of magnitude as `attributed` itself -- and
    #: doc/ORDER_FLOW_GUIDE.md appendix A.4 shows those prints carry real
    #: direction. Excluding them is therefore a choice with a cost, not a
    #: free simplification.
    neutral_on_level: float = 0.0
    #: Attributed volume by the level's depth rank in the reference book:
    #: 0 is the touch (best bid / best ask), 1 the next level down, and so
    #: on. Rank rather than a price offset so it does not depend on tick
    #: size. Worth looking at before trusting any per-level total: 42.9% of
    #: attributed volume names a level BEHIND the touch, and a single print
    #: that large cannot have been filled by a level the quote never reached.
    traded_by_rank: dict[int, float] = field(default_factory=dict)
    n_books: int = 0
    n_intervals: int = 0
    n_gap_intervals: int = 0
    #: Counted in level-intervals, not intervals: one book pair contributes
    #: one for each level that was out of reported depth across it, so this
    #: routinely exceeds n_intervals.
    n_occluded_level_intervals: int = 0
    interval_secs: list[float] = field(default_factory=list)

    @property
    def attributed_frac(self) -> float:
        return self.attributed / self.tick_vol_total if self.tick_vol_total else 0.0

    def balances(self, tol: float = 1e-6) -> bool:
        """The buckets must add back up to the tape. A test, not a comment."""
        total = (self.attributed + self.outside_books + self.in_gap
                 + self.neutral + self.off_grid + self.wrong_side)
        return abs(total - self.tick_vol_total) <= tol


def attribute_level_flow(
    books: list[tuple[datetime, list[dict]]],
    ticks: list[dict],
    book_lag_secs: float,
    max_gap_secs: float = 5.0,
) -> tuple[list[LevelEpisode], FlowAudit]:
    """Split each level's displayed-volume changes into traded / added / cancelled.

    Args:
        books:         Chronological (ts, levels), ONE ENTRY PER STORED
                       SNAPSHOT -- not a flat level list. Two snapshots can
                       share a timestamp (see the ob_snapshots DDL note) and
                       16% of stored rows do, so a caller reading the table
                       must ORDER BY ts, rowid: this function sorts on the
                       timestamp alone and relies on a stable sort to keep
                       whatever order it was handed. Flattening same-stamped
                       books together fuses two of them into one phantom.
                       levels: {"side", "price", "volume"}.
        ticks:         {"ts", "price", "volume", "direction"}; direction is
                       'BUY', 'SELL' or 'NEUTRAL'.
        book_lag_secs: Subtracted from every book timestamp before comparing
                       it with a trade's. REQUIRED, AND THERE IS NO SENSIBLE
                       DEFAULT, because it depends on where the caller got
                       its books.

                       Pass 0.0 for books read through OrderBookStore: those
                       timestamps are already aligned against the tape.

                       Pass a calibrated value for books carrying the raw
                       `ts` column. That one is the receiving machine's
                       clock, which free-runs at -39 ms/h and is stepped back
                       into line every day or two, so the offset is neither
                       small nor constant -- +0.5 s on 2026-10-30, -0.68 s on
                       2026-11-02, -1.35 s by that night, against a median
                       book interval of 0.016 s. Using one day's value on
                       another tripled the share of prints naming a level
                       that cannot have filled them. Rows written before
                       2026-11-03 00:46 are only reachable this way; see
                       feeds.order_book_store.SERVER_LEAD_SECS.
        max_gap_secs:  Intervals longer than this are not decomposed. The feed
                       pauses over 5 s between 147 and 204 times a session, and
                       across such a pause trades are missed and the book has
                       moved, so a drop cannot be attributed to anything.

    Returns:
        (episodes, audit). Episodes carry the three components; audit says
        what fraction of the tape reached them, and at what depth.
    """
    audit = FlowAudit()
    audit.tick_vol_total = float(sum(float(t["volume"]) for t in ticks))

    books = sorted(books, key=lambda b: b[0])
    audit.n_books = len(books)
    if len(books) < 2:
        audit.outside_books = audit.tick_vol_total
        return [], audit

    scale = 10 ** _PRICE_DECIMALS

    def _key(p) -> int:
        return int(round(float(p) * scale))

    lag = timedelta(seconds=book_lag_secs)
    bts = [b[0] - lag for b in books]

    # Per book: level volumes, the visible price span of each side, and each
    # level's depth rank from the touch. The span is what separates
    # "cancelled" from "scrolled out of the window": the feed sends a fixed
    # number of levels per side, so when price moves the far end falls off
    # and a level that is still resting there stops being reported.
    maps: list[dict[tuple[str, int], float]] = []
    spans: list[dict[str, tuple[int, int] | None]] = []
    ranks: list[dict[tuple[str, int], int]] = []
    for _, levels in books:
        m: dict[tuple[str, int], float] = {}
        bid_k: list[int] = []
        ask_k: list[int] = []
        for lv in levels:
            k = _key(lv["price"])
            side = lv["side"]
            m[(side, k)] = m.get((side, k), 0.0) + float(lv["volume"])
            (bid_k if side == "BID" else ask_k).append(k)
        maps.append(m)
        spans.append({
            "BID": (min(bid_k), max(bid_k)) if bid_k else None,
            "ASK": (min(ask_k), max(ask_k)) if ask_k else None,
        })
        r: dict[tuple[str, int], int] = {}
        for i, k in enumerate(sorted(set(bid_k), reverse=True)):
            r[("BID", k)] = i
        for i, k in enumerate(sorted(set(ask_k))):
            r[("ASK", k)] = i
        ranks.append(r)

    n = len(books)
    gap = [False] * n          # gap[k]: interval (k-1 -> k) is untrustworthy
    for k in range(1, n):
        secs = (bts[k] - bts[k - 1]).total_seconds()
        audit.interval_secs.append(secs)
        if secs > max_gap_secs:
            gap[k] = True
            audit.n_gap_intervals += 1
    audit.n_intervals = n - 1

    # ── Attribute each trade to the interval and level it hit ──────────────
    # The reference book is the one BEFORE the trade: that is the liquidity
    # that was standing there to be taken. Reading the book AFTER instead is
    # the single heaviest error available here, because it is exactly what a
    # mis-set book_lag_secs simulates.
    traded: list[dict[tuple[str, int], float]] = [dict() for _ in range(n)]
    touch: list[dict[tuple[str, int], float]] = [dict() for _ in range(n)]
    for t in ticks:
        vol = float(t["volume"])
        k = _bisect.bisect_left(bts, t["ts"])     # bts[k-1] < ts <= bts[k]
        if k <= 0 or k >= n:
            audit.outside_books += vol
            continue
        if gap[k]:
            audit.in_gap += vol
            continue
        pk = _key(t["price"])
        ref = maps[k - 1]
        direction = t.get("direction", "NEUTRAL")
        if direction not in ("BUY", "SELL"):
            audit.neutral += vol
            if ("BID", pk) in ref or ("ASK", pk) in ref:
                audit.neutral_on_level += vol
            continue
        # A seller lifts nobody: aggressive SELL volume hits resting BIDs,
        # aggressive BUY volume hits resting ASKs.
        side = "BID" if direction == "SELL" else "ASK"
        other = "ASK" if side == "BID" else "BID"
        if (side, pk) in ref:
            traded[k][(side, pk)] = traded[k].get((side, pk), 0.0) + vol
            rank = ranks[k - 1].get((side, pk), -1)
            audit.traded_by_rank[rank] = audit.traded_by_rank.get(rank, 0.0) + vol
            if rank == 0:
                touch[k][(side, pk)] = touch[k].get((side, pk), 0.0) + vol
            audit.attributed += vol
        elif (other, pk) in ref:
            # Either the book we hold is stale against this print, or the
            # feed's own side label does not describe this book -- a
            # wholesaler's BUY means the direction on its internal sheet.
            # Measured elsewhere: only 50.6% of BUY volume prints at the ask
            # or above. Either way the level it names cannot be the one it
            # consumed, so it is set aside rather than forced onto a side.
            audit.wrong_side += vol
        else:
            audit.off_grid += vol

    # ── Walk the books, splitting each level's change ──────────────────────
    episodes: list[LevelEpisode] = []
    open_eps: dict[tuple[str, int], LevelEpisode] = {}

    def _open(side: str, pk: int, k: int, vol: float) -> LevelEpisode:
        ep = LevelEpisode(
            side=side, price=pk / scale, t_first=books[k][0], t_last=books[k][0],
            n_books=0, max_displayed=0.0, first_displayed=vol,
            last_displayed=vol, traded=0.0, traded_at_touch=0.0, added=0.0,
            cancelled=0.0, n_occluded=0,
        )
        open_eps[(side, pk)] = ep
        return ep

    def _close_all() -> None:
        episodes.extend(open_eps.values())
        open_eps.clear()

    def _seen(ep: LevelEpisode, k: int, vol: float) -> None:
        ep.t_last = books[k][0]
        ep.n_books += 1
        ep.max_displayed = max(ep.max_displayed, vol)
        ep.last_displayed = vol

    for (side, pk), vol in maps[0].items():
        _seen(_open(side, pk, 0, vol), 0, vol)

    def _occluded(side: str, pk: int, span: tuple[int, int] | None) -> bool:
        """Is this price beyond the far end of what that side reports?"""
        if span is None:
            return True                      # the side went empty: tell nothing
        lo, hi = span
        return pk < lo if side == "BID" else pk > hi

    for k in range(1, n):
        if gap[k]:
            # Nothing about this step can be read, so no episode survives it.
            _close_all()
            for (side, pk), vol in maps[k].items():
                _seen(_open(side, pk, k, vol), k, vol)
            continue

        prev, cur = maps[k - 1], maps[k]
        for key in set(prev) | set(cur) | set(traded[k]) | set(open_eps):
            side, pk = key
            v0 = prev.get(key, 0.0)
            v1 = cur.get(key, 0.0)
            tv = traded[k].get(key, 0.0)
            tt = touch[k].get(key, 0.0)

            ep = open_eps.get(key)
            if ep is None:
                if v1 == 0.0 and tv == 0.0:
                    continue
                ep = _open(side, pk, k, v1)

            if v1 == 0.0 and _occluded(side, pk, spans[k][side]):
                # Out of the reported window, not out of the book. Counting
                # this as a cancellation would invent one every time price
                # moved, and ENDING the episode here would hand the level a
                # fresh, smaller max_displayed on the way back -- which is
                # how 41% of the first version's flagged levels came to have
                # three books or fewer behind them.
                ep.traded += tv
                ep.traded_at_touch += tt
                ep.n_occluded += 1
                audit.n_occluded_level_intervals += 1
                continue

            if v0 == 0.0 and v1 > 0.0 and _occluded(side, pk, spans[k - 1][side]):
                # Scrolled back INTO view. It was resting there all along.
                ep.traded += tv
                ep.traded_at_touch += tt
                ep.n_occluded += 1
                audit.n_occluded_level_intervals += 1
                _seen(ep, k, v1)
                continue

            net = (v1 - v0) + tv             # = added - cancelled
            ep.traded          += tv
            ep.traded_at_touch += tt
            ep.added           += max(net, 0.0)
            ep.cancelled       += max(-net, 0.0)

            if v1 > 0.0:
                _seen(ep, k, v1)
            else:
                ep.t_last = books[k][0]
                ep.last_displayed = 0.0
                episodes.append(ep)
                open_eps.pop(key, None)

    _close_all()
    return episodes, audit


def detect_icebergs(
    snaps: list[dict],
    bucket_to_idx: dict,
    bin_size: float,
    price_min: float,
    N_PRICE: int,
    min_refreshes: int,
    vol_threshold: float,
    best_bid: float | None = None,
    best_ask: float | None = None,
    max_spread_bins: int = 2,
    col_secs: int = 30,
) -> list[tuple]:
    """Detect iceberg orders from order book snapshot series.

    A price level is flagged as an iceberg when resting volume at the TOP OF
    BOOK repeatedly drops (depleted by aggressive executions) then refreshes.

    Only levels within max_spread_bins bins of the best bid (BID) or best ask
    (ASK) are considered — passive orders deep in the book are never consumed.

    Each contiguous appearance of a price level is treated as an independent
    candidate.  A gap > 1.5 × col_secs between consecutive snapshots means the
    level vanished from the book (breakthrough or cancellation); the state
    machine resets so any later reappearance starts a fresh iceberg search.

    Args:
        snaps:           Order book snapshots {ts, side, price, volume}.
        bucket_to_idx:   exact snapshot timestamp -> column index.
        bin_size:        Price bin width in dollars.
        price_min:       Bottom of the price range.
        N_PRICE:         Number of price bins.
        min_refreshes:   Minimum refresh count to classify as iceberg.
        vol_threshold:   Minimum volume per snapshot to include.
        best_bid:        Current best bid price.
        best_ask:        Current best ask price.
        max_spread_bins: Bins away from best bid/ask allowed.
        col_secs:        Polling interval in seconds (used to detect gaps).

    Returns:
        List of (first_bar_idx, last_bar_idx, price, refresh_count).
        Multiple tuples per price level are possible if the level reappeared
        after a gap.
    """
    DROP_FRAC    = 0.25
    RECOVER_FRAC = 0.40
    # Gap larger than this -> level vanished (segment resets). Floored at 5s,
    # and the floor outlived its reason: it was set because the collector
    # once throttled writes to one every 2.0s per code, so a 1.5s gap would
    # have fired between every genuine pair and blocked detection outright
    # (reported as "iceberg detection stopped working entirely" when Col(s)
    # dropped from 30 to 1). That throttle is gone -- order_book_collector
    # now writes every push that yields a complete book.
    #
    # What the floor does NOW, measured on db/order_book.db (the written
    # stream, which is the one this detector reads back), US.SOXL, two
    # sessions: 5.65 snapshots a second, median gap 274 ms on 2026-10-30 and
    # 33 ms on 11-02, longest pause 60 s and 169 s. Gaps over 5 s happen
    # 147-204 times a session. So the floor is not idle -- the FEED's own
    # pauses trip it a couple of hundred times a day, resetting segments at
    # moments that have nothing to do with a level leaving the book.
    #
    # Left as it is because the detector it serves does not work either: it
    # flags 92% of eligible levels, and shuffling its input's time order
    # RAISES the count (383 real against 423 shuffled over 14 windows). Fix
    # the criterion first; the number to derive this from afterwards is the
    # 147-204 pauses, not a constant that no longer exists.
    GAP_SECS     = max(col_secs * 1.5, 5.0)

    # Group by (side, price_bin) — sides must not be mixed
    groups: dict[tuple, list] = defaultdict(list)
    for s in snaps:
        if s["volume"] < vol_threshold:
            continue
        bar_idx = bucket_to_idx.get(s["ts"], -1)
        if bar_idx < 0:
            continue
        p_bin = int((s["price"] - price_min) / bin_size)
        if not (0 <= p_bin < N_PRICE):
            continue
        # (ts, seq, ...) and never (ts, volume, ...): the live path stamps
        # every push inside a column with that column's timestamp, so ts
        # alone does not order them and a plain sort on the old
        # (ts, bar_idx, volume) tuple fell through to VOLUME -- turning the
        # arrival order into an ascending ramp. A level that only ever grew,
        # 60 -> 300 -> 700 -> 1000, then read as three refreshes. `seq` is
        # the caller's own arrival counter; snapshots that carry no seq
        # (replayed from the database, or built by a test) keep the old
        # ordering because their timestamps are already distinct.
        groups[(s["side"], p_bin)].append(
            (s["ts"], s.get("seq", 0), bar_idx, float(s["volume"])))

    bid_bin = int((best_bid - price_min) / bin_size) if best_bid is not None else None
    ask_bin = int((best_ask - price_min) / bin_size) if best_ask is not None else None

    icebergs = []
    for (side, p_bin), entries in groups.items():
        # Only top-of-book levels — executions only happen there
        if side == "BID" and bid_bin is not None:
            if p_bin < bid_bin - max_spread_bins or p_bin > bid_bin + max_spread_bins:
                continue
        elif side == "ASK" and ask_bin is not None:
            if p_bin < ask_bin - max_spread_bins or p_bin > ask_bin + max_spread_bins:
                continue

        entries.sort()

        # Split into contiguous segments; a time gap means the level disappeared
        segments: list[list] = []
        seg: list = [entries[0]]
        for k in range(1, len(entries)):
            gap = (entries[k][0] - entries[k - 1][0]).total_seconds()
            if gap > GAP_SECS:
                segments.append(seg)
                seg = [entries[k]]
            else:
                seg.append(entries[k])
        segments.append(seg)

        price = price_min + (p_bin + 0.5) * bin_size

        # Each contiguous segment is an independent iceberg candidate
        for seg_entries in segments:
            if len(seg_entries) < min_refreshes * 2 + 1:
                continue
            bar_idxs = [e[2] for e in seg_entries]
            vols     = np.array([e[3] for e in seg_entries], dtype=np.float64)

            running_peak = 0.0
            depleted     = False
            refreshes: list[int] = []

            for i, v in enumerate(vols):
                running_peak = max(running_peak, v)
                if running_peak <= 0:
                    continue
                if not depleted and v < DROP_FRAC * running_peak:
                    depleted = True
                elif depleted and v > RECOVER_FRAC * running_peak:
                    depleted = False
                    refreshes.append(i)

            if len(refreshes) < min_refreshes:
                continue

            first_bar = bar_idxs[refreshes[0]]
            last_bar  = bar_idxs[refreshes[-1]]
            icebergs.append((first_bar, last_bar, price, len(refreshes)))

    return icebergs


def detect_spoofs(
    ob_data: list[dict],
    bucket_to_idx: dict,
    bin_size: float,
    price_min: float,
    N_PRICE: int,
    min_vol: float,
    max_duration_secs: float,
) -> list[tuple]:
    """Detect spoofing events from order book snapshots.

    A spoof is a large order that appears at a price level then vanishes within
    max_duration_secs without being executed — creating false price pressure.

    Execution is inferred from spread movement: if the best bid (for a BID
    order) or best ask (for an ASK order) moved through the price level between
    appearance and disappearance, the order was likely filled, not cancelled.

    Args:
        ob_data:           Order book snapshots {ts, side, price, volume}.
        bucket_to_idx:     exact snapshot timestamp -> column index.
        bin_size:          Price bin width.
        price_min:         Bottom of price range.
        N_PRICE:           Number of price bins.
        min_vol:           Min volume to be considered a large order.
                           Pass 0 to auto-set to median of the latest snapshot.
        max_duration_secs: Max lifetime (seconds) of a genuine order; faster
                           disappearance flags as a spoof candidate.

    Returns:
        List of (appear_bar, disappear_bar, price, side).
        side='BID': spoofer pushing price up; side='ASK': pushing price down.
    """
    DISAPPEAR_FRAC = 0.20

    if not ob_data:
        return []

    # Auto min_vol: median volume of the most recent snapshot's levels
    if min_vol <= 0:
        latest_ts   = max(s["ts"] for s in ob_data)
        latest_vols = [s["volume"] for s in ob_data if s["ts"] == latest_ts]
        min_vol     = float(np.median(latest_vols)) if latest_vols else 1.0

    # Build best bid/ask per timestamp for execution-proxy check
    ts_bids: dict = {}
    ts_asks: dict = {}
    for s in ob_data:
        ts = s["ts"]
        p  = float(s["price"])
        if s["side"] == "BID":
            if ts not in ts_bids or p > ts_bids[ts]:
                ts_bids[ts] = p
        else:
            if ts not in ts_asks or p < ts_asks[ts]:
                ts_asks[ts] = p
    sorted_ts = sorted(set(ts_bids) | set(ts_asks))

    def _spread_moved_through(side: str, price: float,
                              t_start: datetime, t_end: datetime) -> bool:
        """True if the spread crossed *price* between t_start and t_end."""
        for ts in sorted_ts:
            if ts <= t_start:
                continue
            if ts > t_end:
                break
            if side == "BID":
                bb = ts_bids.get(ts)
                # Best bid fell below the level → order was consumed by sellers
                if bb is not None and bb < price - bin_size:
                    return True
            else:
                ba = ts_asks.get(ts)
                # Best ask rose above the level → order was consumed by buyers
                if ba is not None and ba > price + bin_size:
                    return True
        return False

    # Group by (side, price_bin)
    groups: dict[tuple, list] = defaultdict(list)
    for s in ob_data:
        bar_idx = bucket_to_idx.get(s["ts"], -1)
        if bar_idx < 0:
            continue
        p_bin = int((s["price"] - price_min) / bin_size)
        if not (0 <= p_bin < N_PRICE):
            continue
        # (ts, seq, ...) -- see the note in detect_icebergs. Without seq a
        # tie on ts fell through to volume, so inside one column the "order"
        # this walk sees was smallest-first rather than first-arrived.
        groups[(s["side"], p_bin)].append(
            (s["ts"], s.get("seq", 0), bar_idx, float(s["volume"])))

    spoofs = []
    for (side, p_bin), entries in groups.items():
        if len(entries) < 2:
            continue
        entries.sort()
        level_price = price_min + (p_bin + 0.5) * bin_size

        i = 0
        while i < len(entries):
            ts_i, _seq_i, bar_i, vol_i = entries[i]
            prev_vol = entries[i - 1][3] if i > 0 else 0.0

            # Order must appear suddenly (prev small) and be large
            if vol_i < min_vol or prev_vol >= min_vol * 0.5:
                i += 1
                continue

            appear_ts  = ts_i
            appear_vol = vol_i
            appear_bar = bar_i

            # Find disappearance within max_duration_secs
            j = i + 1
            disappear_ts  = None
            disappear_bar = None
            while j < len(entries):
                ts_j, _seq_j, bar_j, vol_j = entries[j]
                if (ts_j - appear_ts).total_seconds() > max_duration_secs:
                    break
                if vol_j < appear_vol * DISAPPEAR_FRAC:
                    disappear_ts  = ts_j
                    disappear_bar = bar_j
                    break
                j += 1

            if disappear_ts is None:
                i += 1
                continue

            # If the spread moved through the level the order was executed, not spoofed
            if _spread_moved_through(side, level_price, appear_ts, disappear_ts):
                i = j + 1
                continue

            spoofs.append((appear_bar, disappear_bar, level_price, side))
            i = j + 1

    return spoofs


def detect_absorption(
    ob_data: list[dict],
    raw_ticks: list[dict],
    bin_size: float,
    price_min: float,
    N_PRICE: int,
    passive_k: float,
    active_k: float,
    hit_ratio: float,
) -> list[tuple]:
    """Detect absorption: passive orders holding a price level against significant aggression.

    A price level is flagged when ALL three conditions hold within the window:
      1. Resting volume >= avg_tick_vol * passive_k  (wall is meaningfully large)
      2. Aggressive volume at that level >= avg_tick_vol * active_k  (real pressure applied)
      3. agg_vol / pass_vol >= hit_ratio  (a meaningful fraction of the wall was attempted)
      4. Resting volume at window end > 0  (implicit: level still present = not broken through)

    Aggressive volume accumulates across all ticks at the price bin — small split orders
    are counted together, preventing detection evasion through order fragmentation.

    Args:
        ob_data:    Order book snapshots {ts, side, price, volume} pre-filtered to window,
                    sorted ascending by ts.
        raw_ticks:  Tick records {ts, price, volume, direction} pre-filtered to window.
        bin_size:   Price bin width (use tick size of the instrument).
        price_min:  Lower bound of the price range grid.
        N_PRICE:    Number of price bins.
        passive_k:  Resting volume must be >= avg_tick_vol * passive_k.
        active_k:   Aggressive volume must be >= avg_tick_vol * active_k.
        hit_ratio:  agg_vol / pass_vol must be >= this value (0–1).

    Returns:
        List of (price, side, agg_vol, pass_vol, ratio).
        side='ASK': sellers absorbing buyers (ask wall held).
        side='BID': buyers absorbing sellers (bid wall held).
    """
    if not ob_data or not raw_ticks:
        return []

    # Average volume per tick — instrument/session-adaptive baseline for thresholds
    total_vol = sum(t["volume"] for t in raw_ticks)
    n_ticks   = len(raw_ticks)
    if n_ticks == 0:
        return []
    avg_tick_vol      = total_vol / n_ticks
    passive_threshold = avg_tick_vol * passive_k
    active_threshold  = avg_tick_vol * active_k

    # Last-seen resting volume per (side, price_bin) within the window.
    # ob_data must be time-sorted so the final value reflects end-of-window state.
    last_pass: dict[tuple, float] = {}
    for s in ob_data:
        p_bin = int((s["price"] - price_min) / bin_size)
        if 0 <= p_bin < N_PRICE:
            last_pass[(s["side"], p_bin)] = float(s["volume"])

    # Accumulate aggressive tick volume per price bin (split orders sum naturally)
    agg_buy:  dict[int, float] = defaultdict(float)
    agg_sell: dict[int, float] = defaultdict(float)
    for t in raw_ticks:
        p_bin = int((t["price"] - price_min) / bin_size)
        if not (0 <= p_bin < N_PRICE):
            continue
        if t["direction"] == "BUY":
            agg_buy[p_bin]  += t["volume"]
        elif t["direction"] == "SELL":
            agg_sell[p_bin] += t["volume"]

    results = []

    # ASK absorption: buyers hitting a sell wall that held
    for p_bin, agg in agg_buy.items():
        pass_vol = last_pass.get(("ASK", p_bin), 0.0)
        if pass_vol < passive_threshold:
            continue
        if agg < active_threshold:
            continue
        ratio = agg / pass_vol
        if ratio < hit_ratio:
            continue
        price = price_min + (p_bin + 0.5) * bin_size
        results.append((price, "ASK", agg, pass_vol, ratio))

    # BID absorption: sellers hitting a buy wall that held
    for p_bin, agg in agg_sell.items():
        pass_vol = last_pass.get(("BID", p_bin), 0.0)
        if pass_vol < passive_threshold:
            continue
        if agg < active_threshold:
            continue
        ratio = agg / pass_vol
        if ratio < hit_ratio:
            continue
        price = price_min + (p_bin + 0.5) * bin_size
        results.append((price, "BID", agg, pass_vol, ratio))

    return results


def detect_stacked_imbalance(
    snaps: list[dict],
    bucket_to_idx: dict,
    bin_size: float,
    price_min: float,
    N_PRICE: int,
    min_levels: int = 3,
    imbalance_ratio: float = 3.0,
    min_vol: float = 0.0,
    max_depth: int = 10,
) -> list[tuple]:
    """Detect stacked imbalance from order book snapshots.

    Bid and ask levels in each snapshot are paired by depth rank (rank 0 =
    best bid / best ask, rank 1 = second-best, etc.).  A run of min_levels
    consecutive ranks where bid_vol / ask_vol >= imbalance_ratio is a bullish
    stacked imbalance; ask_vol / bid_vol >= imbalance_ratio is bearish.

    Only the top max_depth ranks are considered.  Levels beyond max_depth are
    ignored because deep-book orders are far from the current price and do not
    reflect near-term market pressure.

    A side holding fewer than max_depth levels has its missing ranks treated
    as 0 volume rather than truncating the rank list — an unpaired level
    registers as an infinite ratio and counts as fully imbalanced.  This is
    about a side being short overall: a level removed by min_vol is dropped
    from the list, not zeroed, so the ranks below it shift up by one.  The two
    behave nothing alike, and only the first one divides by zero.

    Each snapshot is evaluated independently; persistence across snapshots is
    a viewer-layer concern.

    Args:
        snaps:            Order book snapshots {ts, side, price, volume}.
        bucket_to_idx:    exact snapshot timestamp -> column index.
        bin_size:         Price bin width.
        price_min:        Bottom of price range.
        N_PRICE:          Number of price bins.
        min_levels:       Minimum consecutive imbalanced depth levels.
        imbalance_ratio:  bid/ask (or ask/bid) threshold per level.
        min_vol:          Levels below this volume are treated as absent (0).
        max_depth:        Maximum depth ranks to analyse (top-of-book only).

    Returns:
        List of (bar_idx, price_lo, price_hi, direction, mean_ratio).
        direction='BID': buyers dominating — zone spans bid-side prices.
        direction='ASK': sellers dominating — zone spans ask-side prices.
    """
    if not snaps:
        return []

    EPS = 1e-9

    by_ts: dict = defaultdict(lambda: {"BID": [], "ASK": []})
    for s in snaps:
        vol = float(s["volume"])
        if vol < min_vol:
            continue
        by_ts[s["ts"]][s["side"]].append((float(s["price"]), vol))

    results = []
    for ts, sides in by_ts.items():
        bar_idx = bucket_to_idx.get(ts, -1)
        if bar_idx < 0:
            continue

        bids = sorted(sides["BID"], key=lambda x: -x[0])   # best bid first
        asks = sorted(sides["ASK"], key=lambda x:  x[0])   # best ask first

        n_ranks = min(max(len(bids), len(asks)), max_depth)
        if n_ranks == 0:
            continue

        bid_vols   = [bids[i][1] if i < len(bids) else 0.0 for i in range(n_ranks)]
        ask_vols   = [asks[i][1] if i < len(asks) else 0.0 for i in range(n_ranks)]
        bid_prices = [bids[i][0] if i < len(bids) else None for i in range(n_ranks)]
        ask_prices = [asks[i][0] if i < len(asks) else None for i in range(n_ranks)]

        rank_dir: list[str | None] = []
        rank_ratio: list[float]    = []
        for bv, av in zip(bid_vols, ask_vols):
            if bv >= av * imbalance_ratio and bv > EPS:
                rank_dir.append("BID")
                rank_ratio.append(bv / (av + EPS))
            elif av >= bv * imbalance_ratio and av > EPS:
                rank_dir.append("ASK")
                rank_ratio.append(av / (bv + EPS))
            else:
                rank_dir.append(None)
                rank_ratio.append(1.0)

        i = 0
        while i < n_ranks:
            d = rank_dir[i]
            if d is None:
                i += 1
                continue

            j = i + 1
            while j < n_ranks and rank_dir[j] == d:
                j += 1

            if j - i >= min_levels:
                mean_ratio = float(np.mean(rank_ratio[i:j]))
                prices = (
                    [bid_prices[k] for k in range(i, j) if bid_prices[k] is not None]
                    if d == "BID"
                    else [ask_prices[k] for k in range(i, j) if ask_prices[k] is not None]
                )
                if prices:
                    results.append((bar_idx, min(prices), max(prices), d, mean_ratio))

            i = j

    return results


def detect_absorption_bubbles(
    ticks: list[dict],
    col_ts: list[datetime],
    mid_prices: list[float | None],
    col_secs: int,
    min_delta_vol: float = 500.0,
    max_price_move: float | None = None,
) -> list[tuple]:
    """Detect columns where aggressive order flow is absorbed by passive orders.

    For each column window, net delta (buy_vol − sell_vol) is compared against
    the mid-price movement.  True absorption: the aggressive side fails to move
    price in their direction and price stays roughly flat.

    - delta >= +min_delta_vol AND -max_price_move <= price_change <= 0
      → 'BUY' absorbed (aggressive buyers held by passive sell wall — bearish)
    - delta <= -min_delta_vol AND 0 <= price_change <= max_price_move
      → 'SELL' absorbed (aggressive sellers held by passive buy wall — bullish)

    When max_price_move is None the upper bound is not applied (legacy behaviour:
    any price_change in the passive side's direction qualifies).

    Args:
        ticks:           Tick records {ts, price, volume, direction}.
                         direction must be 'BUY', 'SELL', or 'NEUTRAL'.
        col_ts:          Per-column timestamps (col_ts[i] = snapshot time of column i).
        mid_prices:      Per-column mid-price (None if unavailable for that column).
        col_secs:        Column duration in seconds; determines bucket half-width.
        min_delta_vol:   Minimum |delta| required to flag as absorption.
        max_price_move:  Maximum allowed |price_change| for the event to qualify.
                         None means no upper bound (original behaviour).

    Returns:
        List of (col_idx, mid_price, direction, delta_vol).
        direction: 'BUY'  = aggressive buyers absorbed (passive sellers won).
                   'SELL' = aggressive sellers absorbed (passive buyers won).
    """
    if not ticks or not col_ts:
        return []

    # Build a sorted list of col_ts for binary search
    ts_list = list(col_ts)

    # Bucket each tick into the column whose start timestamp <= tick.ts.
    # Each column i owns [col_ts[i], col_ts[i+1]); the last column owns
    # [col_ts[-1], col_ts[-1] + col_secs). bisect_right - 1 gives the correct
    # column index directly — no half-window clipping needed.
    col_buy: dict[int, float] = defaultdict(float)
    col_sell: dict[int, float] = defaultdict(float)

    for tk in ticks:
        direction = tk.get("direction", "NEUTRAL")
        if direction not in ("BUY", "SELL"):
            continue
        tt = tk["ts"]
        idx = _bisect.bisect_right(ts_list, tt) - 1
        if idx < 0:
            continue

        vol = float(tk["volume"])
        if direction == "BUY":
            col_buy[idx] += vol
        else:
            col_sell[idx] += vol

    results = []
    all_cols = set(col_buy) | set(col_sell)
    for i in sorted(all_cols):
        if i >= len(col_ts) or i >= len(mid_prices):
            continue
        mid = mid_prices[i]
        if mid is None:
            continue

        buy_vol  = col_buy.get(i, 0.0)
        sell_vol = col_sell.get(i, 0.0)
        delta    = buy_vol - sell_vol

        # Price change vs previous column with valid mid
        price_change: float | None = None
        for j in range(i - 1, -1, -1):
            if mid_prices[j] is not None:
                price_change = mid - mid_prices[j]
                break

        if abs(delta) < min_delta_vol:
            continue

        buy_ok = price_change is None or (
            price_change <= 0
            and (max_price_move is None or price_change >= -max_price_move)
        )
        sell_ok = price_change is None or (
            price_change >= 0
            and (max_price_move is None or price_change <= max_price_move)
        )
        if delta > 0 and buy_ok:
            results.append((i, mid, "BUY", delta))
        elif delta < 0 and sell_ok:
            results.append((i, mid, "SELL", abs(delta)))

    return results


def detect_aggressor_bubbles(
    ticks: list[dict],
    col_ts: list[datetime],
    mid_prices: list[float | None],
    col_secs: int,
    min_delta_vol: float = 500.0,
) -> list[tuple]:
    """Find columns with dominant directional order flow (aggressors).

    No price-movement filter is applied — any column whose |buy_vol − sell_vol|
    exceeds min_delta_vol produces a bubble.  The bubble direction is the
    dominant side (BUY or SELL); interpreting whether the flow was absorbed
    is left to the user.

    Returns:
        List of (col_idx, mid_price, direction, delta_vol).
    """
    if not ticks or not col_ts:
        return []

    ts_list = list(col_ts)
    col_buy: dict[int, float] = defaultdict(float)
    col_sell: dict[int, float] = defaultdict(float)

    for tk in ticks:
        direction = tk.get("direction", "NEUTRAL")
        if direction not in ("BUY", "SELL"):
            continue
        tt = tk["ts"]
        idx = _bisect.bisect_right(ts_list, tt) - 1
        if idx < 0:
            continue
        vol = float(tk["volume"])
        if direction == "BUY":
            col_buy[idx] += vol
        else:
            col_sell[idx] += vol

    results = []
    for i in sorted(set(col_buy) | set(col_sell)):
        if i >= len(col_ts) or i >= len(mid_prices):
            continue
        mid = mid_prices[i]
        if mid is None:
            continue
        buy_vol  = col_buy.get(i, 0.0)
        sell_vol = col_sell.get(i, 0.0)
        delta    = buy_vol - sell_vol
        if abs(delta) < min_delta_vol:
            continue
        if delta > 0:
            results.append((i, mid, "BUY", delta))
        else:
            results.append((i, mid, "SELL", abs(delta)))

    return results

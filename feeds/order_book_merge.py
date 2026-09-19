"""Merge partial ORDER_BOOK pushes into complete two-sided snapshots.

Extracted verbatim from analysis/order_book_collector.py so the Liquidity
Heatmap can consume the push feed directly without reimplementing any of it.
Every rule here was found by watching the feed misbehave in production; a
second copy would drift and re-earn those bugs.

Deliberately pure: a push goes in, a complete snapshot (or None) comes out.
Write throttling, persistence and logging cadence stay with whoever owns the
sink -- the collector writes to SQLite, the heatmap renders straight to the
screen, and neither wants the other's policy baked in here.
"""

from __future__ import annotations

import logging

_DEFAULT_SIDE_STALE_SECS = 10.0

Level = tuple[float, int]


def parse_side(items) -> list[Level]:
    """Parse bid or ask levels from push data into (price, volume) tuples.

    Push items may be dicts {"price": ..., "volume": ...} or sequences
    [price, volume, ...].
    """
    result: list[Level] = []
    for item in items:
        try:
            if isinstance(item, dict):
                result.append((float(item["price"]), int(item["volume"])))
            else:
                result.append((float(item[0]), int(item[1])))
        except (KeyError, IndexError, TypeError, ValueError):
            pass
    return result


class OrderBookMerger:
    """Per-code side cache turning partial pushes into complete snapshots.

    ORDER_BOOK pushes can be *partial*: Qot_GetOrderBook.proto documents
    svrRecvTimeBid/svrRecvTimeAsk as separate per-side fields precisely
    because a given push can carry a fresh update for one side while the
    other is stale/cached (its recv time reads zero) -- e.g. right after a
    reconnect, or just because that side simply hasn't changed since the
    last push. Treating each push as a complete two-sided snapshot means a
    bid-only push overwrites the stored "latest" row and makes the ask side
    vanish from every reader (the Liquidity Heatmap, depth-to-cursor, etc.)
    until the next push that happens to include asks -- reported as the ask
    (or bid) side going completely blank for stretches, then reappearing.
    Cache the last non-empty list per side per code and always return the
    merged, complete state instead of whatever this one push happened to
    contain.

    One instance per consumer, not per code: it keys its cache by code
    internally, matching how a single subscription delivers every code's
    pushes to one handler.
    """

    def __init__(self, stale_secs: float = _DEFAULT_SIDE_STALE_SECS,
                 log: logging.Logger | None = None) -> None:
        self._stale_secs = stale_secs
        self._log = log or logging.getLogger(__name__)
        self._cache: dict[str, dict] = {}

    def reset(self, code: str | None = None) -> None:
        """Forget cached sides -- for a code, or all of them.

        Call this on reconnect or when switching the code being displayed:
        a cached side from before the gap is not evidence about the book now.
        """
        if code is None:
            self._cache.clear()
        else:
            self._cache.pop(code, None)

    def merge(self, code: str, bid_items, ask_items,
              now: float) -> tuple[list[Level], list[Level]] | None:
        """Fold one push into the cache; return the complete (bids, asks).

        `now` is the caller's clock (time.time()), passed in so the caller can
        keep one consistent timestamp across merge + persist + render.

        Returns None when this push yields nothing usable -- both sides empty,
        or a crossed book that cannot be attributed to one side. A None means
        "don't act on this push", never "the book is empty".
        """
        cache = self._cache.setdefault(
            code, {"bids": [], "asks": [], "bids_ts": 0.0, "asks_ts": 0.0})
        new_bids = parse_side(bid_items)
        new_asks = parse_side(ask_items)

        if new_bids:
            cache["bids"], cache["bids_ts"] = new_bids, now
        elif cache["bids"] and now - cache["bids_ts"] > self._stale_secs:
            self._log.warning(
                "%s bid side stale for >%.0fs, dropping %d cached level(s)",
                code, self._stale_secs, len(cache["bids"]))
            cache["bids"] = []
        if new_asks:
            cache["asks"], cache["asks_ts"] = new_asks, now
        elif cache["asks"] and now - cache["asks_ts"] > self._stale_secs:
            self._log.warning(
                "%s ask side stale for >%.0fs, dropping %d cached level(s)",
                code, self._stale_secs, len(cache["asks"]))
            cache["asks"] = []

        bids, asks = cache["bids"], cache["asks"]
        if not bids and not asks:
            return None

        if bids and asks:
            best_bid = max(p for p, _ in bids)
            best_ask = min(p for p, _ in asks)
            if best_bid >= best_ask:
                # A sustained crossed top-of-book isn't physically valid --
                # a real cross gets arbitraged away in microseconds. Seen
                # in practice: bid frozen at one price for 60+ seconds
                # while ask legitimately moved below it -- moomoo's feed
                # kept re-sending that bid level as a "fresh" (non-empty)
                # push the whole time, so the >0s cache-staleness check
                # above never triggers (it only measures time since the
                # last push, not whether the reported *value* actually
                # changed). Trust whichever side genuinely refreshed this
                # push and drop the other; if both (or neither) refreshed
                # and it's still crossed, there's no way to tell which
                # side is bad -- skip rather than report an impossible
                # snapshot.
                if new_bids and not new_asks:
                    self._log.warning(
                        "%s crossed book bid=%.2f >= ask=%.2f, "
                        "dropping stale ask cache", code, best_bid, best_ask)
                    cache["asks"] = []
                elif new_asks and not new_bids:
                    self._log.warning(
                        "%s crossed book bid=%.2f >= ask=%.2f, "
                        "dropping stale bid cache", code, best_bid, best_ask)
                    cache["bids"] = []
                else:
                    self._log.warning(
                        "%s crossed book bid=%.2f >= ask=%.2f "
                        "(both/neither side fresh), skipping", code, best_bid, best_ask)
                    return None
                bids, asks = cache["bids"], cache["asks"]
                if not bids and not asks:
                    return None

        return bids, asks

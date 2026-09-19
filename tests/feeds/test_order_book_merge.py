"""Unit tests for feeds/order_book_merge.py.

These pin the three rules the collector learned the hard way from the live
feed: partial pushes must not blank the other side, a side cached past its
staleness window must be dropped, and a crossed book must be attributed to
one side or skipped. The heatmap now depends on the same rules, so a
regression here would break both consumers at once.
"""

import logging
import unittest

import pytest

from feeds.order_book_merge import OrderBookMerger, parse_side

C = "US.SOXL"


def _m(**kw) -> OrderBookMerger:
    log = logging.getLogger("test_obm")
    log.addHandler(logging.NullHandler())
    return OrderBookMerger(log=log, **kw)


def _lv(*pairs):
    """Push-shaped level list (the dict form moomoo sends)."""
    return [{"price": p, "volume": v} for p, v in pairs]


# ── parse_side ───────────────────────────────────────────────────────────────

class TestParseSide(unittest.TestCase):

    def setUp(self):
        self._fn = parse_side

    def test_empty_returns_empty(self):
        self.assertEqual(self._fn([]), [])

    def test_dict_format(self):
        items = [{"price": "100.5", "volume": "300"}, {"price": "99.0", "volume": "500"}]
        result = self._fn(items)
        self.assertEqual(result, [(100.5, 300), (99.0, 500)])

    def test_sequence_format(self):
        items = [[101.0, 200, "extra"], [102.5, 150]]
        result = self._fn(items)
        self.assertEqual(result, [(101.0, 200), (102.5, 150)])

    def test_mixed_dict_and_sequence(self):
        items = [{"price": 100.0, "volume": 100}, [101.0, 200]]
        result = self._fn(items)
        self.assertEqual(result, [(100.0, 100), (101.0, 200)])

    def test_malformed_dict_missing_key_skipped(self):
        items = [{"price": 100.0}, {"price": 101.0, "volume": 200}]
        result = self._fn(items)
        self.assertEqual(result, [(101.0, 200)])

    def test_malformed_sequence_too_short_skipped(self):
        items = [[100.0], [101.0, 200]]
        result = self._fn(items)
        self.assertEqual(result, [(101.0, 200)])

    def test_non_numeric_value_skipped(self):
        items = [{"price": "N/A", "volume": 100}, {"price": 100.0, "volume": 200}]
        result = self._fn(items)
        self.assertEqual(result, [(100.0, 200)])

    def test_none_item_skipped(self):
        items = [None, {"price": 100.0, "volume": 100}]
        result = self._fn(items)
        self.assertEqual(result, [(100.0, 100)])

    def test_float_coercion(self):
        items = [{"price": "100", "volume": "50"}]
        price, vol = self._fn(items)[0]
        self.assertIsInstance(price, float)
        self.assertIsInstance(vol, int)

    def test_large_list(self):
        items = [{"price": float(i), "volume": i * 10} for i in range(1, 21)]
        result = self._fn(items)
        self.assertEqual(len(result), 20)
        self.assertEqual(result[0], (1.0, 10))
        self.assertEqual(result[-1], (20.0, 200))


# ── partial pushes ───────────────────────────────────────────────────────────

class TestPartialPushes:
    def test_both_sides_pass_through(self):
        m = _m()
        assert m.merge(C, _lv((10.0, 5)), _lv((10.1, 7)), 0.0) == ([(10.0, 5)], [(10.1, 7)])

    def test_bid_only_push_keeps_the_cached_ask(self):
        """The bug this exists for: a bid-only push used to blank the asks."""
        m = _m()
        m.merge(C, _lv((10.0, 5)), _lv((10.1, 7)), 0.0)
        bids, asks = m.merge(C, _lv((10.02, 9)), [], 1.0)
        assert bids == [(10.02, 9)]
        assert asks == [(10.1, 7)], "ask side vanished on a bid-only push"

    def test_ask_only_push_keeps_the_cached_bid(self):
        m = _m()
        m.merge(C, _lv((10.0, 5)), _lv((10.1, 7)), 0.0)
        bids, asks = m.merge(C, [], _lv((10.09, 2)), 1.0)
        assert bids == [(10.0, 5)]
        assert asks == [(10.09, 2)]

    def test_empty_push_with_no_cache_returns_none(self):
        assert _m().merge(C, [], [], 0.0) is None

    def test_empty_push_with_cache_replays_the_cache(self):
        m = _m()
        m.merge(C, _lv((10.0, 5)), _lv((10.1, 7)), 0.0)
        assert m.merge(C, [], [], 1.0) == ([(10.0, 5)], [(10.1, 7)])

    def test_codes_are_isolated(self):
        m = _m()
        m.merge(C, _lv((10.0, 5)), _lv((10.1, 7)), 0.0)
        assert m.merge("US.SOXS", _lv((3.0, 1)), [], 0.0) == ([(3.0, 1)], [])


# ── staleness ────────────────────────────────────────────────────────────────

class TestStaleness:
    def test_cached_side_survives_inside_the_window(self):
        m = _m(stale_secs=10.0)
        m.merge(C, _lv((10.0, 5)), _lv((10.1, 7)), 0.0)
        _, asks = m.merge(C, _lv((10.0, 6)), [], 9.9)
        assert asks == [(10.1, 7)]

    def test_cached_side_dropped_past_the_window(self):
        m = _m(stale_secs=10.0)
        m.merge(C, _lv((10.0, 5)), _lv((10.1, 7)), 0.0)
        bids, asks = m.merge(C, _lv((10.0, 6)), [], 10.1)
        assert bids == [(10.0, 6)]
        assert asks == [], "stale ask cache should have been dropped"

    def test_refreshing_a_side_resets_its_clock(self):
        m = _m(stale_secs=10.0)
        m.merge(C, _lv((10.0, 5)), _lv((10.1, 7)), 0.0)
        m.merge(C, [], _lv((10.1, 8)), 9.0)          # ask refreshed at t=9
        _, asks = m.merge(C, _lv((10.0, 6)), [], 18.0)  # 9s since that refresh
        assert asks == [(10.1, 8)]

    def test_both_sides_going_stale_returns_none(self):
        m = _m(stale_secs=10.0)
        m.merge(C, _lv((10.0, 5)), _lv((10.1, 7)), 0.0)
        assert m.merge(C, [], [], 100.0) is None


# ── crossed book ─────────────────────────────────────────────────────────────

class TestCrossedBook:
    def test_fresh_bid_wins_stale_ask_dropped(self):
        """Observed live: a frozen bid kept being re-sent as 'fresh' while the
        ask legitimately moved below it."""
        m = _m()
        m.merge(C, _lv((10.0, 5)), _lv((10.1, 7)), 0.0)
        bids, asks = m.merge(C, _lv((10.2, 4)), [], 1.0)   # bid now above cached ask
        assert bids == [(10.2, 4)]
        assert asks == []

    def test_fresh_ask_wins_stale_bid_dropped(self):
        m = _m()
        m.merge(C, _lv((10.0, 5)), _lv((10.1, 7)), 0.0)
        bids, asks = m.merge(C, [], _lv((9.9, 3)), 1.0)
        assert bids == []
        assert asks == [(9.9, 3)]

    def test_both_fresh_and_crossed_is_skipped(self):
        """Neither side can be blamed, so no snapshot is reported at all."""
        assert _m().merge(C, _lv((10.2, 4)), _lv((10.1, 7)), 0.0) is None

    def test_cache_is_never_left_crossed(self):
        """The invariant that makes the "neither side fresh" branch dead code.

        Every merge resolves a cross before returning -- by dropping the stale
        side, or by returning None without touching what it could not judge.
        So a later push can never arrive to find a crossed cache, and the only
        way into that branch is both sides being fresh at once (covered above).
        Asserting the invariant directly is worth more than a test for the
        unreachable branch, and it fails loudly if the resolution is ever
        weakened.
        """
        m = _m()
        pushes = [
            (_lv((10.0, 5)), _lv((10.1, 7))),   # normal
            (_lv((10.2, 4)), []),               # bid-only, crosses
            ([], _lv((10.1, 7))),               # ask-only, crosses back
            (_lv((10.05, 1)), _lv((10.06, 1))), # both fresh, clean
            ([], []),                           # nothing
            (_lv((10.2, 9)), _lv((10.1, 9))),   # both fresh, crossed -> None
            ([], []),                           # nothing, after that skip
        ]
        for i, (b_, a_) in enumerate(pushes):
            got = m.merge(C, b_, a_, float(i))
            if got is None:
                continue
            bids, asks = got
            if bids and asks:
                assert max(p for p, _ in bids) < min(p for p, _ in asks), (
                    f"push {i} returned a crossed book: {bids} / {asks}")

    def test_touching_prices_are_not_crossed(self):
        """bid == ask is treated as crossed (>=), bid < ask is fine."""
        m = _m()
        assert m.merge(C, _lv((10.0, 5)), _lv((10.01, 7)), 0.0) is not None
        assert _m().merge(C, _lv((10.0, 5)), _lv((10.0, 7)), 0.0) is None

    def test_deep_book_uses_best_prices_not_first_entries(self):
        """best bid = max, best ask = min, regardless of list order."""
        m = _m()
        got = m.merge(C, _lv((9.8, 1), (10.0, 5)), _lv((10.3, 2), (10.1, 7)), 0.0)
        assert got is not None, "9.8/10.0 vs 10.1/10.3 is not crossed"


# ── reset ────────────────────────────────────────────────────────────────────

class TestReset:
    def test_reset_one_code(self):
        m = _m()
        m.merge(C, _lv((10.0, 5)), _lv((10.1, 7)), 0.0)
        m.merge("US.SOXS", _lv((3.0, 1)), _lv((3.1, 1)), 0.0)
        m.reset(C)
        assert m.merge(C, [], [], 1.0) is None
        assert m.merge("US.SOXS", [], [], 1.0) is not None

    def test_reset_all(self):
        m = _m()
        m.merge(C, _lv((10.0, 5)), _lv((10.1, 7)), 0.0)
        m.reset()
        assert m.merge(C, [], [], 1.0) is None

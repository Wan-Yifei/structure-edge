"""Where displayed volume goes when it leaves a price level.

The detectors in orderflow_detect.py all read a drop in displayed volume and
guess what caused it. detect_icebergs guesses "it was traded, and the refill
is hidden size"; detect_spoofs guesses from whether the quote moved. Neither
looks at the tape, which is the only place the answer is written down.

attribute_level_flow() reads it, and these tests pin the four cases that
have to come apart -- an execution, a cancellation, a hidden refill, and a
level that merely scrolled out of the reported depth -- plus the three ways
a trade can fail to belong to the level it names.

Every test states book_lag_secs. It has no default because there is no
defensible one: the book clock and the tape clock are different and the
offset between them changed sign between two consecutive trading days.
"""

import unittest
from datetime import datetime, timedelta

from analysis.orderflow_detect import attribute_level_flow

T0 = datetime(2026, 11, 2, 10, 0, 0)

BIDS = [(24.00, 1000), (23.99, 500), (23.98, 300)]
ASKS = [(24.01, 800), (24.02, 500), (24.03, 400)]

#: Price moved up a cent: the old best bid is no longer the best, and the
#: bottom of the reported window has scrolled past 23.98.
BIDS_UP = [(24.01, 700), (24.00, 1000), (23.99, 500)]


def _book(secs, bids=None, asks=None):
    bids = BIDS if bids is None else bids
    asks = ASKS if asks is None else asks
    levels = [{"side": "BID", "price": p, "volume": v} for p, v in bids]
    levels += [{"side": "ASK", "price": p, "volume": v} for p, v in asks]
    return (T0 + timedelta(seconds=secs), levels)


def _tick(secs, price, volume, direction="SELL"):
    return {"ts": T0 + timedelta(seconds=secs), "price": price,
            "volume": volume, "direction": direction}


def _run(books, ticks, lag=0.0, **kw):
    return attribute_level_flow(books, ticks, lag, **kw)


def _at(episodes, side, price):
    """The single episode for one level, so a test fails loudly on a split."""
    hits = [e for e in episodes if e.side == side and abs(e.price - price) < 1e-9]
    if len(hits) != 1:
        raise AssertionError(
            f"expected one {side} episode at {price}, got {len(hits)}")
    return hits[0]


def _drop(top_bid_vol):
    """The same two books every case below varies: the top bid changes size."""
    return [_book(0), _book(1, bids=[(24.00, top_bid_vol)] + BIDS[1:])]


class TestTheThreeComponents(unittest.TestCase):
    """dv = added - cancelled - traded, with the tape supplying traded."""

    def test_a_drop_the_tape_explains_is_an_execution(self):
        eps, audit = _run(_drop(600), [_tick(0.5, 24.00, 400)])
        ep = _at(eps, "BID", 24.00)
        self.assertEqual((ep.traded, ep.added, ep.cancelled), (400.0, 0.0, 0.0))
        self.assertEqual(audit.attributed, 400.0)

    def test_a_drop_with_no_trade_behind_it_is_a_cancellation(self):
        eps, _ = _run(_drop(600), [])
        ep = _at(eps, "BID", 24.00)
        self.assertEqual((ep.traded, ep.added, ep.cancelled), (0.0, 0.0, 400.0))

    def test_more_traded_than_displayed_lost_is_replenishment(self):
        """The iceberg signature: 400 went through, only 100 disappeared, so
        300 was put back while it was being hit."""
        eps, _ = _run(_drop(900), [_tick(0.5, 24.00, 400)])
        ep = _at(eps, "BID", 24.00)
        self.assertEqual((ep.traded, ep.added, ep.cancelled), (400.0, 300.0, 0.0))

    def test_only_the_net_of_add_and_cancel_is_observable(self):
        """Stated rather than hidden: a level that both gained and lost in one
        interval reports the net, and nothing recovers the two halves."""
        eps, _ = _run(_drop(1000), [_tick(0.5, 24.00, 400)])
        ep = _at(eps, "BID", 24.00)
        self.assertEqual((ep.traded, ep.added, ep.cancelled), (400.0, 400.0, 0.0))


class TestTheReferenceBook(unittest.TestCase):
    """A trade is matched against the book BEFORE it -- the liquidity that was
    standing there to be taken. Reading the book after instead is the heaviest
    error available here, and it is what a mis-set book_lag_secs simulates."""

    BOOKS = [_book(0), _book(1, bids=BIDS[1:])]   # 24.00 leaves the book

    def test_a_level_that_the_trade_removed_still_receives_it(self):
        eps, audit = _run(self.BOOKS, [_tick(0.5, 24.00, 1000)])
        self.assertEqual(audit.attributed, 1000.0)
        self.assertEqual(audit.off_grid, 0.0)
        self.assertEqual(_at(eps, "BID", 24.00).traded, 1000.0)

    def test_the_rank_is_read_from_the_same_book(self):
        """24.00 was the touch when the trade hit it, even though by the next
        book the touch is 23.99."""
        eps, audit = _run(self.BOOKS, [_tick(0.5, 24.00, 1000)])
        self.assertEqual(audit.traded_by_rank, {0: 1000.0})
        self.assertEqual(_at(eps, "BID", 24.00).traded_at_touch, 1000.0)

    def test_volume_behind_the_touch_is_separated_not_dropped(self):
        """42.9% of attributed volume on real data names a level behind the
        touch, and a print that large cannot have been filled there. It is
        kept, and kept apart."""
        eps, audit = _run(_drop(600), [_tick(0.5, 23.98, 300)])
        self.assertEqual(audit.traded_by_rank, {2: 300.0})
        ep = _at(eps, "BID", 23.98)
        self.assertEqual((ep.traded, ep.traded_at_touch), (300.0, 0.0))


class TestTradesThatDoNotBelongToTheLevel(unittest.TestCase):

    def test_a_subpenny_print_is_excluded_not_snapped(self):
        """Rounding 23.9987 onto 23.99 would credit a level with an execution
        that happened off-exchange. 45.5% of this tape by volume is sub-penny,
        so the error would not be a rounding detail."""
        eps, audit = _run(_drop(600), [_tick(0.5, 23.9987, 400)])
        self.assertEqual(audit.off_grid, 400.0)
        self.assertEqual(audit.attributed, 0.0)
        self.assertEqual(_at(eps, "BID", 23.99).traded, 0.0)
        # and the unexplained drop stays unexplained
        self.assertEqual(_at(eps, "BID", 24.00).cancelled, 400.0)

    def test_an_aggressive_buy_at_a_bid_price_is_set_aside(self):
        """A buyer cannot lift a bid. Either the book is stale against this
        print or the feed's side label describes someone's internal sheet
        rather than this book."""
        _, audit = _run(_drop(600), [_tick(0.5, 24.00, 400, direction="BUY")])
        self.assertEqual(audit.wrong_side, 400.0)
        self.assertEqual(audit.attributed, 0.0)

    def test_neutral_prints_never_reach_a_level(self):
        _, audit = _run(_drop(600), [_tick(0.5, 24.00, 400, direction="NEUTRAL")])
        self.assertEqual(audit.neutral, 400.0)
        self.assertEqual(audit.attributed, 0.0)

    def test_neutral_volume_sitting_on_a_level_is_counted_separately(self):
        """Excluding NEUTRAL is a choice with a cost, so the cost is visible:
        on real data 15.8% of the whole tape is NEUTRAL volume squarely on a
        displayed level -- the same order of magnitude as `attributed`."""
        _, audit = _run(_drop(600), [
            _tick(0.5, 24.00, 400, direction="NEUTRAL"),     # on a level
            _tick(0.5, 23.9987, 100, direction="NEUTRAL"),   # sub-penny
        ])
        self.assertEqual(audit.neutral, 500.0)
        self.assertEqual(audit.neutral_on_level, 400.0)

    def test_the_buckets_add_back_up_to_the_tape(self):
        ticks = [
            _tick(-5, 24.00, 10),                      # before the first book
            _tick(0.5, 24.00, 400),                    # attributed
            _tick(0.5, 23.9987, 70),                   # off grid
            _tick(0.5, 24.00, 30, direction="BUY"),    # wrong side
            _tick(0.5, 24.00, 20, direction="NEUTRAL"),
            _tick(99, 24.00, 5),                       # after the last book
        ]
        _, audit = _run(_drop(600), ticks)
        self.assertTrue(audit.balances())
        self.assertEqual(audit.tick_vol_total, 535.0)
        self.assertEqual(audit.attributed, 400.0)
        self.assertEqual(audit.outside_books, 15.0)
        # neutral_on_level is a subset of neutral, not a seventh bucket
        self.assertEqual(audit.neutral_on_level, 20.0)


class TestLevelsThatLeaveTheWindow(unittest.TestCase):
    """The feed reports a fixed number of levels a side. When price moves the
    far end stops being reported while the orders are still resting there."""

    def test_scrolling_off_the_bottom_is_not_a_cancellation(self):
        eps, audit = _run([_book(0), _book(1, bids=BIDS_UP)], [])
        ep = _at(eps, "BID", 23.98)
        self.assertEqual(ep.cancelled, 0.0)
        self.assertEqual(ep.n_occluded, 1)
        self.assertEqual(audit.n_occluded_level_intervals, 1)

    def test_trades_against_an_occluded_level_are_still_counted(self):
        """A sweep can reach 23.98 in the same interval that price moves away
        from it. On real data 7745 of 19196 episodes pass through this branch,
        so dropping its volume would be a silent loss at scale."""
        eps, _ = _run([_book(0), _book(1, bids=BIDS_UP)],
                      [_tick(0.5, 23.98, 300)])
        ep = _at(eps, "BID", 23.98)
        self.assertEqual(ep.traded, 300.0)
        self.assertEqual(ep.traded_at_touch, 0.0)

    def test_a_bid_above_the_old_best_is_genuinely_new(self):
        """The guard must not swallow real additions: 24.01 was not out of
        view before, it was not there."""
        eps, _ = _run([_book(0), _book(1, bids=BIDS_UP)], [])
        self.assertEqual(_at(eps, "BID", 24.01).added, 700.0)

    def test_scrolling_back_into_view_is_not_an_addition(self):
        eps, _ = _run([_book(0, bids=BIDS_UP), _book(1)], [])
        ep = _at(eps, "BID", 23.98)
        self.assertEqual(ep.added, 0.0)
        self.assertEqual(ep.n_occluded, 1)

    def test_an_episode_survives_leaving_and_rejoining_the_window(self):
        """Ending it would hand the level a fresh, smaller max_displayed every
        time price wandered past -- which is how 41% of the first version's
        flagged levels came to have three books or fewer behind them."""
        eps, _ = _run([_book(0), _book(1, bids=BIDS_UP), _book(2)], [])
        ep = _at(eps, "BID", 23.98)          # raises if it was split in two
        self.assertEqual(ep.n_occluded, 2)
        self.assertEqual(ep.max_displayed, 300.0)
        self.assertEqual(ep.cancelled, 0.0)


class TestIntervalsThatCannotBeRead(unittest.TestCase):

    def test_a_feed_pause_is_not_decomposed(self):
        """Over a 5 s pause trades are missed and the book has moved, so a
        drop across it cannot be attributed to anything. 147-204 of these a
        session, measured."""
        books = [_book(0), _book(30, bids=[(24.00, 600)] + BIDS[1:])]
        eps, audit = _run(books, [_tick(15, 24.00, 400)], max_gap_secs=5.0)
        self.assertEqual(audit.n_gap_intervals, 1)
        self.assertEqual(audit.in_gap, 400.0)
        self.assertEqual(audit.attributed, 0.0)
        for ep in eps:
            self.assertEqual((ep.added, ep.cancelled), (0.0, 0.0))

    def test_an_episode_does_not_span_a_pause(self):
        eps, _ = _run([_book(0), _book(30), _book(31)], [], max_gap_secs=5.0)
        self.assertEqual(len([e for e in eps if e.side == "BID"
                              and abs(e.price - 24.00) < 1e-9]), 2)


class TestTheTwoClocks(unittest.TestCase):
    """Books are stamped with datetime.now() on receipt, ticks with the feed's
    own row["time"]. The offset decides which interval a trade lands in, and
    therefore whether a drop reads as an execution or as a cancellation plus
    an unexplained addition. Calibrated per session it was +0.5 s on one
    trading day and -0.75 s on the next."""

    BOOKS = [
        (T0,                        _book(0)[1]),
        (T0 + timedelta(seconds=1), _book(0)[1]),
        (T0 + timedelta(seconds=2), _book(2, bids=[(24.00, 600)] + BIDS[1:])[1]),
    ]
    TICK = [_tick(0.5, 24.00, 400)]

    def test_assuming_zero_lag_splits_one_execution_in_two(self):
        eps, _ = _run(self.BOOKS, self.TICK, lag=0.0)
        ep = _at(eps, "BID", 24.00)
        self.assertEqual((ep.traded, ep.added, ep.cancelled), (400.0, 400.0, 400.0))

    def test_the_right_lag_makes_it_one_clean_execution(self):
        eps, _ = _run(self.BOOKS, self.TICK, lag=1.0)
        ep = _at(eps, "BID", 24.00)
        self.assertEqual((ep.traded, ep.added, ep.cancelled), (400.0, 0.0, 0.0))

    def test_the_lag_has_no_default(self):
        with self.assertRaises(TypeError):
            attribute_level_flow(self.BOOKS, self.TICK)


class TestEpisodeBookkeeping(unittest.TestCase):

    def test_max_displayed_is_the_peak_not_the_last(self):
        books = [
            _book(0, bids=[(24.00, 1000)]),
            _book(1, bids=[(24.00, 1500)]),
            _book(2, bids=[(24.00, 200)]),
        ]
        eps, _ = _run(books, [])
        self.assertEqual(_at(eps, "BID", 24.00).max_displayed, 1500.0)

    def test_net_change_does_not_move_with_the_sampling_rate(self):
        """added and cancelled are total variation and grow with how often
        the book is sampled; net_change and max_displayed do not. Dropping
        the middle book here leaves net_change untouched and shrinks the
        other two."""
        dense = [_book(0, bids=[(24.00, 1000)]),
                 _book(1, bids=[(24.00, 1500)]),
                 _book(2, bids=[(24.00, 200)])]
        sparse = [dense[0], dense[2]]
        a, _ = _run(dense, [])
        b, _ = _run(sparse, [])
        ea, eb = _at(a, "BID", 24.00), _at(b, "BID", 24.00)
        self.assertEqual(ea.net_change, eb.net_change)
        self.assertEqual(ea.net_change, -800.0)
        self.assertGreater(ea.added + ea.cancelled, eb.added + eb.cancelled)

    def test_two_books_are_needed(self):
        eps, audit = _run([_book(0)], [_tick(0.5, 24.00, 400)])
        self.assertEqual(eps, [])
        self.assertEqual(audit.outside_books, 400.0)
        self.assertTrue(audit.balances())


if __name__ == "__main__":
    unittest.main()

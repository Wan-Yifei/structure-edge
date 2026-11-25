"""Reusing a snapshot's imbalance zones must not reuse anything else.

detect_stacked_imbalance evaluates each snapshot independently, which is what
makes the result cacheable: once a column's last level has arrived, its zones
are fixed.  Regrouping the whole window every redraw cost 140 ms offline on a
240-column SOXL book (~280k entries), about 3.5x that inside the running
viewer, where it was the largest single block of GUI time.  With the cache the
same window takes 21 ms.

Everything here is a way that saving is wrong if the cache keeps one thing too
many, and each has a visible symptom:

  * bar_idx is NOT a property of the snapshot.  It comes from bucket_to_idx
    and shifts by one every time a column rolls off, so a cache holding it
    pins markers to x-positions the columns have left -- the same symptom as
    the coarsened-bucket bug _build_bucket_to_idx was written to fix, where
    two different moments drew at one column.
  * the newest column is still receiving levels.  Every push inside a column
    carries that column's timestamp, so its row set is final only once the
    next column opens.  Cached on first sight, the live edge -- the part being
    watched -- would stop updating for up to Col(s) seconds.
  * min_levels, imbalance_ratio and max_depth come from spin boxes.  A cache
    that outlived a turn of one would make the control look dead, which is
    worse than slow.
  * an entry per column, kept forever, is a leak in a window that runs all
    session.

The viewer side of this -- dropping the cache wherever _raw_snaps is dropped
-- is pinned in test_liq_hm_coalesce.py, where the window fixture already is.
"""

from datetime import datetime, timedelta

from analysis.orderflow_detect import (
    StackedImbalanceCache,
    detect_stacked_imbalance,
)

T0 = datetime(2026, 11, 24, 10, 0, 0)

# A book imbalanced on the ask side at every one of three ranks: 100/400,
# 90/350, 80/300. Three ranks is min_levels, so it reports exactly one zone.
BEARISH = [("BID", 10.00, 100.0), ("BID", 9.99, 90.0), ("BID", 9.98, 80.0),
           ("ASK", 10.01, 400.0), ("ASK", 10.02, 350.0), ("ASK", 10.03, 300.0)]

# The same shape with the sides swapped.
BULLISH = [(("ASK" if s == "BID" else "BID"), p, v) for s, p, v in BEARISH]


def _book(ts, levels=BEARISH):
    return [{"ts": ts, "side": s, "price": p, "volume": v}
            for s, p, v in levels]


def _run(snaps, bucket_to_idx, cache=None, live_ts=None, **kw):
    params = dict(min_levels=3, imbalance_ratio=3.0, min_vol=0.0, max_depth=10)
    params.update(kw)
    return detect_stacked_imbalance(
        snaps, bucket_to_idx, 0.01, 9.0, 100,
        cache=cache, live_ts=live_ts, **params)


def _norm(res):
    return sorted((b, round(lo, 6), round(hi, 6), d, round(r, 3))
                  for b, lo, hi, d, r in res)


class TestTheColumnIndexIsNeverCached:

    def test_a_roll_moves_the_markers(self):
        """Same snapshots, every column one index to the left."""
        snaps = _book(T0) + _book(T0 + timedelta(seconds=1))
        cache = StackedImbalanceCache()
        live = T0 + timedelta(seconds=2)

        before = _run(snaps, {T0: 1, T0 + timedelta(seconds=1): 2},
                      cache=cache, live_ts=live)
        after = _run(snaps, {T0: 0, T0 + timedelta(seconds=1): 1},
                     cache=cache, live_ts=live)

        assert [r[0] for r in _norm(before)] == [1, 2]
        assert [r[0] for r in _norm(after)] == [0, 1], (
            "the zones were reused with the column index they were first "
            "found at, so the markers stayed where those columns used to be")


class TestTheLiveColumnIsAlwaysReEvaluated:

    def test_levels_arriving_mid_column_still_count(self):
        """A column takes ~7 pushes before it closes; they share one ts.

        Here the first push is not imbalanced and the second one makes it so,
        which is the ordinary case -- the book fills in as the second passes.
        """
        cache = StackedImbalanceCache()
        b2i = {T0: 0}

        first = [{"ts": T0, "side": "BID", "price": 10.00, "volume": 100.0},
                 {"ts": T0, "side": "ASK", "price": 10.01, "volume": 100.0}]
        assert _run(first, b2i, cache=cache, live_ts=T0) == []

        full = first + [
            {"ts": T0, "side": "BID", "price": 9.99, "volume": 90.0},
            {"ts": T0, "side": "BID", "price": 9.98, "volume": 80.0},
            {"ts": T0, "side": "ASK", "price": 10.02, "volume": 350.0},
            {"ts": T0, "side": "ASK", "price": 10.03, "volume": 300.0},
        ]
        # 100/100 at rank 0 is not imbalanced, so the run is ranks 1-2 and
        # two ranks is under min_levels -- what matters is that the detector
        # looked again at all, which the volumes below prove.
        out = _run(full, b2i, cache=cache, live_ts=T0, min_levels=2)
        assert out, (
            "the column was cached from its first push, so every level that "
            "arrived later in the same second was ignored")

    def test_without_live_ts_the_live_column_freezes(self):
        """Why the argument exists, stated as behaviour rather than a comment.

        This is the misuse, not the contract: a caller whose snaps can still
        grow and who passes no live_ts gets a stale answer for the newest
        column. The viewer passes _col_ts[-1] for exactly this reason.
        """
        cache = StackedImbalanceCache()
        b2i = {T0: 0}
        thin = [{"ts": T0, "side": "BID", "price": 10.00, "volume": 100.0},
                {"ts": T0, "side": "ASK", "price": 10.01, "volume": 100.0}]
        _run(thin, b2i, cache=cache)
        assert _run(_book(T0), b2i, cache=cache) == [], (
            "the stale read this argument prevents did not happen, so the "
            "test no longer demonstrates anything -- check the premise "
            "before deleting it")
        # And the same call, told which column is live, sees the real book.
        assert _run(_book(T0), b2i, cache=cache, live_ts=T0)


class TestTurningAControlInvalidatesEverything:

    def test_a_tighter_ratio_is_obeyed_at_once(self):
        cache = StackedImbalanceCache()
        b2i = {T0: 0}
        snaps = _book(T0)

        assert _run(snaps, b2i, cache=cache, live_ts=None)
        assert _run(snaps, b2i, cache=cache, live_ts=None,
                    imbalance_ratio=5.0) == [], (
            "4x at every rank still reported a 5x imbalance: the zones from "
            "the previous setting were served again and the spin box reads "
            "as dead")

    def test_every_setting_counts_not_just_the_ratio(self):
        """min_levels, min_vol and max_depth change the answer too."""
        b2i = {T0: 0}
        snaps = _book(T0)
        for kw in ({"min_levels": 4}, {"max_depth": 2}, {"min_vol": 1000.0}):
            cache = StackedImbalanceCache()
            assert _run(snaps, b2i, cache=cache)
            assert _run(snaps, b2i, cache=cache, **kw) == _run(
                snaps, b2i, **kw), f"{kw} was not honoured on a warm cache"


class TestTheCacheStaysBounded:

    def test_columns_that_rolled_off_are_dropped(self):
        """A grid of 10 columns, 60 of them pushed through it."""
        cache = StackedImbalanceCache()
        snaps, col_ts = [], []
        for i in range(60):
            ts = T0 + timedelta(seconds=i)
            col_ts.append(ts)
            snaps.extend(_book(ts))
            if len(col_ts) > 10:
                col_ts.pop(0)
                snaps = [s for s in snaps if s["ts"] >= col_ts[0]]
            _run(snaps, {t: j for j, t in enumerate(col_ts)},
                 cache=cache, live_ts=col_ts[-1])
            assert len(cache.zones) <= 10, (
                f"{len(cache.zones)} entries for a 10-column grid at step {i}")

    def test_an_empty_book_clears_it(self):
        cache = StackedImbalanceCache()
        _run(_book(T0), {T0: 0}, cache=cache)
        assert cache.zones
        assert _run([], {T0: 0}, cache=cache) == []
        assert not cache.zones, (
            "a replay or a symbol change hands this an empty book; zones "
            "left behind would be served to the next one")


class TestItAgreesWithTheUncachedPath:
    """The premise, checked the only way that counts: against the old path.

    A static window is the one case where no cache can be wrong, so this
    rolls -- appending push by push, re-detecting mid-column, and rolling
    columns off -- and compares every single pass.
    """

    def test_pass_for_pass_over_a_rolling_window(self):
        cache = StackedImbalanceCache()
        snaps, col_ts = [], []
        for i in range(40):
            ts = T0 + timedelta(seconds=i)
            col_ts.append(ts)
            if len(col_ts) > 8:
                col_ts.pop(0)
                snaps = [s for s in snaps if s["ts"] >= col_ts[0]]
            # Three pushes inside the column, the middle one flipping the
            # book, so the live column's answer changes within one second.
            for levels in (BEARISH[:2], BULLISH, BEARISH):
                snaps.extend(_book(ts, levels))
                b2i = {t: j for j, t in enumerate(col_ts)}
                assert _norm(_run(snaps, b2i, cache=cache,
                                  live_ts=col_ts[-1])) == _norm(
                    _run(snaps, b2i)), f"diverged at column {i}"

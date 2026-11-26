"""What the raw-snapshot buffer owes its readers, pinned before it changes.

The buffer behind `_raw_snaps` is being replaced: a flat list of one dict per
book level, rebuilt by list comprehension on every column roll, costs ~193 MB
of resident memory at steady state (measured on a real RTH window) and makes four separate full-window rescans
per second.  Nothing about the OUTPUT is supposed to move.

So this file pins the output and not the container.  Every assertion here is
phrased against something a reader of the buffer can observe -- what the three
detectors return, and what `has_book` decides -- rather than against the list
itself, so the same file runs unchanged on both sides of the refactor.  Tests
that reach into `_raw_snaps` directly live in test_liq_hm_coalesce.py and
test_liq_hm_clock.py and will need rewriting; these should not.

WHAT THIS DOES AND DOES NOT ESTABLISH.  The expected values below were taken
from the code as it stands.  That makes them a record of CURRENT behaviour,
which is the right reference for "did the refactor change anything" and is NOT
evidence that the current behaviour is correct.  Where a value looks wrong,
the place to argue about it is the detector's own test file.

The four properties, and what breaks if the new buffer drops one:

  * arrival order within a column.  Every push inside one column shares that
    column's timestamp, so `ts` alone does not order them; iceberg and spoof
    sort on `(ts, seq, ...)` and a tie falling through to volume turns the
    arrival order into an ascending ramp -- a level that only ever grew,
    60 -> 300 -> 700, then reads as refreshes.  That was a real bug; the
    replacement keeps rows in arrival order inside a column, which has to
    give the same sort.
  * the trim boundary.  Rows older than `_col_ts[0]` are dropped, and rows at
    exactly `_col_ts[0]` are kept.  Off by one column and the oldest column's
    markers vanish or a rolled-off column's markers draw at a borrowed index.
  * fill columns record nothing.  `_push_column(is_fill=True)` repeats the
    last known book so the heatmap stays continuous; recording it would
    manufacture refreshes for iceberg and spoof out of a book that never
    changed.
  * a replayed window has no book.  `_on_bulk_ready` deliberately leaves the
    buffer empty, and `has_book` is what keeps the three book detectors from
    running over columns a bucket apart, where the signal they look for is not
    there to find.
"""

import os
from collections import deque
from datetime import datetime, timedelta

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6.QtWidgets")

from PyQt6.QtWidgets import QApplication   # noqa: E402

T0 = datetime(2026, 11, 24, 14, 0, 0)
MID = 141.0


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture()
def win(qapp):
    from analysis.liq_hm_window import LiqHmWindow

    w = LiqHmWindow()
    w.set_code("US.SOXL")
    # The column timer would roll columns of its own; every test here drives
    # the handlers directly so the column count is exact.
    w._timer.stop()
    # _on_push returns immediately unless the window believes it is live, so
    # without this the whole in-column write path is a silent no-op and every
    # assertion below would be measuring _push_column alone.  _drive asserts
    # that it really did record, so a future change to the gate fails loudly
    # here instead of quietly halving what is under test.
    w._live = True
    yield w
    from qt_window_teardown import destroy_window
    destroy_window(w)


def _rows(mid=MID, scale=1.0):
    """A push payload, shaped like the feed's: no "ts" of its own.

    Imbalanced on the ask side at all three ranks, so the stacked-imbalance
    detector has something to find in every column.
    """
    return ([{"side": "BID", "price": round(mid - 0.01 * i, 2),
              "volume": (100 + i) * scale} for i in range(3)]
            + [{"side": "ASK", "price": round(mid + 0.01 * (i + 1), 2),
                "volume": (400 + i) * scale} for i in range(3)])


def _drive(win, n_cols=4, pushes_per_col=3):
    """Open `n_cols` columns, each carrying `pushes_per_col` book updates.

    Both write paths are exercised on purpose: `_push_column` records the
    first book of a column, `_on_push` records the rest of them against the
    column already open.  A refactor that keeps one and loses the other would
    pass a test that drove only one.

    The price band is set once up front.  Every book here sits at the same
    mid, so the band is never rebuilt mid-session -- a rebuild empties the
    buffer, which is a path worth testing but not inside a fixture that other
    tests read the buffer's contents from.
    """
    win._maybe_init_price_range(_rows())
    for c in range(n_cols):
        ts = T0 + timedelta(seconds=c)
        before = win._snap_rows
        win._push_column(_rows(scale=1.0 + 0.1 * c), ts=ts)
        opened = win._snap_rows - before
        for p in range(1, pushes_per_col):
            # Volume FALLS across the pushes inside a column, so sorting by
            # volume instead of by arrival order reverses them.  A rising
            # ramp would produce the same order under either rule and hide
            # exactly the bug `seq` exists to prevent.
            win._on_push(win._code, _rows(scale=1.0 + 0.1 * c - 0.02 * p),
                         now=0.0)
        if pushes_per_col > 1:
            assert win._snap_rows - before > opened, (
                "_on_push recorded nothing -- the in-column write path is "
                "not being exercised, so these tests cover half of what "
                "their names claim")
    return win


def _simbs(win):
    from analysis.orderflow_detect import detect_stacked_imbalance
    return detect_stacked_imbalance(
        win._flat_snaps(), win._build_bucket_to_idx(), win._bin_size,
        win._price_min, win._bid_grid.shape[1],
        min_levels=3, imbalance_ratio=3.0, max_depth=10)


def _dip_rows(bid_vol):
    """A book whose best bid carries `bid_vol`; everything else is fixed.

    Only the top bid moves, so the iceberg detector has exactly one level
    with a history and the best bid / best ask it bounds by never shift.
    """
    return ([{"side": "BID", "price": MID, "volume": float(bid_vol)},
             {"side": "BID", "price": MID - 0.05, "volume": 100.0},
             {"side": "ASK", "price": MID + 0.01, "volume": 400.0},
             {"side": "ASK", "price": MID + 0.06, "volume": 400.0}])


def _icebergs(win):
    from analysis.orderflow_detect import detect_icebergs
    return detect_icebergs(
        win._flat_snaps(), win._build_bucket_to_idx(), win._bin_size,
        win._price_min, win._bid_grid.shape[1],
        min_refreshes=1, vol_threshold=0.0,
        best_bid=win._best_bid, best_ask=win._best_ask, col_secs=1)


class TestArrivalOrderSurvivesInsideAColumn:
    """Rows in one column keep the order the feed delivered them in.

    Stated through the iceberg detector because that is who pays for getting
    it wrong: it walks a level's history looking for drop-then-refresh, so it
    is the one reader whose answer depends on the order and not just the set.

    Two things have to be true at once for this to test anything, and the
    first two drafts each got one of them wrong.

    The rows must share a TIMESTAMP.  The sort key is `(ts, seq, ...)`, so
    across columns `ts` alone already orders them and `seq` is never
    consulted -- a draft that put the three books in three columns passed
    even with `seq` replaced by volume in the detector.  Everything below
    goes into ONE column: `_push_column` opens it and `_on_push` adds to it,
    which is how the live feed fills a column and the only arrangement where
    arrival order is the only thing separating the rows.

    And the shape must dip.  A monotonic run reads the same sorted either
    way.  1000 -> 200 -> 900 is a drop and a refresh in arrival order;
    200 -> 900 -> 1000, which is what sorting by volume makes of the same
    three rows, is neither.
    """

    def _one_column(self, win, vols):
        win._maybe_init_price_range(_dip_rows(vols[0]))
        win._push_column(_dip_rows(vols[0]), ts=T0)
        for v in vols[1:]:
            win._on_push(win._code, _dip_rows(v), now=0.0)
        assert len({r["ts"] for r in win._flat_snaps()}) == 1, (
            "the books landed in more than one column, so `ts` orders them "
            "and this proves nothing about arrival order")
        return win

    def test_a_dip_and_a_refill_is_found(self, win):
        self._one_column(win, (1000, 200, 900))
        found = _icebergs(win)
        assert found, (
            "1000 -> 200 -> 900 inside one column reported no iceberg; "
            "the detector cannot see the arrival order")
        assert any(abs(p - MID) < win._bin_size for _a, _b, p, _n in found), (
            f"the iceberg was not found at the level that dipped: {found}")

    def test_the_same_rows_in_rising_order_are_not(self, win):
        """The control, and the reason the test above is worth having.

        The identical three books in ascending-volume order -- the order a
        sort that fell through to volume would impose on the dip -- must find
        nothing.  If this one also found an iceberg, the test above would be
        passing on the SET of rows rather than their order, and would go on
        passing straight through the bug it names.
        """
        self._one_column(win, (200, 900, 1000))
        assert _icebergs(win) == [], (
            "a monotonic rise reported an iceberg, so the dip test above "
            "proves nothing about ordering")


class TestTheTrimBoundaryIsTheOldestColumn:

    """Driven past the window's OWN column cap, not a lowered one.

    Lowering `_max_cols_spin` would be the quick way to force a roll, and it
    is the wrong way twice: the spin box clamps at 60 (setRange(60, 1440)), so
    a smaller value is silently ignored, and changing it fires
    _on_max_cols_changed, which starts a prefill worker that goes to the
    database -- it hung this file for two minutes before the cap was read
    instead of set.
    """

    def test_the_oldest_columns_own_rows_are_kept(self, win):
        """Whatever the oldest column holds, it still holds after the roll.

        Phrased as "if it has rows, they are its own" and not as "it has
        rows": a column legitimately holds none.  In a real RTH window 135
        of 720 columns were empty -- every second the feed did not move
        produces one -- so an assertion that the oldest column is non-empty
        would be testing that `_drive` fills every column, not that the trim
        stops where it should.
        """
        _drive(win, n_cols=win._max_cols_spin.value() + 3, pushes_per_col=2)
        oldest = win._col_ts[0]
        first_col = list(win._snap_cols)[0]
        assert all(r["ts"] == oldest for r in first_col), (
            "the oldest column holds rows stamped for a different column")
        assert win._snap_rows == sum(len(c) for c in win._snap_cols), (
            "the row count disagrees with the rows actually held")

    def test_nothing_older_than_the_grid_survives(self, win):
        _drive(win, n_cols=win._max_cols_spin.value() + 3, pushes_per_col=2)
        oldest = win._col_ts[0]
        stale = [r for r in win._flat_snaps() if r["ts"] < oldest]
        assert stale == [], f"{len(stale)} rows older than the grid were kept"

    def test_the_roll_actually_happened(self, win):
        """The premise of the two above.

        Without this, a cap that stopped being enforced would leave both of
        them trivially true: nothing is ever older than `_col_ts[0]` if
        `_col_ts[0]` is still the first column ever pushed.
        """
        cap = win._max_cols_spin.value()
        _drive(win, n_cols=cap + 3, pushes_per_col=2)
        assert len(win._col_ts) == cap, (
            f"{len(win._col_ts)} columns for a cap of {cap}")
        assert win._col_ts[0] > T0, "the grid never rolled"


class TestAFillColumnRecordsNothing:

    def test_the_repeat_is_painted_but_not_recorded(self, win):
        """A fill column exists on the grid and contributes no rows.

        The heatmap stays continuous either way; what changes is whether
        iceberg and spoof see a refresh that never happened.
        """
        _drive(win, n_cols=2, pushes_per_col=2)
        cols_before = len(win._col_ts)
        marks_before = win._snap_rows
        win._push_column(_rows(), ts=T0 + timedelta(seconds=9), is_fill=True)
        assert len(win._col_ts) == cols_before + 1, "no column was added"
        assert win._snap_rows == marks_before, (
            "a fill column recorded rows")


class TestTheDetectorsOutputIsUnchanged:
    """The whole point of the refactor: these numbers must not move.

    A golden, and deliberately so.  It says "the same books produce the same
    zones", which is the only claim a refactor is allowed to make.  It says
    nothing about whether those zones are the right ones.
    """

    def test_stacked_imbalance_over_a_scripted_session(self, win):
        _drive(win, n_cols=4, pushes_per_col=3)
        simbs = _simbs(win)
        assert simbs, "the scripted session produced no zones to compare"
        # One bearish zone per column: the ask side leads 400/100 at every
        # rank, and every push in a column scales both sides together, so the
        # ratio -- and therefore the zone -- is the same in all of them.
        assert len(simbs) == len(win._col_ts), (
            f"{len(simbs)} zones for {len(win._col_ts)} columns")
        assert {d for _i, _lo, _hi, d, _r in simbs} == {"ASK"}, (
            "an ask-dominant book produced a non-ASK zone")
        assert sorted(i for i, *_ in simbs) == list(range(len(win._col_ts))), (
            "zones are not one per column, left to right")


class TestTheColumnarEntryMatchesTheFlatOne:
    """Both entries of detect_stacked_imbalance, on the same books.

    The per-column entry exists to skip columns it has already evaluated.
    The flat entry it replaced walked every row.  While both are in the
    source they can be compared directly, which is a stronger statement than
    any golden: same input, element-for-element same output.

    The sequence below deliberately contains a repeated timestamp.  When the
    feed has not moved, the column timer repeats the last known book as a
    fill column carrying the PREVIOUS column's stamp, so two columns share a
    `ts` -- and this detector keys on `ts`, not on column index.  A per-column
    implementation that keyed on the index would split such a pair into two
    half-books, changing both the zone count and the ratios.  Without this
    segment the test passes either way.
    """

    def _twinned(self, win):
        """Three columns where the third repeats the second's timestamp."""
        win._maybe_init_price_range(_rows())
        t1 = T0
        t2 = T0 + timedelta(seconds=1)
        win._push_column(_rows(scale=1.0), ts=t1)
        win._on_push(win._code, _rows(scale=1.1), now=0.0)
        win._push_column(_rows(scale=1.2), ts=t2)
        # The fill column: same stamp as the one before it, no rows of its
        # own, and then the feed comes back and appends to it.
        win._push_column(_rows(scale=1.2), ts=t2, is_fill=True)
        win._on_push(win._code, _rows(scale=1.3), now=0.0)
        assert len(win._col_ts) > len(set(win._col_ts)), (
            "no two columns share a timestamp, so this proves nothing about "
            "keying by ts rather than by column")
        return win

    def _both(self, win):
        from analysis.orderflow_detect import detect_stacked_imbalance
        args = (win._build_bucket_to_idx(), win._bin_size, win._price_min,
                win._bid_grid.shape[1])
        kw = dict(min_levels=3, imbalance_ratio=3.0, max_depth=10)
        flat = detect_stacked_imbalance(win._flat_snaps(), *args, **kw)
        cols = detect_stacked_imbalance(None, *args,
                                        snap_cols=win._snap_cols, **kw)
        return flat, cols

    def test_they_agree_on_a_window_with_a_repeated_timestamp(self, win):
        flat, cols = self._both(self._twinned(win))
        assert flat, "the scripted window produced no zones to compare"
        assert cols == flat, (
            f"per-column entry returned {cols}, flat entry returned {flat}")

    def test_they_agree_on_an_ordinary_window(self, win):
        flat, cols = self._both(_drive(win, n_cols=5, pushes_per_col=3))
        assert flat, "the scripted window produced no zones to compare"
        assert cols == flat, (
            f"per-column entry returned {cols}, flat entry returned {flat}")

    def test_an_empty_book_clears_the_cache_through_either_entry(self, win):
        """A replay holds one empty list per column.

        The guard at the top of the detector has to read that as "no book"
        -- the container is not empty, its columns are -- or the cache it is
        supposed to clear survives into the next symbol.
        """
        from analysis.orderflow_detect import (
            StackedImbalanceCache, detect_stacked_imbalance)
        _drive(win, n_cols=3, pushes_per_col=2)
        args = (win._build_bucket_to_idx(), win._bin_size, win._price_min,
                win._bid_grid.shape[1])
        cache = StackedImbalanceCache()
        assert detect_stacked_imbalance(None, *args, snap_cols=win._snap_cols,
                                        cache=cache), "nothing was cached"
        assert cache.zones, "the cache stayed empty, so clearing is untestable"
        empty = deque([[] for _ in win._col_ts])
        assert detect_stacked_imbalance(None, *args, snap_cols=empty,
                                        cache=cache) == []
        assert not cache.zones, (
            "columns that hold no rows were read as a book, and the cache "
            "survived")


def _check(win, where):
    """The invariant, in the two halves that can drift independently."""
    assert len(win._snap_cols) == len(win._col_ts), (
        f"{where}: {len(win._snap_cols)} snapshot columns against "
        f"{len(win._col_ts)} grid columns")
    assert win._snap_rows == sum(len(c) for c in win._snap_cols), (
        f"{where}: _snap_rows says {win._snap_rows}, the columns hold "
        f"{sum(len(c) for c in win._snap_cols)}")


class TestTheColumnsStayInStepWithTheGrid:
    """One snapshot column per grid column, and a row count that agrees.

    This is what the whole structure rests on: the trim pops from both ends
    of the pair at once, so `_snap_cols[i]` has to be `_col_ts[i]` on every
    path, and `has_book` has to be able to trust `_snap_rows`.  The code says
    so in a comment -- "everything that writes _col_ts must write here too"
    -- and until this class existed nothing checked it.  Deleting the slot
    that _load_history appends, which is the exact mistake the design review
    caught by reading, left all 226 liq_hm tests green.

    Every path that touches _col_ts is driven here, including the two that
    do not go through _push_column.
    """

    def test_an_ordinary_column(self, win):
        win._maybe_init_price_range(_rows())
        win._push_column(_rows(), ts=T0)
        _check(win, "after _push_column")

    def test_a_fill_column(self, win):
        """Records no rows and must still take a slot.

        Without the slot the deque runs short of _col_ts by one per quiet
        second, and every later trim pops a column that is still on screen.
        """
        win._maybe_init_price_range(_rows())
        win._push_column(_rows(), ts=T0)
        win._push_column(_rows(), ts=T0 + timedelta(seconds=1), is_fill=True)
        _check(win, "after a fill column")
        assert win._snap_rows == len(_rows()), "the fill column recorded rows"

    def test_pushes_into_an_open_column(self, win):
        _drive(win, n_cols=3, pushes_per_col=4)
        _check(win, "after _on_push")

    def test_a_roll(self, win):
        _drive(win, n_cols=win._max_cols_spin.value() + 5, pushes_per_col=2)
        _check(win, "after rolling off the left edge")

    def test_a_replay(self, win, monkeypatch):
        """_load_history appends to _col_ts without going through
        _push_column -- the one write site that a reading of the code missed,
        and the one no other test reaches.

        The real function is driven; only the database behind it is replaced,
        because the path under test is the column bookkeeping and not the
        query.  An earlier version of this test called two methods that do
        not exist, guarded by hasattr, and therefore asserted nothing at all.
        """
        _drive(win, n_cols=3, pushes_per_col=2)
        snaps = [[dict(r, ts=T0 + timedelta(minutes=i)) for r in _rows()]
                 for i in range(4)]

        class _Store:
            def __init__(self, *a, **k):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def earliest_ts(self, code):
                return snaps[0][0]["ts"]

            def latest_ts(self, code):
                return snaps[-1][0]["ts"]

            def bucketed_snapshots(self, code, start, end, bucket):
                return snaps

        import feeds.order_book_store as obs
        monkeypatch.setattr(obs, "OrderBookStore", _Store)
        win._load_history(600)

        assert len(win._col_ts) == len(snaps), (
            f"the replay painted {len(win._col_ts)} columns, not {len(snaps)}")
        _check(win, "after a replay")
        assert win._snap_rows == 0, (
            "a replay recorded book rows; the detectors it feeds look for a "
            "level refreshing between consecutive books, and replayed "
            "columns are a bucket apart")

    def test_clear_columns(self, win):
        _drive(win, n_cols=3, pushes_per_col=2)
        win._clear_columns()
        _check(win, "after _clear_columns")
        assert win._snap_rows == 0

    def test_reset_grid(self, win):
        _drive(win, n_cols=3, pushes_per_col=2)
        win._reset_grid()
        _check(win, "after _reset_grid")
        assert win._snap_rows == 0


class TestAReplayedWindowHasNoBook:
    """Columns without rows must not switch the book detectors back on.

    A replay fills the deque with one empty list per column, so the
    container is full while the book is empty.  Asking `bool(self._snap_cols)`
    instead of the row count reads that as "there is a book" and runs
    iceberg, spoof and imbalance over columns a bucket apart, where the
    refresh they look for cannot be there.

    The output happens to survive that mistake -- all three return nothing
    when handed no rows -- which is why it needs its own test: no assertion
    about markers can see it.  This one watches whether they are CALLED.
    """

    def test_the_detectors_are_not_called(self, win, monkeypatch):
        import analysis.orderflow_detect as od
        called = []
        for name in ("detect_icebergs", "detect_spoofs",
                     "detect_stacked_imbalance"):
            monkeypatch.setattr(
                od, name,
                lambda *a, _n=name, **k: called.append(_n) or [])
        for cb in (win._ice_cb, win._spoof_cb, win._simb_cb):
            cb.setChecked(True)

        _drive(win, n_cols=3, pushes_per_col=2)
        win._snap_cols = deque([[] for _ in win._col_ts])
        win._snap_rows = 0
        called.clear()
        win._redraw_orderflow_markers()

        assert called == [], (
            f"{called} ran over columns that hold no book rows")

    def test_they_are_called_when_there_is_a_book(self, win, monkeypatch):
        """The control: without it the test above passes if the three boxes
        simply never reach the detectors at all."""
        import analysis.orderflow_detect as od
        called = []
        for name in ("detect_icebergs", "detect_spoofs",
                     "detect_stacked_imbalance"):
            monkeypatch.setattr(
                od, name,
                lambda *a, _n=name, **k: called.append(_n) or [])
        for cb in (win._ice_cb, win._spoof_cb, win._simb_cb):
            cb.setChecked(True)

        _drive(win, n_cols=3, pushes_per_col=2)
        called.clear()
        win._redraw_orderflow_markers()

        assert sorted(called) == ["detect_icebergs",
                                  "detect_spoofs",
                                  "detect_stacked_imbalance"], called

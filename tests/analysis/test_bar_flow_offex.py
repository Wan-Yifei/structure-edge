"""Off-exchange flow rides along in _bar_flow's cached pass.

_draw_cvd used to re-sum off_buy/off_sell over every price level of every
bucket on every frame. Measured across three tick days that was 59 / 109 / 66
ms a frame, scaling with the day's distinct price count -- and it ran BEFORE
the `self._ind("offex")` test, so switching the off-exchange line off saved
the drawing and none of the arithmetic.

They live in the same per-price dict as buy/sell, so _bar_flow is already
walking them; two more accumulators cost ~0.01ms on the incremental path.

What makes caching them sound is that _dirty_bars compares the IDENTITY of
each bar's price dict, not just its OHLC, so a replaced bucket marks its bar
dirty and off_* is inside the replaced dict. The loaders only ever swap whole
buckets. These tests pin that property rather than assuming it.

The .get() on off_buy/off_sell is load-bearing and has its own test: live
buckets come from _live_ticks, whose factory carries only buy/sell/neutral.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6.QtWidgets")

from PyQt6.QtWidgets import QApplication        # noqa: E402

import analysis.trade_viewer_qt as tv           # noqa: E402

T0 = datetime(2026, 11, 11, 9, 30)
COLS = ["time_key", "open", "high", "low", "close", "volume"]


def klines(n: int = 6) -> pd.DataFrame:
    rows = [((T0 + timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M:%S"),
             54.0, 54.5, 53.8, 54.2, 1000.0 + i) for i in range(n)]
    return pd.DataFrame(rows, columns=COLS)


def bin_(buy=0, sell=0, off_buy=0, off_sell=0) -> dict:
    """A price bin shaped like _new_tick_bin's, i.e. a DB-loaded one."""
    b = tv._new_tick_bin()
    b["buy"], b["sell"] = buy, sell
    b["off_buy"], b["off_sell"] = off_buy, off_sell
    return b


def live_bin(buy=0, sell=0) -> dict:
    """A price bin shaped like _live_ticks' factory: no off_* keys at all."""
    return {"buy": buy, "sell": sell, "neutral": 0}


def buckets_for(viewer, df: pd.DataFrame, spec: dict) -> dict:
    """spec maps bar index -> {price: bin}.

    Keys come from the viewer's own _bucket_keys, never rebuilt here: a
    kline's time_key is the bar's END, so the tick bucket is one candle
    length earlier. Rebuilding it got that wrong and every bucket missed,
    which made the first version of these tests compare two sets of zeros
    and pass.
    """
    keys = viewer._bucket_keys(df, 1)
    return {keys[i]: b for i, b in spec.items() if keys[i] is not pd.NaT}


def loop_off_delta(viewer, df, buckets, cm):
    """The loop that was removed, kept here as the reference implementation."""
    n = len(df)
    off_delta = np.zeros(n, dtype=float)
    off_seen = np.zeros(n, dtype=bool)
    for i, bk in enumerate(viewer._bucket_keys(df, cm)):
        if bk is pd.NaT:
            continue
        pd_ = buckets.get(bk)
        if not pd_:
            continue
        ob = sum(pd_[p].get("off_buy", 0) for p in pd_)
        os_ = sum(pd_[p].get("off_sell", 0) for p in pd_)
        if ob or os_:
            off_seen[i] = True
        off_delta[i] = ob - os_
    return off_delta, off_seen


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app
    app.processEvents()


@pytest.fixture(scope="module")
def viewer(qapp):
    v = tv.TradeViewerQt()
    v.resize(1000, 700)
    yield v
    v.close()
    qapp.processEvents()


@pytest.fixture()
def fresh(viewer):
    viewer._flow_cache = None
    viewer._prev_frame = None
    viewer._frame_dirty = None
    yield viewer
    viewer._flow_cache = None
    viewer._prev_frame = None
    viewer._frame_dirty = None


class TestItMatchesTheLoopItReplaced:

    def test_a_mixed_frame(self, fresh):
        df = klines()
        bk = buckets_for(fresh, df, {
            1: {54.0: bin_(buy=10, sell=2, off_buy=100, off_sell=40),
                54.1: bin_(buy=5, sell=5, off_buy=7, off_sell=0)},
            3: {54.2: bin_(buy=1, sell=9, off_sell=250)},
            4: {54.3: bin_(buy=3, sell=3)},        # on-exchange only
        })
        *_, off_b, off_s = fresh._bar_flow(df, bk, 1)
        want_delta, want_seen = loop_off_delta(fresh, df, bk, 1)

        assert np.array_equal(off_b - off_s, want_delta)
        assert np.array_equal((off_b != 0) | (off_s != 0), want_seen)

    def test_a_bar_that_nets_to_zero_is_still_seen(self, fresh):
        """Why off_seen is its own question: 100 bought against 100 sold is
        real data, and `off_delta == 0` cannot tell it from no data."""
        df = klines()
        bk = buckets_for(fresh, df, {2: {54.0: bin_(off_buy=100, off_sell=100)}})
        *_, off_b, off_s = fresh._bar_flow(df, bk, 1)
        seen = (off_b != 0) | (off_s != 0)
        assert (off_b - off_s)[2] == 0.0
        assert seen[2], "a netted bar was mistaken for an empty one"

    def test_a_frame_with_no_off_exchange_data(self, fresh):
        df = klines()
        bk = buckets_for(fresh, df, {1: {54.0: bin_(buy=10, sell=2)}})
        *_, off_b, off_s = fresh._bar_flow(df, bk, 1)
        assert not ((off_b != 0) | (off_s != 0)).any()

    def test_buy_and_sell_are_unchanged(self, fresh):
        df = klines()
        bk = buckets_for(fresh, df, {1: {54.0: bin_(buy=10, sell=2, off_buy=99)}})
        buy, sell, *_ = fresh._bar_flow(df, bk, 1)
        assert buy[1] == 10.0 and sell[1] == 2.0


class TestLiveBucketsHaveNoOffColumns:
    """_live_ticks' factory is {"buy", "sell", "neutral"} and _render copies
    it straight into buckets, so a plain subscript raises KeyError on the
    Live render path -- in a GUI slot, which ends the process."""

    def test_a_plain_subscript_would_raise(self):
        with pytest.raises(KeyError):
            live_bin(buy=5)["off_buy"]

    def test_bar_flow_handles_them(self, fresh):
        df = klines()
        bk = buckets_for(fresh, df, {1: {54.0: live_bin(buy=10, sell=2)}})
        buy, sell, _, off_b, off_s = fresh._bar_flow(df, bk, 1)
        assert buy[1] == 10.0 and sell[1] == 2.0
        assert off_b[1] == 0.0 and off_s[1] == 0.0

    def test_a_mix_of_both_shapes(self, fresh):
        df = klines()
        bk = buckets_for(fresh, df, {
            1: {54.0: live_bin(buy=10), 54.1: bin_(buy=1, off_buy=70)},
        })
        *_, off_b, off_s = fresh._bar_flow(df, bk, 1)
        assert off_b[1] == 70.0

    def test_buy_and_sell_subscript_is_safe_on_both(self):
        """Why buy/sell may subscript while off_* may not: both factories
        carry those three keys, and there is no third source."""
        for b in (live_bin(), tv._new_tick_bin()):
            assert b["buy"] == 0 and b["sell"] == 0 and b["neutral"] == 0


class TestTheCacheKeepsOffExchangeHonest:
    """The incremental path is only sound because _dirty_bars marks a bar
    dirty when its price dict is a different OBJECT. These drive the real
    _dirty_bars rather than setting _frame_dirty by hand."""

    @staticmethod
    def _frame(viewer, df, bk):
        """One frame, through the same entry point _render uses."""
        viewer._frame_dirty = viewer._begin_frame(df, bk, 1)
        return viewer._bar_flow(df, bk, 1)

    @staticmethod
    def _frame_without_flow(viewer, df, bk):
        """A frame where every overlay wanting _bar_flow was switched off."""
        viewer._frame_dirty = viewer._begin_frame(df, bk, 1)

    def test_a_replaced_bucket_is_picked_up(self, fresh):
        df = klines()
        bk = buckets_for(fresh, df, {1: {54.0: bin_(off_buy=10)}})
        self._frame(fresh, df, bk)

        # What the loaders do: swap the whole bucket for a new object.
        key = next(iter(bk))
        bk[key] = {54.0: bin_(off_buy=999)}
        *_, off_b, _ = self._frame(fresh, df, bk)

        assert off_b[1] == 999.0, "a replaced bucket was served from cache"

    def test_an_untouched_bar_is_served_from_cache(self, fresh):
        df = klines()
        bk = buckets_for(fresh, df, {1: {54.0: bin_(off_buy=10)}})
        self._frame(fresh, df, bk)
        *_, off_b, _ = self._frame(fresh, df, bk)
        assert off_b[1] == 10.0

    def test_a_grown_frame_keeps_the_earlier_bars(self, fresh):
        df = klines(5)
        bk = buckets_for(fresh, df, {1: {54.0: bin_(off_buy=10)}})
        self._frame(fresh, df, bk)

        df2 = klines(6)
        bk2 = dict(bk)
        bk2.update(buckets_for(fresh, df2, {5: {54.4: bin_(off_sell=33)}}))
        *_, off_b, off_s = self._frame(fresh, df2, bk2)

        assert off_b[1] == 10.0 and off_s[5] == 33.0

    def test_a_change_during_a_skipped_frame_is_not_lost(self, fresh):
        """The dirty set describes ONE frame. If a frame goes by without
        _bar_flow being called -- every overlay that wants it switched off --
        that frame's changes never reach the cache, and the next frame's
        dirty set does not mention them again. Measured before the frame
        counter: the change was silently dropped forever."""
        df = klines()
        bk = buckets_for(fresh, df, {1: {54.0: bin_(off_buy=10)}})
        self._frame(fresh, df, bk)

        key = next(iter(bk))
        bk[key] = {54.0: bin_(off_buy=999)}
        self._frame_without_flow(fresh, df, bk)      # nobody asked this frame

        *_, off_b, _ = self._frame(fresh, df, bk)
        assert off_b[1] == 999.0, "a change made during a skipped frame was lost"

    def test_two_calls_in_one_frame_both_see_it(self, fresh):
        """Five overlays call _bar_flow; the 2nd through 5th must hit cache,
        not be mistaken for a stale frame."""
        df = klines()
        bk = buckets_for(fresh, df, {1: {54.0: bin_(off_buy=10)}})
        fresh._frame_dirty = fresh._begin_frame(df, bk, 1)
        first = fresh._bar_flow(df, bk, 1)
        second = fresh._bar_flow(df, bk, 1)
        assert np.array_equal(first[3], second[3])
        assert second[3][1] == 10.0

    def test_a_bar_whose_bucket_went_away_is_zeroed(self, fresh):
        """The incremental path writes into arrays seeded from the cache, so
        a bar that no longer has a bucket has to be cleared explicitly or it
        keeps yesterday's figure. buy/sell were cleared; off_* was the one
        that could be forgotten."""
        df = klines()
        bk = buckets_for(fresh, df, {1: {54.0: bin_(buy=7, off_buy=10)}})
        self._frame(fresh, df, bk)

        bk.pop(next(iter(bk)))                   # the bucket is gone
        buy, _, _, off_b, _ = self._frame(fresh, df, bk)

        assert buy[1] == 0.0
        assert off_b[1] == 0.0, "a stale off-exchange figure survived"

    def test_off_arrays_are_in_the_cache(self, fresh):
        df = klines()
        fresh._bar_flow(df, buckets_for(fresh, df, {1: {54.0: bin_(off_buy=1)}}), 1)
        assert "off_buy" in fresh._flow_cache
        assert "off_sell" in fresh._flow_cache


class TestWhatDrawCvdMakesOfIt:
    """The classes above assert on arrays _bar_flow returns, which cannot see
    how _draw_cvd combines them -- a mutation replacing its off_seen with
    `(off_buy - off_sell) != 0` survived them all. These drive the real
    _draw_cvd and read back the curve it published."""

    @staticmethod
    def _draw(viewer, df, bk):
        viewer._ind_checks["cvd"].setChecked(True)
        viewer._ind_checks["offex"].setChecked(True)
        viewer._frame_dirty = viewer._begin_frame(df, bk, 1)
        viewer._draw_cvd(df, bk, 1)
        return viewer._offex_arr

    def test_a_netted_bar_does_not_end_the_curve(self, fresh):
        """100 bought against 100 sold is real data. Deriving "was there any
        off-exchange flow here" from the DELTA calls that bar empty, and the
        curve stops one bar early -- which reads as "no data since", the one
        thing the NaN tail exists to say truthfully."""
        df = klines()
        bk = buckets_for(fresh, df, {
            1: {54.0: bin_(buy=5, sell=5, off_buy=300, off_sell=0)},
            3: {54.0: bin_(buy=5, sell=5, off_buy=100, off_sell=100)},
        })
        off = self._draw(fresh, df, bk)

        assert off is not None, "the off-exchange curve was not drawn at all"
        assert np.isfinite(off[3]), \
            "a bar whose off-exchange flow netted to zero ended the curve"

    def test_the_curve_still_stops_after_the_last_bar_with_data(self, fresh):
        df = klines()
        bk = buckets_for(fresh, df, {1: {54.0: bin_(off_buy=300)}})
        off = self._draw(fresh, df, bk)
        assert np.isfinite(off[1])
        assert not np.isfinite(off[-1]), \
            "the curve ran past the last bar that had data"

    def test_no_off_exchange_data_draws_no_curve(self, fresh):
        df = klines()
        bk = buckets_for(fresh, df, {1: {54.0: bin_(buy=10, sell=2)}})
        assert self._draw(fresh, df, bk) is None


class TestTheReturnShape:
    def test_it_returns_five(self, fresh):
        df = klines()
        got = fresh._bar_flow(df, buckets_for(fresh, df, {}), 1)
        assert len(got) == 5

    def test_the_off_arrays_are_per_bar_floats(self, fresh):
        df = klines()
        *_, off_b, off_s = fresh._bar_flow(df, buckets_for(fresh, df, {}), 1)
        for a in (off_b, off_s):
            assert a.shape == (len(df),)
            assert a.dtype == np.dtype(float)

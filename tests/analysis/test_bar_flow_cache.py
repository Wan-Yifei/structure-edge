"""_bar_flow must return what an uncached computation would return.

_draw_delta, _draw_cvd and _draw_dcvd each walked all 1440 buckets and
summed every price level in them -- the same two sums three times a
frame. They now share _bar_flow, which keeps the totals across renders
and recomputes only the bars _dirty_bars reported.

A cache that is right is invisible and a cache that is stale is a chart
quietly showing yesterday's flow, so every test here compares against a
reference that does the obvious thing, kept in this file so the two
cannot drift into agreeing with each other.

The cases that matter are the ones where "only the newest bar changed"
is false: a print arriving late into an older bucket, a bar whose stamp
does not parse, and the first render, where there is nothing to reuse.
"""

import os
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6.QtWidgets")

from PyQt6.QtWidgets import QApplication        # noqa: E402

import analysis.trade_viewer_qt as tv           # noqa: E402
from core.time_utils import candle_start        # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(scope="module")
def viewer(qapp):
    v = tv.TradeViewerQt()
    yield v
    v.close()


@pytest.fixture(autouse=True)
def fresh(viewer):
    viewer._flow_cache = None
    viewer._frame_dirty = None
    viewer._prev_frame = None
    viewer._bk_sig = None
    yield


def make(n, seed=11):
    rng = np.random.default_rng(seed)
    t = pd.date_range("2026-10-23 09:31", periods=n, freq="1min")
    base = 100 + np.cumsum(rng.normal(0, 0.05, n))
    kl = pd.DataFrame({
        "time_key": t.strftime("%Y-%m-%d %H:%M:%S"),
        "open": base, "close": base, "high": base + 0.1, "low": base - 0.1,
        "volume": rng.integers(500, 5000, n).astype(float),
    })
    buckets = {}
    for j in range(n):
        bk = candle_start(t[j].to_pydatetime() - timedelta(minutes=1), 1)
        buckets[bk] = {
            float(round(p, 2)): {"buy": int(rng.integers(0, 400)),
                                 "sell": int(rng.integers(0, 400)),
                                 "neutral": int(rng.integers(0, 900))}
            for p in np.linspace(base[j] - 0.08, base[j] + 0.08, 5)
        }
    return kl, buckets


def reference(viewer, kl, buckets, cm=1):
    """The obvious computation, the way the three drawers used to do it."""
    n = len(kl)
    buy = np.zeros(n)
    sell = np.zeros(n)
    day = [None] * n
    for i, bk in enumerate(tv.bar_start_keys(kl["time_key"], cm)):
        if bk is pd.NaT:
            continue
        day[i] = tv._trading_day_of_bucket(bk)
        pd_ = buckets.get(bk)
        if not pd_:
            continue
        buy[i] = sum(pd_[p]["buy"] for p in pd_)
        sell[i] = sum(pd_[p]["sell"] for p in pd_)
    return buy, sell, day


def same(got, want):
    assert np.array_equal(got[0], want[0]), "buy 不一致"
    assert np.array_equal(got[1], want[1]), "sell 不一致"
    assert got[2] == want[2], "trading day 不一致"


def step(viewer, kl, buckets, cm=1):
    """One render's worth: dirty set then flow, as _render does."""
    viewer._frame_dirty = viewer._dirty_bars(kl, buckets, cm)
    return viewer._bar_flow(kl, buckets, cm)


class TestItMatchesTheReference:
    def test_cold(self, viewer):
        kl, buckets = make(40)
        same(step(viewer, kl, buckets), reference(viewer, kl, buckets))

    def test_growing_one_bar_at_a_time(self, viewer):
        kl, buckets = make(60)
        for m in range(20, 61):
            f = kl.iloc[:m].reset_index(drop=True)
            same(step(viewer, f, buckets), reference(viewer, f, buckets))

    def test_a_tick_into_the_newest_bucket(self, viewer):
        kl, buckets = make(30)
        step(viewer, kl, buckets)
        last = sorted(buckets)[-1]
        buckets = dict(buckets)
        buckets[last] = dict(buckets[last])
        p0 = next(iter(buckets[last]))
        buckets[last][p0] = {"buy": 5000, "sell": 1, "neutral": 0}
        same(step(viewer, kl, buckets), reference(viewer, kl, buckets))

    def test_a_late_print_in_an_old_bucket(self, viewer):
        """Off-exchange prints arrive late; the bar is not the newest
        one and the cache must still pick it up."""
        kl, buckets = make(30)
        step(viewer, kl, buckets)
        old = sorted(buckets)[4]
        buckets = dict(buckets)
        buckets[old] = dict(buckets[old])
        p0 = next(iter(buckets[old]))
        buckets[old][p0] = {"buy": 7777, "sell": 3, "neutral": 0}
        got = step(viewer, kl, buckets)
        same(got, reference(viewer, kl, buckets))
        assert got[0][4] == pytest.approx(
            reference(viewer, kl, buckets)[0][4])

    def test_a_bar_that_loses_its_bucket(self, viewer):
        """A bucket going away must zero that bar, not leave the old
        total sitting in the cache."""
        kl, buckets = make(30)
        step(viewer, kl, buckets)
        victim = sorted(buckets)[6]
        buckets = {k: v for k, v in buckets.items() if k != victim}
        got = step(viewer, kl, buckets)
        same(got, reference(viewer, kl, buckets))
        assert got[0][6] == 0.0 and got[1][6] == 0.0

    def test_an_unparseable_stamp(self, viewer):
        kl, buckets = make(20)
        kl.loc[5, "time_key"] = "not a timestamp"
        same(step(viewer, kl, buckets), reference(viewer, kl, buckets))

    def test_a_window_that_moved(self, viewer):
        """A different first bar is a different series; nothing from the
        old one may survive."""
        kl, buckets = make(40)
        step(viewer, kl, buckets)
        moved = kl.iloc[8:].reset_index(drop=True)
        same(step(viewer, moved, buckets),
             reference(viewer, moved, buckets))

    def test_a_timeframe_change(self, viewer):
        kl, buckets = make(40)
        step(viewer, kl, buckets)
        same(step(viewer, kl, buckets, cm=5),
             reference(viewer, kl, buckets, cm=5))


class TestItActuallySkipsWork:
    def test_a_steady_frame_rescans_one_bar(self, viewer, monkeypatch):
        """Counting rescanned bars, because a cache that silently does
        the full sweep would pass every test above."""
        kl, buckets = make(50)
        step(viewer, kl, buckets)

        seen = []
        real = tv._trading_day_of_bucket
        monkeypatch.setattr(tv, "_trading_day_of_bucket",
                            lambda bk: (seen.append(bk), real(bk))[1])

        last = sorted(buckets)[-1]
        buckets = dict(buckets)
        buckets[last] = dict(buckets[last])
        step(viewer, kl, buckets)
        assert len(seen) == 1, f"重扫了 {len(seen)} 根，应为 1"

    def test_the_first_render_scans_everything(self, viewer, monkeypatch):
        kl, buckets = make(50)
        seen = []
        real = tv._trading_day_of_bucket
        monkeypatch.setattr(tv, "_trading_day_of_bucket",
                            lambda bk: (seen.append(bk), real(bk))[1])
        step(viewer, kl, buckets)
        assert len(seen) == 50

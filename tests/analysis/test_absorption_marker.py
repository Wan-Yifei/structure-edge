"""Absorption markers: drawn to be watched, never to be traded.

The condition is backtest/absorption.py's, pre-registered there and
measured out of sample over 60 sessions -- where TRADING it lost
(-0.361 R long, -0.566 R short). So the thing these tests guard is not
"does it predict" but "does it mark the same event the study defined,
and does the UI still say not to act on it".

The constants test is the important one: the tooltip quotes the study's
numbers, so the viewer must be applying the study's gate. Retuning one
without the other would leave the panel citing an experiment it no
longer runs.
"""
import pathlib
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

pytest.importorskip("PyQt6.QtWidgets")

CM = 1
T0 = datetime(2026, 10, 26, 10, 0)


def _frame(closes, buys, sells, t0=T0):
    """closes/buys/sells per bar -> (klines, buckets)."""
    from core.time_utils import candle_start

    rows, buckets = [], {}
    prev = closes[0]
    for i, (c, b, s) in enumerate(zip(closes, buys, sells)):
        hi, lo = max(c, prev) + 0.05, min(c, prev) - 0.05
        end = t0 + timedelta(minutes=CM * (i + 1))
        rows.append((end.strftime("%Y-%m-%d %H:%M"), prev, hi, lo, c,
                     float(b + s)))
        bk = candle_start(end - timedelta(minutes=CM), CM)
        buckets[bk] = {float(c): {"buy": float(b), "sell": float(s),
                                  "neutral": 0.0}}
        prev = c
    kl = pd.DataFrame(rows, columns=["time_key", "open", "high", "low",
                                     "close", "volume"])
    return kl, buckets


def _quiet(n, base=100.0, vol=2000.0, slope=4e-5, seed=7):
    """An unremarkable stretch that still has an impact relationship.

    The first version made buy and sell pure noise around each other,
    which left lambda fitted on nothing -- it came out NEGATIVE, so the
    trailing fit "expected" heavy selling to lift price and every sign
    downstream inverted. That is not a bug in the marker, it is what a
    market with no price impact would mean; but it is not the condition
    a planted absorption bar is supposed to be judged against. So flow
    moves price here, modestly, and lambda lands near +1.5.
    """
    rng = np.random.default_rng(seed)
    ofi = rng.normal(0, 400, n)
    buys = vol + ofi / 2
    sells = vol - ofi / 2
    dp = slope * ofi + rng.normal(0, 0.01, n)
    closes = base + np.cumsum(dp)
    return list(closes), list(buys), list(sells)


def _hl(c):
    c = np.asarray(c, float)
    prev = np.r_[c[0], c[:-1]]
    return np.maximum(c, prev) + 0.05, np.minimum(c, prev) - 0.05


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app
    app.processEvents()


@pytest.fixture(scope="module")
def viewer(qapp):
    from analysis.trade_viewer_qt import TradeViewerQt
    v = TradeViewerQt()
    v.resize(1200, 800)
    for _ in range(20):
        qapp.processEvents()
    yield v
    v.close()
    qapp.processEvents()


class TestItMarksTheStudysEvent:
    def test_the_constants_are_the_studys(self):
        """The tooltip quotes numbers measured under these settings."""
        from analysis import trade_viewer_qt as v
        from backtest import absorption as study
        from backtest.delta_atr_matrix import MIN_SIGNED

        assert v.DCVD_FLOW_WIN == study.WIN
        assert v.DCVD_FLOW_PCT == study.FLOW_PCT
        assert v.ABS_THRESHOLD == study.A_MAIN
        assert v.DCVD_MIN_SIGNED == MIN_SIGNED

    def test_heavy_selling_that_holds_price_up_fires_bullish(self, viewer):
        c, b, s = _quiet(90)
        # One bar: a wall of selling, and price closes HIGHER.
        c[80] = c[79] + 0.9
        s[80] = 40000.0
        h, l = _hl(c)
        bull, bear = viewer._abs_day(h, l, np.asarray(c, float),
                                     np.asarray(b, float),
                                     np.asarray(s, float))
        assert bull[80], "heavy selling into a rise is the bullish case"
        assert not bear[80]

    def test_heavy_buying_that_fails_fires_bearish(self, viewer):
        c, b, s = _quiet(90)
        c[80] = c[79] - 0.9
        b[80] = 40000.0
        h = np.maximum(c, np.r_[c[0], c[:-1]]) + 0.05
        l = np.minimum(c, np.r_[c[0], c[:-1]]) - 0.05
        bull, bear = viewer._abs_day(h, l, np.asarray(c, float),
                                     np.asarray(b, float),
                                     np.asarray(s, float))
        assert bear[80]
        assert not bull[80]

    def test_a_big_move_on_no_flow_does_not_fire(self, viewer):
        """The gate is not optional: price moving with nothing behind it
        is the OPPOSITE event, and without the gate it scores highest."""
        c, b, s = _quiet(90)
        c[80] = c[79] + 1.2           # a large move
        # A SMALL net sell, not a balanced one. Exactly balanced makes x
        # zero, and then the sign condition rejects the bar whether the
        # gate is there or not -- the test would pass without ever
        # reaching the thing it is named after. Tilted this way the
        # residual is large and positive, so with the gate removed this
        # bar fires bullish; the gate is the only thing stopping it.
        # Volume stays well over DCVD_MIN_SIGNED for the same reason.
        b[80], s[80] = 1900.0, 2100.0
        h = np.maximum(c, np.r_[c[0], c[:-1]]) + 0.05
        l = np.minimum(c, np.r_[c[0], c[:-1]]) - 0.05
        bull, bear = viewer._abs_day(h, l, np.asarray(c, float),
                                     np.asarray(b, float),
                                     np.asarray(s, float))
        assert not bull[80] and not bear[80]


class TestItCannotSeeForward:
    def test_a_later_bar_cannot_change_an_earlier_mark(self, viewer):
        """Every window here ends at t-1. If that slips, the marks move
        under you when the next bar arrives -- the worst failure a live
        chart can have, because it is invisible in a screenshot."""
        c, b, s = _quiet(90)
        c[80] = c[79] + 0.9
        s[80] = 40000.0

        def flags(cl, bu, se):
            h = np.maximum(cl, np.r_[cl[0], cl[:-1]]) + 0.05
            l = np.minimum(cl, np.r_[cl[0], cl[:-1]]) - 0.05
            return viewer._abs_day(h, l, np.asarray(cl, float),
                                   np.asarray(bu, float),
                                   np.asarray(se, float))

        base_bull, base_bear = flags(c, b, s)
        c2, b2, s2 = list(c), list(b), list(s)
        c2[85] = c2[84] + 3.0         # a violent bar AFTER the mark
        s2[85] = 90000.0
        bull2, bear2 = flags(c2, b2, s2)
        assert np.array_equal(base_bull[:85], bull2[:85])
        assert np.array_equal(base_bear[:85], bear2[:85])


class TestTheDayBoundary:
    def test_a_short_day_cannot_borrow_the_previous_one(self, viewer):
        """Windows are trailing, and one reaching over 20:00 would fit
        this session's flow against the last session's -- with the
        overnight gap entering as a move no flow explains."""
        # Starting 18:30, bar i covers 18:30+i, so the 20:00 boundary puts
        # bars 0..89 in one trading day and 90..99 in the next.
        c, b, s = _quiet(90, base=100.0)
        c[80] = c[79] + 0.9                       # a real event in day one
        s[80] = 40000.0
        c2, b2, s2 = _quiet(10, base=120.0)       # a new day, 20 points up
        kl, buckets = _frame(c + c2, b + b2, s + s2,
                             t0=datetime(2026, 10, 26, 18, 30))
        bull, bear = viewer._absorption_flags(kl, buckets, CM)
        fired = set(np.where(bull | bear)[0].tolist())
        assert 80 in fired, "the day-one event was lost in the split"
        assert all(i < 90 for i in fired), \
            f"a mark landed in the 10-bar second day: {sorted(fired)}"

    def test_a_session_shorter_than_the_warm_up_marks_nothing(self, viewer):
        """Three trailing windows run in series -- 20 bars for the volume
        scale, 20 more before lambda has a fit, 20 more before the
        residual has a spread to be measured against. So nothing can be
        marked until roughly bar 60 of a session, and a short day marks
        nothing at all. That is correct, and it is better stated than
        discovered."""
        c, b, s = _quiet(55)
        c[45] = c[44] + 0.9
        s[45] = 40000.0
        h, l = _hl(c)
        bull, bear = viewer._abs_day(h, l, np.asarray(c, float),
                                     np.asarray(b, float),
                                     np.asarray(s, float))
        assert not bull.any() and not bear.any()


class TestTheViewerDrawsAndClears:
    def test_items_appear_and_go(self, viewer):
        c, b, s = _quiet(90)
        c[80] = c[79] + 0.9
        s[80] = 40000.0
        kl, buckets = _frame(c, b, s)
        viewer._clear_abs_items()
        viewer._draw_absorption(kl, buckets, CM)
        assert viewer._abs_items, "nothing was drawn for a planted event"
        viewer._clear_abs_items()
        assert viewer._abs_items == []

    def test_it_is_off_by_default(self, viewer):
        """It is a marker for watching, not part of the standard read."""
        assert not viewer._ind_checks["absorb"].isChecked()

    def test_the_tooltip_refuses_to_be_an_entry(self, viewer):
        tip = viewer._ind_checks["absorb"].toolTip()
        assert "NOT AN ENTRY" in tip
        assert "-0.361" in tip and "-0.566" in tip, \
            "the measured loss is not stated"
        assert "+0.352" in tip, "the alternative that does pay is not stated"

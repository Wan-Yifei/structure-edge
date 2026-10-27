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


def _frame_days(days, day0=datetime(2026, 10, 19), lo_min=9 * 60 + 30):
    """Several RTH sessions in one frame.

    `days` is a list of (closes, buys, sells) per session; session k's
    bar i ends at 09:31 + i minutes on day0 + k. The time-of-day profile
    reads previous sessions, so anything exercising it needs at least
    three of them -- a single day marks nothing by construction.
    """
    from core.time_utils import candle_start

    rows, buckets = [], {}
    for k, (closes, buys, sells) in enumerate(days):
        base = day0 + timedelta(days=k, minutes=lo_min)
        prev = closes[0]
        for i, (c, b, s) in enumerate(zip(closes, buys, sells)):
            hi, lo = max(c, prev) + 0.05, min(c, prev) - 0.05
            end = base + timedelta(minutes=CM * (i + 1))
            rows.append((end.strftime("%Y-%m-%d %H:%M"), prev, hi, lo, c,
                         float(b + s)))
            bk = candle_start(end - timedelta(minutes=CM), CM)
            buckets[bk] = {float(c): {"buy": float(b), "sell": float(s),
                                      "neutral": 0.0}}
            prev = c
    kl = pd.DataFrame(rows, columns=["time_key", "open", "high", "low",
                                     "close", "volume"])
    return kl, buckets


def _ramped(n, base=100.0, seed=7, peak=10.0, decay=40.0, vol0=2000.0):
    """A session whose volume opens high and decays, as a real one does.

    The flat-volume stretch _quiet builds cannot show what went wrong:
    the bug was a trailing window straddling a volume regime change, and
    with no change there is nothing to straddle. Here bar 0 trades
    `peak` x the late-session rate, which is what left the gate stuck
    far above anything the rest of the session could reach.
    """
    rng = np.random.default_rng(seed)
    i = np.arange(n, dtype=float)
    vol = vol0 * (1.0 + (peak - 1.0) * np.exp(-i / decay))
    ofi = rng.normal(0, 0.2, n) * vol
    buys = vol + ofi / 2
    sells = vol - ofi / 2
    dp = 4e-5 * ofi / (vol / 2000.0) + rng.normal(0, 0.01, n)
    closes = base + np.cumsum(dp)
    return list(closes), list(buys), list(sells)


def _pre_then_rth(n_pre, n_rth, base=100.0, seed=7, thin=6.0):
    """A thin premarket stretch followed by a real session.

    This is the shape the bug lived in: premarket trades a fraction of
    the RTH rate -- on the real feed the median was 1,650 shares a bar
    against RTH's 10,944 -- so a trailing window spanning the open
    measures a bar against a market that is gone, and the |x| that
    produces holds the percentile above anything RTH can reach for the
    hour behind it.

    Flat, not ramped, and not _quiet: _quiet's imbalance is a fixed 400
    shares whatever the volume, so at a premarket rate it drives buy and
    sell NEGATIVE and every bar falls under the 300-share floor -- the
    contamination then never happens and the control passes vacuously.
    """
    pc, pb, ps = _ramped(n_pre, base=base, seed=seed, peak=1.0,
                         vol0=2000.0 / thin)
    rc, rb, rs = _ramped(n_rth, base=pc[-1], seed=seed + 100)
    return pc + rc, pb + rb, ps + rs


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
    def test_only_rth_is_marked(self, viewer):
        """The study is RTH 09:31-16:00, overnight and premarket taking
        no part in it -- backtest/absorption.py says so. This used to
        split on the 20:00 trading-day boundary instead, which fed 2,000-
        share overnight bars through the same windows as the 50,000-share
        RTH ones -- 55% of the marks landed outside RTH."""
        days = [_ramped(150, base=100.0 + 2 * k, seed=11 + k)
                for k in range(4)]
        # An unmistakable absorption bar in the premarket of the last
        # session: heavy one-sided selling, price up.
        c, b, s = days[-1]
        c[20] = c[19] + 0.9
        s[20] = 400_000.0
        b[20] = 2_000.0
        kl, buckets = _frame_days(days, lo_min=8 * 60)     # 08:01 onward
        bull, bear = viewer._absorption_flags(kl, buckets, CM)
        mins = [int(str(t)[11:13]) * 60 + int(str(t)[14:16])
                for t in kl["time_key"]]
        for i in np.where(bull | bear)[0]:
            assert 9 * 60 + 30 < mins[i] <= 16 * 60, \
                f"标记落在 RTH 之外：{kl['time_key'].iloc[i]}"

    def test_the_first_bar_of_a_session_is_never_marked(self, viewer):
        """`dp` is the only quantity that does not continue across the
        boundary, because the overnight gap is a price move no flow
        explains -- the original reason for splitting by day at all."""
        days = [_ramped(150, base=100.0 + 9 * k, seed=3 + k)
                for k in range(5)]
        # Each first bar is made into something that WOULD be marked if
        # the gap counted as its move: heavily one-sided, several times
        # the usual size. Without this the check passes whether or not
        # dp is carried over, because no first bar happens to qualify --
        # mutation testing is what says which of the two it is.
        for k in range(1, 5):
            c, b, s = days[k]
            tot = (b[0] + s[0]) * 4
            s[0], b[0] = tot * 0.95, tot * 0.05
        kl, buckets = _frame_days(days)
        bull, bear = viewer._absorption_flags(kl, buckets, CM)
        firsts = {k * 150 for k in range(5)}
        hit = firsts & set(np.where(bull | bear)[0].tolist())
        assert not hit, f"交易日第一根被标记了：{sorted(hit)}"

    def test_a_session_is_marked_before_its_sixtieth_bar(self, viewer):
        """lambda, the residual spread and ATR continue across sessions,
        which is what makes the first hour reachable. Warming all three
        up inside the session left nothing markable before 10:30, and on
        the real feed that was 0 marks in 09:31-10:30 over 136
        sessions."""
        days = [_ramped(200, base=100.0, seed=21 + k) for k in range(6)]
        c, b, s = days[-1]
        c[12] = c[11] + 0.5                   # 09:43, inside the old warm-up
        s[12] = 120_000.0
        b[12] = 2_000.0
        kl, buckets = _frame_days(days)
        bull, _bear = viewer._absorption_flags(kl, buckets, CM)
        assert bull[5 * 200 + 12], \
            "续窗之后第 12 根仍然没标 —— 热身还是在交易日内重来"

    def test_the_open_does_not_set_the_gate_for_the_whole_morning(
            self, viewer):
        """The bug, stated as a test.

        A trailing percentile of |x| keeps the open's blown-up values for
        sixty bars after the open, so the hour behind it is judged
        against a market that has already gone. On the real feed that hour
        held 1 mark in 136 sessions. Each planted bar here is one-sided
        and fails to move price, and is sized to what ITS OWN time of day
        trades -- not to the open -- so only a time-of-day gate can see
        them.
        """
        n_pre, n_rth = 60, 150                    # 08:31-09:30, 09:31-12:00
        days = [_pre_then_rth(n_pre, n_rth, base=100.0, seed=41 + k)
                for k in range(8)]
        # One plant a session, so each is judged against an unshaken
        # series: several in one session make the later ones unremarkable
        # by inflating the ATR and the residual spread the earlier ones
        # left behind.
        at = {3: 60, 4: 70, 5: 80, 6: 90, 7: 100}     # 10:31 - 11:11
        for k, j in at.items():
            c, b, s = days[k]
            i = n_pre + j
            tot = (b[i] + s[i]) * 1.6             # big for this minute,
            s[i], b[i] = tot * 0.92, tot * 0.08   # tiny next to the open
            for m in range(i, len(c)):            # and price goes the
                c[m] += 0.30                      # other way
        kl, buckets = _frame_days(days, lo_min=8 * 60 + 30)
        bull, _bear = viewer._absorption_flags(kl, buckets, CM)
        per = n_pre + n_rth
        got = [k for k, j in at.items() if bull[k * per + n_pre + j]]

        # The control: the whole day through one trailing window, which
        # is what this did before -- premarket and RTH in the same
        # percentile, so the open's blown-up |x| sets the gate for the
        # hour behind it.
        old = []
        for k, j in at.items():
            c, b, s = days[k]
            cc = np.asarray(c, float)
            hh, ll = _hl(cc)
            ob, _ = viewer._abs_day(hh, ll, cc, np.asarray(b, float),
                                    np.asarray(s, float))
            if ob[n_pre + j]:
                old.append(k)
        assert len(got) >= 4, f"时段门槛只标到第 {got} 天，预期大部分"
        assert len(got) > len(old), \
            f"换门槛没有带来差别：时段 {got} vs 整天滚动 {old}"

    def test_a_session_is_not_in_its_own_reference(self, viewer):
        """The profile is appended after it is used, never before. The
        other way round a bar helps set the bar it is judged against,
        which reads as a gate that quietly tracks whatever just
        happened."""
        days = [_ramped(150, base=100.0, seed=61 + k) for k in range(4)]
        kl, buckets = _frame_days(days)
        before = viewer._absorption_flags(kl, buckets, CM)[0].copy()
        # Blow up ONE bar of the last session. Nothing in an earlier
        # session may move, and the bucket's own gate must not absorb it.
        c, b, s = days[-1]
        s[140] = (b[140] + s[140]) * 40
        kl2, buckets2 = _frame_days(days)
        after = viewer._absorption_flags(kl2, buckets2, CM)[0]
        assert (before[:3 * 150] == after[:3 * 150]).all(), \
            "改最后一个交易日的一根，前面几天的标记变了"

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
        days = [_ramped(150, base=100.0, seed=5 + k) for k in range(4)]
        c, b, s = days[-1]
        c[100] = c[99] + 0.6
        s[100] = 200_000.0
        b[100] = 2_000.0
        kl, buckets = _frame_days(days)
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

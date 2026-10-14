"""The entry bar is not part of the holding period.

The fill is the close of the signal bar, so everything that happened inside
that bar happened before the position existed. Scanning it for the stop
declared 20% of SOXL's regular-session trades losers on their own entry
candle, and it bit hardest on the best setups: a strong reclaim closes near
POC, which makes R small, which puts that same bar's low below the stop.
Over 2025-02-13..2026-03-31 it turned +0.063 average R into -0.142 -- a sign
flip from one off-by-one.

backtest/engine.py's own call to _find_exit is different and correct: it
passes the current loop bar for a trade opened on an earlier one, so that bar
is genuinely part of the holding period. The fix therefore belongs in the
session_vp caller, not in the shared simulator, and the first class here pins
that distinction so a later tidy-up cannot "unify" the two call sites.
"""

import numpy as np
import pandas as pd
import pytest

from backtest.engine import _find_exit
from backtest.session_vp_engine import SessionVPParams, run_backtest_session_vp

SESSIONS = {"regular": {"start": "09:30", "end": "16:00", "enabled": True}}


def session(bars):
    """bars: list of (open, high, low, close), one minute apart from 09:31."""
    t0 = pd.Timestamp("2026-06-01 09:31")
    return pd.DataFrame({
        "time_key": [str(t0 + pd.Timedelta(minutes=i)) for i in range(len(bars))],
        "open": [b[0] for b in bars],
        "high": [b[1] for b in bars],
        "low": [b[2] for b in bars],
        "close": [b[3] for b in bars],
        "volume": [1000] * len(bars),
    })


def warmup():
    """40 bars with a clear peak at 100 and tails either side.

    Closes alternate by two cents so RSI is defined -- a perfectly flat close
    series makes Wilder's average gain and loss both zero and the indicator
    NaN, which the detector treats as "filter fails" and no signal fires.
    """
    out = []
    for n, (hi, lo) in enumerate([(100.1, 99.9)] * 16 + [(100.4, 99.6)] * 10
                                 + [(100.8, 99.2)] * 8 + [(101.2, 98.8)] * 6):
        out.append((100.0, hi, lo, 99.98 if n % 2 else 100.02))
    return out


# VAL lands at 99.82 and POC at 99.94 on this profile. The touch dips to
# 99.00, and the entry bar reclaims to close at 99.93 -- one cent under POC,
# so R is one cent and the stop sits at 99.92, far above that same bar's low.
TOUCH = (99.90, 99.92, 99.00, 99.10)
ENTRY = (99.10, 99.95, 99.00, 99.93)
RALLY = (99.93, 100.50, 99.93, 100.40)


@pytest.fixture
def params():
    return SessionVPParams(
        warmup_minutes=40, va_pct=0.70, n_bins=20, rsi_period=6,
        rsi_threshold=100.0,              # take the signal whatever RSI says
        tradeable_sessions=("regular",), max_bars=60,
        min_val_poc_dist_pct=0.0, target_rr=1.0, val_proximity_pct=0.0)


class TestFindExitItself:
    """The shared simulator is unchanged: from_bar is inclusive, by design."""

    def test_from_bar_is_inclusive(self):
        lows = np.array([5.0, 10.0, 10.0])
        highs = np.array([10.0, 10.0, 10.0])
        closes = np.array([10.0, 10.0, 10.0])
        bar, _, outcome = _find_exit(lows, highs, closes, from_bar=0, sl=6.0,
                                     tp=20.0, direction="bull", max_bars=3)
        assert outcome == "loss" and bar == 0

    def test_starting_one_bar_later_changes_the_answer(self):
        lows = np.array([5.0, 10.0, 10.0])
        highs = np.array([10.0, 10.0, 21.0])
        closes = np.array([10.0, 10.0, 21.0])
        bar, _, outcome = _find_exit(lows, highs, closes, from_bar=1, sl=6.0,
                                     tp=20.0, direction="bull", max_bars=3)
        assert outcome == "win" and bar == 2


class TestEntryBarIsExcluded:
    """The regression: a deep-wicked entry candle must not stop the trade."""

    def build(self):
        return session(warmup() + [TOUCH, ENTRY] + [RALLY] * 30)

    def test_a_trade_is_opened(self, params):
        assert len(run_backtest_session_vp(self.build(), params, SESSIONS).trades) == 1

    def test_the_entry_bar_wick_really_is_below_the_stop(self, params):
        """Without this the test could pass for the wrong reason."""
        t = run_backtest_session_vp(self.build(), params, SESSIONS).trades[0]
        assert ENTRY[2] < t.sl, "fixture no longer exercises the bug"

    def test_it_does_not_exit_on_its_own_entry_bar(self, params):
        t = run_backtest_session_vp(self.build(), params, SESSIONS).trades[0]
        assert t.exit_time != t.entry_time

    def test_the_rally_is_what_resolves_it(self, params):
        t = run_backtest_session_vp(self.build(), params, SESSIONS).trades[0]
        assert t.result == "win"
        assert t.r_multiple == pytest.approx(1.0)


class TestTheBarAfterEntryStillCounts:
    """Excluding the entry bar must not spill into excluding the next one."""

    def test_a_stop_on_the_following_bar_is_a_loss(self, params):
        bars = warmup() + [TOUCH, ENTRY,
                           (99.93, 99.94, 99.00, 99.05)] + [RALLY] * 30
        r = run_backtest_session_vp(session(bars), params, SESSIONS)
        assert len(r.trades) == 1
        assert r.trades[0].r_multiple == pytest.approx(-1.0)

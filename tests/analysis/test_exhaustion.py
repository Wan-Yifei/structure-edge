"""Sell-exhaustion marks: a burst of aggressive selling that then stops.

Both gates exist because either alone marks the wrong things. RSI alone let a
677-share "burst" in a quiet oversold minute score the same as the 31,499-share
flush at 2026-10-13 11:35; burst size alone marked mid-range flushes that had
nothing to do with a low. The thresholds were chosen by how often the mark
should appear -- about six a session over 30 SOXL sessions -- and never by
what price did afterwards, which is also why nothing here asserts an outcome.

The alignment is the subtle part and gets its own tests: kline time_key is the
bar's END, so the bar labelled 11:36 carries RSI for the ticks of 11:35.
Getting that backwards shifts every mark a minute and was a real bug earlier
in this work.
"""

import os
from datetime import datetime, timedelta

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6.QtWidgets")

from analysis.trade_viewer_qt import (            # noqa: E402
    EXH_BURST_MULT, EXH_DROP, EXH_RSI_MAX,
    compute_exhaustion_marks, exhaustion_rsi_map,
)

DAY = datetime(2026, 10, 13)


def klines(closes, start="09:31", n_pad=0):
    """1m bars whose time_key is the bar END, as moomoo reports it."""
    t0 = datetime.strptime(f"2026-10-13 {start}", "%Y-%m-%d %H:%M")
    rows = []
    for i, c in enumerate(list(closes) + [closes[-1]] * n_pad):
        rows.append({"time_key": (t0 + timedelta(minutes=i)
                                  ).strftime("%Y-%m-%d %H:%M:%S"),
                     "close": float(c)})
    return pd.DataFrame(rows)


def ticks(spec, price=100.0):
    """spec: {'HH:MM:SS': sell_volume} -> raw tick rows."""
    out = []
    for hms, vol in spec.items():
        out.append({"ts": datetime.strptime(f"2026-10-13 {hms}",
                                            "%Y-%m-%d %H:%M:%S"),
                    "price": price, "volume": vol, "direction": "SELL"})
    return out


def falling_day():
    """A session that declines steadily, so RSI(6) sits far below the gate."""
    return klines([200 - i * 0.5 for i in range(120)])


def steady_day():
    """A session going nowhere, so RSI(6) sits near 50."""
    return klines([100 + (1 if i % 2 else -1) for i in range(120)])


def background(n=600, vol=100, start_sec=0):
    """Enough small prints to establish the session's five-second scale."""
    out = {}
    for i in range(n):
        s = 9 * 3600 + 31 * 60 + start_sec + i * 5
        out[f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"] = vol
    return out


class TestRsiMap:
    def test_keys_are_the_tick_minute_not_the_bar_label(self):
        """time_key is the bar END, so 11:36's RSI belongs to 11:35's ticks.

        The last bar of falling_day() is labelled 11:10, so the last key must
        be 11:09. Reading the label straight through would put it at 11:10 and
        shift every mark a minute late.
        """
        k = falling_day()
        last_bar = pd.to_datetime(k["time_key"].iloc[-1])
        labels = sorted({key[1] for key in exhaustion_rsi_map(k)})
        assert labels[-1] == (last_bar - timedelta(minutes=1)
                              ).strftime("%H:%M")
        assert labels[-1] != last_bar.strftime("%H:%M")
        # RSI(6) has no value until it warms up, so the first bars are absent.
        assert labels[0] > "09:30"

    def test_a_session_too_short_to_warm_up_is_skipped(self):
        assert exhaustion_rsi_map(klines([100] * 5)) == {}

    def test_a_falling_session_is_oversold(self):
        m = exhaustion_rsi_map(falling_day())
        assert min(m.values()) < EXH_RSI_MAX

    def test_a_flat_session_is_not(self):
        m = exhaustion_rsi_map(steady_day())
        assert min(m.values()) >= EXH_RSI_MAX

    def test_bars_outside_regular_hours_are_left_out(self):
        m = exhaustion_rsi_map(klines([200 - i for i in range(120)],
                                      start="07:00"))
        assert m == {}


class TestMarks:
    def test_a_burst_that_stops_is_marked(self):
        rsi = exhaustion_rsi_map(falling_day())
        spec = background()
        spec["09:40:00"] = 100 * EXH_BURST_MULT * 3
        for s in range(5, 35, 5):                       # then near silence
            spec[f"09:40:{s:02d}"] = 1
        got = compute_exhaustion_marks(ticks(spec), rsi)
        assert len(got) == 1
        assert got[0]["burst_x"] >= EXH_BURST_MULT

    def test_a_burst_too_small_for_the_session_is_not(self):
        rsi = exhaustion_rsi_map(falling_day())
        spec = background()
        spec["09:40:00"] = 100 * (EXH_BURST_MULT - 2)
        for s in range(5, 35, 5):
            spec[f"09:40:{s:02d}"] = 1
        assert compute_exhaustion_marks(ticks(spec), rsi) == []

    def test_a_burst_that_keeps_selling_is_not(self):
        rsi = exhaustion_rsi_map(falling_day())
        big = 100 * EXH_BURST_MULT * 3
        spec = background()
        spec["09:40:00"] = big
        # Below the burst gate itself, so it cannot become its own candidate,
        # but well above EXH_DROP of the first burst -- selling eased without
        # stopping. Covers the whole 30s window, or the background behind it
        # would read as the collapse.
        for s in range(5, 40, 5):
            spec[f"09:40:{s:02d}"] = big * 0.3
        assert compute_exhaustion_marks(ticks(spec), rsi) == []

    def test_an_oversold_gate_that_is_not_met_blocks_the_mark(self):
        """Same flow, a session that is not oversold."""
        rsi = exhaustion_rsi_map(steady_day())
        spec = background()
        spec["09:40:00"] = 100 * EXH_BURST_MULT * 3
        for s in range(5, 35, 5):
            spec[f"09:40:{s:02d}"] = 1
        assert compute_exhaustion_marks(ticks(spec), rsi) == []

    def test_one_flush_gives_one_mark(self):
        """Consecutive five-second bursts are one event, not six."""
        rsi = exhaustion_rsi_map(falling_day())
        big = 100 * EXH_BURST_MULT * 3
        spec = background()
        for s in range(0, 30, 5):
            spec[f"09:40:{s:02d}"] = big - s      # all over the gate
        for s in range(30, 60, 5):
            spec[f"09:40:{s:02d}"] = 1
        assert len(compute_exhaustion_marks(ticks(spec), rsi)) == 1

    def test_the_mark_sits_at_the_lull_not_the_burst(self):
        rsi = exhaustion_rsi_map(falling_day())
        spec = background()
        spec["09:40:00"] = 100 * EXH_BURST_MULT * 3
        for s in range(5, 35, 5):
            spec[f"09:40:{s:02d}"] = 1
        got = compute_exhaustion_marks(ticks(spec), rsi)
        assert got[0]["ts"] > datetime(2026, 10, 13, 9, 40, 0)
        assert got[0]["ts"] <= datetime(2026, 10, 13, 9, 40, 30)

    def test_buy_prints_do_not_count_as_selling(self):
        rsi = exhaustion_rsi_map(falling_day())
        spec = background()
        spec["09:40:00"] = 100 * EXH_BURST_MULT * 3
        for s in range(5, 35, 5):
            spec[f"09:40:{s:02d}"] = 1
        rows = ticks(spec)
        for r in rows:
            r["direction"] = "BUY"
        assert compute_exhaustion_marks(rows, rsi) == []


class TestDegenerate:
    def test_no_ticks_is_empty_not_an_error(self):
        assert compute_exhaustion_marks([], exhaustion_rsi_map(
            falling_day())) == []

    def test_no_rsi_is_empty_not_an_error(self):
        assert compute_exhaustion_marks(ticks(background()), {}) == []

    def test_a_day_too_thin_to_set_a_scale_is_empty(self):
        rsi = exhaustion_rsi_map(falling_day())
        assert compute_exhaustion_marks(ticks({"09:40:00": 50_000}), rsi) == []

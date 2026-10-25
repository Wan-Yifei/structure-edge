"""Yesterday's open and close, as the profile panel marks them.

Two things decide whether these are the right prices, and both are easy
to get wrong in a way that still produces a plausible number:

  the boundary   a trading day runs 20:00 ET to 20:00 ET, so the
                 overnight session belongs to the day it leads INTO.
                 Using the calendar date puts the evening's bars on the
                 wrong side and moves "yesterday's open" by hours.
  the frame      the levels come from the unfiltered klines. The panel's
                 histogram is built from a session-filtered frame, and
                 with "Night" unchecked that frame has no overnight bars
                 -- yesterday's open would quietly become its 09:30 open.

_prev_session_levels touches only self._klines and self._candle_mins, so
it is exercised here without building the widget, which would need Qt
and a live OpenD.
"""
import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import analysis.trade_viewer_qt as tv  # noqa: E402

_levels = tv.TradeViewerQt.__dict__["_prev_session_levels"]


def got(stub):
    return {tag: p for p, tag in _levels(stub)}


class Stub:
    """Only what the method reads."""

    def __init__(self, df, cm=1):
        self._klines = df
        self._candle_mins = cm


def frame(rows):
    """rows: [(time_key, open, close)]"""
    return pd.DataFrame(
        {"time_key": [r[0] for r in rows],
         "open": [r[1] for r in rows],
         "high": [max(r[1], r[2]) for r in rows],
         "low": [min(r[1], r[2]) for r in rows],
         "close": [r[2] for r in rows],
         "volume": [1000.0] * len(rows)})


class TestBoundary:
    def test_the_evening_belongs_to_the_next_day(self):
        """20:00 on the 20th opens the 21st's trading day, so the 21st's
        open is that evening's open -- not the 09:31 one."""
        df = frame([
            ("2026-10-20 09:31:00", 100.0, 100.5),   # the 20th
            ("2026-10-20 16:00:00", 100.5, 101.0),
            ("2026-10-20 20:01:00", 111.0, 111.5),   # starts the 21st
            ("2026-10-21 09:31:00", 112.0, 112.5),
            ("2026-10-21 16:00:00", 112.5, 113.0),
            ("2026-10-22 09:31:00", 120.0, 120.5),   # today
        ])
        g = got(Stub(df))
        assert g["PDO"] == 111.0, "20:01 的开盘才是前一交易日的开盘"
        assert g["PDC"] == 113.0

    def test_calendar_grouping_would_give_a_different_answer(self):
        """The control. If this ever starts matching the calendar-date
        answer, the boundary has been dropped."""
        df = frame([
            ("2026-10-20 20:01:00", 111.0, 111.5),
            ("2026-10-21 09:31:00", 112.0, 112.5),
            ("2026-10-21 16:00:00", 112.5, 113.0),
            ("2026-10-22 09:31:00", 120.0, 120.5),
        ])
        assert got(Stub(df))["PDO"] != 112.0, \
            "取成了 09:31 的开盘，边界没生效"


class TestEdges:
    def test_one_day_of_data_has_no_previous_session(self):
        df = frame([("2026-10-22 09:31:00", 100.0, 100.5),
                    ("2026-10-22 09:32:00", 100.5, 101.0)])
        assert _levels(Stub(df)) == []

    def test_too_few_rows_is_not_an_error(self):
        one = frame([("2026-10-22 09:31:00", 1.0, 1.0)])
        assert _levels(Stub(one)) == []
        assert _levels(Stub(None)) == []

    def test_a_gap_over_a_weekend_still_finds_the_last_session(self):
        """"Previous" is the last trading day present, not yesterday's
        date -- a Monday's previous session is the Friday."""
        df = frame([
            ("2026-10-16 09:31:00", 90.0, 90.5),     # Friday
            ("2026-10-16 16:00:00", 90.5, 91.0),
            ("2026-10-19 09:31:00", 95.0, 95.5),     # Monday
        ])
        g = got(Stub(df))
        assert g["PDO"] == 90.0 and g["PDC"] == 91.0

    def test_the_close_is_the_last_bar_of_that_day(self):
        """Not the day's high or its 16:00 bar specifically -- the last
        bar present for that trading day, whatever time it carries."""
        df = frame([
            ("2026-10-20 09:31:00", 100.0, 100.5),
            ("2026-10-20 12:00:00", 100.5, 104.0),
            ("2026-10-20 15:59:00", 104.0, 99.0),
            ("2026-10-21 09:31:00", 99.0, 99.5),
        ])
        assert got(Stub(df))["PDC"] == 99.0

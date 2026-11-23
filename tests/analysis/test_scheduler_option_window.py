"""The option collector runs on its own window, not the stock sessions.

The four stock sessions are chained 20:00 -> 04:00 -> 09:30 -> 16:00 ->
20:00 with no weekday test, so a minute-by-minute scan finds no moment when
none is active. The condition that stops a collector -- "this session ended
and nothing else is running" -- therefore never fires. A collector hung off
them starts once and is never stopped, and so never re-picks its contracts:
by Friday it still holds Monday's set, whose near expiry has expired.

A window with real gaps is the whole point, so these tests pin the gaps.
"""
from datetime import datetime

import pytest

from analysis.scheduler import SchedulerApp


def _app(**over):
    """A SchedulerApp with no Tk behind it -- only cfg and the methods."""
    app = object.__new__(SchedulerApp)
    app.cfg = {"option_window": dict(SchedulerApp._DEFAULT_OPTION, **over)}
    return app


def _at(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M")


class TestTheWindowHasRealGaps:

    @pytest.mark.parametrize("when,want", [
        ("2026-11-23 09:14", False),   # Monday, one minute early
        ("2026-11-23 09:15", True),    # open edge, inclusive
        ("2026-11-23 12:00", True),
        ("2026-11-23 16:25", True),    # close edge, inclusive
        ("2026-11-23 16:26", False),
        ("2026-11-23 03:00", False),   # the overnight stock session
        ("2026-11-23 19:00", False),   # after-hours stock session
    ])
    def test_the_edges(self, when, want):
        assert _app(enabled=True)._option_window_active(_at(when)) is want

    def test_it_closes_every_night(self):
        """The property the whole design rests on: there IS an off period,
        so the collector stops and re-picks contracts the next morning."""
        app = _app(enabled=True)
        day = "2026-11-23"
        closed = [h for h in range(24)
                  if not app._option_window_active(_at(f"{day} {h:02d}:30"))]
        assert closed, "the window never closes -- contracts would never be re-picked"
        assert 0 in closed and 23 in closed, closed

    @pytest.mark.parametrize("day", ["2026-11-28", "2026-11-29"])
    def test_weekends_are_off(self, day):
        assert _app(enabled=True)._option_window_active(_at(f"{day} 12:00")) is False


class TestItIsOffUntilSomebodyTurnsItOn:

    def test_disabled_by_default(self):
        assert SchedulerApp._DEFAULT_OPTION["enabled"] is False

    def test_disabled_means_never_active(self):
        app = _app(enabled=False)
        assert app._option_window_active(_at("2026-11-23 12:00")) is False

    def test_a_config_without_the_key_does_not_raise(self):
        """Older config files have no option_window. _load_config only
        setdefaults a handful of keys, so everything else must be read
        through .get(key, DEFAULT)."""
        app = object.__new__(SchedulerApp)
        app.cfg = {}
        assert app._option_window_active(_at("2026-11-23 12:00")) is False


class TestBadTimesDoNotTakeTheSchedulerDown:

    @pytest.mark.parametrize("start,end", [
        ("", "16:25"), ("9:15", "16:25"), ("ninefifteen", "16:25"),
        ("09:15", ""), ("09:15", "25:99"),
    ])
    def test_a_malformed_time_falls_back(self, start, end):
        """These come from free-text entry boxes. A ValueError in the 1 Hz
        tick would take the scheduler's whole loop with it."""
        app = _app(enabled=True, start=start, end=end)
        assert app._option_window_active(_at("2026-11-23 12:00")) is True
        assert app._option_window_active(_at("2026-11-23 03:00")) is False

    def test_a_custom_window_is_honoured(self):
        app = _app(enabled=True, start="10:00", end="11:00")
        assert app._option_window_active(_at("2026-11-23 09:30")) is False
        assert app._option_window_active(_at("2026-11-23 10:30")) is True
        assert app._option_window_active(_at("2026-11-23 11:30")) is False


class TestTheDefaultsMatchWhatOptionsActuallyDo:

    def test_the_window_covers_the_session_and_the_late_prints(self):
        """Options trade 09:30-16:00 with late reports to about 16:15 --
        5.6% of prints and 7.8% of contracts, measured."""
        app = _app(enabled=True)
        assert app._option_window_active(_at("2026-11-23 09:29")) is True, \
            "selection must happen before the open"
        assert app._option_window_active(_at("2026-11-23 16:16")) is True, \
            "late prints would be missed"

    def test_reconcile_is_scheduled_after_the_late_prints(self):
        cron = SchedulerApp._DEFAULT_OPTION["reconcile_cron"].split()
        minute, hour = int(cron[0]), int(cron[1])
        assert (hour, minute) > (16, 15), cron
        assert cron[4] == "1-5", "weekdays only"

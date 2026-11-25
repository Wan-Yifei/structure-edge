"""The calendar is checked against OTHER people's calendars, not its own.

Two independent rulers are available in this environment and both are used,
because a holiday table that is wrong is wrong silently -- the collector
simply runs on a closed day again, which is the whole failure this module
was written to end.

* `pandas.tseries.holiday.USFederalHolidayCalendar` for the fixed and
  nth-weekday holidays. It is not the exchange calendar, and the two
  differences are asserted explicitly rather than papered over: the
  exchange is OPEN on Columbus Day and Veterans Day, and CLOSED on Good
  Friday, which is not a federal holiday at all.
* `dateutil.easter` for Good Friday, the one moving holiday and therefore
  the one the federal calendar cannot supply.

Both are test-only dependencies. Production computes the rules itself so it
does not inherit a third-party calendar's own staleness.
"""
from __future__ import annotations

from datetime import date, time, timedelta

import pytest

from core.market_calendar import (
    CLOSE_FULL_ET, CLOSE_HALF_ET, MIN_MODELLED_YEAR, easter, good_friday,
    holidays, is_half_day, is_market_holiday, is_trading_day,
    market_close_et, thanksgiving)

YEARS = range(MIN_MODELLED_YEAR, 2031)

# Federal but not exchange holidays: the market trades these.
_FEDERAL_ONLY = {"Columbus Day", "Veterans Day"}


def _federal_holidays(year: int) -> set[date]:
    from pandas.tseries.holiday import USFederalHolidayCalendar
    cal = USFederalHolidayCalendar()
    # Ask by name so the two the exchange ignores can be removed by name
    # rather than by date -- a date-based exclusion would silently also drop
    # a real closure that happened to collide.
    keep = [r for r in cal.rules if r.name not in _FEDERAL_ONLY]
    cal.rules = keep
    stamps = cal.holidays(start=f"{year}-01-01", end=f"{year}-12-31")
    return {s.date() for s in stamps}


class TestEasterAgainstDateutil:

    @pytest.mark.parametrize("year", list(YEARS))
    def test_easter_matches_dateutil(self, year):
        from dateutil.easter import easter as ruler
        assert easter(year) == ruler(year), year

    @pytest.mark.parametrize("year", list(YEARS))
    def test_good_friday_is_the_friday_before_easter(self, year):
        gf = good_friday(year)
        assert gf.weekday() == 4, (year, gf)
        assert (easter(year) - gf).days == 2


class TestAgainstTheFederalCalendar:
    """The exchange set differs from the federal set in exactly two ways."""

    @pytest.mark.parametrize("year", list(YEARS))
    def test_exchange_set_is_federal_minus_two_plus_good_friday(self, year):
        mine = set(holidays(year))
        theirs = _federal_holidays(year)
        # The federal rule rolls a Saturday New Year's Day back to the
        # Friday before, which lands in the PREVIOUS year -- so the federal
        # set for 2027 contains 2027-12-31 for the 2028 New Year. The
        # exchange does not observe it (see _observed), and this is the one
        # documented divergence, named here rather than absorbed into a
        # loose comparison.
        rollback = date(year, 12, 31)
        if date(year + 1, 1, 1).weekday() == 5:
            assert rollback in theirs, "fixture no longer shows the roll-back"
            theirs = theirs - {rollback}
            assert rollback not in mine
        assert mine == theirs | {good_friday(year)}, {
            "only mine": sorted(mine - theirs - {good_friday(year)}),
            "only theirs": sorted(theirs - mine),
        }

    @pytest.mark.parametrize("year", list(YEARS))
    def test_the_market_trades_columbus_day_and_veterans_day(self, year):
        from pandas.tseries.holiday import USFederalHolidayCalendar
        cal = USFederalHolidayCalendar()
        cal.rules = [r for r in cal.rules if r.name in _FEDERAL_ONLY]
        for stamp in cal.holidays(start=f"{year}-01-01", end=f"{year}-12-31"):
            d = stamp.date()
            assert not is_market_holiday(d), d
            if d.weekday() < 5:
                assert is_trading_day(d), d


class TestTheDayThisModuleWasWrittenFor:
    """2026-11-26. Before this module both session guards tested only
    `weekday() >= 5`, so the collector would have run a full session."""

    def test_thanksgiving_2026_is_a_full_closure(self):
        d = date(2026, 11, 26)
        assert thanksgiving(2026) == d
        assert is_market_holiday(d)
        assert not is_trading_day(d)

    def test_black_friday_2026_is_a_half_day_not_a_closure(self):
        d = date(2026, 11, 27)
        assert is_trading_day(d)
        assert is_half_day(d)
        assert market_close_et(d) == CLOSE_HALF_ET

    def test_the_wednesday_before_is_an_ordinary_full_session(self):
        d = date(2026, 11, 25)
        assert is_trading_day(d)
        assert not is_half_day(d)
        assert market_close_et(d) == CLOSE_FULL_ET


class TestObservanceRolls:
    """Saturday rolls back to Friday and Sunday forward to Monday -- except
    New Year's Day, which does not roll back at all."""

    def test_a_saturday_new_year_is_not_observed_on_the_friday_before(self):
        # 2022-01-01 was a Saturday. The federal calendar observes
        # 2021-12-31; the NYSE traded a full session that day.
        assert date(2021, 12, 31) not in holidays(2021)
        assert is_trading_day(date(2021, 12, 31))

    def test_a_saturday_christmas_does_roll_back(self):
        # 2021-12-25 was a Saturday and 2021-12-24 was closed -- so the
        # New Year exception really is specific to New Year, and not a
        # general "no rolling back" rule.
        assert date(2021, 12, 24) in holidays(2021)
        assert not is_trading_day(date(2021, 12, 24))

    def test_a_sunday_new_year_is_observed_on_the_monday(self):
        assert date(2023, 1, 2) in holidays(2023)       # 2023-01-01 was a Sun

    @pytest.mark.parametrize("year", list(YEARS))
    def test_no_holiday_ever_lands_on_a_weekend(self, year):
        for d in holidays(year):
            assert d.weekday() < 5, (year, d)

    @pytest.mark.parametrize("year", list(YEARS))
    def test_every_holiday_falls_inside_its_own_year(self, year):
        for d in holidays(year):
            assert d.year == year, (year, d)


class TestHalfDays:

    def test_july_third_is_an_early_close_when_it_is_a_session(self):
        # 2025-07-03 was a Thursday and the Fourth a Friday.
        assert is_trading_day(date(2025, 7, 3))
        assert is_half_day(date(2025, 7, 3))

    def test_july_third_is_a_closure_not_an_early_close_in_2026(self):
        """July 4th 2026 is a Saturday, so the Third is the OBSERVED
        holiday. Reading it as a 13:00 session would have the collector
        running on a day with no tape, which is this module's whole
        subject."""
        d = date(2026, 7, 3)
        assert is_market_holiday(d)
        assert not is_trading_day(d)
        assert not is_half_day(d), "a closure is not an early close"
        # The close time still answers, for time-to-expiry maths.
        assert market_close_et(d) == CLOSE_FULL_ET

    def test_christmas_eve_is_an_early_close_when_it_is_a_session(self):
        # 2025-12-24 was a Wednesday.
        assert is_trading_day(date(2025, 12, 24))
        assert is_half_day(date(2025, 12, 24))

    @pytest.mark.parametrize("year", list(YEARS))
    def test_a_half_day_is_always_a_trading_day(self, year):
        d = date(year, 1, 1)
        while d.year == year:
            if is_half_day(d):
                assert is_trading_day(d), d
            d += timedelta(days=1)

    @pytest.mark.parametrize("year", list(YEARS))
    def test_close_is_one_of_the_two_times_and_follows_is_half_day(self, year):
        d = date(year, 1, 1)
        got = set()
        while d.year == year:
            close = market_close_et(d)
            assert close in (CLOSE_FULL_ET, CLOSE_HALF_ET), (d, close)
            assert (close == CLOSE_HALF_ET) == is_half_day(d), (d, close)
            if is_half_day(d):
                got.add(d)
            d += timedelta(days=1)

        # The exact set, derived from the rule rather than read off the
        # implementation. July 3rd is an early close precisely when the
        # Fourth falls Tue-Fri: a Saturday Fourth makes the Third the
        # OBSERVED holiday, and a Monday or Sunday Fourth leaves the Third
        # on a weekend. Christmas Eve is the same statement about the 25th.
        want = {thanksgiving(year) + timedelta(days=1)}
        if date(year, 7, 4).weekday() in (1, 2, 3, 4):
            want.add(date(year, 7, 3))
        if date(year, 12, 25).weekday() in (1, 2, 3, 4):
            want.add(date(year, 12, 24))
        assert got == want, {"extra": sorted(got - want),
                             "missing": sorted(want - got)}


class TestWeekendsAreNotHolidays:
    """Kept apart on purpose, so a log line can say which reason applies."""

    def test_a_saturday_is_not_a_market_holiday_but_is_not_a_trading_day(self):
        d = date(2026, 11, 28)
        assert d.weekday() == 5
        assert not is_market_holiday(d)
        assert not is_trading_day(d)


class TestTradingDayCountIsSane:
    """A year has 250 +/- 3 sessions. This is the one assertion that would
    survive an implementation rewrite, and it catches a rule that fires far
    too often or not at all."""

    @pytest.mark.parametrize("year", list(YEARS))
    def test_sessions_per_year(self, year):
        d = date(year, 1, 1)
        n = 0
        while d.year == year:
            n += is_trading_day(d)
            d += timedelta(days=1)
        assert 248 <= n <= 254, (year, n)

    @pytest.mark.parametrize("year", list(YEARS))
    def test_closure_count_is_ten_except_for_a_saturday_new_year(self, year):
        """Nine fixed plus Good Friday -- minus one in a year whose January
        1st is a Saturday, because that one is not observed at all and does
        not roll back into December. 2022 and 2028 are such years.

        Stated as an equation rather than a range: a range wide enough to
        admit both counts would also admit a rule that stopped firing.
        """
        want = 10 - (date(year, 1, 1).weekday() == 5)
        assert len(holidays(year)) == want, sorted(holidays(year))


class TestAnnouncedClosures:
    """`UNSCHEDULED_CLOSURES` was an empty set, which made it dead code and
    made one real date wrong. A mutation that deleted the branch entirely
    left all 100 tests green."""

    def test_the_carter_day_of_mourning_is_a_closure(self):
        from core.market_calendar import UNSCHEDULED_CLOSURES
        d = date(2025, 1, 9)
        assert d in UNSCHEDULED_CLOSURES
        assert is_market_holiday(d)
        assert not is_trading_day(d)

    def test_it_is_not_reachable_by_any_rule(self):
        """The point of the set: a day of mourning is announced, not
        derived, so no amount of rule-fixing would have caught it. This
        also pins that `holidays()` stays rule-only."""
        assert date(2025, 1, 9) not in holidays(2025)

    def test_the_days_around_it_still_trade(self):
        """The control. A closure that swallowed its neighbours would
        satisfy the test above just as well."""
        assert is_trading_day(date(2025, 1, 8))
        assert is_trading_day(date(2025, 1, 10))

    def test_the_rule_count_is_unchanged_by_it(self):
        """holidays(2025) is the ten rule-derived closures; the announced
        one is counted separately, so adding it must not have moved that."""
        assert len(holidays(2025)) == 10

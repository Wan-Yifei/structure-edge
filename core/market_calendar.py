"""US equity market calendar: full closures and early closes.

Why this exists, and what it cost not to have it: nothing in the collection
chain knew that a full closure is different from a weekend. Both session
guards tested `weekday() >= 5` and nothing else, so on a holiday the option
collector would start at 09:15, spend chain and snapshot calls against an
account-wide rate limit, raise its "NO PRINTS in 3 minutes" alarm against a
market that is not trading, write a session's worth of greeks rows carrying
the *previous* day's spot, and finish with a 16:50 reconcile reporting every
subscribed contract in `caught_nothing` -- the one detector for a dropped
subscription, spending its credibility on a day the exchange was shut.

The rules are COMPUTED, not tabulated, so the module does not expire: a
hardcoded list of dates is right until the year it silently is not, and the
failure is invisible (the collector just runs on a closed day again).

Scope and limits, stated rather than implied:

* NYSE / Nasdaq regular-session rules for US equities and equity options.
* Models the CURRENT rule set, so it is not valid before 2022 -- Juneteenth
  became an exchange holiday in 2022, and there is no reason to model its
  absence for a forward-looking scheduler. `MIN_MODELLED_YEAR` states that
  floor. NOTHING ASSERTS ON IT: ask about 2021 and `holidays` answers with a
  straight face, one Juneteenth too many. Checked against a real exchange
  calendar, 2018-06-19, 2019-06-19, 2020-06-19 and 2021-06-18 are all days
  this module calls open and the exchange did not.
* Unscheduled closures (weather, a national day of mourning, 9/11) cannot
  be derived -- they are announced -- so they live in
  `UNSCHEDULED_CLOSURES` and must be added when announced. An unlisted one
  fails in the UNSAFE direction: the collector runs on a day with no tape,
  which is precisely the behaviour this module exists to end. An earlier
  version of this note called that "the safe direction", which is
  backwards.
"""
from __future__ import annotations

from datetime import date, time, timedelta

__all__ = [
    "MIN_MODELLED_YEAR",
    "UNSCHEDULED_CLOSURES",
    "CLOSE_FULL_ET",
    "CLOSE_HALF_ET",
    "easter",
    "good_friday",
    "thanksgiving",
    "holidays",
    "is_market_holiday",
    "is_half_day",
    "is_trading_day",
    "market_close_et",
]

MIN_MODELLED_YEAR = 2022        # Juneteenth joined the exchange calendar here

CLOSE_FULL_ET = time(16, 0)
CLOSE_HALF_ET = time(13, 0)

#: Closures that are announced rather than derived. There is exactly one
#: inside the modelled range, and leaving it out made this branch dead code
#: as well as wrong: a national day of mourning is not derivable from any
#: rule, so the set is the only place it can live.
UNSCHEDULED_CLOSURES: frozenset[date] = frozenset({
    # Day of mourning for Jimmy Carter. NYSE and Nasdaq shut for the whole
    # session. Anything that reads this calendar for a date in January 2025
    # -- a backtest, a gex run over that month -- was counting it as a
    # trading day until this line existed.
    date(2025, 1, 9),
})


def easter(year: int) -> date:
    """Western (Gregorian) Easter Sunday -- the anonymous Gregorian algorithm.

    Good Friday is the only moving exchange holiday, and it is the only one
    the federal holiday calendar does not contain, so it cannot be borrowed
    from there. Checked against dateutil.easter in the tests rather than
    against itself.
    """
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    ll = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ll) // 451
    month, day = divmod(h + ll - 7 * m + 114, 31)
    return date(year, month, day + 1)


def good_friday(year: int) -> date:
    return easter(year) - timedelta(days=2)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """The n-th `weekday` (Mon=0) of a month, 1-based."""
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    """The last `weekday` (Mon=0) of a month."""
    nxt = date(year + (month == 12), month % 12 + 1, 1)
    last = nxt - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def thanksgiving(year: int) -> date:
    """Fourth Thursday of November."""
    return _nth_weekday(year, 11, 3, 4)


def _observed(d: date) -> date:
    """NYSE observance: Saturday rolls back to Friday, Sunday forward to Monday.

    New Year's Day is handled by the caller, not here -- it is the one
    holiday that does NOT roll back. Measured: 2022-01-01 was a Saturday and
    the federal calendar observes Friday 2021-12-31, but the NYSE traded a
    full session that day. Christmas on a Saturday does roll back, and
    2021-12-24 was closed, so the exception is New Year's alone.
    """
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def holidays(year: int) -> frozenset[date]:
    """Every full exchange closure observed in `year`.

    A holiday observed in January for a January 1st that fell on a Sunday
    belongs to this year's set; one rolled back from a Saturday January 1st
    would belong to the previous December, and does not happen (see
    `_observed`).
    """
    out: set[date] = set()

    # New Year's Day -- forward only. A Saturday January 1st is not observed
    # at all, and the Friday before it is a full trading session.
    new_year = date(year, 1, 1)
    if new_year.weekday() == 6:
        out.add(date(year, 1, 2))
    elif new_year.weekday() < 5:
        out.add(new_year)

    out.add(_nth_weekday(year, 1, 0, 3))         # MLK Day, 3rd Monday Jan
    out.add(_nth_weekday(year, 2, 0, 3))         # Washington's Birthday
    out.add(good_friday(year))
    out.add(_last_weekday(year, 5, 0))           # Memorial Day
    out.add(_observed(date(year, 6, 19)))        # Juneteenth
    out.add(_observed(date(year, 7, 4)))         # Independence Day
    out.add(_nth_weekday(year, 9, 0, 1))         # Labor Day
    out.add(thanksgiving(year))
    out.add(_observed(date(year, 12, 25)))       # Christmas

    # A December 25th on a Sunday is observed on the 26th, which is still
    # December; a January 1st on a Sunday is observed on the 2nd. Neither
    # crosses a year boundary, so the set is self-contained by construction.
    return frozenset(out)


def is_market_holiday(d: date) -> bool:
    """True when the exchange is shut for the whole day (weekends excluded).

    Weekends are deliberately NOT folded in: a caller that wants "is there a
    session" should ask `is_trading_day`, and keeping the two separate makes
    a log line able to say which of the two reasons applies.
    """
    return d in UNSCHEDULED_CLOSURES or d in holidays(d.year)


def is_trading_day(d: date) -> bool:
    """True when the exchange holds a regular session, half or full."""
    return d.weekday() < 5 and not is_market_holiday(d)


def is_half_day(d: date) -> bool:
    """True when the regular session ends 13:00 ET.

    The three recurring 1pm closes. A date that is not a trading day at all
    is not a half day either -- 2026-07-03 is the observed Independence Day
    (July 4th is a Saturday that year), so it is a full closure and not the
    early close the July 3rd rule would otherwise make it.
    """
    if not is_trading_day(d):
        return False
    if d == thanksgiving(d.year) + timedelta(days=1):
        return True                              # Black Friday
    if (d.month, d.day) == (12, 24):
        return True                              # Christmas Eve
    if (d.month, d.day) == (7, 3):
        return True                              # July 4th eve
    return False


def market_close_et(d: date) -> time:
    """Regular-session close for `d`: 13:00 on a half day, else 16:00.

    Answers for a date that is not a trading day too, returning the full-day
    close, because the time-to-expiry maths wants a clock and an expiry never
    lands on a closure. Callers deciding whether to RUN must ask
    `is_trading_day` -- this function has never been able to answer that, and
    reading a 16:00 out of it for Thanksgiving is how the gap survived.
    """
    return CLOSE_HALF_ET if is_half_day(d) else CLOSE_FULL_ET

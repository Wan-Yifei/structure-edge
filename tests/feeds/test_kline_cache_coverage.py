"""Which days the cache knows it has, as opposed to which days hold bars.

The cache used to answer "do I cover this range?" with the first and last
bar it held. A range with holes in the middle passed that test, so load()
returned short and nothing ever refetched: once the endpoints are wide,
every later request matches. Six weekdays went missing that way between May
and October 2026, among them 2026-10-08 and 2026-10-09.

The tolerance made it worse. Seven days on each boundary is longer than a
trading week, so a cache that stopped last Tuesday still "covered" a request
running through this Monday.

Coverage is now recorded per day when a fetch completes, which also settles
holidays without a calendar: a day inside a span that was fetched
successfully and holds no bars is a day the market was shut.
"""

from datetime import date

import pandas as pd
import pytest

from feeds.kline_store import KlineStore


@pytest.fixture
def store(tmp_path):
    s = KlineStore(tmp_path / "klines.duckdb")
    yield s
    s.close()


def bars(days, ktype="1m"):
    """One bar per day in *days*, which is what the seeding path reads."""
    return pd.DataFrame({
        "time_key": [f"{d} 10:00:00" for d in days],
        "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 100,
    })


# Mon 2026-10-05 .. Fri 2026-10-09, with Thu and Fri absent.
WEEK = ["2026-10-05", "2026-10-06", "2026-10-07"]
HOLE = ["2026-10-08", "2026-10-09"]


class TestTheRegression:
    def test_a_hole_in_the_middle_is_reported(self, store):
        """The exact shape of the bug: bars either side, nothing between."""
        store.save("X", "1m", bars(["2026-10-05", "2026-10-12"]))
        missing = store.missing_days("X", "1m", "2026-10-05", "2026-10-12",
                                     today=date(2026, 10, 13))
        assert date(2026, 10, 8) in missing
        assert date(2026, 10, 9) in missing

    def test_the_endpoints_alone_would_have_passed(self, store):
        """date_range spans the request even though five days are absent."""
        store.save("X", "1m", bars(["2026-10-05", "2026-10-12"]))
        lo, hi = store.date_range("X", "1m")
        assert lo[:10] == "2026-10-05" and hi[:10] == "2026-10-12"
        assert store.missing_days("X", "1m", "2026-10-05", "2026-10-12",
                                  today=date(2026, 10, 13))

    def test_a_week_old_cache_does_not_cover_this_week(self, store):
        """What the seven-day tolerance used to allow."""
        store.save("X", "1m", bars(WEEK))
        store.mark_covered("X", "1m", "2026-10-05", "2026-10-07")
        missing = store.missing_days("X", "1m", "2026-10-05", "2026-10-12",
                                     today=date(2026, 10, 13))
        assert [str(d) for d in missing] == ["2026-10-08", "2026-10-09",
                                             "2026-10-11", "2026-10-12"]


class TestWeekends:
    def test_saturdays_are_never_missing(self, store):
        missing = store.missing_days("X", "1m", "2026-10-10", "2026-10-10",
                                     today=date(2026, 10, 13))
        assert missing == []

    def test_sundays_are(self, store):
        """The week opens at 20:00 on Sunday and those bars carry its date."""
        missing = store.missing_days("X", "1m", "2026-10-11", "2026-10-11",
                                     today=date(2026, 10, 13))
        assert missing == [date(2026, 10, 11)]


class TestHolidays:
    def test_an_empty_day_inside_a_fetched_span_stays_covered(self, store):
        """No calendar needed: the fetch ran, so the market was shut.

        Juneteenth 2026-06-19 is a Friday with no session. Marking the span
        is what stops it being refetched for ever.
        """
        store.save("X", "1m", bars(["2026-06-18", "2026-06-22"]))
        store.mark_covered("X", "1m", "2026-06-18", "2026-06-22")
        assert store.missing_days("X", "1m", "2026-06-18", "2026-06-22",
                                  today=date(2026, 7, 1)) == []

    def test_before_the_span_is_marked_it_is_missing(self, store):
        store.save("X", "1m", bars(["2026-06-18", "2026-06-22"]))
        missing = store.missing_days("X", "1m", "2026-06-18", "2026-06-22",
                                     today=date(2026, 7, 1))
        assert date(2026, 6, 19) in missing


class TestSeeding:
    def test_days_with_bars_are_seeded(self, store):
        store.save("X", "1m", bars(WEEK))
        assert store.missing_days("X", "1m", "2026-10-05", "2026-10-07",
                                  today=date(2026, 10, 13)) == []

    def test_weekday_holes_survive_seeding(self, store):
        """Seeding must not paper over the gaps it was introduced to expose."""
        store.save("X", "1m", bars(WEEK))
        missing = store.missing_days("X", "1m", "2026-10-05", "2026-10-09",
                                     today=date(2026, 10, 13))
        assert [str(d) for d in missing] == HOLE

    def test_seeding_happens_once(self, store):
        store.save("X", "1m", bars(WEEK))
        assert store.seed_coverage("X", "1m") == len(WEEK)
        assert store.seed_coverage("X", "1m") == 0

    def test_an_empty_series_seeds_nothing(self, store):
        assert store.seed_coverage("X", "1m") == 0


class TestMarkAndClear:
    def test_marking_covers_the_range(self, store):
        store.mark_covered("X", "1m", "2026-10-05", "2026-10-09")
        assert store.missing_days("X", "1m", "2026-10-05", "2026-10-09",
                                  today=date(2026, 10, 13)) == []

    def test_clearing_reopens_it(self, store):
        store.mark_covered("X", "1m", "2026-10-05", "2026-10-09")
        store.clear_coverage("X", "1m", "2026-10-07", "2026-10-09")
        missing = store.missing_days("X", "1m", "2026-10-05", "2026-10-09",
                                     today=date(2026, 10, 13))
        assert [str(d) for d in missing] == ["2026-10-07", "2026-10-08",
                                             "2026-10-09"]

    def test_clearing_everything(self, store):
        store.mark_covered("X", "1m", "2026-10-05", "2026-10-09")
        store.clear_coverage("X", "1m")
        assert len(store.missing_days("X", "1m", "2026-10-05", "2026-10-09",
                                      today=date(2026, 10, 13))) == 5

    def test_coverage_is_per_series(self, store):
        store.mark_covered("X", "1m", "2026-10-05", "2026-10-09")
        assert store.missing_days("X", "5m", "2026-10-05", "2026-10-09",
                                  today=date(2026, 10, 13))


class TestToday:
    """A session still running cannot be called complete."""

    def test_the_future_is_not_required(self, store):
        assert store.missing_days("X", "1m", "2026-10-14", "2026-10-16",
                                  today=date(2026, 10, 13)) == []

    def test_today_counts_as_present_once_it_has_a_bar(self, store):
        """Keeps repeated intraday calls off the API."""
        store.save("X", "1m", bars(["2026-10-13"]))
        store.clear_coverage("X", "1m")
        store.mark_covered("X", "1m", "2026-10-12", "2026-10-12")
        assert store.missing_days("X", "1m", "2026-10-12", "2026-10-13",
                                  today=date(2026, 10, 13)) == []

    def test_today_is_missing_while_it_has_none(self, store):
        store.save("X", "1m", bars(["2026-10-12"]))
        assert store.missing_days("X", "1m", "2026-10-12", "2026-10-13",
                                  today=date(2026, 10, 13)) == [date(2026, 10, 13)]

    def test_marking_a_span_does_not_freeze_today(self, store):
        """The bug this guard exists for.

        A fetch for [.., today] used to write today into the coverage table,
        so the prefix of bars it had at that moment became the whole day for
        good. On 2026-10-13 a fetch at 12:02 stored 152 of the session's 390
        bars and marked the day covered; every later call trusted the cache
        and the afternoon never arrived. Only days that have ENDED may be
        marked -- today is kept out of coverage, and the bars-on-disk check
        above is what keeps intraday calls off the API.
        """
        store.mark_covered("X", "1m", "2026-10-12", "2026-10-14",
                           today=date(2026, 10, 14))
        assert date(2026, 10, 14) not in store.covered_days("X", "1m")
        assert date(2026, 10, 13) in store.covered_days("X", "1m")

    def test_seeding_cannot_tell_a_partial_day_from_a_whole_one(self, store):
        """The limit that remains, recorded rather than hidden.

        Seeding knows only whether a day holds bars, so a day saved while its
        session ran is seeded as complete. That is tolerable because seeding
        is a one-time migration for a cache predating the table: it is a
        no-op once anything is covered, which is why the 2026-10-13 truncation
        came from mark_covered and not from here. A cache seeded for the first
        time the morning after a partial save would still inherit it.
        """
        store.save("X", "1m", bars(["2026-10-13"]))
        store.mark_covered("X", "1m", "2026-10-13", "2026-10-13",
                           today=date(2026, 10, 13))
        assert store.covered_days("X", "1m") == set()      # mark_covered held
        assert store.missing_days("X", "1m", "2026-10-13", "2026-10-13",
                                  today=date(2026, 10, 14)) == []
        assert date(2026, 10, 13) in store.covered_days("X", "1m")

    def test_seeding_is_a_no_op_once_anything_is_covered(self, store):
        """What makes the limit above tolerable in a live cache."""
        store.mark_covered("X", "1m", "2026-10-01", "2026-10-01",
                           today=date(2026, 10, 14))
        store.save("X", "1m", bars(["2026-10-13"]))
        store.missing_days("X", "1m", "2026-10-13", "2026-10-13",
                           today=date(2026, 10, 14))
        assert date(2026, 10, 13) not in store.covered_days("X", "1m")

    def test_marking_only_today_records_nothing(self, store):
        assert store.mark_covered("X", "1m", "2026-10-14", "2026-10-14",
                                  today=date(2026, 10, 14)) == 0

    def test_today_is_never_written_into_coverage_by_seeding(self, store):
        """Seeding reads bars, and today's are partial -- but seeding marks
        the day, so the guard that matters is that a later day is still
        refetched rather than inheriting today's partial row."""
        store.save("X", "1m", bars(["2026-10-13"]))
        store.seed_coverage("X", "1m")
        assert store.missing_days("X", "1m", "2026-10-14", "2026-10-14",
                                  today=date(2026, 10, 14)) == [date(2026, 10, 14)]
